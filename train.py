"""Train a FontStyleViT-conditioned GAN for 256x256 ASCII glyph generation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import string
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from dataset.font_mimic_dataset import FontMimicDataset, ID_TO_LETTER
from model import ConditionalDiscriminator, ConditionalGenerator
from model.parseq_loss import (
    make_parseq_lines,
    load_frozen_parseq,
    parseq_logits_loss,
)
from utils.rand_util import seed_everything
from utils.train_util import (
    choose_device,
    load_frozen_style_encoder,
    optimizer_from_config,
    scheduler_from_config,
    create_dataloader,
    generated_grid_style_view,
    bboxes_to_square_positions,
    reconstruct_generated_crops,
    move_style_view,
    update_ema,
)
from utils.loss_util import (
    foreground_l1,
    gradient_edge_loss,
    create_perceptual_loss,
    style_embedding_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--resume", default=None, help="Override train.resume")
    parser.add_argument("--device", default=None, help="Override train.device")
    return parser.parse_args()


def unpack_generator_output(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    predict_position: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize ConditionalGenerator's optional position output."""
    if predict_position:
        images, positions = output
        return images, positions
    return output, None


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def parse_grid_columns_range(value: Any) -> tuple[int, int]:
    """Parse an inclusive grid-column range, retaining scalar compatibility."""
    if isinstance(value, int):
        minimum = maximum = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        minimum, maximum = value
    else:
        raise ValueError("grid_columns must be a positive integer or [min, max]")
    return minimum, maximum


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    ema: nn.Module,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    scaler,
    config: Mapping[str, Any],
    style_encoder_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_ema": ema.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict() if scheduler_g else None,
            "scheduler_d": scheduler_d.state_dict() if scheduler_d else None,
            "scaler": scaler.state_dict(),
            "config": dict(config),
            "style_encoder_checkpoint": style_encoder_checkpoint,
            "letters": string.ascii_letters,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    generator: nn.Module,
    discriminator: nn.Module,
    ema: nn.Module,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    scaler,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    ema.load_state_dict(checkpoint.get("generator_ema", checkpoint["generator"]))
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if scheduler_g and checkpoint.get("scheduler_g"):
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    if scheduler_d and checkpoint.get("scheduler_d"):
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def train(
    config: Dict[str, Any],
    resume_override: Optional[str] = None,
    device_override: Optional[str] = None,
) -> None:
    train_config = config["train"]
    seed = int(train_config.get("seed", 42))
    seed_everything(seed)
    device = choose_device(device_override or str(train_config.get("device", "cuda")))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_config.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(train_config.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train_config.get("allow_tf32", True))

    dataloader = create_dataloader(config, seed)
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader has zero batches")
    dataset: FontMimicDataset = dataloader.dataset
    model_config = config["model"]
    style_encoder, encoder_path, encoder_state_key = load_frozen_style_encoder(
        model_config["style_encoder"], device
    )

    generator_options = copy.deepcopy(model_config["generator"])
    style_dim = int(
        generator_options.setdefault("style_embedding_dim", style_encoder.backbone.embed_dim)
    )
    if style_dim != style_encoder.backbone.embed_dim:
        raise ValueError(
            f"Generator style_embedding_dim={style_dim} != "
            f"FontStyleViT style_dim={style_encoder.backbone.embed_dim}"
        )
    generator = ConditionalGenerator(**generator_options).to(device)
    predict_position = generator.pos_predictor is not None
    generator_ema = copy.deepcopy(generator).eval().requires_grad_(False)

    discriminator_options = copy.deepcopy(model_config["discriminator"])
    discriminator = ConditionalDiscriminator(**discriminator_options).to(device)

    parseq_options = copy.deepcopy(model_config.get("parseq", {}))
    characters_per_line = int(parseq_options.pop("characters_per_line", 4))
    parseq_loss_type = str(parseq_options.pop("logits_loss", "kl"))
    parseq_temperature = float(parseq_options.pop("temperature", 1.0))
    parseq, parseq_image_size, parseq_max_length = load_frozen_parseq(parseq_options, device)

    optimizer_g = optimizer_from_config(
        generator.parameters(), config["optimizer"]["generator"]
    )
    optimizer_d = optimizer_from_config(
        discriminator.parameters(), config["optimizer"]["discriminator"]
    )
    scheduler_g, scheduler_g_interval = scheduler_from_config(
        optimizer_g, config.get("scheduler", {}).get("generator")
    )
    scheduler_d, scheduler_d_interval = scheduler_from_config(
        optimizer_d, config.get("scheduler", {}).get("discriminator")
    )

    amp_enabled = bool(train_config.get("amp", True) and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if str(train_config.get("amp_dtype", "float16")).lower() == "bfloat16"
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    output_dir = Path(str(train_config.get("output_dir", "outputs/font_cgan")))
    checkpoint_dir, sample_dir = output_dir / "checkpoints", output_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (output_dir / "font_index.json").open("w", encoding="utf-8") as file:
        json.dump(
            {Path(font).name: index for index, font in enumerate(dataset.fonts)},
            file,
            indent=2,
            ensure_ascii=False,
        )

    start_epoch, global_step = 0, 0
    resume = resume_override or train_config.get("resume")
    if resume:
        start_epoch, global_step = load_checkpoint(
            str(resume),
            device,
            generator,
            discriminator,
            generator_ema,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            scaler,
        )
    writer = SummaryWriter(
        output_dir / "tensorboard", purge_step=global_step if resume else None
    )

    losses = config.get("losses", {})
    d_adv_weight = float(losses.get("d_adversarial_weight", 1.0))
    g_adv_weight = float(losses.get("g_adversarial_weight", 1.0))
    g_letter_weight = float(losses.get("g_letter_weight", 1.0))
    pixel_weight = float(losses.get("pixel_weight", 10.0))
    foreground_weight = float(losses.get("foreground_weight", 5.0))
    edge_weight = float(losses.get("edge_weight", 0.0))
    edge_operator = str(losses.get("edge_operator", "scharr")).lower()
    style_weight = float(losses.get("style_weight", 5.0))
    style_loss_type = str(losses.get("style_type", "cosine"))
    position_weight = float(losses.get("position_weight", 1.0))
    perceptual_content_weight = float(losses.get("perceptual_content_weight", 0.0))
    perceptual_style_weight = float(losses.get("perceptual_style_weight", 0.0))

    perceptual_criterion = None
    if perceptual_content_weight > 0.0 or perceptual_style_weight > 0.0:
        perceptual_criterion = create_perceptual_loss(
            losses, perceptual_content_weight, perceptual_style_weight
        )

    patch_size = style_encoder.patch_size
    grid_mean = dataset.reference_dataset.mean
    grid_std = dataset.reference_dataset.std
    grid_max_side = dataset.reference_dataset.max_long_side
    grid_max_pixels = dataset.reference_dataset.max_pixels
    grid_columns_range = parse_grid_columns_range(config.get("grid_columns", 4))
    grid_columns_rng = random.Random(seed)

    epochs = int(train_config.get("epochs", 200))
    d_every = int(train_config.get("d_every", 1))
    ema_decay = float(train_config.get("ema_decay", 0.999))
    grad_clip = train_config.get("gradient_clip_norm")
    log_every = int(train_config.get("log_every", 20))
    sample_every = int(train_config.get("sample_every", 500))
    sample_font_count = int(train_config.get("sample_font_count", 4))
    checkpoint_every = int(train_config.get("checkpoint_every_epochs", 5))
    non_blocking = bool(config.get("loader", {}).get("pin_memory", False))

    print(
        f"Training on {device}: {len(dataset.fonts)} fonts, "
        f"{dataset.letters_per_font} glyphs per font and batch item; "
        f"PARSeq={characters_per_line} characters per line",
        flush=True,
    )
    try:
        for epoch in range(start_epoch, epochs):
            generator.train()
            discriminator.train()
            style_encoder.eval()
            parseq.eval()
            epoch_start = time.perf_counter()
            sums = {
                "d": 0.0,
                "g": 0.0,
                "style": 0.0,
                "pixel": 0.0,
                "edge": 0.0,
                "perc_content": 0.0,
                "perc_style": 0.0,
                "letter": 0.0,
                "position": 0.0,
            }
            font_count = generator_font_count = 0

            for batch_index, batch in enumerate(dataloader):
                grid_columns = grid_columns_rng.randint(*grid_columns_range)
                target_key = "target_crops" if predict_position else "target_images"
                target_groups = batch[target_key].to(device, non_blocking=non_blocking)
                letter_groups = batch["letter_ids"].to(device, non_blocking=non_blocking)
                font_batch, letters_per_font = letter_groups.shape
                real, letter_ids = target_groups.flatten(0, 1), letter_groups.flatten()
                position_targets = None
                if predict_position:
                    position_targets = bboxes_to_square_positions(
                        batch["target_bboxes"].to(device, non_blocking=non_blocking)
                    ).flatten(0, 1)
                real_for_style = real
                if position_targets is not None:
                    real_for_style = reconstruct_generated_crops(
                        real, position_targets, dataset.letter_image_size
                    )
                real_for_style = real_for_style.reshape(
                    font_batch, letters_per_font, *real_for_style.shape[1:]
                )
                reference_view = move_style_view(batch["style_view"], device, non_blocking)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    input_cls, input_style = style_encoder(reference_view)
                    real_grid_view = generated_grid_style_view(
                        real_for_style,
                        patch_size,
                        grid_mean,
                        grid_std,
                        grid_max_side,
                        grid_max_pixels,
                        grid_columns,
                    )
                    real_style_outputs = style_encoder.encode_view(real_grid_view)
                    real_style = real_style_outputs["style_embedding"]
                    real_patch_features = real_style_outputs["patch_features"]
                    real_parseq_lines = make_parseq_lines(
                        real_for_style, characters_per_line, parseq_image_size
                    )
                    real_letter_logits = parseq(
                        real_parseq_lines, max_length=characters_per_line
                    ).detach()
                flat_style = input_cls.repeat_interleave(letters_per_font, dim=0)

                d_updated = global_step % d_every == 0
                nan = torch.tensor(float("nan"), device=device)
                d_loss = d_real = d_fake = real_score = fake_score = nan
                if d_updated:
                    set_requires_grad(discriminator, True)
                    optimizer_d.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        noise = torch.randn(real.shape[0], generator.latent_dim, device=device)
                        with torch.no_grad():
                            fake_for_d, _ = unpack_generator_output(
                                generator(noise, flat_style, letter_ids),
                                predict_position,
                            )
                        real_score = discriminator(real)
                        fake_score = discriminator(fake_for_d)
                        d_real = F.relu(1.0 - real_score).mean()
                        d_fake = F.relu(1.0 + fake_score).mean()
                        d_loss = d_adv_weight * (d_real + d_fake)
                    if not torch.isfinite(d_loss):
                        raise FloatingPointError(
                            f"Non-finite D loss at epoch={epoch}, batch={batch_index}"
                        )
                    scaler.scale(d_loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer_d)
                        d_grad_norm = nn.utils.clip_grad_norm_(
                            discriminator.parameters(), float(grad_clip)
                        )
                    else:
                        d_grad_norm = torch.tensor(float("nan"))
                    scaler.step(optimizer_d)

                set_requires_grad(discriminator, False)
                optimizer_g.zero_grad(set_to_none=True)
                with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    noise = torch.randn(real.shape[0], generator.latent_dim, device=device)
                    fake, predicted_positions = unpack_generator_output(
                        generator(noise, flat_style, letter_ids),
                        predict_position,
                    )
                    fake_score_g = discriminator(fake)
                    g_adv = -fake_score_g.mean()
                    g_position = fake.new_zeros(())
                    if predict_position:
                        g_position = F.smooth_l1_loss(predicted_positions.float(), position_targets.float())
                    g_pixel = foreground_l1(fake, real, foreground_weight)
                    g_edge = gradient_edge_loss(fake, real, edge_operator)

                    generated_for_style = fake
                    if predicted_positions is not None:
                        generated_for_style = reconstruct_generated_crops(
                            fake, predicted_positions, dataset.letter_image_size
                        )
                    generated_for_style = generated_for_style.reshape(
                        font_batch, letters_per_font, *generated_for_style.shape[1:],
                    )
                    generated_parseq_lines = make_parseq_lines(
                        generated_for_style, characters_per_line, parseq_image_size,
                    )
                    # Match the generated line's OCR distribution to the real
                    # line's detached distribution. PARSeq remains frozen while
                    # gradients flow through its generated-image input.
                    generated_letter_logits = parseq(
                        generated_parseq_lines, max_length=characters_per_line
                    )
                    g_letter = parseq_logits_loss(
                        generated_letter_logits, real_letter_logits, parseq_loss_type, parseq_temperature,
                    )
                    generated_view = generated_grid_style_view(
                        generated_for_style,
                        patch_size,
                        grid_mean,
                        grid_std,
                        grid_max_side,
                        grid_max_pixels,
                        grid_columns,
                    )
                    generated_style_outputs = style_encoder.encode_view(generated_view)
                    generated_style = generated_style_outputs["style_embedding"]
                    g_perc_content, g_perc_style = fake.new_zeros(()), fake.new_zeros(())
                    if perceptual_criterion is not None:
                        g_perc_content, g_perc_style = perceptual_criterion(
                            generated_style_outputs["patch_features"],
                            real_patch_features,
                            generated_view["patch_padding_mask"],
                        )
                    g_style = style_embedding_loss(
                        generated_style, input_style.detach(), style_loss_type
                    ) * 0.5 + style_embedding_loss(
                        generated_style, real_style.detach(), style_loss_type
                    ) * 0.5

                    g_loss = (
                            g_adv_weight * g_adv
                            + g_letter_weight * g_letter
                            + pixel_weight * g_pixel
                            + edge_weight * g_edge
                            + perceptual_content_weight * g_perc_content
                            + perceptual_style_weight * g_perc_style
                            + style_weight * g_style
                            + position_weight * g_position
                    )
                if not torch.isfinite(g_loss):
                    raise FloatingPointError(
                        f"Non-finite G loss at epoch={epoch}, batch={batch_index}"
                    )
                scaler.scale(g_loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer_g)
                    g_grad_norm = nn.utils.clip_grad_norm_(
                        generator.parameters(), float(grad_clip)
                    )
                else:
                    g_grad_norm = torch.tensor(float("nan"))
                previous_scale = scaler.get_scale()
                scaler.step(optimizer_g)

                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    update_ema(generator_ema, generator, ema_decay)

                if scheduler_d and scheduler_d_interval == "step" and d_updated:
                    scheduler_d.step()
                if scheduler_g and scheduler_g_interval == "step":
                    scheduler_g.step()

                if d_updated:
                    font_count += font_batch
                    sums["d"] += float(d_loss.detach()) * font_batch
                generator_font_count += font_batch
                sums["g"] += float(g_loss.detach()) * font_batch
                sums["style"] += float(g_style.detach()) * font_batch
                sums["pixel"] += float(g_pixel.detach()) * font_batch
                sums["edge"] += float(g_edge.detach()) * font_batch
                sums["perc_content"] += float(g_perc_content.detach()) * font_batch
                sums["perc_style"] += float(g_perc_style.detach()) * font_batch
                sums["letter"] += float(g_letter.detach()) * font_batch
                if predict_position:
                    sums["position"] += float(g_position.detach()) * font_batch

                if global_step % log_every == 0:
                    scalars = {
                            "loss/generator": g_loss,
                            "loss/g_adversarial": g_adv,
                            "loss/g_letter": g_letter,
                            "loss/g_pixel_foreground_l1": g_pixel,
                            f"loss/g_edge_{edge_operator}": g_edge,
                            "loss/g_perc_content": g_perc_content,
                            "loss/g_perc_style": g_perc_style,
                            "loss/g_style_embedding": g_style,
                        }
                    if predict_position:
                        scalars["loss/g_position"] = g_position
                    if d_updated:
                        scalars.update(
                            {
                                "loss/discriminator": d_loss,
                                "loss/d_real_hinge": d_real,
                                "loss/d_fake_hinge": d_fake,
                                "score/real": real_score.mean(),
                                "score/fake": fake_score.mean(),
                            }
                        )
                    for name, value in scalars.items():
                        writer.add_scalar(name, float(value), global_step)
                    writer.add_scalar(
                        "lr/generator", optimizer_g.param_groups[0]["lr"], global_step,
                    )
                    writer.add_scalar(
                        "lr/discriminator", optimizer_d.param_groups[0]["lr"], global_step,
                    )
                    if grad_clip is not None:
                        if d_updated:
                            writer.add_scalar(
                                "gradient_norm/discriminator", float(d_grad_norm), global_step,
                            )
                        writer.add_scalar(
                            "gradient_norm/generator", float(g_grad_norm), global_step,
                        )
                    print(
                        f"[E {epoch + 1:04d}/{epochs:04d}] [B {batch_index + 1:04d}/{len(dataloader):04d}] "
                        f"D={float(d_loss):.4f} G={float(g_loss):.4f} "
                        f"style={float(g_style):.4f} edge={float(g_edge):.4f} "
                        f"perc_content={float(g_perc_content):.4f} perc_style={float(g_perc_style):.4f} "
                        f"letter={float(g_letter):.4f} position={float(g_position):.4f}",
                        flush=True,
                    )

                if global_step % sample_every == 0:
                    current = min(sample_font_count, font_batch)
                    targets = target_groups[:current]
                    target_ids = letter_groups[:current].flatten()
                    sample_cls = input_cls[:current].repeat_interleave(letters_per_font, dim=0)
                    with torch.no_grad(), torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        sample_noise = torch.randn(
                            current * letters_per_font,
                            generator.latent_dim,
                            device=device,
                        )
                        sample_images, sample_positions = unpack_generator_output(
                            generator_ema(sample_noise, sample_cls, target_ids),
                            predict_position,
                        )
                        samples = sample_images.reshape_as(targets)
                        comparison_parts = [targets, samples]
                        if predict_position:
                            reconstructed = reconstruct_generated_crops(
                                sample_images,
                                sample_positions,
                                dataset.letter_image_size,
                            )
                            reconstructed = F.interpolate(
                                reconstructed, size=targets.shape[3:], mode="bilinear", align_corners=False,
                            ).reshape_as(targets)
                            comparison_parts.append(reconstructed)
                    comparisons = torch.cat(comparison_parts, dim=-2)
                    output_name = [ID_TO_LETTER[t.data.item()] for t in target_ids]
                    output_name = ''.join(output_name)
                    save_image(
                        comparisons.flatten(0, 1),
                        sample_dir / f"step_{global_step:08d}_{output_name}.jpg",
                        nrow=letters_per_font,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                global_step += 1

            if scheduler_d and scheduler_d_interval == "epoch":
                scheduler_d.step()
            if scheduler_g and scheduler_g_interval == "epoch":
                scheduler_g.step()
            writer.add_scalar("epoch/loss_d", sums["d"] / font_count, epoch + 1)
            writer.add_scalar("epoch/loss_g", sums["g"] / generator_font_count, epoch + 1)
            writer.add_scalar("epoch/loss_style", sums["style"] / generator_font_count, epoch + 1)
            writer.add_scalar("epoch/loss_pixel", sums["pixel"] / generator_font_count, epoch + 1)
            writer.add_scalar("epoch/loss_edge", sums["edge"] / generator_font_count, epoch + 1)
            writer.add_scalar("epoch/loss_perc_content", sums["perc_content"] / generator_font_count, epoch + 1)
            writer.add_scalar("epoch/loss_perc_style", sums["perc_style"] / generator_font_count, epoch + 1)
            writer.add_scalar("epoch/loss_letter", sums["letter"] / generator_font_count, epoch + 1)
            if predict_position:
                writer.add_scalar("epoch/loss_position", sums["position"] / generator_font_count, epoch + 1)
            writer.add_scalar("time/epoch_seconds", time.perf_counter() - epoch_start, epoch + 1)
            writer.flush()

            if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
                checkpoint_args = (
                    epoch,
                    global_step,
                    generator,
                    discriminator,
                    generator_ema,
                    optimizer_g,
                    optimizer_d,
                    scheduler_g,
                    scheduler_d,
                    scaler,
                    config,
                    encoder_path,
                )
                save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", *checkpoint_args)
                save_checkpoint(checkpoint_dir / "latest.pt", *checkpoint_args)

        torch.save(
            {
                "generator": generator_ema.state_dict(),
                "generator_config": generator_options,
                "letters": string.ascii_letters,
                "style_encoder_checkpoint": encoder_path,
                "style_encoder_state_key": encoder_state_key,
                "image_size": 256,
            },
            output_dir / "font_generator.pt",
        )
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    arguments = parse_args()
    with open(arguments.config, "r", encoding="utf-8") as config_file:
        train(
            yaml.safe_load(config_file),
            resume_override=arguments.resume,
            device_override=arguments.device,
        )
