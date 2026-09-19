"""Pretrain a variable-resolution font-style ViT with MoCo, DINO and iBOT."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FontStyleCollator, FontStyleDataset, ImageFontStyleDataset, ComposeDataset
from model.font_style_vit import FontStyleViT
from model.self_supervised_losses import (
    DINOLoss,
    IBOTPlusPlusPatchLoss,
    MoCoQueueLoss,
)
from utils.rand_util import seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train_font_style.yaml", help="YAML configuration path"
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path; overrides train.resume")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:0 or cpu")
    return parser.parse_args()


def create_dataloader(config: Mapping[str, Any], seed: int) -> DataLoader:
    dataset = FontStyleDataset(copy.deepcopy(config["dataset"]))
    if "img_dataset" in config:
        img_dataset = ImageFontStyleDataset(copy.deepcopy(config["img_dataset"]))
        dataset = ComposeDataset([dataset, img_dataset])
    collator = FontStyleCollator(**copy.deepcopy(config.get("collator", {})))
    loader_config = copy.deepcopy(config.get("loader", {}))
    num_workers = int(loader_config.get("num_workers", 0))
    if num_workers == 0:
        loader_config["persistent_workers"] = False
        loader_config.pop("prefetch_factor", None)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_config,
    )


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def move_view_to_device(
    view: Mapping[str, Any], device: torch.device, non_blocking: bool
) -> Dict[str, Any]:
    moved = dict(view)
    for key in ("patches", "patch_padding_mask", "patch_valid_mask", "ibot_mask"):
        if key in moved:
            moved[key] = moved[key].to(device, non_blocking=non_blocking)
    # grid_sizes remains on CPU. The backbone uses it as small shape metadata
    # when interpolating a different 2-D position map for every sample.
    return moved


def parameter_groups(model: nn.Module, weight_decay: float) -> Sequence[Dict[str, Any]]:
    regularized, unregularized = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim == 1
            or name.endswith(".bias")
            or any(token in name for token in ("cls_token", "mask_token", "position"))
        ):
            unregularized.append(parameter)
        else:
            regularized.append(parameter)
    return (
        {"params": regularized, "weight_decay": weight_decay, "apply_weight_decay": True},
        {"params": unregularized, "weight_decay": 0.0, "apply_weight_decay": False},
    )


def create_optimizer(model: nn.Module, config: Mapping[str, Any], batch_size: int):
    optimizer_config = dict(config)
    name = str(optimizer_config.pop("name", "AdamW"))
    learning_rate = float(optimizer_config.pop("lr"))
    scale_by_batch = bool(optimizer_config.pop("scale_lr_by_batch", True))
    reference_batch_size = int(optimizer_config.pop("reference_batch_size", 256))
    if scale_by_batch:
        learning_rate *= batch_size / reference_batch_size
    weight_decay = float(optimizer_config.pop("weight_decay", 0.04))
    optimizer_config.pop("min_lr", None)
    try:
        optimizer_class = getattr(torch.optim, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown optimizer: {name}") from exc
    optimizer = optimizer_class(
        parameter_groups(model, weight_decay), lr=learning_rate, **optimizer_config
    )
    return optimizer, learning_rate


def cosine_value(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def learning_rate_at_step(
    base_lr: float,
    min_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    cosine_steps = max(1, total_steps - warmup_steps)
    return cosine_value(base_lr, min_lr, step - warmup_steps, cosine_steps)


def temperature_at_step(
    warmup_temperature: float,
    target_temperature: float,
    step: int,
    warmup_steps: int,
) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return target_temperature
    progress = step / max(1, warmup_steps - 1)
    return warmup_temperature + progress * (target_temperature - warmup_temperature)


@torch.no_grad()
def update_teacher(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    teacher_parameters = dict(teacher.named_parameters())
    for name, parameter in student.named_parameters():
        teacher_parameters[name].mul_(momentum).add_(parameter.detach(), alpha=1.0 - momentum)
    teacher_buffers = dict(teacher.named_buffers())
    for name, buffer in student.named_buffers():
        teacher_buffers[name].copy_(buffer)


def font_signature(dataset) -> Sequence[str]:
    signature = []
    root = Path(dataset.font_dir).resolve()
    for font in dataset.fonts:
        path = Path(font).resolve()
        try:
            signature.append(path.relative_to(root).as_posix())
        except ValueError:
            signature.append(path.name)
    return signature


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    student: FontStyleViT,
    teacher: FontStyleViT,
    moco_loss: MoCoQueueLoss,
    dino_loss: DINOLoss,
    ibot_loss: IBOTPlusPlusPatchLoss,
    optimizer,
    scaler,
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "moco_loss": moco_loss.state_dict(),
            "dino_loss": dino_loss.state_dict(),
            "ibot++_loss": ibot_loss.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": dict(config),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    student: FontStyleViT,
    teacher: FontStyleViT,
    moco_loss: MoCoQueueLoss,
    dino_loss: DINOLoss,
    ibot_loss: IBOTPlusPlusPatchLoss,
    optimizer,
    scaler,
) -> Tuple[int, int]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location=device)
    student.load_state_dict(checkpoint["student"])
    teacher.load_state_dict(checkpoint.get("teacher", checkpoint["student"]))
    moco_loss.load_state_dict(checkpoint["moco_loss"])
    dino_loss.load_state_dict(checkpoint["dino_loss"])
    ibot_loss.load_state_dict(checkpoint["ibot++_loss"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def validate_config(config: Mapping[str, Any]) -> None:
    dataset_patch_size = int(config["dataset"].get("patch_size", 16))
    collator_patch_size = int(config.get("collator", {}).get("patch_size", 16))
    model_patch_size = int(config["model"].get("patch_size", 16))
    if len({dataset_patch_size, collator_patch_size, model_patch_size}) != 1:
        raise ValueError("dataset, collator and model patch_size values must match")
    num_views = int(config["dataset"].get("num_views", 2))
    num_global_views = int(config["losses"].get("num_global_views", 2))
    if num_views < 2 or num_global_views < 2 or num_global_views > num_views:
        raise ValueError("Need at least two views and 2 <= num_global_views <= num_views")


def train(
    config: Dict[str, Any],
    resume_override: Optional[str] = None,
    device_override: Optional[str] = None,
) -> None:
    validate_config(config)
    train_config = config["train"]
    seed = int(train_config.get("seed", 42))
    seed_everything(seed)
    requested_device = device_override or str(train_config.get("device", "cuda"))
    device = choose_device(requested_device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(train_config.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train_config.get("allow_tf32", True))

    dataloader = create_dataloader(config, seed)
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader has zero batches; reduce batch_size or disable drop_last")
    dataset = dataloader.dataset
    if isinstance(dataset, ComposeDataset):
        fonts = []
        for sub_dataset in dataset.datasets:
            fonts += font_signature(sub_dataset)
    else:
        fonts = font_signature(dataset)

    model_config = copy.deepcopy(config["model"])
    student = FontStyleViT(**model_config).to(device)
    teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
    loader_batch_size = int(config.get("loader", {}).get("batch_size", 1))
    optimizer_config = config["optimizer"]
    optimizer, base_lr = create_optimizer(
        student, optimizer_config, loader_batch_size
    )

    losses_config = config["losses"]
    moco_config = dict(losses_config.get("moco", {}))
    moco_loss = MoCoQueueLoss(feature_dim=128, **moco_config).to(device)
    dino_config = dict(losses_config.get("dino", {}))
    dino_teacher_temperature = float(dino_config.pop("teacher_temperature", 0.07))
    dino_warmup_temperature = float(dino_config.pop("warmup_teacher_temperature", 0.04))
    dino_warmup_temperature_epochs = int(dino_config.pop("warmup_temperature_epochs", 30))
    dino_loss = DINOLoss(output_dim=int(model_config["dino_out_dim"]), **dino_config).to(device)
    # Accept the old section name so existing experiments remain resumable.
    ibot_config = dict(losses_config.get("ibot++", {}))
    ibot_teacher_temperature = float(ibot_config.pop("teacher_temperature", 0.07))
    ibot_warmup_temperature = float(ibot_config.pop("warmup_teacher_temperature", 0.04))
    ibot_warmup_temperature_epochs = int(ibot_config.pop("warmup_temperature_epochs", 30))
    ibot_loss = IBOTPlusPlusPatchLoss(
        output_dim=int(model_config["ibot_out_dim"]), **ibot_config
    ).to(device)

    amp_enabled = bool(train_config.get("amp", True) and device.type == "cuda")
    amp_dtype_name = str(train_config.get("amp_dtype", "float16"))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    output_dir = Path(train_config.get("output_dir", "outputs/font_style_vit"))
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (output_dir / "font_index.json").open("w", encoding="utf-8") as file:
        json.dump({name: index for index, name in enumerate(fonts)}, file, indent=2, ensure_ascii=False)

    start_epoch, global_step = 0, 0
    resume = resume_override or train_config.get("resume")
    if resume:
        start_epoch, global_step = load_checkpoint(
            str(resume),
            device,
            student,
            teacher,
            moco_loss,
            dino_loss,
            ibot_loss,
            optimizer,
            scaler,
        )
    writer = SummaryWriter(output_dir / "tensorboard", purge_step=global_step if resume else None)
    writer.add_text("data/font_count", str(len(fonts)))

    epochs = int(train_config.get("epochs", 100))
    total_steps = max(1, epochs * len(dataloader))
    warmup_steps = int(train_config.get("warmup_epochs", 10)) * len(dataloader)
    min_lr = float(optimizer_config.get("min_lr", 1e-6))
    base_teacher_momentum = float(losses_config.get("teacher_momentum", 0.996))
    num_global_views = int(losses_config.get("num_global_views", 2))
    weights = losses_config.get("weights", {})
    moco_weight = float(weights.get("moco", 1.0))
    dino_weight = float(weights.get("dino", 1.0))
    ibot_weight = float(weights.get("ibot++", 1.0))
    freeze_last_layer_epochs = int(train_config.get("freeze_last_layer_epochs", 1))
    gradient_clip_norm = train_config.get("gradient_clip_norm", 3.0)
    log_every = int(train_config.get("log_every", 20))
    checkpoint_every = int(train_config.get("checkpoint_every_epochs", 5))
    non_blocking = bool(config.get("loader", {}).get("pin_memory", False))
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]

    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    print(
        f"Training FontStyleViT with {parameter_count / 1e6:.2f}M parameters, "
        f"{len(fonts)} fonts and {len(dataloader)} batches/epoch on {device}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, epochs):
            student.train()
            teacher.eval()
            epoch_start = time.perf_counter()
            epoch_sums = {
                "total": 0.0, "moco": 0.0, "dino": 0.0, "ibot++": 0.0
            }
            samples_seen = 0
            log_start = time.perf_counter()
            log_samples = 0
            for batch_index, batch in enumerate(dataloader):
                learning_rate = learning_rate_at_step(
                    base_lr, min_lr, global_step, total_steps, warmup_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                teacher_momentum = cosine_value(
                    base_teacher_momentum, 0.9999, global_step, total_steps
                )
                dino_temperature = temperature_at_step(
                    dino_warmup_temperature,
                    dino_teacher_temperature,
                    global_step,
                    dino_warmup_temperature_epochs * len(dataloader),
                )
                ibot_temperature = temperature_at_step(
                    ibot_warmup_temperature,
                    ibot_teacher_temperature,
                    global_step,
                    ibot_warmup_temperature_epochs * len(dataloader),
                )

                views = [
                    move_view_to_device(view, device, non_blocking) for view in batch["views"]
                ]
                style_ids = batch["style_ids"].to(device, non_blocking=non_blocking)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    student_outputs = [
                        student.forward_view(
                            view,
                            use_ibot_mask=view_index < num_global_views,
                        )
                        for view_index, view in enumerate(views)
                    ]
                    with torch.no_grad():
                        teacher_outputs = [
                            teacher.forward_view(view, use_ibot_mask=False)
                            for view in views[:num_global_views]
                        ]

                    moco_loss_01, moco_stats_01 = moco_loss(
                        student_outputs[0]["style_embedding"],
                        teacher_outputs[1]["style_embedding"],
                        style_ids,
                    )
                    moco_loss_10, moco_stats_10 = moco_loss(
                        student_outputs[1]["style_embedding"],
                        teacher_outputs[0]["style_embedding"],
                        style_ids,
                    )
                    loss_moco = 0.5 * (moco_loss_01 + moco_loss_10)
                    loss_dino, dino_stats = dino_loss(
                        [output["dino_logits"] for output in student_outputs],
                        [output["dino_logits"] for output in teacher_outputs],
                        dino_temperature,
                    )
                    loss_ibot, ibot_stats = ibot_loss(
                        [output["ibot_logits"] for output in student_outputs[:num_global_views]],
                        [output["ibot_logits"] for output in teacher_outputs],
                        [view["ibot_mask"] for view in views[:num_global_views]],
                        [view["patch_padding_mask"] for view in views[:num_global_views]],
                        ibot_temperature,
                    )
                    total_loss = (
                        moco_weight * loss_moco
                        + dino_weight * loss_dino
                        + ibot_weight * loss_ibot
                    )

                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, batch={batch_index}: "
                        f"moco={loss_moco.item()}, dino={loss_dino.item()}, "
                        f"ibot++={loss_ibot.item()}"
                    )

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if epoch < freeze_last_layer_epochs:
                    student.cancel_last_layer_gradients()
                max_norm = float(gradient_clip_norm) if gradient_clip_norm is not None else float("inf")
                gradient_norm = nn.utils.clip_grad_norm_(trainable_parameters, max_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_succeeded = scaler.get_scale() >= previous_scale
                if optimizer_step_succeeded:
                    update_teacher(teacher, student, teacher_momentum)
                    queued_keys = torch.cat(
                        (
                            teacher_outputs[0]["style_embedding"],
                            teacher_outputs[1]["style_embedding"],
                        )
                    )
                    moco_loss.enqueue(queued_keys, style_ids.repeat(2))

                batch_size = style_ids.shape[0]
                samples_seen += batch_size
                log_samples += batch_size
                scalar_losses = {
                    "total": float(total_loss.detach()),
                    "moco": float(loss_moco.detach()),
                    "dino": float(loss_dino.detach()),
                    "ibot++": float(loss_ibot.detach()),
                }
                for name, value in scalar_losses.items():
                    epoch_sums[name] += value * batch_size

                if global_step % log_every == 0:
                    elapsed = max(time.perf_counter() - log_start, 1e-6)
                    valid_patches = sum(
                        int((~view["patch_padding_mask"]).sum()) for view in views
                    )
                    padded_patches = sum(int(view["patch_padding_mask"].sum()) for view in views)
                    ibot_views = views[:num_global_views]
                    ibot_valid_patches = sum(
                        int((~view["patch_padding_mask"]).sum()) for view in ibot_views
                    )
                    masked_patches = sum(int(view["ibot_mask"].sum()) for view in ibot_views)
                    visible_patches = ibot_valid_patches - masked_patches
                    total_patch_slots = valid_patches + padded_patches
                    moco_positive = 0.5 * (
                        moco_stats_01["positive_similarity"]
                        + moco_stats_10["positive_similarity"]
                    )
                    moco_accuracy = 0.5 * (
                        moco_stats_01["top1_accuracy"] + moco_stats_10["top1_accuracy"]
                    )
                    writer.add_scalar("loss/total", scalar_losses["total"], global_step)
                    writer.add_scalar("loss/moco", scalar_losses["moco"], global_step)
                    writer.add_scalar("loss/dino", scalar_losses["dino"], global_step)
                    writer.add_scalar("loss/ibot++", scalar_losses["ibot++"], global_step)
                    writer.add_scalar("optimizer/learning_rate", learning_rate, global_step)
                    writer.add_scalar("optimizer/gradient_norm", float(gradient_norm), global_step)
                    writer.add_scalar("teacher/momentum", teacher_momentum, global_step)
                    writer.add_scalar("teacher/dino_temperature", dino_temperature, global_step)
                    writer.add_scalar("teacher/ibot++_temperature", ibot_temperature, global_step)
                    writer.add_scalar("teacher/dino_entropy", dino_stats["teacher_entropy"], global_step)
                    writer.add_scalar("ibot++/masked_token_loss", ibot_stats["masked_token_loss"], global_step)
                    writer.add_scalar("ibot++/visible_token_loss", ibot_stats["visible_token_loss"], global_step)
                    writer.add_scalar("moco/positive_similarity", moco_positive, global_step)
                    writer.add_scalar("moco/top1_accuracy", moco_accuracy, global_step)
                    writer.add_scalar("moco/queue_pointer", int(moco_loss.queue_pointer), global_step)
                    writer.add_scalar("moco/queue_filled", int(moco_loss.queue_filled), global_step)
                    writer.add_scalar("tokens/padding_fraction", padded_patches / max(1, total_patch_slots), global_step)
                    writer.add_scalar("tokens/ibot++_masked_fraction", masked_patches / max(1, ibot_valid_patches), global_step)
                    writer.add_scalar("tokens/ibot++_visible_fraction", visible_patches / max(1, ibot_valid_patches), global_step)
                    writer.add_scalar("performance/samples_per_second", log_samples / elapsed, global_step)
                    print(
                        f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
                        f"[Batch {batch_index + 1:05d}/{len(dataloader):05d}] "
                        f"[Step {global_step:08d}] total={scalar_losses['total']:.4f} "
                        f"moco={scalar_losses['moco']:.4f} dino={scalar_losses['dino']:.4f} "
                        f"ibot++={scalar_losses['ibot++']:.4f} "
                        f"lr={learning_rate:.3e} "
                        f"m={teacher_momentum:.6f}",
                        flush=True,
                    )
                    log_start, log_samples = time.perf_counter(), 0
                global_step += 1

            epoch_seconds = time.perf_counter() - epoch_start
            for name, value in epoch_sums.items():
                writer.add_scalar(f"epoch_loss/{name}", value / max(1, samples_seen), epoch + 1)
            writer.add_scalar("performance/epoch_seconds", epoch_seconds, epoch + 1)
            writer.flush()

            if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
                checkpoint_arguments = (
                    epoch,
                    global_step,
                    student,
                    teacher,
                    moco_loss,
                    dino_loss,
                    ibot_loss,
                    optimizer,
                    scaler,
                    config,
                )
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", *checkpoint_arguments
                )
                save_checkpoint(checkpoint_dir / "latest.pt", *checkpoint_arguments)

        # A compact final artifact still contains every head so training can be
        # inspected, while inference only calls FontStyleViT.forward().
        torch.save(
            {
                "model": student.state_dict(),
                "model_config": model_config,
                "embedding_dim": 128,
            },
            output_dir / "font_style_encoder.pt",
        )
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    arguments = parse_args()
    with open(arguments.config, "r", encoding="utf-8") as config_file:
        train(yaml.safe_load(config_file), arguments.resume, arguments.device)
