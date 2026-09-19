"""Shared inference helpers for train.py checkpoints and reference images."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from dataset.font_mimic_dataset import _patchify_reference_images
from model import ConditionalGenerator
from utils.train_util import (
    choose_device, load_frozen_style_encoder, move_style_view,
    reconstruct_generated_crops,
)

IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}


def find_input_images(input_path: Path, recursive: bool = True) -> list[Path]:
    """Return supported images below a directory in stable path order."""
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input must be an image directory: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")
    return images


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Config must contain a mapping: {path}")
    return copy.deepcopy(dict(loaded))


def _as_mapping(value: Any) -> dict[str, Any]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _checkpoint_config(
    checkpoint: Mapping[str, Any], fallback_config: Mapping[str, Any]
) -> dict[str, Any]:
    saved = checkpoint.get("config")
    return _as_mapping(saved) if isinstance(saved, Mapping) else copy.deepcopy(dict(fallback_config))


def _resolve_existing_path(raw_path: str | Path, search_roots: Sequence[Path]) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [root / path for root in search_roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint does not exist. Checked: {checked}")


def _select_generator_state(
    checkpoint: Mapping[str, Any], requested: str
) -> tuple[Mapping[str, torch.Tensor], str]:
    keys = ("generator_ema", "generator") if requested == "auto" else (requested,)
    for key in keys:
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            return state, key
    raise KeyError(f"No generator state found for keys: {keys}")


def _generator_options(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    exported = checkpoint.get("generator_config")
    if isinstance(exported, Mapping):
        return copy.deepcopy(dict(exported))
    model_options = _as_mapping(config.get("model"))
    options = model_options.get("generator")
    if not isinstance(options, Mapping):
        raise ValueError(
            "Generator architecture is missing from the checkpoint and fallback config"
        )
    return copy.deepcopy(dict(options))


def load_models(
    checkpoint_path: Path,
    fallback_config_path: Path,
    style_checkpoint_override: Path | None,
    generator_state: str,
    device: torch.device,
    *,
    checkpoint: Mapping[str, Any] | None = None,
) -> tuple[ConditionalGenerator, torch.nn.Module, dict[str, Any], str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Generator checkpoint does not exist: {checkpoint_path}")
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Generator checkpoint must contain a mapping")

    fallback_config = _load_yaml(fallback_config_path)
    config = _checkpoint_config(checkpoint, fallback_config)
    options = _generator_options(checkpoint, config)
    generator = ConditionalGenerator(**options)
    state, state_name = _select_generator_state(checkpoint, generator_state)
    generator.load_state_dict(state, strict=True)
    generator.to(device).eval().requires_grad_(False)

    model_config = _as_mapping(config.get("model"))
    style_options = _as_mapping(model_config.get("style_encoder"))
    recorded_style_path = checkpoint.get("style_encoder_checkpoint")
    configured_style_path = style_options.get("checkpoint")
    selected_style_path = style_checkpoint_override or recorded_style_path or configured_style_path
    if selected_style_path is None:
        raise ValueError(
            "FontStyleViT checkpoint is missing; pass --style-checkpoint explicitly"
        )
    search_roots = (
        Path.cwd(),
        checkpoint_path.resolve().parent,
        fallback_config_path.resolve().parent,
    )
    style_path = _resolve_existing_path(selected_style_path, search_roots)
    style_options["checkpoint"] = str(style_path)
    exported_state_key = checkpoint.get("style_encoder_state_key")
    if exported_state_key:
        style_options["state_key"] = str(exported_state_key)
    style_encoder, _, _ = load_frozen_style_encoder(style_options, device)

    if generator.style_embedding_dim != style_encoder.backbone.embed_dim:
        raise ValueError(
            f"Generator style dimension {generator.style_embedding_dim} does not match "
            f"FontStyleViT style dimension {style_encoder.backbone.embed_dim}"
        )
    return generator, style_encoder, config, state_name


def _aligned_image_size(
    width: int,
    height: int,
    patch_size: int,
    max_long_side: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Match FontStyleDataset's resolution limiting and patch alignment."""
    scale = min(
        max_long_side / max(width, height),
        math.sqrt(max_pixels / width * height),
    )
    aligned_max = max(patch_size, max_long_side // patch_size * patch_size)
    width = min(
        max(patch_size, round(width * scale / patch_size) * patch_size),
        aligned_max,
    )
    height = min(
        max(patch_size, round(height * scale / patch_size) * patch_size),
        aligned_max,
    )
    while width * height > max_pixels:
        if width >= height and width > patch_size:
            width -= patch_size
        elif height > patch_size:
            height -= patch_size
        else:
            break
    return width, height


def prepare_reference_image(
    image_path: Path,
    patch_size: int,
    reference_options: Mapping[str, Any],
) -> torch.Tensor:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("L")
        else:
            image = image.convert("L")
        size = _aligned_image_size(
            image.width,
            image.height,
            patch_size,
            int(reference_options.get("max_long_side", 512)),
            int(reference_options.get("max_pixels", 262_144)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0

    mean = float(reference_options.get("mean", 0.5))
    std = float(reference_options.get("std", 0.5))
    return (torch.from_numpy(pixels).unsqueeze(0) - mean) / std


def generate_letters(
    reference: torch.Tensor,
    generator: ConditionalGenerator,
    style_encoder: torch.nn.Module,
    letter_ids: torch.Tensor,
    noise: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    view = _patchify_reference_images([reference], style_encoder.patch_size)
    view = move_style_view(view, device, non_blocking=False)
    amp_enabled = bool(amp and device.type == "cuda")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        style, _ = style_encoder(view)
        image_chunks = []
        position_chunks = []
        output_has_positions = None
        for start in range(0, len(letter_ids), batch_size):
            end = min(start + batch_size, len(letter_ids))
            count = end - start
            output = generator(
                noise[start:end].to(device),
                style.expand(count, -1),
                letter_ids[start:end].to(device),
            )
            current_has_positions = isinstance(output, tuple)
            if output_has_positions is None:
                output_has_positions = current_has_positions
            elif output_has_positions != current_has_positions:
                raise RuntimeError("Generator changed its return type between batches")
            if current_has_positions:
                images, positions = output
                image_chunks.append(images.float().cpu())
                position_chunks.append(positions.float().cpu())
            else:
                image_chunks.append(output.float().cpu())

    predicted_positions = (
        torch.cat(position_chunks, dim=0) if position_chunks else None
    )
    return torch.cat(image_chunks, dim=0), predicted_positions


def restore_predicted_glyphs(
    predicted_crops: torch.Tensor,
    predicted_positions: torch.Tensor,
    canvas_size: int,
) -> torch.Tensor:
    """Restore generated crops onto white canvases at predicted positions."""
    if predicted_crops.ndim != 4:
        raise ValueError(
            f"Expected predicted crops [N, C, H, W], got {tuple(predicted_crops.shape)}"
        )
    if predicted_positions.shape != (predicted_crops.shape[0], 3):
        raise ValueError(
            f"Expected predicted positions [{predicted_crops.shape[0]}, 3], "
            f"got {tuple(predicted_positions.shape)}"
        )
    return reconstruct_generated_crops(
        predicted_crops, predicted_positions, int(canvas_size)
    )


def make_grid(images: torch.Tensor, columns: int, gap: int) -> Image.Image:
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N, C, H, W], got {tuple(images.shape)}")
    count, channels, height, width = images.shape
    if channels not in (1, 3):
        raise ValueError(f"Only one- or three-channel output is supported, got {channels}")
    rows = math.ceil(count / columns)
    mode = "L" if channels == 1 else "RGB"
    canvas = Image.new(
        mode,
        (columns * width + (columns - 1) * gap, rows * height + (rows - 1) * gap),
        "white",
    )
    values = images.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    for index, value in enumerate(values):
        array = value.squeeze(0).numpy() if channels == 1 else value.permute(1, 2, 0).numpy()
        cell = Image.fromarray(array, mode=mode)
        left = (index % columns) * (width + gap)
        top = (index // columns) * (height + gap)
        canvas.paste(cell, (left, top))
    return canvas
