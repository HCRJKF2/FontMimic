from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.font_mimic_bbox_dataset import FontMimicBBoxCollator, FontMimicBBoxDataset
from dataset.font_mimic_dataset import FontMimicCollator, FontMimicDataset
from model import FontStyleViT
from utils.rand_util import seed_worker


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_frozen_style_encoder(
    config: Mapping[str, Any], device: torch.device
) -> tuple[FontStyleViT, str, str]:
    checkpoint_path = config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("model.style_encoder.checkpoint is required")
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"FontStyleViT checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("FontStyleViT checkpoint must contain a mapping")

    requested_key = str(config.get("state_key", "auto"))
    candidates = ("model", "teacher", "student") if requested_key == "auto" else (requested_key,)
    state_key = next(
        (key for key in candidates if isinstance(checkpoint.get(key), Mapping)), None
    )
    if state_key is None:
        raise KeyError(f"No FontStyleViT state found for keys {candidates}")

    architecture = config.get("architecture") or checkpoint.get("model_config")
    if architecture is None and isinstance(checkpoint.get("config"), Mapping):
        architecture = checkpoint["config"].get("model")
    if not isinstance(architecture, Mapping):
        raise ValueError(
            "FontStyleViT architecture missing; set model.style_encoder.architecture"
        )

    encoder = FontStyleViT(**copy.deepcopy(dict(architecture)))
    incompatible = encoder.load_state_dict(
        checkpoint[state_key], strict=bool(config.get("strict", True))
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Non-strict FontStyleViT load: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}",
            flush=True,
        )
    encoder.to(device).eval().requires_grad_(False)
    return encoder, str(path.resolve()), state_key


def optimizer_from_config(
    parameters: Iterable[nn.Parameter], config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    options = copy.deepcopy(dict(config))
    name = str(options.pop("name"))
    try:
        optimizer_class = getattr(torch.optim, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown optimizer: {name}") from exc
    return optimizer_class(parameters, **options)


def scheduler_from_config(optimizer, config: Optional[Mapping[str, Any]]):
    if not config or str(config.get("name", "none")).lower() == "none":
        return None, "epoch"
    options = copy.deepcopy(dict(config))
    name = str(options.pop("name"))
    interval = str(options.pop("interval", "epoch")).lower()
    if interval not in {"step", "epoch"}:
        raise ValueError("scheduler.interval must be 'step' or 'epoch'")
    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown scheduler: {name}") from exc
    return scheduler_class(optimizer, **options), interval


def create_dataloader(config: Mapping[str, Any], seed: int) -> DataLoader:
    model_options = config.get("model", {})
    generator_options = (
        model_options.get("generator", {}) if isinstance(model_options, Mapping) else {}
    )
    flow_options = (
        model_options.get("flow_model", {}) if isinstance(model_options, Mapping) else {}
    )
    predict_position = any(
        isinstance(options, Mapping)
        and options.get("predict_pos_args") is not None
        for options in (generator_options, flow_options)
    )
    dataset_class = FontMimicBBoxDataset if predict_position else FontMimicDataset
    collator_class = FontMimicBBoxCollator if predict_position else FontMimicCollator
    dataset = dataset_class(copy.deepcopy(config["dataset"]))
    collator = collator_class(**copy.deepcopy(config.get("collator", {})))
    options = copy.deepcopy(config.get("loader", {}))
    if int(options.get("num_workers", 0)) == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=generator,
        **options,
    )


def make_grouped_grid(
    images: torch.Tensor, columns: int
) -> torch.Tensor:
    """Arrange [B, N, C, H, W] into one differentiable grid per font."""
    if images.ndim != 5:
        raise ValueError(f"Expected [B, N, C, H, W], got {tuple(images.shape)}")
    batch_size, count, channels, height, width = images.shape
    rows = math.ceil(count / columns)
    missing = rows * columns - count
    if missing:
        images = torch.cat(
            (
                images,
                torch.ones(
                    batch_size,
                    missing,
                    channels,
                    height,
                    width,
                    dtype=images.dtype,
                    device=images.device,
                ),
            ),
            dim=1,
        )
    return (
        images.reshape(batch_size, rows, columns, channels, height, width)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch_size, channels, rows * height, columns * width)
    )


def _aligned_size(
    height: int,
    width: int,
    patch_size: int,
    max_long_side: int,
    max_pixels: int,
) -> tuple[int, int]:
    scale = min(
        1.0,
        max_long_side / max(height, width),
        math.sqrt(max_pixels / (height * width)),
    )
    height = max(patch_size, int(height * scale) // patch_size * patch_size)
    width = max(patch_size, int(width * scale) // patch_size * patch_size)
    while height * width > max_pixels:
        if width >= height and width > patch_size:
            width -= patch_size
        elif height > patch_size:
            height -= patch_size
        else:
            break
    return height, width


def generated_grid_style_view(
    images: torch.Tensor,
    patch_size: int,
    mean: float,
    std: float,
    max_long_side: int,
    max_pixels: int,
    columns: int,
) -> Dict[str, torch.Tensor]:
    """Convert grouped RGB outputs to a differentiable FontStyleViT view."""
    if std <= 0:
        raise ValueError("style_grid.std must be positive")
    pixels = make_grouped_grid(images, columns).add(1.0).mul(0.5)
    if pixels.shape[1] == 3:
        weights = pixels.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        grayscale = (pixels * weights).sum(1, keepdim=True)
    elif pixels.shape[1] == 1:
        grayscale = pixels
    else:
        raise ValueError("Style grids support one or three image channels")
    normalized = (grayscale - mean) / std
    size = _aligned_size(
        normalized.shape[-2],
        normalized.shape[-1],
        patch_size,
        max_long_side,
        max_pixels,
    )
    if size != normalized.shape[-2:]:
        normalized = F.interpolate(
            normalized, size=size, mode="bicubic", align_corners=False
        )
    grid_h, grid_w = size[0] // patch_size, size[1] // patch_size
    patches = normalized.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        normalized.shape[0], grid_h * grid_w, -1
    )
    return {
        "patches": patches.contiguous(),
        "patch_padding_mask": torch.zeros(
            normalized.shape[0],
            grid_h * grid_w,
            dtype=torch.bool,
            device=normalized.device,
        ),
        "grid_sizes": torch.tensor(
            [[grid_h, grid_w]] * normalized.shape[0], dtype=torch.long
        ),
    }


def bboxes_to_square_positions(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized xyxy square boxes to (side, center_x, center_y)."""
    widths = bboxes[..., 2] - bboxes[..., 0]
    heights = bboxes[..., 3] - bboxes[..., 1]
    side = (widths + heights) * 0.5
    center_x = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    center_y = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    return torch.stack((side, center_x, center_y), dim=-1)


def reconstruct_generated_crops(
    crops: torch.Tensor, positions: torch.Tensor, canvas_size: int
) -> torch.Tensor:
    """Paste generated square crops onto white canvases using normalized positions."""
    canvases = crops.new_ones(
        crops.shape[0], crops.shape[1], canvas_size, canvas_size
    )
    for index, (side, center_x, center_y) in enumerate(
        positions.detach().float().cpu().tolist()
    ):
        side_pixels = max(1, min(canvas_size, round(side * canvas_size)))
        left = round(center_x * canvas_size - side_pixels * 0.5)
        top = round(center_y * canvas_size - side_pixels * 0.5)
        left = max(0, min(left, canvas_size - side_pixels))
        top = max(0, min(top, canvas_size - side_pixels))
        resized = F.interpolate(
            crops[index:index + 1], size=(side_pixels, side_pixels), mode="bilinear", align_corners=False,
        )
        canvases[index:index + 1, :, top:top + side_pixels, left:left + side_pixels] = resized
    return canvases


def move_style_view(
    view: Mapping[str, torch.Tensor], device: torch.device, non_blocking: bool
) -> Dict[str, torch.Tensor]:
    return {
        "patches": view["patches"].to(device, non_blocking=non_blocking),
        "patch_padding_mask": view["patch_padding_mask"].to(
            device, non_blocking=non_blocking
        ),
        "grid_sizes": view["grid_sizes"],
    }


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter, alpha=1.0 - decay)
    for name, buffer in model.named_buffers():
        dict(ema.named_buffers())[name].copy_(buffer)
