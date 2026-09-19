from __future__ import annotations

import string
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def load_frozen_parseq(
    options: Mapping[str, Any], device: torch.device
) -> tuple[nn.Module, tuple[int, int], int]:
    """Load an official PARSeq Torch Hub model as a frozen OCR teacher."""
    repo_or_dir = str(options.get("repo_or_dir", "baudm/parseq"))
    source = str(options.get("source", "github")).lower()
    if source not in {"github", "local"}:
        raise ValueError("model.parseq.source must be 'github' or 'local'")
    model_name = str(options.get("model_name", "parseq"))

    hub_options: dict[str, Any] = {
        "pretrained": True,
        "decode_ar": bool(options.get("decode_ar", True)),
        "refine_iters": int(options.get("refine_iters", 1)),
        "source": source,
        "force_reload": bool(options.get("force_reload", False)),
        "verbose": bool(options.get("verbose", True)),
    }
    if source == "github":
        hub_options["trust_repo"] = bool(options.get("trust_repo", True))
        hub_options["skip_validation"] = bool(options.get("skip_validation", False))
    parseq = torch.hub.load(repo_or_dir, model_name, **hub_options)
    parseq = parseq.to(device).eval().requires_grad_(False)

    try:
        image_size = tuple(int(value) for value in parseq.hparams.img_size)
        max_label_length = int(parseq.hparams.max_label_length)
        charset = str(parseq.hparams.charset_train)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Loaded Torch Hub model is not a compatible PARSeq model") from error
    if len(image_size) != 2:
        raise ValueError(f"PARSeq img_size must contain two entries, got {image_size}")

    missing = sorted(set(string.ascii_letters) - set(charset))
    if missing:
        raise ValueError(f"PARSeq charset is missing required ASCII letters: {''.join(missing)}")
    return parseq, (image_size[0], image_size[1]), max_label_length


def make_parseq_lines(
    images: torch.Tensor,
    characters_per_line: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Compose grouped glyph canvases into differentiable PARSeq text lines.

    ``images`` is expected in [-1, 1] with shape [font, glyph, channel, H, W].
    Each output line is resized with its aspect ratio intact and padded with white
    to PARSeq's input size. The final incomplete line is padded with blank cells.
    """
    if images.ndim != 5:
        raise ValueError(f"Expected [B, N, C, H, W], got {tuple(images.shape)}")
    target_height, target_width = (int(value) for value in image_size)

    batch_size, glyph_count, channels, height, width = images.shape
    if channels not in (1, 3):
        raise ValueError(f"PARSeq expects one or three image channels, got {channels}")

    line_count = (glyph_count + characters_per_line - 1) // characters_per_line
    padded_count = line_count * characters_per_line
    if padded_count != glyph_count:
        padding = images.new_ones(batch_size, padded_count - glyph_count, channels, height, width)
        images = torch.cat((images, padding), dim=1)

    lines = (
        images.reshape(batch_size, line_count, characters_per_line, channels, height, width)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(batch_size * line_count, channels, height, characters_per_line * width)
    ).float()

    scale = min(target_height / lines.shape[-2], target_width / lines.shape[-1])
    resized_height = max(1, min(target_height, round(lines.shape[-2] * scale)))
    resized_width = max(1, min(target_width, round(lines.shape[-1] * scale)))
    if (resized_height, resized_width) != lines.shape[-2:]:
        lines = F.interpolate(
            lines,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    lines = F.pad(
        lines,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
        value=1.0,
    )
    if channels == 1:
        lines = lines.repeat(1, 3, 1, 1)
    # lines = lines.clamp(-1.0, 1.0)

    return lines


def parseq_logits_loss(
    generated_logits: torch.Tensor,
    real_logits: torch.Tensor,
    kind: str = "kl",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Match PARSeq outputs from generated and real glyph lines."""
    kind = kind.lower()
    generated = generated_logits.float()
    real = real_logits.detach().float()
    if kind in {"kl", "kld", "distillation"}:
        log_prediction = F.log_softmax(generated / temperature, dim=-1)
        target = F.softmax(real / temperature, dim=-1)
        token_kl = F.kl_div(log_prediction, target, reduction="none").sum(dim=-1)
        return token_kl.mean() * temperature**2
    if kind in {"mse", "l2"}:
        return F.mse_loss(generated, real)
    if kind in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(generated, real)
    raise ValueError(
        f"Unknown PARSeq logits loss {kind!r}; expected kl, mse, or smooth_l1"
    )
