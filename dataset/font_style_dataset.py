"""DINO multi-crop extension for the on-the-fly font-style dataset.

The rendering primitives live in :mod:`dataset.font_style_dataset_base`.  This
module adds an explicit view contract:

``[global_0, global_1, local_0, ..., local_n]``

Global and local views have separate crop and resolution distributions. Local
views are cropped from the same rendered sources as the global views, while the
two global sources may contain different text from the same font. This keeps the
style-positive signal strong without reducing DINO to content matching.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from dataset.font_style_dataset_base import (
    FONT_EXTENSIONS,
    FontStyleCollator as _BaseFontStyleCollator,
    FontStyleDataset as _BaseFontStyleDataset,
)


ViewProfile = Dict[str, Any]


class FontStyleDataset(_BaseFontStyleDataset):
    """Generate two global and multiple local font-style views.

    ``num_views`` remains supported for old configurations. For real multi-crop
    training, set ``num_global_views`` and ``num_local_views`` explicitly. The
    first views are always global because the trainer sends only that prefix to
    the EMA teacher and iBOT objective.
    """

    def __init__(self, opt: Mapping[str, Any]):
        options = dict(opt)
        explicit_multicrop = "num_global_views" in options or "num_local_views" in options
        if explicit_multicrop:
            self.num_global_views = int(options.get("num_global_views", 2))
            self.num_local_views = int(options.get("num_local_views", 4))
            expected_total = self.num_global_views + self.num_local_views
            if "num_views" in options and int(options["num_views"]) != expected_total:
                raise ValueError(
                    "num_views must equal num_global_views + num_local_views when all are set"
                )
        else:
            # Preserve the old two-view behavior for existing callers and tests.
            self.num_global_views = int(options.get("num_views", 2))
            self.num_local_views = 0

        if self.num_global_views < 1:
            raise ValueError("num_global_views must be at least one")
        if self.num_local_views < 0:
            raise ValueError("num_local_views cannot be negative")

        self._profile_options = options
        self.global_view_profile = self._build_global_profile(options)
        self.local_view_profile = self._build_local_profile(options)
        self._active_view_profile = self.global_view_profile
        super().__init__(options)
        self.num_views = self.num_global_views + self.num_local_views

    @staticmethod
    def _pair(value: Sequence[float], name: str) -> Tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain two values")
        pair = (float(value[0]), float(value[1]))
        if not 0 < pair[0] <= pair[1]:
            raise ValueError(f"{name} must contain positive ordered values")
        return pair

    def _validate_profile(self, profile: ViewProfile, name: str) -> ViewProfile:
        profile = dict(profile)
        profile["min_short_side"] = int(profile["min_short_side"])
        profile["max_long_side"] = int(profile["max_long_side"])
        profile["max_pixels"] = int(profile["max_pixels"])
        profile["aspect_ratio_range"] = self._pair(
            profile["aspect_ratio_range"], f"{name}.aspect_ratio_range"
        )
        profile["crop_scale"] = self._pair(profile["crop_scale"], f"{name}.crop_scale")
        profile["crop_aspect_ratio"] = self._pair(
            profile["crop_aspect_ratio"], f"{name}.crop_aspect_ratio"
        )
        if profile["crop_scale"][1] > 1.0:
            raise ValueError(f"{name}.crop_scale cannot exceed 1")
        if profile["min_short_side"] < self._profile_options.get("patch_size", 16):
            raise ValueError(f"{name}.min_short_side must be at least one patch")
        if profile["max_long_side"] < profile["min_short_side"]:
            raise ValueError(f"{name}.max_long_side must be >= min_short_side")
        if profile["max_pixels"] < self._profile_options.get("patch_size", 16) ** 2:
            raise ValueError(f"{name}.max_pixels is too small")
        return profile

    def _build_global_profile(self, options: Mapping[str, Any]) -> ViewProfile:
        configured = dict(options.get("global_view", {}))
        aspect = configured.get(
            "aspect_ratio_range", options.get("aspect_ratio_range", (0.35, 2.85))
        )
        profile = {
            "min_short_side": configured.get(
                "min_short_side", options.get("min_short_side", 96)
            ),
            "max_long_side": configured.get(
                "max_long_side", options.get("max_long_side", 512)
            ),
            "max_pixels": configured.get("max_pixels", options.get("max_pixels", 196_608)),
            "aspect_ratio_range": aspect,
            "crop_scale": configured.get("crop_scale", (0.65, 1.0)),
            "crop_aspect_ratio": configured.get("crop_aspect_ratio", aspect),
        }
        return self._validate_profile(profile, "global_view")

    def _build_local_profile(
        self, options: Mapping[str, Any]
    ) -> ViewProfile:
        configured = dict(options.get("local_view", {}))
        profile = {
            "min_short_side": configured.get("min_short_side", 96),
            "max_long_side": configured.get("max_long_side", 192),
            "max_pixels": configured.get("max_pixels", 192**2),
            "aspect_ratio_range": configured.get("aspect_ratio_range", (0.5, 2.0)),
            "crop_scale": configured.get("crop_scale", (0.15, 0.45)),
            "crop_aspect_ratio": configured.get("crop_aspect_ratio", (0.5, 2.0)),
        }
        return self._validate_profile(profile, "local_view")

    def _sample_canvas_size(self, rng: random.Random) -> Tuple[int, int]:
        profile = self._active_view_profile
        low_aspect, high_aspect = profile["aspect_ratio_range"]
        aspect = math.exp(rng.uniform(math.log(low_aspect), math.log(high_aspect)))
        long_to_short = max(aspect, 1.0 / aspect)
        capacity_from_side = profile["max_long_side"] / long_to_short
        capacity_from_area = math.sqrt(profile["max_pixels"] / long_to_short)
        lower = max(self.patch_size, profile["min_short_side"])
        upper = int(max(lower, capacity_from_side, capacity_from_area))
        short_side = rng.randint(lower, upper)
        if aspect >= 1.0:
            height, width = short_side, round(short_side * aspect)
        else:
            width, height = short_side, round(short_side / aspect)
        return max(self.patch_size, width), max(self.patch_size, height)

    def _limit_resolution(self, image: Image.Image) -> Image.Image:
        profile = self._active_view_profile
        width, height = image.size
        scale = min(
            profile["max_long_side"] / max(width, height),
            math.sqrt(profile["max_pixels"] / width * height),
        )
        width = max(self.patch_size, round(width * scale / self.patch_size) * self.patch_size)
        height = max(self.patch_size, round(height * scale / self.patch_size) * self.patch_size)
        aligned_max = max(
            self.patch_size,
            profile["max_long_side"] // self.patch_size * self.patch_size,
        )
        width, height = min(width, aligned_max), min(height, aligned_max)
        while width * height > profile["max_pixels"]:
            if width >= height and width > self.patch_size:
                width -= self.patch_size
            elif height > self.patch_size:
                height -= self.patch_size
            else:
                break
        return image.resize((width, height), Image.Resampling.LANCZOS).convert("L")

    @staticmethod
    def _crop_signal(image: Image.Image) -> float:
        values = np.asarray(image, dtype=np.float32)
        if not values.size:
            return 0.0
        return float(np.percentile(values, 95) - np.percentile(values, 5))

    def _random_resized_crop(
        self,
        image: Image.Image,
        profile: ViewProfile,
        rng: random.Random,
    ) -> Image.Image:
        width, height = image.size
        area = width * height
        scale_low, scale_high = profile["crop_scale"]
        aspect_low, aspect_high = profile["crop_aspect_ratio"]
        best_crop = image
        best_signal = -1.0
        for _ in range(15):
            target_area = area * rng.uniform(scale_low, scale_high)
            aspect = math.exp(rng.uniform(math.log(aspect_low), math.log(aspect_high)))
            crop_width = max(1, round(math.sqrt(target_area * aspect)))
            crop_height = max(1, round(math.sqrt(target_area / aspect)))
            if crop_width > width or crop_height > height:
                continue
            left = rng.randint(0, width - crop_width)
            top = rng.randint(0, height - crop_height)
            candidate = image.crop((left, top, left + crop_width, top + crop_height))
            signal = self._crop_signal(candidate)
            if signal > best_signal:
                best_crop, best_signal = candidate, signal
            # Avoid local crops containing only illuminated paper/background.
            if signal >= 32.0:
                return candidate
        return best_crop.copy()

    def _render_source(
        self, style_id: int, rng: random.Random
    ) -> Tuple[Image.Image, str]:
        last_error: Exception | None = None
        self._active_view_profile = self.global_view_profile
        for _ in range(self.max_render_attempts):
            mode = "layout" if rng.random() < self.layout_probability else "glyph_montage"
            try:
                image = (
                    super()._render_layout(style_id, rng)
                    if mode == "layout"
                    else super()._render_glyph_montage(style_id, rng)
                )
                return image.convert("L"), mode
            except (OSError, ValueError, RuntimeError) as exc:
                last_error = exc
        raise RuntimeError(
            f"Failed to render source for {Path(self.fonts[style_id]).name}: {last_error}"
        ) from last_error

    def _finalize_view(
        self,
        source: Image.Image,
        profile: ViewProfile,
        rng: random.Random,
    ) -> torch.Tensor:
        image = self._random_resized_crop(source, profile, rng)
        image = super()._camera_augmentation(image, rng)
        self._active_view_profile = profile
        image = self._limit_resolution(image)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return (torch.from_numpy(array).unsqueeze(0) - self.mean) / self.std

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        style_id, rng = index % len(self.fonts), self._rng()

        views: List[torch.Tensor] = []
        modes: List[str] = []
        view_types: List[str] = []
        source_indices: List[int] = []
        sources: List[Tuple[Image.Image, str]] = []
        for global_index in range(self.num_global_views):
            source, mode = self._render_source(style_id, rng)
            view = self._finalize_view(source.copy(), self.global_view_profile, rng)
            sources.append((source, mode))
            views.append(view)
            modes.append(mode)
            view_types.append("global")
            source_indices.append(global_index)

        for _ in range(self.num_local_views):
            source_index = rng.randrange(len(sources))
            source, mode = sources[source_index]
            view = self._finalize_view(source.copy(), self.local_view_profile, rng)
            views.append(view)
            modes.append(mode)
            view_types.append("local")
            source_indices.append(source_index)

        font_path = self.fonts[style_id]
        return {
            "views": views,
            "view_types": view_types,
            "source_indices": source_indices,
            "style_id": style_id,
            "font_name": Path(font_path).name,
            "font_path": font_path,
            "generation_modes": modes,
            "image_sizes": [(int(view.shape[-2]), int(view.shape[-1])) for view in views],
        }


class FontStyleCollator(_BaseFontStyleCollator):
    """Collate multi-crop views and disable iBOT masking on local views."""

    def __init__(
        self,
        patch_size: int = 16,
        ibot_mask_ratio: float | Sequence[float] = (0.1, 0.5),
        keep_images: bool = False,
        num_global_views: int = 2,
    ) -> None:
        super().__init__(patch_size, ibot_mask_ratio, keep_images, num_global_views)

    def __call__(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        items = list(batch)
        result = super().__call__(items)
        if len(result["views"]) < self.num_global_views:
            raise ValueError("Batch contains fewer views than num_global_views")
        expected_types = ["global"] * self.num_global_views + ["local"] * (
            len(result["views"]) - self.num_global_views
        )
        for item in items:
            if "view_types" in item and list(item["view_types"]) != expected_types:
                raise ValueError("Views must be ordered as global views followed by local views")
        for view in result["views"][self.num_global_views :]:
            view["ibot_mask"].zero_()
        result["view_types"] = expected_types
        result["source_indices"] = [list(item.get("source_indices", [])) for item in items]
        return result


def collate_font_style(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    return FontStyleCollator(patch_size=16, num_global_views=2)(batch)


__all__ = [
    "FONT_EXTENSIONS",
    "FontStyleDataset",
    "FontStyleCollator",
    "collate_font_style",
]
