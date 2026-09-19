"""Load font-style training views directly from handwriting images.

Each image file is treated as an independent style.  Multiple global/local
views are augmented crops of that image, and the returned sample follows the
same contract as :mod:`dataset.font_style_dataset` so its collator and training
loop can be reused unchanged.

Required option:
    image_dir: Directory containing the source images.

Useful options:
    samples_per_image: Virtual samples generated from each image (default 1000).
    recursive_images: Search child directories recursively (default true).
    image_extensions: Optional iterable of filename extensions.
    strict_images: Raise on an unreadable image instead of skipping it.

All view, resolution, normalization, and augmentation options accepted by
``dataset.font_style_dataset.FontStyleDataset`` are also supported.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from dataset.font_style_dataset import (
    FontStyleCollator,
    FontStyleDataset,
)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class ImageFontStyleDataset(FontStyleDataset):
    """Create DINO multi-crop style views from existing handwriting images.

    The inheritance is used only to share the augmentation and multi-crop
    implementation.  Font discovery and text rendering are deliberately
    bypassed: one file in ``image_dir`` always maps to exactly one ``style_id``.
    """

    def __init__(self, opt: Mapping[str, Any]):
        # Do not call the rendered dataset initializer: it requires font files
        # and a wordlist.  Initialize the common Dataset state directly.
        Dataset.__init__(self)
        options = dict(opt)
        self.opt = options

        self.image_dir = Path(options["image_dir"]).expanduser().resolve()

        self.samples_per_image = int(options.get("samples_per_image", 8))
        self.patch_size = int(options.get("patch_size", 16))
        self.mean = float(options.get("mean", 0.5))
        self.std = float(options.get("std", 0.5))
        self.recursive_images = bool(options.get("recursive_images", True))
        self.image_extensions = IMAGE_EXTENSIONS

        explicit_multicrop = "num_global_views" in options or "num_local_views" in options
        if explicit_multicrop:
            self.num_global_views = int(options.get("num_global_views", 2))
            self.num_local_views = int(options.get("num_local_views", 4))
        else:
            self.num_global_views = int(options.get("num_views", 2))
            self.num_local_views = 0
        if self.num_global_views < 1:
            raise ValueError("num_global_views must be at least one")
        self.num_views = self.num_global_views + self.num_local_views

        # These parent helpers validate and build the exact same view profiles
        # used by the rendered-font dataset.
        self._profile_options = options
        self.global_view_profile = self._build_global_profile(options)
        self.local_view_profile = self._build_local_profile(options)
        self._active_view_profile = self.global_view_profile

        glob_method = self.image_dir.rglob if self.recursive_images else self.image_dir.glob
        candidates = sorted(
            path.resolve()
            for path in glob_method("*")
            if path.is_file() and path.suffix.lower() in self.image_extensions
        )
        if not candidates:
            raise RuntimeError(f"No supported image files found in: {self.image_dir}")

        self.images: List[str] = []
        for image_path in candidates:
            self.images.append(str(image_path))

        # Compatibility aliases used by the existing checkpoint-signature code.
        self.font_dir = self.image_dir
        self.fonts = self.images

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_image

    @staticmethod
    def _load_grayscale(image_path: str) -> Image.Image:
        """Load one frame, apply EXIF orientation, and flatten alpha on white."""
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba)
            return image.convert("L").copy()

    def _render_source(
        self, style_id: int, rng: random.Random
    ) -> Tuple[Image.Image, str]:
        # ``rng`` is intentionally accepted to match the rendered dataset's
        # method contract; randomness is introduced during crop/augmentation.
        del rng
        return self._load_grayscale(self.images[style_id]), "image"


__all__ = [
    "ImageFontStyleDataset",
]
