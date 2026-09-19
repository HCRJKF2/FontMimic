"""Font mimic dataset with separate glyph localization and generation targets."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from dataset.font_mimic_dataset import FontMimicDataset, _patchify_reference_images


class FontMimicBBoxDataset(FontMimicDataset):
    """Return a bounding box and a tight crop instead of a full target image.

    ``target_bboxes`` uses normalized ``(left, top, right, bottom)`` coordinates.
    Right and bottom are exclusive, so a box covering the complete source image is
    ``(0, 0, 1, 1)``. ``target_crops`` contains the corresponding padded regions,
    resized to ``crop_image_size`` and kept in the same ``[-1, 1]`` range as the
    original :class:`FontMimicDataset` targets.

    Extra dataset options:

    - ``crop_image_size``: output crop size, either an integer or ``[height, width]``;
      defaults to ``letter_image_size``.
    - ``bbox_padding``: fixed padding in source-image pixels; defaults to 0.
    - ``bbox_padding_ratio``: padding on each side relative to the unpadded glyph
      width/height; defaults to 0.1.
    - ``foreground_threshold``: pixels darker than this value in the source
      ``[0, 1]`` range are treated as glyph pixels; defaults to 0.99.
    """

    def __init__(self, opt: Mapping[str, Any]) -> None:
        options = copy.deepcopy(dict(opt))
        super().__init__(options)

        crop_size = options.get("crop_image_size", self.letter_image_size)
        self.crop_image_size = (int(crop_size), int(crop_size))

        self.bbox_padding = int(options.get("bbox_padding", 0))
        self.bbox_padding_ratio = float(options.get("bbox_padding_ratio", 0.1))
        self.foreground_threshold = float(options.get("foreground_threshold", 0.95))
        if not 0.0 <= self.foreground_threshold <= 1.0:
            raise ValueError("foreground_threshold must be in [0, 1]")

    def _find_padded_bbox(self, image: torch.Tensor) -> tuple[int, int, int, int]:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(f"Expected grayscale [1, H, W], got {tuple(image.shape)}")
        height, width = image.shape[-2:]

        # FontMimicDataset normalizes [0, 1] images to [-1, 1].
        normalized_threshold = self.foreground_threshold * 2.0 - 1.0
        foreground = image[0] < normalized_threshold
        rows, columns = torch.where(foreground)
        if rows.numel() == 0:
            return 0, 0, width, height

        top = int(rows.min().item())
        bottom = int(rows.max().item()) + 1
        left = int(columns.min().item())
        right = int(columns.max().item()) + 1

        glyph_height = bottom - top
        glyph_width = right - left
        padding_y = self.bbox_padding + int(glyph_height * self.bbox_padding_ratio)
        padding_x = self.bbox_padding + int(glyph_width * self.bbox_padding_ratio)

        # Use the larger padded dimension as the side length so the crop can be
        # resized to a square without changing the glyph's aspect ratio. Near an
        # image boundary, shift the complete square inward instead of clipping an
        # individual side, which would make the result rectangular again.
        side = max(glyph_width + 2 * padding_x, glyph_height + 2 * padding_y)
        side = min(side, width, height)
        square_left = (left + right - side) // 2
        square_top = (top + bottom - side) // 2
        square_left = max(0, min(square_left, width - side))
        square_top = max(0, min(square_top, height - side))
        return square_left, square_top, square_left + side, square_top + side

    def _split_target(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, top, right, bottom = self._find_padded_bbox(image)
        height, width = image.shape[-2:]
        bbox = image.new_tensor(
            [left / width, top / height, right / width, bottom / height]
        )
        crop = image[:, top:bottom, left:right].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=self.crop_image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return bbox, crop

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        target_images = item.pop("target_images")
        split_targets = [self._split_target(image) for image in target_images]
        item["target_bboxes"] = torch.stack([target[0] for target in split_targets])
        item["target_crops"] = torch.stack([target[1] for target in split_targets])
        return item


class FontMimicBBoxCollator:
    """Batch :class:`FontMimicBBoxDataset` samples."""

    def __init__(self, patch_size: int = 16, keep_reference_images: bool = True) -> None:
        self.patch_size = int(patch_size)
        self.keep_reference_images = bool(keep_reference_images)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

    def __call__(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        items = list(batch)
        if not items:
            raise ValueError("Cannot collate an empty batch")
        target_counts = {int(item["target_crops"].shape[0]) for item in items}
        if len(target_counts) != 1:
            raise ValueError("All fonts in a batch must have the same number of target glyphs")

        references = [item["style_reference"] for item in items]
        result: Dict[str, Any] = {
            "style_view": _patchify_reference_images(references, self.patch_size),
            "target_bboxes": torch.stack([item["target_bboxes"] for item in items]),
            "target_crops": torch.stack([item["target_crops"] for item in items]),
            "letter_ids": torch.stack([item["letter_ids"] for item in items]),
            "style_ids": torch.tensor(
                [int(item["style_id"]) for item in items], dtype=torch.long
            ),
            "letters": [list(item["letters"]) for item in items],
            "font_names": [str(item["font_name"]) for item in items],
            "font_paths": [str(item["font_path"]) for item in items],
            "reference_modes": [str(item["reference_mode"]) for item in items],
        }
        if self.keep_reference_images:
            result["style_reference_images"] = references
        return result


def collate_font_mimic_bbox(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    return FontMimicBBoxCollator()(batch)
