"""Paired font-style references and grouped target glyphs for GAN training."""

from __future__ import annotations

import copy
import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import ImageFont
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.base_dataset import BaseDataset
from dataset.font_style_dataset_base import FontStyleDataset

LETTERS = string.ascii_letters
LETTER_TO_ID = {letter: index for index, letter in enumerate(LETTERS)}
ID_TO_LETTER = {index: letter for index, letter in enumerate(LETTERS)}


class FontMimicDataset(Dataset):
    """Return one style reference and N target glyphs from the same font.

    Reference rendering is delegated to FontStyleDataset. Target rendering calls
    BaseDataset.draw_letter, which is the same primitive used by MimicDataset.
    """

    def __init__(self, opt: Mapping[str, Any]) -> None:
        super().__init__()
        options = copy.deepcopy(dict(opt))
        self.font_root_path = str(options["font_root_path"])
        self.samples_per_font = int(options.get("samples_per_font", 1))
        self.letters_per_font = int(options.get("letters_per_font", 8))
        self.letter_image_size = int(options.get("letter_image_size", 256))
        self.letter_size = int(options.get("letter_size", 220))
        self.characters = "".join(
            dict.fromkeys(str(options.get("characters", string.ascii_letters)))
        )
        self.max_glyph_attempts = int(options.get("max_glyph_attempts", 104))

        if not 1 <= self.letters_per_font <= len(LETTERS):
            raise ValueError(f"letters_per_font must be in [1, {len(LETTERS)}]")
        if self.letter_image_size != 256:
            raise ValueError("The conditional GAN requires letter_image_size=256")

        reference_options = copy.deepcopy(options.get("reference", {}))
        reference_options["font_root_path"] = self.font_root_path
        reference_options["samples_per_font"] = self.samples_per_font
        reference_options["num_views"] = 1
        self.reference_dataset = FontStyleDataset(reference_options)
        self.fonts = list(self.reference_dataset.fonts)

        allowed = set(self.characters) & set(LETTERS)
        self.supported_letters: list[tuple[str, ...]] = []
        insufficient = []
        for font_path, supported in zip(
            self.fonts, self.reference_dataset.supported_characters
        ):
            candidates = tuple(ch for ch in LETTERS if ch in allowed and ch in supported)
            self.supported_letters.append(candidates)
            if len(candidates) < self.letters_per_font:
                insufficient.append((Path(font_path).name, len(candidates)))
        if insufficient:
            preview = ", ".join(f"{name}({count})" for name, count in insufficient[:8])
            raise RuntimeError(
                f"{len(insufficient)} fonts have fewer than {self.letters_per_font} "
                f"supported ASCII letters: {preview}"
            )

        self.target_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.reference_dataset)

    @staticmethod
    def _rng() -> random.Random:
        return random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        reference_item = self.reference_dataset[index]
        style_id = int(reference_item["style_id"])
        font_path = self.fonts[style_id]
        font = ImageFont.truetype(font_path, self.letter_size, index=0)
        candidates = list(self.supported_letters[style_id])
        self._rng().shuffle(candidates)

        images, letters = [], []
        attempts = 0
        while candidates and len(images) < self.letters_per_font:
            attempts += 1
            letter = candidates.pop()
            try:
                image = BaseDataset.draw_letter(self, letter, font=font)
            except (OSError, ValueError):
                if attempts >= self.max_glyph_attempts:
                    break
                continue
            if image.mode in ("RGB", "RGBA"):
                image = image.convert("L")
            images.append(self.target_transform(image))
            letters.append(letter)
        if len(images) != self.letters_per_font:
            raise RuntimeError(
                f"Rendered only {len(images)}/{self.letters_per_font} glyphs for "
                f"{Path(font_path).name}"
            )

        return {
            "style_reference": reference_item["views"][0],
            "target_images": torch.stack(images),
            "letter_ids": torch.tensor(
                [LETTER_TO_ID[letter] for letter in letters], dtype=torch.long
            ),
            "letters": letters,
            "style_id": style_id,
            "font_name": reference_item["font_name"],
            "font_path": font_path,
            "reference_mode": reference_item["generation_modes"][0],
        }


def _patchify_reference_images(
    images: Sequence[torch.Tensor], patch_size: int
) -> Dict[str, torch.Tensor]:
    patch_rows = []
    grid_sizes = []
    for image in images:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(f"Expected grayscale [1, H, W], got {tuple(image.shape)}")
        pad_h = (-image.shape[-2]) % patch_size
        pad_w = (-image.shape[-1]) % patch_size
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        grid_h = image.shape[-2] // patch_size
        grid_w = image.shape[-1] // patch_size
        patches = image.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )
        patches = patches.permute(1, 2, 0, 3, 4).reshape(grid_h * grid_w, -1)
        patch_rows.append(patches.contiguous())
        grid_sizes.append((grid_h, grid_w))

    max_tokens = max(row.shape[0] for row in patch_rows)
    patch_dim = patch_rows[0].shape[1]
    padded = torch.zeros(len(images), max_tokens, patch_dim, dtype=patch_rows[0].dtype)
    padding_mask = torch.ones(len(images), max_tokens, dtype=torch.bool)
    for index, row in enumerate(patch_rows):
        padded[index, : row.shape[0]] = row
        padding_mask[index, : row.shape[0]] = False
    return {
        "patches": padded,
        "patch_padding_mask": padding_mask,
        "grid_sizes": torch.tensor(grid_sizes, dtype=torch.long),
    }


class FontMimicCollator:
    def __init__(self, patch_size: int = 16, keep_reference_images: bool = True) -> None:
        self.patch_size = int(patch_size)
        self.keep_reference_images = bool(keep_reference_images)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

    def __call__(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        items = list(batch)
        if not items:
            raise ValueError("Cannot collate an empty batch")
        target_counts = {int(item["target_images"].shape[0]) for item in items}
        if len(target_counts) != 1:
            raise ValueError("All fonts in a batch must have the same number of target glyphs")

        references = [item["style_reference"] for item in items]
        result: Dict[str, Any] = {
            "style_view": _patchify_reference_images(references, self.patch_size),
            "target_images": torch.stack([item["target_images"] for item in items]),
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


def collate_font_mimic(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    return FontMimicCollator()(batch)
