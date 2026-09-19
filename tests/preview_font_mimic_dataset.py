"""Save FontMimicDataset samples to disk for visual inspection.

Examples:
    python tests/preview_font_mimic_dataset.py
    python tests/preview_font_mimic_dataset.py --samples 8 --start-index 10
    python tests/preview_font_mimic_dataset.py --font-root D:/data/fonts_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.font_mimic_dataset import FontMimicDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train.yaml",
        help="Training YAML whose dataset section is used.",
    )
    parser.add_argument(
        "--font-root",
        type=Path,
        default=None,
        help="Optional override for dataset.font_root_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "font_mimic_preview",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--letters-per-font", type=int, default=None)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_dataset_options(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping) or not isinstance(config.get("dataset"), Mapping):
        raise ValueError(f"{config_path} must contain a dataset mapping")

    options = dict(config["dataset"])
    if args.font_root is not None:
        font_root = args.font_root.expanduser().resolve()
    else:
        if "font_root_path" not in options:
            raise ValueError("dataset.font_root_path is missing; pass --font-root")
        font_root = Path(str(options["font_root_path"])).expanduser()
        if not font_root.is_absolute():
            font_root = (PROJECT_ROOT / font_root).resolve()
    options["font_root_path"] = str(font_root)
    if args.letters_per_font is not None:
        options["letters_per_font"] = args.letters_per_font
    return options


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return value or "font"


def reference_to_image(
    tensor: torch.Tensor, mean: float, std: float
) -> Image.Image:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected reference [1, H, W], got {tuple(tensor.shape)}")
    pixels = tensor.detach().cpu().float().mul(std).add(mean).clamp(0, 1)
    array = pixels.squeeze(0).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L")


def target_to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
        raise ValueError(f"Expected target [C, H, W], got {tuple(tensor.shape)}")
    pixels = tensor.detach().cpu().float().add(1).mul(0.5).clamp(0, 1)
    array = pixels.mul(255).round().to(torch.uint8)
    if tensor.shape[0] == 1:
        return Image.fromarray(array.squeeze(0).numpy(), mode="L")
    return Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")


def make_target_grid(
    images: Sequence[Image.Image],
    letters: Sequence[str],
    columns: int,
) -> Image.Image:
    if not images or len(images) != len(letters):
        raise ValueError("images and letters must have the same non-zero length")
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    image_width = max(image.width for image in images)
    image_height = max(image.height for image in images)
    label_height, gap = 28, 6
    cell_width = image_width + gap * 2
    cell_height = image_height + label_height + gap * 2
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (image, letter) in enumerate(zip(images, letters)):
        row, column = divmod(index, columns)
        left, top = column * cell_width, row * cell_height
        rgb = image.convert("RGB")
        x = left + (cell_width - rgb.width) // 2
        sheet.paste(rgb, (x, top + gap))
        draw.text(
            (left + gap, top + gap + image_height + 4),
            f"#{index:02d}  letter={letter!r}",
            fill="black",
        )
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline=(190, 190, 190),
        )
    return sheet


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.grid_columns <= 0:
        raise ValueError("--grid-columns must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    options = load_dataset_options(args)
    dataset = FontMimicDataset(options)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_mean = float(dataset.reference_dataset.mean)
    reference_std = float(dataset.reference_dataset.std)
    records: list[dict[str, Any]] = []

    for preview_index in range(args.samples):
        dataset_index = (args.start_index + preview_index) % len(dataset)
        item = dataset[dataset_index]
        font_stem = safe_filename(Path(item["font_name"]).stem)
        sample_dir = output_dir / (
            f"sample_{preview_index:03d}_dataset_{dataset_index:06d}_"
            f"style_{item['style_id']:04d}_{font_stem}"
        )
        targets_dir = sample_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)

        reference = reference_to_image(
            item["style_reference"], reference_mean, reference_std
        )
        reference_path = sample_dir / "style_reference.jpg"
        reference.save(reference_path, format="JPEG")

        target_images = [
            target_to_image(tensor) for tensor in item["target_images"]
        ]
        target_records = []
        for target_index, (image, letter, letter_id) in enumerate(
            zip(target_images, item["letters"], item["letter_ids"].tolist())
        ):
            target_path = targets_dir / (
                f"target_{target_index:02d}_id_{letter_id:02d}_{letter}.jpg"
            )
            image.save(target_path, format="JPEG")
            target_records.append(
                {
                    "index": target_index,
                    "letter": letter,
                    "letter_id": int(letter_id),
                    "path": str(target_path.relative_to(output_dir)),
                }
            )

        target_grid = make_target_grid(
            target_images, item["letters"], args.grid_columns
        )
        grid_path = sample_dir / "target_grid.jpg"
        target_grid.save(grid_path, format="JPEG")
        records.append(
            {
                "preview_index": preview_index,
                "dataset_index": dataset_index,
                "style_id": int(item["style_id"]),
                "font_name": item["font_name"],
                "font_path": str(item["font_path"]),
                "reference_mode": item["reference_mode"],
                "reference_size": list(reference.size),
                "reference_path": str(reference_path.relative_to(output_dir)),
                "target_grid_path": str(grid_path.relative_to(output_dir)),
                "targets": target_records,
            }
        )
        print(sample_dir)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": args.seed,
                "config": str(args.config.expanduser().resolve()),
                "dataset_length": len(dataset),
                "font_count": len(dataset.fonts),
                "letters_per_font": dataset.letters_per_font,
                "samples": records,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved {len(records)} samples to: {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

