"""Save ImageFontStyleDataset outputs as viewable grayscale PNG files.

Examples:
    python tests/preview_image_font_style_dataset.py \
        --image-dir D:/data/handwriting_images

    python tests/preview_image_font_style_dataset.py \
        --image-dir D:/data/handwriting_images \
        --config config/train_font_style.yaml \
        --samples 8 \
        --output-dir outputs/image_font_style_preview
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.image_font_style_dataset import ImageFontStyleDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing handwriting images; each image is one style.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train_font_style.yaml",
        help="YAML file whose dataset crop/normalization options are reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "image_font_style_preview",
        help="Directory in which PNG previews and metadata are saved.",
    )
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to save.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--global-views",
        type=int,
        default=None,
        help="Override dataset.num_global_views from the YAML file.",
    )
    parser.add_argument(
        "--local-views",
        type=int,
        default=None,
        help="Override dataset.num_local_views from the YAML file.",
    )
    parser.add_argument("--contact-sheet-columns", type=int, default=3)
    return parser.parse_args()


def load_dataset_options(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping) or not isinstance(config.get("dataset"), Mapping):
        raise ValueError(f"{config_path} must contain a mapping named 'dataset'")

    options = dict(config["dataset"])
    options["image_dir"] = args.image_dir.expanduser().resolve()
    # Preview length only; all crop and augmentation settings remain unchanged.
    options["samples_per_image"] = max(1, int(options.get("samples_per_image", 1)))

    if args.global_views is not None:
        options["num_global_views"] = args.global_views
    if args.local_views is not None:
        options["num_local_views"] = args.local_views
    if args.global_views is not None or args.local_views is not None:
        options["num_views"] = int(options.get("num_global_views", 2)) + int(
            options.get("num_local_views", 4)
        )
    return options


def tensor_to_image(tensor: torch.Tensor, mean: float, std: float) -> Image.Image:
    """Undo dataset normalization and convert [1, H, W] to a grayscale image."""
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected [1, H, W], got {tuple(tensor.shape)}")
    if std <= 0:
        raise ValueError("std must be positive")
    pixels = tensor.detach().cpu().float().mul(std).add(mean).clamp(0.0, 1.0)
    array = pixels.squeeze(0).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L")


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return value or "image"


def make_contact_sheet(
    previews: List[Tuple[Image.Image, str]],
    output_path: Path,
    columns: int,
    cell_size: Tuple[int, int] = (384, 288),
) -> None:
    if not previews:
        return
    columns = max(1, min(columns, len(previews)))
    rows = (len(previews) + columns - 1) // columns
    cell_width, cell_height = cell_size
    label_height = 42
    sheet = Image.new("L", (columns * cell_width, rows * cell_height), color=230)
    draw = ImageDraw.Draw(sheet)

    for index, (image, label) in enumerate(previews):
        row, column = divmod(index, columns)
        x0, y0 = column * cell_width, row * cell_height
        thumbnail = image.copy()
        thumbnail.thumbnail(
            (cell_width - 16, cell_height - label_height - 16),
            Image.Resampling.LANCZOS,
        )
        x = x0 + (cell_width - thumbnail.width) // 2
        y = y0 + label_height + (cell_height - label_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
        draw.text((x0 + 8, y0 + 7), label[:72], fill=20)
        draw.rectangle(
            (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
            outline=160,
            width=1,
        )
    sheet.save(output_path, format="PNG")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.contact_sheet_columns <= 0:
        raise ValueError("--contact-sheet-columns must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    options = load_dataset_options(args)
    dataset = ImageFontStyleDataset(options)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mean = float(options.get("mean", 0.5))
    std = float(options.get("std", 0.5))
    previews: List[Tuple[Image.Image, str]] = []
    sample_records: List[dict[str, Any]] = []

    for preview_index in range(args.samples):
        dataset_index = (args.start_index + preview_index) % len(dataset)
        item = dataset[dataset_index]
        image_name = str(item["font_name"])
        image_stem = safe_filename(Path(image_name).stem)
        sample_dir = output_dir / (
            f"sample_{preview_index:03d}_dataset_{dataset_index:06d}_"
            f"style_{int(item['style_id']):04d}_{image_stem}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        view_records: List[dict[str, Any]] = []
        for view_index, tensor in enumerate(item["views"]):
            view_type = str(item["view_types"][view_index])
            source_index = int(item["source_indices"][view_index])
            image = tensor_to_image(tensor, mean=mean, std=std)
            filename = (
                f"view_{view_index:02d}_{view_type}_source_{source_index}_"
                f"{image.width}x{image.height}.png"
            )
            image_path = sample_dir / filename
            image.save(image_path, format="PNG")

            relative_path = image_path.relative_to(output_dir)
            label = (
                f"{image_stem} | v{view_index} {view_type} | "
                f"{image.width}x{image.height}"
            )
            previews.append((image, label))
            view_records.append(
                {
                    "view_index": view_index,
                    "view_type": view_type,
                    "source_index": source_index,
                    "width": image.width,
                    "height": image.height,
                    "path": relative_path.as_posix(),
                }
            )
            print(image_path)

        sample_records.append(
            {
                "preview_index": preview_index,
                "dataset_index": dataset_index,
                "style_id": int(item["style_id"]),
                "source_name": image_name,
                "source_path": str(item["font_path"]),
                "views": view_records,
            }
        )

    contact_sheet_path = output_dir / "contact_sheet.png"
    make_contact_sheet(previews, contact_sheet_path, args.contact_sheet_columns)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": args.seed,
                "config": str(args.config.expanduser().resolve()),
                "dataset_options": json_safe(options),
                "samples": sample_records,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Metadata: {metadata_path}")
    print(
        f"Saved {len(previews)} views from {args.samples} samples "
        f"({dataset.num_global_views} global + {dataset.num_local_views} local each)."
    )


if __name__ == "__main__":
    main()
