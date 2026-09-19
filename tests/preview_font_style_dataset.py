"""Generate FontStyleDataset images for visual inspection.

The script reads the same dataset section as training, keeps the latest
``[global views..., local views...]`` ordering, and saves every normalized
single-channel tensor back to a lossless grayscale PNG.

Examples:
    python tests/preview_font_style_dataset.py --samples 4
    python tests/preview_font_style_dataset.py \
        --config config/train_font_style.yaml \
        --font-root D:/data/fonts_dataset \
        --output-dir outputs/font_style_preview
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

# Running this file directly puts tests/ rather than the repository root on
# sys.path, so add the project root before importing the dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.font_style_dataset import FontStyleDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train_font_style.yaml",
        help="Training YAML whose dataset section is used for rendering.",
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
        default=PROJECT_ROOT / "outputs" / "font_style_preview",
    )
    parser.add_argument("--samples", type=int, default=4, help="Number of dataset items.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--global-views",
        type=int,
        default=None,
        help="Optional override; otherwise use dataset.num_global_views.",
    )
    parser.add_argument(
        "--local-views",
        type=int,
        default=None,
        help="Optional override; otherwise use dataset.num_local_views.",
    )
    parser.add_argument("--contact-sheet-columns", type=int, default=3)
    return parser.parse_args()


def load_dataset_options(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping) or not isinstance(config.get("dataset"), Mapping):
        raise ValueError(f"{config_path} must contain a mapping named 'dataset'")

    options = dict(config["dataset"])
    if args.font_root is not None:
        options["font_root_path"] = args.font_root.resolve()
    elif "font_root_path" not in options:
        raise ValueError("dataset.font_root_path is missing; pass --font-root")
    else:
        font_root = Path(options["font_root_path"]).expanduser()
        if not font_root.is_absolute():
            # Training is normally launched from the repository root.
            font_root = (PROJECT_ROOT / font_root).resolve()
        options["font_root_path"] = font_root

    if args.global_views is not None:
        options["num_global_views"] = args.global_views
    if args.local_views is not None:
        options["num_local_views"] = args.local_views
    if args.global_views is not None or args.local_views is not None:
        options["num_views"] = int(options.get("num_global_views", 2)) + int(
            options.get("num_local_views", 4)
        )

    # Only dataset length changes; rendering settings remain identical to training.
    options["samples_per_font"] = max(1, int(options.get("samples_per_font", 1)))
    return options


def tensor_to_grayscale_image(
    tensor: torch.Tensor,
    mean: float,
    std: float,
) -> Image.Image:
    """Undo normalization and convert a [1, H, W] tensor to an L-mode image."""
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected a single-channel [1, H, W] tensor, got {tuple(tensor.shape)}")
    if std <= 0:
        raise ValueError("Dataset std must be positive")
    pixels = tensor.detach().cpu().float().mul(std).add(mean).clamp(0.0, 1.0)
    array = pixels.squeeze(0).mul(255).round().to(torch.uint8).numpy()
    image = Image.fromarray(array, mode="L")
    if image.mode != "L":
        raise AssertionError(f"Expected grayscale output, got mode={image.mode}")
    return image


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return value or "font"


def make_contact_sheet(
    previews: List[Tuple[Image.Image, str]],
    output_path: Path,
    columns: int,
    cell_size: Tuple[int, int] = (384, 288),
) -> None:
    """Save labeled thumbnails while preserving original PNG dimensions."""
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
        draw.text((x0 + 8, y0 + 7), label[:64], fill=20)
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

    dataset_options = load_dataset_options(args)
    dataset = FontStyleDataset(dataset_options)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mean = float(dataset_options.get("mean", 0.5))
    std = float(dataset_options.get("std", 0.5))
    previews: List[Tuple[Image.Image, str]] = []
    sample_records: List[dict[str, Any]] = []

    for preview_index in range(args.samples):
        dataset_index = (args.start_index + preview_index) % len(dataset)
        item = dataset[dataset_index]
        font_stem = safe_filename(Path(item["font_name"]).stem)
        sample_dir = output_dir / (
            f"sample_{preview_index:03d}_dataset_{dataset_index:06d}_"
            f"style_{item['style_id']:04d}_{font_stem}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        view_records: List[dict[str, Any]] = []

        for view_index, tensor in enumerate(item["views"]):
            view_type = item["view_types"][view_index]
            source_index = int(item["source_indices"][view_index])
            generation_mode = safe_filename(item["generation_modes"][view_index])
            image = tensor_to_grayscale_image(tensor, mean=mean, std=std)
            filename = (
                f"view_{view_index:02d}_{view_type}_source_{source_index}_"
                f"{generation_mode}_{image.width}x{image.height}.png"
            )
            image_path = sample_dir / filename
            image.save(image_path, format="PNG")

            label = (
                f"#{preview_index} {font_stem} v{view_index} {view_type} "
                f"src{source_index} {image.width}x{image.height}"
            )
            previews.append((image, label))
            view_records.append(
                {
                    "view_index": view_index,
                    "view_type": view_type,
                    "source_index": source_index,
                    "generation_mode": item["generation_modes"][view_index],
                    "height": image.height,
                    "width": image.width,
                    "mode": image.mode,
                    "path": str(image_path.relative_to(output_dir)),
                }
            )
            print(image_path)

        sample_records.append(
            {
                "preview_index": preview_index,
                "dataset_index": dataset_index,
                "style_id": int(item["style_id"]),
                "font_name": item["font_name"],
                "font_path": str(item["font_path"]),
                "views": view_records,
            }
        )

    contact_sheet_path = output_dir / "contact_sheet.png"
    make_contact_sheet(
        previews,
        contact_sheet_path,
        columns=args.contact_sheet_columns,
    )
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": args.seed,
                "config": str(args.config.resolve()),
                "dataset_options": json_safe(dataset_options),
                "samples": sample_records,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Metadata: {metadata_path}")
    print(
        f"Saved {len(previews)} grayscale views from {args.samples} samples "
        f"({dataset.num_global_views} global + {dataset.num_local_views} local per sample)."
    )


if __name__ == "__main__":
    main()
