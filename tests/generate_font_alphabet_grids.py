"""Render every training font's supported ASCII letters into JPEG grids.

The font order and character support checks match the training dataset: fonts
are sorted by path and cmap inspection is used when available.  Within each
image, supported lowercase letters are rendered first, followed by supported
uppercase letters.

Examples:
    python generate_font_alphabet_grids.py
    python generate_font_alphabet_grids.py --config config/train_fm.yaml
    python generate_font_alphabet_grids.py --font-root D:/data/fonts_dataset
"""

from __future__ import annotations

import argparse
import string
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.font_style_dataset_base import (  # noqa: E402
    FONT_EXTENSIONS,
    _probe_font_characters,
)

ASCII_LETTERS = string.ascii_lowercase + string.ascii_uppercase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train.yaml",
        help="Training YAML whose dataset section supplies the rendering defaults.",
    )
    parser.add_argument(
        "--font-root",
        type=Path,
        default=None,
        help="Override dataset.font_root_path (the directory containing font/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "font_alphabet_grids",
        help="Directory in which the JPEG grids are saved.",
    )
    parser.add_argument("--columns", type=int, default=13)
    parser.add_argument(
        "--cell-size",
        type=int,
        default=None,
        help="Square glyph cell size; defaults to dataset.letter_image_size or 256.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Initial glyph size; defaults to dataset.letter_size or 78%% of cell size.",
    )
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=None,
        help="Process at most this many fonts (useful for a quick preview).",
    )
    parser.add_argument(
        "--recursive-fonts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recursively scan font/; defaults to the training configuration.",
    )
    parser.add_argument(
        "--strict-fonts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop on an unusable font instead of warning and continuing.",
    )
    return parser.parse_args()


def _load_dataset_options(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping) or not isinstance(config.get("dataset"), Mapping):
        raise ValueError(f"{config_path} must contain a dataset mapping")
    return dict(config["dataset"])


def _nested_default(options: Mapping[str, Any], key: str, fallback: Any) -> Any:
    if key in options:
        return options[key]
    reference = options.get("reference")
    if isinstance(reference, Mapping) and key in reference:
        return reference[key]
    return fallback


def _resolve_font_root(
    options: Mapping[str, Any], override: Path | None
) -> Path:
    if override is not None:
        root = override.expanduser()
    else:
        configured = options.get("font_root_path")
        if configured is None:
            raise ValueError("dataset.font_root_path is missing; pass --font-root")
        root = Path(str(configured)).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def find_fonts(font_root: Path, recursive: bool) -> list[Path]:
    font_dir = font_root / "font"
    if not font_dir.is_dir():
        raise FileNotFoundError(f"Font directory does not exist: {font_dir}")
    glob_method = font_dir.rglob if recursive else font_dir.glob
    fonts = sorted(
        path.resolve()
        for path in glob_method("*")
        if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
    )
    if not fonts:
        raise RuntimeError(f"No supported font files found in: {font_dir}")
    return fonts


def supported_ascii_letters(font_path: Path) -> tuple[str, ...]:
    """Return supported characters in lowercase-then-uppercase order."""
    supported = set(_probe_font_characters(font_path, ASCII_LETTERS))
    return tuple(letter for letter in ASCII_LETTERS if letter in supported)


def _load_fitted_font(
    font_path: Path,
    letter: str,
    initial_size: int,
    available_size: int,
) -> ImageFont.FreeTypeFont:
    """Keep the requested size unless the glyph would be clipped by its cell."""
    font = ImageFont.truetype(str(font_path), initial_size, index=0)
    bbox = font.getbbox(letter)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        raise ValueError(f"Font rendered {letter!r} as an empty glyph")
    scale = min(1.0, available_size / width, available_size / height)
    if scale < 1.0:
        fitted_size = max(1, int(initial_size * scale))
        font = ImageFont.truetype(str(font_path), fitted_size, index=0)
    return font


def render_glyph(
    font_path: Path,
    letter: str,
    cell_size: int,
    font_size: int,
    padding: int,
) -> Image.Image:
    available_size = cell_size - 2 * padding
    font = _load_fitted_font(font_path, letter, font_size, available_size)
    bbox = font.getbbox(letter)
    glyph_width = bbox[2] - bbox[0]
    glyph_height = bbox[3] - bbox[1]
    if glyph_width <= 0 or glyph_height <= 0:
        raise ValueError(f"Font rendered {letter!r} as an empty glyph")

    cell = Image.new("RGB", (cell_size, cell_size), "white")
    x = (cell_size - glyph_width) // 2 - bbox[0]
    y = (cell_size - glyph_height) // 2 - bbox[1]
    ImageDraw.Draw(cell).text((x, y), letter, font=font, fill="black")
    return cell


def make_grid(images: Sequence[Image.Image], columns: int, gap: int) -> Image.Image:
    if not images:
        raise ValueError("At least one glyph image is required")
    columns = min(columns, len(images))
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    width = columns * cell_width + (columns - 1) * gap
    height = rows * cell_height + (rows - 1) * gap
    grid = Image.new("RGB", (width, height), "white")
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        grid.paste(image, (column * (cell_width + gap), row * (cell_height + gap)))
    return grid


def output_names(fonts: Sequence[Path]) -> list[str]:
    """Build stable names while retaining the original font stem as the prefix."""
    stem_counts = Counter(font.stem.casefold() for font in fonts)
    occurrences: defaultdict[str, int] = defaultdict(int)
    names = []
    for font in fonts:
        key = font.stem.casefold()
        occurrences[key] += 1
        duplicate_suffix = (
            f"_{occurrences[key]:02d}" if stem_counts[key] > 1 else ""
        )
        names.append(f"{font.stem}_alphabet_grid{duplicate_suffix}.jpg")
    return names


def _validate_args(args: argparse.Namespace, cell_size: int, font_size: int) -> None:
    if args.columns <= 0:
        raise ValueError("--columns must be positive")
    if cell_size <= 0 or font_size <= 0:
        raise ValueError("--cell-size and --font-size must be positive")
    if args.padding < 0 or args.padding * 2 >= cell_size:
        raise ValueError("--padding must be non-negative and less than half the cell size")
    if args.gap < 0:
        raise ValueError("--gap cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.max_fonts is not None and args.max_fonts <= 0:
        raise ValueError("--max-fonts must be positive")


def main() -> None:
    args = parse_args()
    options = _load_dataset_options(args.config)
    font_root = _resolve_font_root(options, args.font_root)
    cell_size = int(
        args.cell_size
        if args.cell_size is not None
        else options.get("letter_image_size", 256)
    )
    font_size = int(
        args.font_size
        if args.font_size is not None
        else options.get("letter_size", round(cell_size * 0.78))
    )
    _validate_args(args, cell_size, font_size)

    recursive = bool(
        _nested_default(options, "recursive_fonts", True)
        if args.recursive_fonts is None
        else args.recursive_fonts
    )
    strict = bool(
        _nested_default(options, "strict_fonts", False)
        if args.strict_fonts is None
        else args.strict_fonts
    )
    all_fonts = find_fonts(font_root, recursive)
    selected_fonts = all_fonts[args.start_index :]
    if args.max_fonts is not None:
        selected_fonts = selected_fonts[: args.max_fonts]
    if not selected_fonts:
        raise ValueError(
            f"No fonts selected: start index {args.start_index}, "
            f"available font count {len(all_fonts)}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(all_fonts)[
        args.start_index : args.start_index + len(selected_fonts)
    ]
    saved_count = 0
    skipped_count = 0

    for offset, (font_path, filename) in enumerate(zip(selected_fonts, names)):
        dataset_index = args.start_index + offset
        try:
            letters = supported_ascii_letters(font_path)
            if not letters:
                raise RuntimeError("no supported ASCII letters")
            glyphs = [
                render_glyph(
                    font_path,
                    letter,
                    cell_size=cell_size,
                    font_size=font_size,
                    padding=args.padding,
                )
                for letter in letters
            ]
            grid = make_grid(glyphs, columns=args.columns, gap=args.gap)
            output_path = output_dir / filename
            grid.save(
                output_path,
                format="JPEG",
                quality=args.jpeg_quality,
                subsampling=0,
                optimize=True,
            )
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Cannot render font {font_path}: {exc}") from exc
            skipped_count += 1
            warnings.warn(f"Skipping unusable font {font_path}: {exc}", RuntimeWarning)
            continue

        saved_count += 1
        print(
            f"[{dataset_index + 1}/{len(all_fonts)}] {font_path.name} -> "
            f"{output_path.name} ({len(letters)} letters)"
        )

    print(f"Saved {saved_count} JPEG grids to: {output_dir}")
    if skipped_count:
        print(f"Skipped {skipped_count} unusable fonts")


if __name__ == "__main__":
    main()

    """
    python tests/generate_font_alphabet_grids.py --config config/train_fm.yaml \
        --cell-size 96 --jpeg-quality 90 \
        --font-root ./data/fonts_dataset
    """
