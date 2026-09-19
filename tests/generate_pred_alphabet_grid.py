"""Generate a 52-letter alphabet grid from one or more font reference images.

The generator and frozen FontStyleViT are loaded from checkpoints produced by
``train.py``.  Both a regular training checkpoint (for example ``latest.pt``)
and the compact ``font_generator.pt`` export are supported. When the
generator predicts crop positions, predicted crops are restored to the original
canvas before the alphabet grid is placed beside the input reference image.
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path
from typing import Mapping

import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.font_inference import (
    _as_mapping, load_models, prepare_reference_image, generate_letters,
    restore_predicted_glyphs, make_grid, choose_device,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A reference image or a directory containing reference images.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="A train.py checkpoint or exported font_generator.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/generated_pred_alphabet_grids"),
        help=(
            "Output directory. For a single input image, this may instead be an "
            "explicit .png/.jpg/.jpeg/.webp file path."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train.yaml",
        help="Fallback config, mainly needed by compact font_generator.pt exports.",
    )
    parser.add_argument(
        "--style-checkpoint",
        type=Path,
        default=None,
        help="Override the FontStyleViT checkpoint recorded by the GAN checkpoint.",
    )
    parser.add_argument("--device", default="cuda", help="For example cuda, cuda:0, or cpu.")
    parser.add_argument("--columns", type=int, default=13, help="Grid column count.")
    parser.add_argument("--gap", type=int, default=4, help="White gap between cells.")
    parser.add_argument(
        "--batch-size", type=int, default=13, help="Letters generated per forward pass."
    )
    parser.add_argument("--seed", type=int, default=42, help="Latent-noise seed.")
    parser.add_argument(
        "--generator-state",
        choices=("auto", "generator_ema", "generator"),
        default="auto",
        help="Generator state stored in a full training checkpoint.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search input directories recursively (default: true).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 autocast on CUDA (default: true).",
    )
    parser.add_argument(
        "--strict-inputs",
        action="store_true",
        help="Stop on the first unreadable image instead of skipping it.",
    )
    return parser.parse_args()


def find_input_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")
    return images


def concat_reference(
    grid: Image.Image,
    reference_path: Path,
    cell_width: int,
    gap: int,
) -> Image.Image:
    """Place the original reference in a white panel to the left of the grid."""
    panel_width = cell_width * 2 + gap
    with Image.open(reference_path) as opened:
        reference = ImageOps.exif_transpose(opened)
        if reference.mode in {"RGBA", "LA"} or "transparency" in reference.info:
            rgba = reference.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            reference = Image.alpha_composite(background, rgba)
        reference = reference.convert(grid.mode)
        reference.thumbnail(
            (panel_width, grid.height),
            Image.Resampling.LANCZOS,
        )

    panel = Image.new(grid.mode, (panel_width, grid.height), "white")
    left = (panel_width - reference.width) // 2
    top = (grid.height - reference.height) // 2
    panel.paste(reference, (left, top))

    combined = Image.new(
        grid.mode,
        (panel.width + gap + grid.width, grid.height),
        "white",
    )
    combined.paste(panel, (0, 0))
    combined.paste(grid, (panel.width + gap, 0))
    return combined


def output_path_for(
    image_path: Path,
    input_path: Path,
    output: Path,
    single_explicit_file: bool,
) -> Path:
    if single_explicit_file:
        return output
    relative_parent = (
        image_path.relative_to(input_path.resolve()).parent if input_path.is_dir() else Path()
    )
    return output / relative_parent / f"{image_path.stem}_alphabet_grid.png"


def individual_output_paths(combined_path: Path) -> tuple[Path, Path]:
    """Return paths for the standalone reference and generated grid images."""
    stem = combined_path.stem
    if stem.endswith("_alphabet_grid"):
        stem = stem[: -len("_alphabet_grid")]
    return (
        combined_path.with_name(f"{stem}_reference{combined_path.suffix}"),
        combined_path.with_name(f"{stem}_generated{combined_path.suffix}"),
    )


def save_reference_image(reference_path: Path, destination: Path) -> None:
    """Save an EXIF-oriented, opaque copy of the original reference image."""
    with Image.open(reference_path) as opened:
        reference = ImageOps.exif_transpose(opened)
        if reference.mode in {"RGBA", "LA"} or "transparency" in reference.info:
            rgba = reference.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            reference = Image.alpha_composite(background, rgba)
        reference.convert("RGB").save(destination)


def main() -> None:
    args = parse_args()
    if args.columns <= 0 or args.gap < 0 or args.batch_size <= 0:
        raise ValueError("--columns and --batch-size must be positive; --gap cannot be negative")

    input_path = args.input.expanduser()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    images = find_input_images(input_path, args.recursive)
    output = args.output.expanduser()
    explicit_output = (
        input_path.is_file()
        and output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if input_path.is_dir() and output.suffix.lower() in IMAGE_EXTENSIONS:
        raise ValueError("--output must be a directory when --input is a directory")

    device = choose_device(str(args.device))
    generator, style_encoder, config, state_name = load_models(
        checkpoint_path,
        config_path,
        args.style_checkpoint,
        args.generator_state,
        device,
    )

    saved_letters = str(string.ascii_letters)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw_checkpoint, Mapping) and raw_checkpoint.get("letters") is not None:
        saved_letters = "".join(raw_checkpoint["letters"])
    missing = [letter for letter in string.ascii_letters if letter not in saved_letters]
    if missing:
        raise ValueError(f"Checkpoint is missing required letters: {''.join(missing)}")
    letter_ids = torch.tensor(
        [saved_letters.index(letter) for letter in string.ascii_letters], dtype=torch.long
    )
    if int(letter_ids.max()) >= generator.letter_embedding.num_embeddings:
        raise ValueError("Checkpoint letter ids exceed the generator's embedding table")

    cpu_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(letter_ids), generator.latent_dim, generator=cpu_rng)
    dataset_options = _as_mapping(config.get("dataset"))
    reference_options = _as_mapping(dataset_options.get("reference"))

    saved, failed = 0, []
    print(
        f"Loaded {state_name} on {device}; generating {len(images)} alphabet grid(s)",
        flush=True,
    )
    for index, image_path in enumerate(images, start=1):
        try:
            reference = prepare_reference_image(
                image_path, style_encoder.patch_size, reference_options
            )
            glyphs, predicted_positions = generate_letters(
                reference,
                generator,
                style_encoder,
                letter_ids,
                noise,
                args.batch_size,
                device,
                args.amp,
            )
            if predicted_positions is not None:
                glyphs = restore_predicted_glyphs(
                    predicted_crops=glyphs,
                    predicted_positions=predicted_positions,
                    canvas_size=int(
                        dataset_options.get("letter_image_size", glyphs.shape[-1])
                    ),
                )
            generated_grid = make_grid(glyphs, args.columns, args.gap)
            combined_grid = concat_reference(
                generated_grid,
                image_path,
                cell_width=int(glyphs.shape[-1]),
                gap=args.gap,
            )
            destination = output_path_for(
                image_path, input_path, output, explicit_output
            ).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            reference_destination, generated_destination = individual_output_paths(
                destination
            )
            combined_grid.save(destination)
            save_reference_image(image_path, reference_destination)
            generated_grid.save(generated_destination)
            saved += 1
            print(
                f"[{index}/{len(images)}] {image_path.name} -> "
                f"{destination}, {reference_destination}, {generated_destination}",
                flush=True,
            )
        except Exception as exc:
            if args.strict_inputs:
                raise
            failed.append((image_path, exc))
            print(f"[{index}/{len(images)}] Skipped {image_path}: {exc}", flush=True)

    if not saved:
        details = "; ".join(f"{path.name}: {error}" for path, error in failed)
        raise RuntimeError(f"No grids were generated. {details}")
    print(f"Saved {saved} grid(s); skipped {len(failed)} image(s).", flush=True)


if __name__ == "__main__":
    main()

    """
    CUDA_VISIBLE_DEVICES=0 python tests/generate_pred_alphabet_grid.py \
      --input ./data/fonts_dataset/test_data \
      --checkpoint outputs/font_cgan_parseq/checkpoints/epoch_0400.pt \
      --output outputs/pred_alphabet_grids
    """
