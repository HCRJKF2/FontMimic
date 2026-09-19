"""Generate one TTF and preview for every style-reference image in a directory.

Example:
    python generate_font.py --input references --checkpoint font_generator.pt \
        --style-checkpoint font_style_encoder.pt --output outputs/generated_fonts
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from collections.abc import Mapping

from PIL import Image

from utils.font_export import DEFAULT_TEXT, build_ttf, render_specimen

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing style-reference images.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="train.py checkpoint or font_generator.pt.")
    parser.add_argument("--style-checkpoint", type=Path, help="Override the recorded FontStyleViT checkpoint path.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/train.yaml",
                        help="For compact exports, use the original training config.")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_fonts"),
                        help="Directory receiving one TTF and asset directory per input image.")
    parser.add_argument("--family-name", default="Mimic Font")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator-state", choices=("auto", "generator_ema", "generator"), default="auto")
    parser.add_argument("--batch-size", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True,
                        help="Search subdirectories and preserve their layout (default: true).")
    parser.add_argument("--strict-inputs", action="store_true",
                        help="Stop on the first failed image instead of continuing.")
    parser.add_argument("--threshold", type=int, default=160, help="Generated pixels darker than this become ink (1..255).")
    parser.add_argument("--min-component-area", type=int, default=3, help="Minimum connected ink area in source pixels.")
    parser.add_argument("--simplify", type=float, default=0.35, help="Contour tolerance in source pixels (0..2).")
    parser.add_argument("--baseline", type=float, help="Override baseline as a fraction of canvas height, e.g. 0.75.")
    parser.add_argument("--side-bearing", type=int, default=50, help="Spacing on either side, in 1000-unit em coordinates.")
    parser.add_argument("--space-width", type=int, default=300, help="Space advance in font units.")
    parser.add_argument("--font-size", type=int, default=64, help="Preview font size in pixels.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Preview text: ASCII letters, spaces and newlines only.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.font_size <= 0:
        parser.error("--batch-size and --font-size must be positive")
    if args.output.suffix.lower() == ".ttf":
        parser.error("--output must be a directory, not a .ttf path")
    if not args.text.strip() or set(args.text) - set(string.ascii_letters + " \n"):
        parser.error("--text must contain only ASCII letters, spaces and newlines, and cannot be empty")
    if not 1 <= args.threshold <= 255 or args.min_component_area < 1:
        parser.error("--threshold must be 1..255; --min-component-area must be positive")
    if not 0 <= args.simplify <= 2 or (args.baseline is not None and not 0 < args.baseline < 1):
        parser.error("--simplify must be 0..2; --baseline must be between 0 and 1")
    if not 0 <= args.side_bearing <= 1000 or not 1 <= args.space_width <= 2000:
        parser.error("--side-bearing must be 0..1000; --space-width must be 1..2000")
    if not args.family_name.strip():
        parser.error("--family-name cannot be empty")
    return args


def concat_reference_and_preview(
    reference_path: Path,
    preview_path: Path,
    output_path: Path,
    gap: int = 20,
) -> None:
    from PIL import ImageOps

    with Image.open(reference_path) as opened_reference:
        reference = ImageOps.exif_transpose(opened_reference)
        if reference.mode in {"RGBA", "LA"} or "transparency" in reference.info:
            rgba = reference.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            reference = Image.alpha_composite(background, rgba)
        reference = reference.convert("RGB")
    with Image.open(preview_path) as opened_preview:
        preview = opened_preview.convert("RGB")

    # Put the source image in a square panel as tall as the preview. This keeps
    # very wide reference images from overwhelming the generated specimen.
    panel_size = preview.height
    reference.thumbnail((panel_size - gap * 2, panel_size - gap * 2), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_size, panel_size), "white")
    panel.paste(reference, ((panel_size - reference.width) // 2,
                            (panel_size - reference.height) // 2))
    canvas = Image.new("RGB", (panel.width + gap + preview.width, preview.height), "white")
    canvas.paste(panel, (0, 0))
    canvas.paste(preview, (panel.width + gap, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate_one(reference_path, output, family_name, args, generator, encoder,
                 dataset, ids, noise, state, checkpoint_path, device):
    """Generate and save one font. Model objects are shared by the batch."""
    import torch
    from utils.font_inference import (
        prepare_reference_image, generate_letters, restore_predicted_glyphs, make_grid,
    )
    reference = prepare_reference_image(reference_path, encoder.patch_size, dataset.get("reference", {}))
    glyphs, positions = generate_letters(reference, generator, encoder, ids, noise,
                                          args.batch_size, device, args.amp)
    if not torch.isfinite(glyphs).all() or (positions is not None and not torch.isfinite(positions).all()):
        raise ValueError("Model generated non-finite values; try --no-amp or check the checkpoint")
    if positions is not None:
        glyphs = restore_predicted_glyphs(glyphs, positions, int(dataset.get("letter_image_size", 256)))
    if glyphs.ndim != 4 or glyphs.shape[0] != 52 or glyphs.shape[1] not in (1, 3):
        raise ValueError(f"Unexpected generated image shape: {tuple(glyphs.shape)}")
    assets = output.parent / (output.stem + "_assets")
    glyph_dir = assets / "glyphs"
    glyph_dir.mkdir(parents=True, exist_ok=True)
    pixels = glyphs.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    images = {}
    for ch, pixel in zip(string.ascii_letters, pixels):
        array = pixel[0].numpy() if pixel.shape[0] == 1 else pixel.permute(1, 2, 0).numpy()
        images[ch] = Image.fromarray(array).convert("L")
        # Unicode filenames avoid a/A collisions on Windows and macOS.
        images[ch].save(glyph_dir / f"U{ord(ch):04X}.png")
    make_grid(glyphs, columns=13, gap=4).save(assets / "alphabet_grid.png")
    metrics = build_ttf(images, output, family_name=family_name, threshold=args.threshold,
                        min_component_area=args.min_component_area, simplify=args.simplify,
                        baseline_ratio=args.baseline, side_bearing=args.side_bearing,
                        space_width=args.space_width)
    font_preview = assets / "font_preview.png"
    render_specimen(output, font_preview, args.text, args.font_size)
    preview = assets / "specimen.png"
    concat_reference_and_preview(reference_path, font_preview, preview)

    (assets / "specimen.txt").write_text(args.text + "\n", encoding="utf-8")
    metadata = {"reference": str(reference_path), "checkpoint": str(checkpoint_path),
                "generator_state": state, "seed": args.seed, "letters": string.ascii_letters,
                "positions_restored": positions is not None,
                "predicted_positions": positions.tolist() if positions is not None else None,
                "options": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}, **metrics}
    (assets / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output, preview


def main() -> None:
    args = parse_args()
    # Delay model imports so --help and font-export utilities need no training stack.
    import torch
    from utils.font_inference import choose_device, find_input_images, load_models

    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be an image directory: {input_dir}")
    references = find_input_images(input_dir, args.recursive)
    # A previous output tree inside the input directory must not become new input.
    references = [path for path in references if path != output_dir and output_dir not in path.parents]
    if not references:
        raise RuntimeError(f"No input images remain after excluding output directory: {input_dir}")

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a train.py checkpoint mapping")
    generator, encoder, config, state = load_models(
        checkpoint_path, args.config.expanduser().resolve(), args.style_checkpoint,
        args.generator_state, device, checkpoint=checkpoint,
    )
    saved_letters = "".join(checkpoint.get("letters", string.ascii_letters))
    if len(set(saved_letters)) != len(saved_letters) or set(string.ascii_letters) - set(saved_letters):
        raise ValueError("Checkpoint must contain all 52 unique ASCII letters")
    ids = torch.tensor([saved_letters.index(ch) for ch in string.ascii_letters], dtype=torch.long)
    if int(ids.max()) >= generator.letter_embedding.num_embeddings:
        raise ValueError("Checkpoint letter IDs exceed the generator embedding table")
    del checkpoint
    dataset = config.get("dataset", {})
    noise = torch.randn(52, generator.latent_dim,
                        generator=torch.Generator().manual_seed(args.seed))
    print(f"Loaded {state} on {device}; processing {len(references)} image(s)", flush=True)

    used_outputs = set()
    saved, failed = [], []
    for index, reference_path in enumerate(references, start=1):
        relative = reference_path.relative_to(input_dir)
        output = output_dir / relative.parent / f"{reference_path.stem}.ttf"
        if output in used_outputs:
            output = output.with_name(f"{output.stem}_{reference_path.suffix[1:].lower()}.ttf")
        used_outputs.add(output)
        family_name = f"{args.family_name} {reference_path.stem}"
        try:
            font, preview = generate_one(
                reference_path, output, family_name, args, generator, encoder,
                dataset, ids, noise, state, checkpoint_path, device,
            )
            saved.append(font)
            print(f"[{index}/{len(references)}] {relative} -> {font}; preview: {preview}", flush=True)
        except Exception as exc:
            if args.strict_inputs:
                raise
            failed.append((relative, exc))
            print(f"[{index}/{len(references)}] Failed {relative}: {exc}", flush=True)

    summary = {
        "input_directory": str(input_dir), "output_directory": str(output_dir),
        "total": len(references), "saved": [str(path) for path in saved],
        "failed": [{"image": str(path), "error": str(error)} for path, error in failed],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not saved:
        raise RuntimeError(f"No fonts were generated; see {output_dir / 'batch_summary.json'}")
    print(f"Finished: {len(saved)} saved, {len(failed)} failed. Summary: "
          f"{output_dir / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
