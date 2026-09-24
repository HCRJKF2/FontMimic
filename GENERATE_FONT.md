# Generate a TTF Font from Reference Images

`generate_font.py` uses the generator trained by `train.py` together with FontStyleViT to process a directory of style reference images in batches. For each reference image, it generates all 52 uppercase and lowercase English letters and exports them as a TTF file. It then **reloads the generated TTF file** through Pillow/FreeType to render an English preview and places the reference image and font preview side by side.

## Usage

Install the additional contour-tracing dependencies in your existing training environment (see `requirements.txt` for the remaining dependencies):

```bash
python -m pip install opencv-python-headless fonttools Pillow
```

Run the following command from the project root, replacing the example paths with your own. The command is shown on one line and works in both Windows PowerShell and Linux:

```bash
python generate_font.py --input references --checkpoint outputs/font_cgan_parseq/checkpoints/latest.pt --output outputs/generated_fonts --family-name "My Mimic Font"
```

A regular training checkpoint contains the training configuration. By default, the script prefers `generator_ema` and falls back to `generator` when EMA weights are unavailable. You can explicitly select the non-EMA weights with `--generator-state generator`.

If the FontStyleViT weights have been moved, add `--style-checkpoint path/to/checkpoint.pt`. The style encoder must match the one used to train the generator.

When using the `font_generator.pt` exported at the end of training, also pass that run's configuration file so the reference-image preprocessing and glyph-canvas settings can be restored. The `generator` field in this exported file already contains the EMA weights:

```bash
python generate_font.py --input references --checkpoint outputs/font_cgan_parseq/font_generator.pt --config outputs/font_cgan_parseq/resolved_config.yaml --style-checkpoint outputs/font_style_vit_5/checkpoints/epoch_0080.pt --output outputs/generated_fonts
```

Reference images should resemble the training inputs as closely as possible: dark text on a light background, without complex scenery. Transparent regions are automatically composited onto a white background. Images are converted to grayscale, resized, aligned to the patch size, and normalized according to the training configuration. CUDA is used by default; the script automatically falls back to the CPU when CUDA is unavailable. You can also select it explicitly with `--device cpu`.

## Output Files

The commands above produce:

```text
outputs/generated_fonts/
  reference_1.ttf
  reference_1_assets/
    alphabet_grid.png
    font_preview.png
    specimen.png
    specimen.txt
    metadata.json
    glyphs/
      U0061.png ... U007A.png   # a-z
      U0041.png ... U005A.png   # A-Z
```

- `reference_1.ttf`: An outline font for each input image containing 52 letters, a space, and `.notdef`. It can be loaded by software with TTF support and does not need to be installed system-wide for previewing.
- `alphabet_grid.png`: Model-generated glyphs in a-z, A-Z order. When a position-prediction model is available, each cropped glyph is first restored to its original canvas.
- `glyphs/`: The 52 grayscale glyph images. Unicode code points are used as filenames to prevent `a.png` and `A.png` from overwriting each other on Windows.
- `font_preview.png`: English text rendered exclusively with the exported font file.
- `specimen.png`: The corresponding style reference on the left and `font_preview.png` on the right for direct style comparison.
- `metadata.json`: Input and weight paths, inference settings, predicted positions (when available), estimated baseline, scale, and per-glyph metrics.

By default, the input directory is processed recursively and its subdirectory structure is preserved in the output directory. Use `--no-recursive` to process only the top level of the input directory. If an image fails, the script continues with the remaining images and writes the results to `batch_summary.json` in the output root. Use `--strict-inputs` to stop immediately after the first failure.

The default preview contains the lowercase and uppercase alphabets, followed by an English pangram in all-lowercase and all-uppercase forms:

```text
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
the quick brown fox jumps over the lazy dog
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
```

This ensures that the default preview covers all 52 uppercase and lowercase letters. Use `--text "The quick brown fox jumps over the lazy dog"` to provide custom text. The font generates only letters and a space; custom preview text containing unsupported characters such as digits or punctuation produces an explicit error.

## Contour and Layout Tuning

The model generates raster images. The script uses [OpenCV contour extraction](https://docs.opencv.org/4.10.0/d3/dc0/group__imgproc__shape.html) to preserve outer contours, counters, and disconnected strokes, then uses [fontTools](https://fonttools.readthedocs.io/en/latest/) to write TrueType outlines and font tables. The outlines are polygonal approximations.

All letters share a single vertical coordinate system and scale, which prevents lowercase letters from being stretched to uppercase height. By default, the baseline is estimated from lowercase letters without descenders, and the estimated cap height is mapped to 700 font units in a 1,000-units-per-em coordinate system. Horizontally centered image padding is removed, and each advance width is based on the actual outline width plus side bearings. The model does not predict the original font's true advance widths, kerning, or hinting, so these layout metrics are approximate and do not fully reproduce the source font's professional typesetting parameters.

| Option | Default | Purpose |
| --- | --- | --- |
| `--threshold` | `160` | Treat pixels below this value on the 0–255 grayscale range as strokes. Raising it preserves lighter strokes but may introduce noise. |
| `--min-component-area` | `3` | Remove isolated components with a smaller area. Set it to `1` if small dotted strokes are removed at low resolutions. |
| `--simplify` | `0.35` | Contour-simplification tolerance in source-image pixels. Lower it to preserve more detail, or set it to `0` to disable simplification. |
| `--baseline` | Automatically estimated | Baseline as a ratio of the original canvas height, from 0 to 1; for example, `0.75`. |
| `--side-bearing` | `50` | Blank space on each side of a letter, in font units. |
| `--space-width` | `300` | Width of the space character, in font units. |
| `--font-size` | `64` | English preview size in pixels. |
| `--batch-size` | `13` | Number of letters generated per inference batch. Lower it if GPU memory is insufficient. |
| `--seed` | `42` | Generator noise seed. |
| `--no-amp` | Disabled | Disable CUDA mixed precision; try this if the model produces non-finite outputs. |

If a letter is blank or no usable outline remains after thresholding, the script identifies the failed letter and reports an error. The saved glyph images can be used to diagnose model output and threshold settings.
