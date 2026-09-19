<div align="center">

# Font Mimic

### Turn any text image into a style-matched English alphabet — and export it as a TTF font.

[简体中文](README_ZH.md) · **English**

![Python](https://img.shields.io/badge/Python-PyTorch-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-Conditional%20GAN-E69F00)
![Characters](https://img.shields.io/badge/Characters-a--z%20%2B%20A--Z-009E73)
![Input](https://img.shields.io/badge/Input-Any%20text%20image-CC79A7)

</div>

Font Mimic learns a reusable style representation from text images, then generates all 52 uppercase and lowercase English glyphs in that style. The reference can be a rendering from a TTF/OTF font, a photo of handwriting, a word, a sentence, or an entire paragraph. Generated raster glyphs can also be traced and assembled into an installable TrueType font.

> Most of this codebase was developed with [OpenAI Codex](https://openai.com/codex/).

> [!NOTE]
> The current release supports `a-z` and `A-Z`. Digits, punctuation, and eventually Chinese characters are possible future extensions. Model checkpoints are not included; train the two stages below before running export.

## Results

The model reads style from the complete reference image; it does not require pre-segmented characters.

<table>
  <tr><th width="34%">Reference image</th><th width="66%">Generated alphabet</th></tr>
  <tr><td><img src="assets/2_reference.png" alt="Cursive handwriting reference" width="100%"></td><td><img src="assets/2_generated.png" alt="Generated cursive alphabet" width="100%"></td></tr>
  <tr><td><img src="assets/8_reference.png" alt="Historical manuscript reference" width="100%"></td><td><img src="assets/8_generated.png" alt="Generated historical-style alphabet" width="100%"></td></tr>
  <tr><td><img src="assets/11_reference.png" alt="Geometric type reference" width="100%"></td><td><img src="assets/11_generated.png" alt="Generated geometric alphabet" width="100%"></td></tr>
</table>

More examples are available in [`assets/`](assets/).

## Why this project

- **Simple dataset construction.** Collect commercially reusable public font files and render training views on demand. In our experience, roughly 1,000–2,000 cleaned fonts already produce convincing results. Cleaning matters: remove broken fonts, missing glyphs, duplicates, and especially fonts that do not meaningfully distinguish uppercase from lowercase.
- **Flexible style references.** The input can contain a single letter, a word, a sentence, or a paragraph. Printed fonts and photographed handwriting follow the same inference path.
- **Lightweight, fast generation.** The glyph generator is a conditional GAN, so inference uses direct forward passes rather than iterative diffusion sampling or autoregressive language-model decoding.
- **Font export included.** The pipeline generates 52 glyph images, extracts contours with OpenCV, builds a TTF with fontTools, reloads that font, and renders a validation specimen.

## Architecture

![Font Mimic architecture](assets/architecture.svg)

The system is trained in two stages:

1. **Style encoder pretraining — `train_font_style.py`.** A variable-resolution `FontStyleViT` learns from multi-crop text views with a student/EMA-teacher setup. The objective combines a MoCo queue loss, DINO cross-view self-distillation, and iBOT++ patch distillation over both masked and visible valid patches.
2. **Glyph generator training — `train.py`.** The frozen style encoder conditions a GAN generator together with a letter ID and random noise. Training combines hinge adversarial loss, frozen PARSeq OCR-logit distillation, pixel/foreground and edge losses, style and patch-perceptual losses, and optional glyph-position regression. An EMA generator is exported for inference.
3. **Inference and export — `generate_font.py`.** Each reference image is encoded once; the generator produces `a-z` and `A-Z`, OpenCV converts raster ink to contours, and fontTools writes a TTF plus preview assets.

## Quick start

### 1. Install dependencies

Create an isolated Python environment with a PyTorch build appropriate for your CUDA setup, then install the project requirements:

```bash
pip install -r requirements.txt
```

The default generator configuration loads the official PARSeq model through Torch Hub. On an offline machine, clone PARSeq locally and set `model.parseq.repo_or_dir` and `model.parseq.source: local` in `config/train.yaml`.

### 2. Prepare the font dataset

Point `dataset.font_root_path` in both training configs to a directory like this:

```text
fonts_dataset/
├── font/
│   ├── font_0001.ttf
│   ├── font_0002.otf
│   └── ...
└── wordlist.txt
```

Use only fonts whose licenses permit your intended use. Before training, validate glyph coverage and remove unreadable files, duplicates, symbol-only fonts, and case-insensitive designs. The checked-in configs contain example absolute paths and must be edited for your machine.

### 3. Pretrain the font-style encoder

```bash
python train_font_style.py --config config/train_font_style.yaml
```

Main entry point: [`train_font_style.py`](train_font_style.py)  
Default config: [`config/train_font_style.yaml`](config/train_font_style.yaml)

Set the resulting checkpoint under `model.style_encoder.checkpoint` in `config/train.yaml`.

### 4. Train the conditional GAN

```bash
python train.py --config config/train.yaml
```

Main entry point: [`train.py`](train.py)  
Default config: [`config/train.yaml`](config/train.yaml)

Training writes regular checkpoints and a compact EMA export named `font_generator.pt` to the configured output directory.

### 5. Generate glyphs and export TTF files

Place one or more reference images in a directory, then run:

```bash
python generate_font.py \
  --input references \
  --checkpoint outputs/font_cgan_parseq/font_generator.pt \
  --config outputs/font_cgan_parseq/resolved_config.yaml \
  --style-checkpoint outputs/font_style_vit/checkpoints/latest.pt \
  --output outputs/generated_fonts \
  --family-name "My Mimic Font"
```

`generate_font.py` processes the input directory recursively by default. A typical result is:

```text
outputs/generated_fonts/
├── reference_1.ttf
├── batch_summary.json
└── reference_1_assets/
    ├── alphabet_grid.png
    ├── font_preview.png
    ├── specimen.png
    ├── metadata.json
    └── glyphs/
        ├── U0061.png
        └── ...
```

See [`GENERATE_FONT_ZH.md`](GENERATE_FONT_ZH.md) for detailed export parameters, contour tuning, and troubleshooting.

## Current limitations

- Only the 52 ASCII letters are generated; digits, punctuation, kerning pairs, and non-Latin scripts are not yet supported.
- Font metrics, baseline, side bearings, and contour simplification are estimated from raster outputs. Exported TTFs are usable, but they do not reproduce professional kerning and hinting from the source font.
- Clean, high-contrast text on a simple background is closest to the training distribution. Complex scenes may need preprocessing.
- Results depend strongly on dataset licensing, coverage, cleaning, and checkpoint quality.

## References and acknowledgements

The style-encoder training objectives and OCR supervision build on the following work:

- **MoCo:** He et al., *Momentum Contrast for Unsupervised Visual Representation Learning* — [paper](https://arxiv.org/abs/1911.05722) · [official code](https://github.com/facebookresearch/moco)
- **DINO:** Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* — [paper](https://arxiv.org/abs/2104.14294) · [official code](https://github.com/facebookresearch/dino)
- **iBOT:** Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer* — [paper](https://arxiv.org/abs/2111.07832) · [official code](https://github.com/bytedance/ibot)
- **iBOT++:** Cao et al., *TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment* — [paper](https://arxiv.org/abs/2604.12012) · [project page and code](https://gdm-tipsv2.github.io/)
- **PARSeq:** Bautista and Atienza, *Scene Text Recognition with Permuted Autoregressive Sequence Models* — [paper](https://arxiv.org/abs/2207.06966) · [official code](https://github.com/baudm/parseq)

If you publish work built on this repository, please cite the original methods above as appropriate.
