"""On-the-fly font-style images and token-padded batches for variable-size ViTs.

Each font file is one style. A sample contains independently rendered views from
the same font, so a positive pair shares style without having to share content.

Mask conventions: ``patch_padding_mask=True`` means ignore the token;
``attention_mask=True`` means valid and includes a leading CLS position;
``ibot_mask=True`` marks a valid patch to replace with a learned mask token.
"""

from __future__ import annotations

import io
import math
import os
import random
import string
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset

try:  # Pillow often renders .notdef without raising, so cmap inspection is preferred.
    from fontTools.ttLib import TTCollection, TTFont
except ImportError:  # pragma: no cover
    TTCollection = None
    TTFont = None


FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}
DEFAULT_CHARACTERS = string.ascii_letters + string.digits + ".,!?;:'-()&"


def _unique_characters(value: str) -> str:
    return "".join(dict.fromkeys(value))


def _font_cmap(font_path: Path) -> Optional[set[int]]:
    """Return the first face's Unicode cmap, or ``None`` without fontTools."""
    if TTFont is None:
        return None
    font_object = None
    try:
        if font_path.suffix.lower() in {".ttc", ".otc"}:
            font_object = TTCollection(str(font_path), lazy=True)
            if not font_object.fonts:
                return set()
            font = font_object.fonts[0]
        else:
            font_object = TTFont(str(font_path), lazy=True)
            font = font_object
        cmap: set[int] = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                cmap.update(table.cmap.keys())
        return cmap
    except Exception as exc:
        warnings.warn(f"Could not read cmap from {font_path}: {exc}", RuntimeWarning)
        return None
    finally:
        if font_object is not None:
            try:
                font_object.close()
            except Exception:
                pass


def _glyph_signature(font: ImageFont.FreeTypeFont, character: str) -> Optional[Tuple[Any, ...]]:
    """Fingerprint a visible glyph, including metrics used by a .notdef box."""
    try:
        mask = font.getmask(character, mode="L")
        if mask.getbbox() is None:
            return None
        return mask.size, font.getbbox(character), bytes(mask)
    except Exception:
        return None


def _probe_font_characters(font_path: Path, candidates: str, probe_size: int = 48) -> List[str]:
    """Find visible supported characters without mistaking tofu for a glyph."""
    font = ImageFont.truetype(str(font_path), probe_size, index=0)
    cmap = _font_cmap(font_path)
    missing_signatures = set()
    if cmap is None:
        for missing in ("\u0378", "\u0380", "\ufdd0", "\U0010ffff"):
            signature = _glyph_signature(font, missing)
            if signature is not None:
                missing_signatures.add(signature)
    supported = []
    for character in _unique_characters(candidates):
        if character.isspace() or (cmap is not None and ord(character) not in cmap):
            continue
        signature = _glyph_signature(font, character)
        if signature is not None and signature not in missing_signatures:
            supported.append(character)
    return supported


def _perspective_coefficients(
    destination: Sequence[Tuple[float, float]], source: Sequence[Tuple[float, float]]
) -> Tuple[float, ...]:
    """Solve Pillow's output-to-input perspective coefficients."""
    matrix, target = [], []
    for (x_out, y_out), (x_in, y_in) in zip(destination, source):
        matrix.append([x_out, y_out, 1.0, 0.0, 0.0, 0.0, -x_in * x_out, -x_in * y_out])
        matrix.append([0.0, 0.0, 0.0, x_out, y_out, 1.0, -y_in * x_out, -y_in * y_out])
        target.extend([x_in, y_in])
    return tuple(float(x) for x in np.linalg.solve(np.asarray(matrix), np.asarray(target)))


class FontStyleDataset(Dataset):
    """Generate grayscale text views from font files on the fly.

    Options are read from a mapping. Fonts are read from ``font_dir`` or from the
    ``font`` child of ``font_root_path``. Resolution is bounded by
    ``max_long_side`` and ``max_pixels``, then aligned to ``patch_size``.
    """

    def __init__(self, opt: Mapping[str, Any]):
        super().__init__()
        self.opt = dict(opt)
        font_root_path = self.opt["font_root_path"]
        self.font_dir = Path(font_root_path) / "font"
        if not self.font_dir.is_dir():
            raise FileNotFoundError(f"Font directory does not exist: {self.font_dir}")

        self.samples_per_font = int(self.opt.get("samples_per_font", 8))
        self.num_views = int(self.opt.get("num_views", 2))
        self.layout_probability = float(self.opt.get("layout_probability", 0.7))
        self.patch_size = int(self.opt.get("patch_size", 16))
        self.min_short_side = int(self.opt.get("min_short_side", 96))
        self.max_long_side = int(self.opt.get("max_long_side", 512))
        self.max_pixels = int(self.opt.get("max_pixels", 196_608))
        self.aspect_ratio_range = self.opt.get("aspect_ratio_range", (0.35, 2.85))
        self.min_font_size = int(self.opt.get("min_font_size", 18))
        self.max_font_size = int(self.opt.get("max_font_size", 128))
        self.max_render_attempts = int(self.opt.get("max_render_attempts", 8))
        self.mean = float(self.opt.get("mean", 0.5))
        self.std = float(self.opt.get("std", 0.5))
        self.strict_fonts = bool(self.opt.get("strict_fonts", False))
        self.recursive_fonts = bool(self.opt.get("recursive_fonts", True))
        self.characters = _unique_characters(str(self.opt.get("characters", DEFAULT_CHARACTERS)))

        if self.samples_per_font <= 0 or self.num_views <= 0:
            raise ValueError("samples_per_font and num_views must be positive")
        if not 0.0 <= self.layout_probability <= 1.0:
            raise ValueError("layout_probability must lie in [0, 1]")
        if self.patch_size <= 0 or self.max_long_side < self.patch_size:
            raise ValueError("patch_size must be positive and <= max_long_side")
        if self.max_pixels < self.patch_size**2:
            raise ValueError("max_pixels is too small for one patch")
        if self.std <= 0:
            raise ValueError("std must be positive")
        if not (0 < self.aspect_ratio_range[0] <= self.aspect_ratio_range[1]):
            raise ValueError("aspect_ratio_range must contain two positive ordered values")

        with open(os.path.join(font_root_path, "wordlist.txt")) as f:
            wordlist = f.readlines()
        self.words = tuple(w.strip() for w in wordlist)
        glob_method = self.font_dir.rglob if self.recursive_fonts else self.font_dir.glob
        candidates = sorted(
            path.resolve()
            for path in glob_method("*")
            if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
        )
        if not candidates:
            raise RuntimeError(f"No supported font files found in: {self.font_dir}")

        self.fonts: List[str] = []
        self.supported_characters: List[Tuple[str, ...]] = []
        self.words_by_font: List[Tuple[str, ...]] = []
        self.skipped_fonts: List[Tuple[str, str]] = []
        for font_path in candidates:
            try:
                supported = _probe_font_characters(font_path, self.characters)
                if not supported:
                    raise RuntimeError("none of the configured visible characters are supported")
            except Exception as exc:
                if self.strict_fonts:
                    raise RuntimeError(f"Cannot use font {font_path}: {exc}") from exc
                self.skipped_fonts.append((str(font_path), str(exc)))
                warnings.warn(f"Skipping unusable font {font_path}: {exc}", RuntimeWarning)
                continue
            supported_set = set(supported)
            usable_words = tuple(
                word for word in self.words if len(word) >= 2 and all(ch in supported_set for ch in word)
            )
            self.fonts.append(str(font_path))
            self.supported_characters.append(tuple(supported))
            self.words_by_font.append(usable_words)
        if not self.fonts:
            details = "; ".join(f"{Path(p).name}: {e}" for p, e in self.skipped_fonts)
            raise RuntimeError(f"No renderable fonts found in {self.font_dir}. {details}")

    def __len__(self) -> int:
        return len(self.fonts) * self.samples_per_font

    @staticmethod
    def _rng() -> random.Random:
        # DataLoader seeds torch independently in each worker.
        return random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    @staticmethod
    def _np_rng(rng: random.Random) -> np.random.Generator:
        return np.random.default_rng(rng.randrange(0, 2**63 - 1))

    def _sample_canvas_size(self, rng: random.Random) -> Tuple[int, int]:
        low_aspect, high_aspect = self.aspect_ratio_range
        aspect = math.exp(rng.uniform(math.log(low_aspect), math.log(high_aspect)))
        largest_short = min(
            max(self.min_short_side, self.patch_size),
            max(self.patch_size, int(math.sqrt(self.max_pixels / max(aspect, 1 / aspect)))),
        )
        upper = max(self.patch_size, min(self.max_long_side, largest_short * 2))
        lower = min(max(self.patch_size, self.min_short_side), upper)
        short = rng.randint(lower, upper)
        if aspect >= 1:
            height, width = short, round(short * aspect)
        else:
            width, height = short, round(short / aspect)
        return width, height

    def _paper_background(
        self, width: int, height: int, rng: random.Random, lightweight: bool = False
    ) -> np.ndarray:
        np_rng = self._np_rng(rng)
        base = rng.uniform(185, 252)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx, yy = xx / max(width - 1, 1) - 0.5, yy / max(height - 1, 1) - 0.5
        theta = rng.uniform(0, 2 * math.pi)
        directional = (xx * math.cos(theta) + yy * math.sin(theta)) * rng.uniform(-55, 55)
        radial = (xx**2 + yy**2) * rng.uniform(-45, 25)
        texture = np.sin((xx * rng.uniform(2, 9) + yy * rng.uniform(2, 9)) * math.pi)
        background = base + directional + radial + texture * rng.uniform(0, 5)
        sigma = rng.uniform(0.5, 3.0 if lightweight else 5.5)
        background += np_rng.normal(0, sigma, (height, width))
        background = np.clip(background, 0, 255).astype(np.uint8)
        if lightweight:
            return background

        paper = Image.fromarray(background, mode="L")
        draw = ImageDraw.Draw(paper)
        for _ in range(min(500, int(width * height * rng.uniform(0.00015, 0.0012)))):
            x, y = rng.randrange(width), rng.randrange(height)
            radius = rng.choice((0, 0, 0, 1, 1, 2))
            shade = int(np.clip(base + rng.uniform(-100, 80), 0, 255))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=shade)
        for _ in range(rng.randint(0, 5)):
            x0, y0 = rng.randrange(width), rng.randrange(height)
            length, angle = rng.uniform(0.05, 0.45) * max(width, height), rng.uniform(0, 2 * math.pi)
            x1, y1 = int(x0 + length * math.cos(angle)), int(y0 + length * math.sin(angle))
            shade = int(np.clip(base + rng.uniform(-60, 40), 0, 255))
            draw.line((x0, y0, x1, y1), fill=shade, width=rng.choice((1, 1, 2)))
        return np.asarray(paper, dtype=np.uint8)

    def _composite_ink(self, background: np.ndarray, mask: Image.Image, rng: random.Random) -> Image.Image:
        background = background.astype(np.float32)
        mean = float(background.mean())
        ink_level = (
            rng.uniform(3, max(5, mean - 65))
            if mean >= 128
            else rng.uniform(min(250, mean + 65), 252)
        )
        alpha = np.asarray(mask, dtype=np.float32) / 255
        ink = ink_level + self._np_rng(rng).normal(0, rng.uniform(0, 7), background.shape)
        result = background * (1 - alpha) + ink * alpha
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L")

    def _sample_words(self, style_id: int, count: int, rng: random.Random) -> List[str]:
        words, chars = self.words_by_font[style_id], self.supported_characters[style_id]
        if words:
            return [rng.choice(words) for _ in range(count)]
        return ["".join(rng.choice(chars) for _ in range(rng.randint(2, 10))) for _ in range(count)]

    @staticmethod
    def _wrap_words(draw: ImageDraw.ImageDraw, words: Sequence[str], font: Any, max_width: int) -> str:
        lines, current = [], ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    def _render_layout(self, style_id: int, rng: random.Random) -> Image.Image:
        width, height = self._sample_canvas_size(rng)
        background = self._paper_background(width, height, rng)
        mask, layout = Image.new("L", (width, height), 0), rng.choices(
            ("word", "sentence", "paragraph"), weights=(0.2, 0.3, 0.5)
        )[0]
        draw = ImageDraw.Draw(mask)

        count = {"word": rng.randint(1, 3), "sentence": rng.randint(4, 15), "paragraph": rng.randint(16, 100)}[layout]
        words = self._sample_words(style_id, count, rng)
        mx, my = max(3, int(width * rng.uniform(0.025, 0.12))), max(3, int(height * rng.uniform(0.025, 0.14)))
        usable_w, usable_h = max(1, width - 2 * mx), max(1, height - 2 * my)

        relative = int(height * (0.62 if layout == "word" else 0.32 if layout == "sentence" else 0.2))
        low = self.min_font_size
        size = rng.randint(low, max(low, min(self.max_font_size, relative)))
        align, spacing_factor = rng.choice(("left", "center", "right")), rng.uniform(0.10, 0.55)
        font = ImageFont.truetype(self.fonts[style_id], size, index=0)
        text = " ".join(words) if layout == "word" else self._wrap_words(draw, words, font, usable_w)
        spacing = max(1, int(size * spacing_factor))
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if font is None or not text:
            raise RuntimeError("could not construct layout text")
        # if tw > usable_w or th > usable_h:
        #     raise RuntimeError("text did not fit canvas")
        x = mx if align == "left" else width - mx - tw if align == "right" else (width - tw) // 2
        x += rng.randint(-max(1, mx // 3), max(1, mx // 3)) - bbox[0]
        y = my + rng.randint(0, max(0, usable_h - th)) - bbox[1]
        draw.multiline_text(
            (x, y), text, fill=rng.randint(190, 255), font=font, spacing=spacing,
            align=align, stroke_width=1 if rng.random() < 0.08 else 0,
        )
        text_bbox = mask.getbbox()
        if text_bbox is None:
            raise RuntimeError("font produced an empty text mask")
        image = self._composite_ink(background, mask, rng)

        # Crop from the text mask rather than image contrast, so lighting,
        # stains and dark paper cannot be mistaken for foreground glyphs.
        left, top, right, bottom = text_bbox
        text_width = max(1, right - left)
        text_height = max(1, bottom - top)
        padding_ratio = rng.uniform(0.035, 0.125)
        padding_x = max(2, round(text_width * padding_ratio))
        padding_y = max(2, round(text_height * padding_ratio))
        crop_box = (
            max(0, left - padding_x),
            max(0, top - padding_y),
            min(width, right + padding_x),
            min(height, bottom + padding_y),
        )

        return image.crop(crop_box)

    def _render_glyph_cell(
        self, style_id: int, character: str, width: int, height: int, size: int, rng: random.Random
    ) -> Image.Image:
        background = self._paper_background(width, height, rng, lightweight=True)
        font = ImageFont.truetype(self.fonts[style_id], size, index=0)
        bbox = font.getbbox(character)
        gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if gw <= 0 or gh <= 0:
            raise ValueError(f"empty glyph for {character!r}")
        mask = Image.new("L", (width, height), 0)
        x = (width - gw) // 2 - bbox[0] + rng.randint(-max(1, width // 18), max(1, width // 18))
        y = (height - gh) // 2 - bbox[1] + rng.randint(-max(1, height // 18), max(1, height // 18))
        ImageDraw.Draw(mask).text((x, y), character, fill=rng.randint(205, 255), font=font)
        if mask.getbbox() is None:
            raise ValueError(f"font rendered {character!r} as blank")
        return self._composite_ink(background, mask, rng)

    def _render_glyph_montage(self, style_id: int, rng: random.Random) -> Image.Image:
        count = rng.randint(4, 14)
        grid = count > 8 or rng.random() < 0.38
        columns = rng.randint(2, min(5, count)) if grid else count
        rows = math.ceil(count / columns)
        size = rng.randint(self.min_font_size, self.max_font_size)
        cell_h, cell_w = int(size * rng.uniform(1.05, 1.35)), int(size * rng.uniform(1.05, 1.35))
        gap = rng.choice((0, 0, 0, 1, 2, 3))
        width, height = columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap
        canvas = Image.fromarray(self._paper_background(width, height, rng), mode="L")
        characters, rendered, attempts = self.supported_characters[style_id], 0, 0
        while rendered < count and attempts < count * 5:
            attempts += 1
            try:
                cell = self._render_glyph_cell(
                    style_id, rng.choice(characters), cell_w, cell_h,
                    round(size * rng.uniform(0.9, 1.12)), rng,
                )
            except (OSError, ValueError):
                continue
            row, column = divmod(rendered, columns)
            canvas.paste(cell, (column * (cell_w + gap), row * (cell_h + gap)))
            rendered += 1
        if rendered < 2:
            raise RuntimeError("fewer than two glyph tiles could be rendered")
        return canvas

    def _camera_augmentation(self, image: Image.Image, rng: random.Random) -> Image.Image:
        fill, (width, height) = int(np.median(np.asarray(image))), image.size
        if rng.random() < 0.75 and min(width, height) >= 8:
            jitter = rng.uniform(0.015, 0.10) * min(width, height)
            source = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))
            destination = (
                (rng.uniform(0, jitter), rng.uniform(0, jitter)),
                (width - rng.uniform(0, jitter), rng.uniform(0, jitter)),
                (width - rng.uniform(0, jitter), height - rng.uniform(0, jitter)),
                (rng.uniform(0, jitter), height - rng.uniform(0, jitter)),
            )
            try:
                image = image.transform(
                    image.size, Image.Transform.PERSPECTIVE,
                    _perspective_coefficients(destination, source),
                    resample=Image.Resampling.BICUBIC, fillcolor=fill,
                )
            except np.linalg.LinAlgError:
                pass
        if rng.random() < 0.70:
            image = image.rotate(rng.uniform(-8, 8), Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.72, 1.35))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.20))
        if rng.random() < 0.4:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.85)))
        if rng.random() < 0.2 and min(image.size) >= 32:
            original, factor = image.size, rng.uniform(0.5, 0.85)
            small = (max(8, int(image.width * factor)), max(8, int(image.height * factor)))
            image = image.resize(small, Image.Resampling.BILINEAR).resize(original, Image.Resampling.BICUBIC)
        if rng.random() < 0.18:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=rng.randint(60, 92))
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = compressed.convert("L")
        array = np.asarray(image, dtype=np.float32)
        array += self._np_rng(rng).normal(0, rng.uniform(0, 4.5), array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")

    def _limit_resolution(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = min(1.0, self.max_long_side / max(width, height), math.sqrt(self.max_pixels / max(1, width * height)))
        width, height = max(self.patch_size, int(width * scale)), max(self.patch_size, int(height * scale))
        width = round(width / self.patch_size) * self.patch_size
        height = round(height / self.patch_size) * self.patch_size
        aligned_max = max(self.patch_size, self.max_long_side // self.patch_size * self.patch_size)
        width, height = min(width, aligned_max), min(height, aligned_max)
        while width * height > self.max_pixels:
            if width >= height and width > self.patch_size:
                width -= self.patch_size
            elif height > self.patch_size:
                height -= self.patch_size
            else:
                break
        return image.resize((width, height), Image.Resampling.LANCZOS).convert("L")

    def _render_view(self, style_id: int, rng: random.Random) -> Tuple[torch.Tensor, str]:
        last_error: Optional[Exception] = None
        for _ in range(self.max_render_attempts):
            mode = "layout" if rng.random() < self.layout_probability else "glyph_montage"
            try:
                image = self._render_layout(style_id, rng) if mode == "layout" else self._render_glyph_montage(style_id, rng)
                image = self._limit_resolution(self._camera_augmentation(image, rng))
                array = np.asarray(image, dtype=np.float32).copy() / 255
                tensor = (torch.from_numpy(array).unsqueeze(0) - self.mean) / self.std
                return tensor, mode
            except (OSError, ValueError, RuntimeError) as exc:
                last_error = exc
        raise RuntimeError(
            f"Failed to render {Path(self.fonts[style_id]).name} after "
            f"{self.max_render_attempts} attempts: {last_error}"
        ) from last_error

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        style_id, rng = index % len(self.fonts), self._rng()
        views, modes = [], []
        for _ in range(self.num_views):
            view, mode = self._render_view(style_id, rng)
            views.append(view)
            modes.append(mode)
        font_path = self.fonts[style_id]
        return {
            "views": views,
            "style_id": style_id,
            "font_name": Path(font_path).name,
            "font_path": font_path,
            "generation_modes": modes,
            "image_sizes": [(int(x.shape[-2]), int(x.shape[-1])) for x in views],
        }


class FontStyleCollator:
    """Patchify each image independently, then pad only token sequences."""

    def __init__(
        self,
        patch_size: int = 16,
        ibot_mask_ratio: float | Sequence[float] = (0.1, 0.5),
        keep_images: bool = False,
        num_global_views: int = 2,
    ) -> None:
        self.patch_size = int(patch_size)
        if isinstance(ibot_mask_ratio, (int, float)):
            self.ibot_mask_ratio = (float(ibot_mask_ratio),) * 2
        else:
            self.ibot_mask_ratio = (float(ibot_mask_ratio[0]), float(ibot_mask_ratio[1]))
        self.keep_images = bool(keep_images)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not 0 <= self.ibot_mask_ratio[0] <= self.ibot_mask_ratio[1] <= 1:
            raise ValueError("ibot_mask_ratio must lie in [0, 1]")

        self.num_global_views = int(num_global_views)
        if self.num_global_views < 2:
            raise ValueError("num_global_views must be at least 2")

    def _patchify(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(f"Expected grayscale [1, H, W], got {tuple(image.shape)}")
        _, height, width = image.shape
        pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
        if pad_h or pad_w:  # defensive; dataset emits aligned dimensions.
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        grid_h, grid_w = image.shape[-2] // self.patch_size, image.shape[-1] // self.patch_size
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(grid_h * grid_w, -1)
        return patches.contiguous(), (grid_h, grid_w)

    @staticmethod
    def _block_mask(grid_h: int, grid_w: int, ratio: float) -> torch.Tensor:
        token_count, target = grid_h * grid_w, round(grid_h * grid_w * ratio)
        mask, attempts = torch.zeros((grid_h, grid_w), dtype=torch.bool), 0
        while int(mask.sum()) < target and attempts < 16:
            attempts += 1
            remaining = target - int(mask.sum())
            area = max(1, min(remaining, round(token_count * random.uniform(0.03, 0.18))))
            aspect = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
            bh = min(grid_h, max(1, round(math.sqrt(area / aspect))))
            bw = min(grid_w, max(1, round(math.sqrt(area * aspect))))
            top, left = random.randrange(grid_h - bh + 1), random.randrange(grid_w - bw + 1)
            mask[top : top + bh, left : left + bw] = True
        flat, current = mask.flatten(), int(mask.sum())
        if current > target:
            indices = torch.nonzero(flat).flatten()
            flat[indices[torch.randperm(current)[: current - target]]] = False
        elif current < target:
            indices = torch.nonzero(~flat).flatten()
            flat[indices[torch.randperm(len(indices))[: target - current]]] = True
        return flat

    def __call__(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        items = list(batch)
        if not items:
            raise ValueError("Cannot collate an empty batch")
        num_views = len(items[0]["views"])
        if num_views == 0 or any(len(item["views"]) != num_views for item in items):
            raise ValueError("Every item must contain the same non-zero number of views")

        collated_views: List[Dict[str, Any]] = []
        for view_index in range(num_views):
            images = [item["views"][view_index] for item in items]
            patch_data = [self._patchify(image) for image in images]
            counts = [patches.shape[0] for patches, _ in patch_data]
            max_tokens, patch_dim = max(counts), patch_data[0][0].shape[1]
            dtype, device = patch_data[0][0].dtype, patch_data[0][0].device
            padded = torch.zeros((len(items), max_tokens, patch_dim), dtype=dtype, device=device)
            padding_mask = torch.ones((len(items), max_tokens), dtype=torch.bool, device=device)
            ibot_mask = torch.zeros_like(padding_mask)
            grid_sizes = torch.tensor([grid for _, grid in patch_data], dtype=torch.long)
            image_sizes = torch.tensor([[x.shape[-2], x.shape[-1]] for x in images], dtype=torch.long)
            for batch_index, ((patches, grid), count) in enumerate(zip(patch_data, counts)):
                padded[batch_index, :count] = patches
                padding_mask[batch_index, :count] = False
                ratio = random.uniform(*self.ibot_mask_ratio)
                if view_index < self.num_global_views:
                    ibot_mask[batch_index, :count] = self._block_mask(*grid, ratio).to(device)
            valid_mask = ~padding_mask
            attention_mask = torch.cat(
                [torch.ones((len(items), 1), dtype=torch.bool, device=device), valid_mask], dim=1
            )
            view_batch: Dict[str, Any] = {
                "patches": padded,
                "patch_padding_mask": padding_mask,
                "patch_valid_mask": valid_mask,
                "attention_mask": attention_mask,
                "ibot_mask": ibot_mask,
                "grid_sizes": grid_sizes,
                "image_sizes": image_sizes,
                "token_counts": torch.tensor(counts, dtype=torch.long),
            }
            if self.keep_images:
                view_batch["images"] = images
            collated_views.append(view_batch)
        return {
            "views": collated_views,
            "style_ids": torch.tensor([int(item["style_id"]) for item in items], dtype=torch.long),
            "font_names": [str(item["font_name"]) for item in items],
            "font_paths": [str(item["font_path"]) for item in items],
            "generation_modes": [list(item["generation_modes"]) for item in items],
        }


def collate_font_style(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Default picklable collate entry point for a 16-pixel patch size."""
    return FontStyleCollator(patch_size=16)(batch)
