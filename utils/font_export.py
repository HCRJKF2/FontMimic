"""Trace grayscale glyphs into a TrueType font, then render the saved font.

Only Pillow, NumPy, OpenCV and fontTools are required by this module.
The model does not predict font metrics: baseline and scale are estimated from
the shared training canvas, while horizontal spacing uses fixed side bearings.
"""
from __future__ import annotations

import math
import re
import string
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

LETTERS = string.ascii_letters
PANGRAM = "The quick brown fox jumps over the lazy dog"
DEFAULT_TEXT = "\n".join((string.ascii_lowercase, string.ascii_uppercase,
                          PANGRAM.lower(), PANGRAM.upper()))


def clean_mask(image: Image.Image, threshold: int, min_component_area: int) -> np.ndarray:
    """Remove only tiny disconnected specks; keep counters and i/j dots."""
    mask = (np.asarray(image.convert("L")) < threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_area
    return keep[labels]


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty glyph after thresholding; adjust --threshold/--min-component-area")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2)


def trace_glyph(mask: np.ndarray, baseline: float, scale: float,
                side_bearing: int, simplify: float):
    """Trace all nested contours with alternating TrueType winding directions.

    Upsampling the binary mask preserves even a one-pixel stroke as an outline
    with nonzero area. Simplification tolerance is in original image pixels.
    """
    factor = 4
    expanded = cv2.resize(mask, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    expanded = np.pad(expanded, 1)
    contours, hierarchy = cv2.findContours(expanded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError("Cannot trace an empty glyph")
    x_origin = _bounds(mask)[0]
    pen = TTGlyphPen(None)
    written = 0
    for index, contour in enumerate(contours):
        polygon = cv2.approxPolyDP(contour, simplify * factor, True)
        if len(polygon) < 3 or abs(cv2.contourArea(polygon)) == 0:
            polygon = contour  # Never erase a small dot or counter by simplification.
        points = (polygon.reshape(-1, 2).astype(float) - 1 + 0.5) / factor
        points[:, 0] = (points[:, 0] - x_origin) * scale + side_bearing
        points[:, 1] = (baseline - points[:, 1]) * scale
        points = np.rint(points).astype(np.int32)
        # Rounding can merge adjacent vertices.
        points = points[np.any(points != np.roll(points, 1, axis=0), axis=1)]
        if len(points) < 3 or _signed_area(points) == 0:
            raise ValueError("Outline collapsed in font units; reduce --simplify")
        depth, parent = 0, int(hierarchy[0, index, 3])
        while parent != -1:
            depth += 1
            parent = int(hierarchy[0, parent, 3])
        # Font coordinates point upward: outer paths clockwise, holes CCW.
        clockwise = depth % 2 == 0
        if (_signed_area(points) < 0) != clockwise:
            points = points[::-1]
        pen.moveTo(tuple(points[0]))
        for point in points[1:]:
            pen.lineTo(tuple(point))
        pen.closePath()
        written += 1
    if not written:
        raise ValueError("No usable glyph contours")
    glyph = pen.glyph()
    glyph.recalcBounds(None)
    return glyph


def _notdef():
    pen = TTGlyphPen(None)
    for points in (((50, 0), (50, 700), (450, 700), (450, 0)),
                   ((100, 50), (400, 50), (400, 650), (100, 650))):
        pen.moveTo(points[0])
        for point in points[1:]:
            pen.lineTo(point)
        pen.closePath()
    return pen.glyph()


def build_ttf(images: Mapping[str, Image.Image], output: Path, *,
              family_name: str = "Mimic Font", threshold: int = 160,
              min_component_area: int = 3, simplify: float = 0.35,
              baseline_ratio: float | None = None, side_bearing: int = 50,
              space_width: int = 300) -> dict:
    """Build 52 letters plus space and .notdef. Return reproducible metrics."""
    if not 1 <= threshold <= 255 or min_component_area < 1:
        raise ValueError("threshold must be 1..255 and min_component_area must be positive")
    if not math.isfinite(simplify) or not 0 <= simplify <= 2:
        raise ValueError("simplify must be between 0 and 2 source pixels")
    if not 0 <= side_bearing <= 1000 or not 1 <= space_width <= 2000:
        raise ValueError("side_bearing must be 0..1000; space_width must be 1..2000")
    if baseline_ratio is not None and not 0 < baseline_ratio < 1:
        raise ValueError("baseline_ratio must be between 0 and 1")
    if not family_name.strip():
        raise ValueError("family_name cannot be empty")
    if Path(output).suffix.lower() != ".ttf":
        raise ValueError("Output font must have a .ttf extension")
    missing = set(LETTERS) - images.keys()
    if missing:
        raise ValueError(f"Missing glyph images: {''.join(sorted(missing))}")
    if len({images[ch].size for ch in LETTERS}) != 1:
        raise ValueError("All glyphs must use the same restored canvas size")
    masks, boxes = {}, {}
    for ch in LETTERS:
        masks[ch] = clean_mask(images[ch], threshold, min_component_area)
        try:
            boxes[ch] = _bounds(masks[ch])
        except ValueError as exc:
            raise ValueError(f"Glyph {ch!r}: {exc}") from exc
    height = images["a"].height
    baseline = (baseline_ratio * height if baseline_ratio is not None else
                float(np.median([boxes[ch][3] for ch in "aceimnorsuvwxz"])))
    cap_height_px = baseline - float(np.median([boxes[ch][1] for ch in "BDEFHIKLMNPRTXYZ"]))
    if cap_height_px < 2:
        raise ValueError("Cannot estimate cap height; check generated glyphs or --baseline")
    scale = 700.0 / cap_height_px
    glyphs = {".notdef": _notdef(), "space": TTGlyphPen(None).glyph()}
    metrics = {".notdef": (500, 50), "space": (space_width, 0)}
    glyph_info = {}
    for ch in LETTERS:
        try:
            glyph = trace_glyph(masks[ch], baseline, scale, side_bearing, simplify)
        except ValueError as exc:
            raise ValueError(f"Glyph {ch!r}: {exc}") from exc
        if min(glyph.xMin, glyph.yMin) < -32768 or max(glyph.xMax, glyph.yMax) > 32767:
            raise ValueError(f"Glyph {ch!r} exceeds TrueType coordinate limits")
        advance = glyph.xMax + side_bearing
        glyphs[ch] = glyph
        metrics[ch] = (advance, glyph.xMin)
        glyph_info[ch] = {"source_bbox": boxes[ch], "advance_width": advance,
                          "left_side_bearing": glyph.xMin,
                          "contours": glyph.numberOfContours}
    ascent = max(750, max(glyphs[ch].yMax for ch in LETTERS) + 50)
    descent = min(-200, min(glyphs[ch].yMin for ch in LETTERS) - 50)
    if ascent > 32767 or descent < -32768:
        raise ValueError("Estimated line metrics exceed TrueType limits")
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(list(glyphs))
    builder.setupCharacterMap({32: "space", **{ord(ch): ch for ch in LETTERS}})
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=ascent, descent=descent)
    ps_name = (re.sub(r"[^A-Za-z0-9-]", "", family_name) or "MimicFont")[:54] + "-Regular"
    builder.setupNameTable({"familyName": family_name, "styleName": "Regular",
                            "uniqueFontIdentifier": ps_name + ";1.0",
                            "fullName": family_name + " Regular", "psName": ps_name,
                            "version": "Version 1.0"})
    x_height = round((baseline - float(np.median([boxes[ch][1] for ch in "aceosuxz"]))) * scale)
    builder.setupOS2(sTypoAscender=ascent, sTypoDescender=descent, sTypoLineGap=0,
                     usWinAscent=ascent, usWinDescent=-descent, fsType=0,
                     sxHeight=max(0, x_height), sCapHeight=700)
    builder.setupPost()
    builder.setupMaxp()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    builder.save(str(output))
    return {"units_per_em": 1000, "baseline_pixels": baseline,
            "baseline_ratio": baseline / height, "font_units_per_pixel": scale,
            "ascent": ascent, "descent": descent, "glyphs": glyph_info}


def render_specimen(font_path: Path, output: Path, text: str = DEFAULT_TEXT,
                     font_size: int = 64) -> None:
    """Reload the actual TTF and verify character coverage before rendering."""
    if font_size < 1 or not text.strip():
        raise ValueError("font_size and nonempty text are required")
    with TTFont(str(font_path)) as saved:
        cmap = saved.getBestCmap() or {}
        missing = {ch for ch in text if ch != "\n" and ord(ch) not in cmap}
        if missing:
            raise ValueError(f"Font does not support specimen characters: {sorted(missing)!r}")
        if any(ord(ch) not in cmap for ch in LETTERS):
            raise ValueError("Exported font is missing ASCII letters")
        for ch in LETTERS:
            if saved["glyf"][cmap[ord(ch)]].numberOfContours <= 0:
                raise ValueError(f"Exported letter {ch!r} has no outline")
    font = ImageFont.truetype(str(font_path), size=font_size)
    ascent, descent = font.getmetrics()
    lines = text.split("\n")
    bounds = [font.getbbox(line, anchor="ls") for line in lines]
    left = min(0, min(box[0] for box in bounds))
    right = max(max(box[2], math.ceil(font.getlength(line))) for box, line in zip(bounds, lines))
    margin = max(16, font_size // 2)
    line_height = ascent + descent + max(8, font_size // 4)
    canvas = Image.new("RGB", (right - left + margin * 2,
                               line_height * len(lines) + margin * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((margin - left, margin + ascent + index * line_height), line,
                  font=font, fill="black", anchor="ls")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
