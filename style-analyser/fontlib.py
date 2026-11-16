"""Font rendering shared by font_index.py and match_font.py.

Text is shaped with HarfBuzz (so kerning from GPOS tables is applied, which Pillow's basic
layout skips) and each glyph is drawn with Pillow at a fixed em size. Callers downscale the
result to the size they need, so only one Pillow font object exists per font instance.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from common import ROOT

FONTS_DIR = ROOT / "models" / "google-fonts"
INDEX_PATH = ROOT / "models" / "font_index.json"
EM = 160                 # render size in px per em; results are downscaled from this
NO_LIGATURES = {"liga": False, "clig": False, "dlig": False}


def font_axes(path: Path) -> list[tuple[str, float, float, float]]:
    """Variable-font axes as (tag, min, default, max); empty for static fonts."""
    f = TTFont(path, lazy=True)
    if "fvar" not in f:
        return []
    return [(a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in f["fvar"].axes]


@lru_cache(maxsize=8)
def _font_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


class FontInstance:
    """One font file at one point in its design space (e.g. Inter at wght=800)."""

    def __init__(self, path: str | Path, axes: dict[str, float] | None = None):
        self.path = str(path)
        self.axes = axes or {}
        data = _font_bytes(self.path)
        face = hb.Face(data)
        self.upem = face.upem
        self.hb_font = hb.Font(face)
        all_axes = font_axes(Path(self.path)) if self.axes else []
        if self.axes:
            self.hb_font.set_variations(self.axes)
        self.pil = ImageFont.truetype(io.BytesIO(data), EM)
        if all_axes:
            self.pil.set_variation_by_axes([self.axes.get(tag, dflt) for tag, _, dflt, _ in all_axes])

    def shape(self, text: str) -> list[tuple[str, float, float]] | None:
        """[(char, x_px, advance_px)] at EM size, or None if a character is missing."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, NO_LIGATURES)
        scale = EM / self.upem
        out, pen = [], 0.0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            ch = text[info.cluster]
            if info.codepoint == 0 and not ch.isspace():
                return None
            out.append((ch, (pen + pos.x_offset) * scale, pos.x_advance * scale))
            pen += pos.x_advance
        return out

    def render(self, text: str, track_em: float = 0.0, glyphs=None) -> np.ndarray | None:
        """Coverage image (float32, 0..1) of `text` at EM px/em with extra letter spacing."""
        glyphs = glyphs if glyphs is not None else self.shape(text)
        if glyphs is None:
            return None
        track = track_em * EM
        width = int(sum(a for _, _, a in glyphs) + max(0.0, track) * len(glyphs) + EM)
        img = Image.new("L", (width, int(EM * 1.7)), 0)
        d = ImageDraw.Draw(img)
        base_x, base_y = EM * 0.4, EM * 1.25
        for i, (ch, x, _) in enumerate(glyphs):
            if not ch.isspace():
                d.text((base_x + x + i * track, base_y), ch, font=self.pil, fill=255, anchor="ls")
        return np.asarray(img, np.float32) / 255.0


def ink_box(cov: np.ndarray, thr: float = 0.5) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(cov >= thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def stroke_width(mask: np.ndarray) -> float:
    """Average stroke thickness in px: for long strokes, area ~ width * length and
    perimeter ~ 2 * length, so width ~ 2 * area / perimeter."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    return float(2 * m.sum() / perim) if perim else 0.0
