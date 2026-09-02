"""Shared visual language for the Purcell (radiation-as-a-kink) scenes.

This is the channel's visual style for the published animation scenes. The
vignette is included in ``assets/vignette.png`` so this repository remains
self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import *

ASSETS_DIR = Path(__file__).parent / "assets"
VIGNETTE_PATH = ASSETS_DIR / "vignette.png"

# ---------------------------------------------------------------------------
# Palette — never pure black/white; exactly one ACCENT live on screen at a
# time, everything else MUTED. SECOND is the default/recurring element
# (e.g. field lines); ACCENT is reserved for the one thing to look at
# (e.g. the kink itself).
# ---------------------------------------------------------------------------

BG = "#12131A"
FG = "#E8E6E3"
ACCENT = "#FF6B4A"
SECOND = "#4AC5FF"
MUTED = "#6B7080"

# ---------------------------------------------------------------------------
# Fonts — Barlow for Text(), default LaTeX (Computer Modern) for MathTex/Tex.
# DIN Alternate is the accent face for numeric/technical callouts.
#
# Pango sees "Barlow" as a single family with weight faces (not separate
# family names per weight), so pass font=FONT + weight=FONT_WEIGHT_* rather
# than baking the weight into the family string.
# ---------------------------------------------------------------------------

FONT = "Barlow"
FONT_NUMERIC = "DIN Alternate"

FONT_WEIGHT_TITLE = "SEMIBOLD"
FONT_WEIGHT_LABEL = "MEDIUM"
FONT_WEIGHT_BODY = "NORMAL"

# ---------------------------------------------------------------------------
# Pacing — Manim's animation defaults read slightly fast for physics content.
# Pass these explicitly rather than relying on the library default.
# ---------------------------------------------------------------------------

RUN_TIME = 1.35        # a typical Create/Transform
RUN_TIME_SLOW = 1.8    # a transform that needs to be read, not just seen
BEAT_PAUSE = 1.0        # self.wait() after a conceptual beat lands
LONG_PAUSE = 2.0        # after something the viewer needs to sit with


class BrandScene(Scene):
    """Base class for every scene in this project.

    Sets the background color and vignette once, consistently. This does
    NOT rely on Manim's `background_image` config key: CairoRenderer
    instantiates `Camera()` with no arguments (see
    site-packages/manim/renderer/cairo_renderer.py), so that config value
    is silently ignored in this Manim version. Adding the vignette as a
    real ImageMobject is renderer-agnostic and actually works.
    """

    def setup(self) -> None:
        self.camera.background_color = BG
        if VIGNETTE_PATH.exists():
            bg = ImageMobject(str(VIGNETTE_PATH))
            bg.stretch_to_fit_width(config.frame_width)
            bg.stretch_to_fit_height(config.frame_height)
            bg.set_z_index(-1000)
            self.add(bg)


def make_vignette(path: Path = VIGNETTE_PATH, w: int = 1920, h: int = 1080,
                   lift: float = 20.0, strength: float = 1.0) -> None:
    """Regenerate the background vignette: flat BG with a very soft radial
    lift toward the center. Only needs re-running if BG changes.

    The lift spans only ~20 of 256 levels, so a naive quantise to 8-bit
    lands every pixel on one of ~20 values and the gradient shows up as
    visible concentric rings — very obvious against a near-black frame on
    a big screen. Triangular-PDF dither (one LSB of zero-mean noise)
    breaks those contours into imperceptible grain. Do not remove it.
    """
    from PIL import Image

    def hex2rgb(hx: str) -> np.ndarray:
        hx = hx.lstrip("#")
        return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)

    bg_rgb = hex2rgb(BG)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = w / 2, h / 2
    # normalise on the diagonal, so the falloff is a true circle softly
    # cropped by the frame rather than an ellipse whose edge runs parallel
    # to the frame border (which reads as a drawn shape, not as lighting)
    norm = np.hypot(w / 2, h / 2)
    d = np.clip(np.hypot(xx - cx, yy - cy) / norm, 0.0, 1.0)
    # cosine falloff: zero slope at both ends, so there's no bright seam at
    # the centre and no visible edge where the lift runs out
    t = strength * 0.5 * (1.0 + np.cos(np.pi * d))
    img = bg_rgb[None, None, :] + lift * t[..., None]

    rng = np.random.default_rng(7)
    dither = rng.random((h, w, 1)) - rng.random((h, w, 1))  # TPDF, +-1 LSB
    img = np.clip(img + dither, 0, 255)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)


if __name__ == "__main__":
    make_vignette()
    print(f"wrote {VIGNETTE_PATH}")
