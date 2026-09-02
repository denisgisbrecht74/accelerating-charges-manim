# Why Accelerating Charges Radiate — Manim source

This repository contains the Manim animation source for the Physical Intuition
video **Why Accelerating Charges Radiate**.

It includes every production animation scene from the video, together with the
shared drawing helpers and background needed to render them. Narration, audio,
thumbnails, research notes, raw renders, and archived drafts are intentionally
not included.

## Included scenes

| File | Scene classes | Content |
| --- | --- | --- |
| `scenes/scene_00_hook.py` | `Hook` | Opening visual hook |
| `scenes/scene_01_setup.py` | `Rule1Coulomb`, `Rule2Motion`, `Rule3SpeedLimit`, `ThreeRegions`, `Mechanism` | The three rules and the retarded field-line construction |
| `scenes/scene_02_geometry.py` | `Derivation`, `TheWave`, `PowerLaws`, `Identification` | Derivation and visualization of the transverse radiation field |
| `scenes/scene_03_donut.py` | `Donut` | Angular radiation pattern, wave, and energy argument |
| `scenes/scene_03_relativity.py` | `Squeeze`, `Chasing`, `Headlight`, `HeadlightCircle` | Relativistic field compression and beaming |

## Implementation notes

### Retarded field-line geometry

The charge follows a prescribed trajectory. Each field line is reconstructed
on every frame from points at different light-travel distances `R`. A point at
distance `R` can only know about the source at the retarded time

```text
tau = t - R/c.
```

The sampled points are joined with `VMobject.set_points_smoothly`. Advancing a
Manim `ValueTracker` therefore leaves earlier changes in motion farther from the
charge, producing bends that propagate outward at `c`.

### Transverse radiation field

`TheWave` evaluates the non-relativistic far-zone field

```text
E_t(r, theta, t) proportional to a(t - r/c) sin(theta) / r
```

on a NumPy grid. Field magnitude controls brightness, while orange and blue
show the two signs of the same transverse field. Each grid is converted to an
RGBA array and displayed as a Manim `ImageMobject`.

The donut-like angular envelope comes from `sin(theta)`: radiation vanishes
along the acceleration axis and is strongest perpendicular to it. The screen
shows a two-dimensional slice through that three-dimensional angular pattern.

## Installation

The source was rendered with Python 3.13 and Manim Community 0.21.0. Manim also
requires its normal system dependencies, including FFmpeg and a LaTeX
installation. See the [Manim installation guide](https://docs.manim.community/en/stable/installation.html)
if you do not already have them.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The visual design uses the Barlow font. Manim will fall back to another font if
Barlow is unavailable, but the line breaks and spacing can differ.

## Rendering

Run these commands from the repository root. Replace `-pql` with `-pqh` for a
high-quality render.

```bash
manim -pql scenes/scene_00_hook.py Hook

manim -pql scenes/scene_01_setup.py \
  Rule1Coulomb Rule2Motion Rule3SpeedLimit ThreeRegions Mechanism

manim -pql scenes/scene_02_geometry.py \
  Derivation TheWave PowerLaws Identification

manim -pql scenes/scene_03_donut.py Donut

manim -pql scenes/scene_03_relativity.py \
  Squeeze Chasing Headlight HeadlightCircle
```

Two representative scenes can be rendered individually with:

```bash
manim -pql scenes/scene_01_setup.py Rule3SpeedLimit
manim -pql scenes/scene_02_geometry.py TheWave
```

## Scope and accuracy

These scenes are explanatory visualizations, not a general electromagnetic
field solver. The field-line scenes visualize the retarded construction used
in the video; they do not numerically integrate Maxwell's equations or display
relativistic changes in field-line density. The raster radiation scenes
evaluate the field expressions documented directly in the source.

## License

All files in this repository are available under the [MIT License](LICENSE).
The license applies only to the contents of this repository. The original
video, narration, channel identity, thumbnails, and rendered media are not part
of this repository and remain separately copyrighted.
