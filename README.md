# Why Accelerating Charges Radiate — Manim source

This repository contains the Manim animation source for the Physical Intuition
video **Why Accelerating Charges Radiate**.

It includes every production animation scene from the video, together with the
shared drawing helpers and background needed to render them. Narration, audio,
thumbnails, research notes, raw renders, and archived drafts are intentionally
not included.

Part 2, [But Why Do Moving Charges Have a Magnetic Field?](https://github.com/denisgisbrecht74/magnetism-manim),
has its own repository.

## Included scenes

| File | Scene classes | Content |
| --- | --- | --- |
| `scenes/scene_00_hook.py` | `Hook` | Opening visual hook |
| `scenes/scene_01_setup.py` | `Rule1Coulomb`, `Rule2Motion`, `Rule3SpeedLimit`, `ThreeRegions`, `Mechanism` | The three rules and the retarded field-line construction |
| `scenes/scene_02_geometry.py` | `Derivation`, `TheWave`, `PowerLaws`, `Identification` | Derivation and visualization of the transverse radiation field |
| `scenes/scene_03_donut.py` | `Donut` | Angular radiation pattern, wave, and energy argument |
| `scenes/scene_03_relativity.py` | `Squeeze`, `Chasing`, `Headlight`, `HeadlightCircle` | Relativistic field compression and beaming |

## How the animations work

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

### Radiation shell and angular pattern

The outgoing shell is divided into short `AnnularSector` elements whose
opacity follows `sin²(theta)`. This makes the shell brightest broadside to the
acceleration and dark along its axis.

The familiar donut is drawn separately as a plot of radiated power per
direction. A two-dimensional polar curve is swept around the acceleration axis
to create a wireframe surface of revolution, then projected into the Manim
frame. The donut is therefore an angular diagram; the object that propagates
through space is the spherical radiation shell.

### Relativistic motion

For a uniformly moving charge, directions sampled uniformly in the rest frame
are mapped into the lab frame with

```text
tan(theta_lab) = gamma tan(theta_rest).
```

The relativistic radiation scenes solve for retarded emission time on a NumPy
grid and evaluate the angular field pattern there. This produces the forward
beaming visible as the charge's speed approaches `c`. Expanding shells are
centred on the source positions from which they were emitted, making the
front-to-back bunching a consequence of the geometry rather than a decorative
effect.

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

## Scope and accuracy

These scenes are explanatory visualizations, not a general electromagnetic
field solver. The field-line scenes visualize the retarded construction used
in the video; they do not numerically integrate Maxwell's equations or display
relativistic changes in field-line density. The raster radiation scenes
evaluate the field expressions documented directly in the source.

## License

Copyright (c) 2026 Physical Intuition.

Unless otherwise noted, the contents of this repository are available under
the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License](LICENSE) ([license summary](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

You may share and adapt the material for noncommercial purposes, provided that
you credit Physical Intuition, link to this repository and the license, state
whether you made changes, and distribute adaptations under the same license.
Commercial use requires separate permission from Physical Intuition.

This license applies only to the contents of this repository. The original
video, narration, channel identity, thumbnails, and rendered media are not part
of this repository and remain separately copyrighted.
