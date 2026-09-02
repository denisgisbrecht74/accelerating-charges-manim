"""Shared drawing helpers for the §1 "Setup" scene (and beyond) — kept
local to this project rather than exported across project boundaries.
"""

import math

import numpy as np
from manim import *

from theme import (ACCENT, BG, FG, FONT, FONT_WEIGHT_BODY, FONT_WEIGHT_LABEL,
                   MUTED, SECOND)

CHARGE_RADIUS = 0.16

# field-line geometry, shared by every beat that draws Q's field.
# FIELD_LENGTH runs well past any frame edge so lines get clipped by the
# viewport rather than ending in a visible tip — a real Coulomb field
# doesn't stop at some finite radius, and a tapered tip quietly claims that
# it does. The bar is the frame half-diagonal (8.16) PLUS however far the
# charge wanders from centre, since the starburst travels with it; Beat 2's
# box path reaches 3.72 out, so 14 clears every frame corner with margin.
N_LINES = 20
LINE_STEP = TAU / N_LINES
FIELD_LENGTH = 14.0

# force-arrow length as a function of distance, standing in for 1/r^2.
# Calibrated so the arrow (charge position + arrow length) stays on screen
# even at the closest orbit radius we use (~2.2) swept through top/bottom,
# where the vertical half-frame is only ~4.0 units.
REF_R = 3.0
REF_LEN = 0.6
MIN_LEN, MAX_LEN = 0.45, 2.0


def charge_dot(pos, radius=CHARGE_RADIUS, sign="+", glow=True, sw=1.0):
    """The charge itself: a ringed disc with a soft halo.

    The halo is a stack of concentric strokes at falling opacity rather
    than a real blurred glow — Cairo has no cheap blur, and at these radii the
    stacked rings read as an aura anyway. It also keeps the charge findable
    once Beat 2's zoom-out shrinks the disc to a few pixels across.
    """
    parts = []
    if glow:
        # twelve closely-spaced rings, not three: at 1080p a 3-ring version
        # renders as three visibly separate circles. Spacing them tighter
        # than their own stroke width is what makes them read as one wash.
        for i in range(12):
            parts.append(Circle(radius=radius * (1.10 + 0.12 * i), color=FG,
                                stroke_width=4 * sw, fill_opacity=0,
                                stroke_opacity=0.115 * (1 - i / 12) ** 1.6).move_to(pos))
    dot = Circle(radius=radius, color=FG, fill_color=BG, fill_opacity=1,
                 stroke_width=3 * sw)
    dot.move_to(pos)
    plus = Text(sign, font=FONT, weight=FONT_WEIGHT_LABEL, color=FG).scale(0.5 * radius / CHARGE_RADIUS)
    plus.move_to(pos)
    return VGroup(*parts, dot, plus)


def force_arrow_length(r):
    length = REF_LEN * (REF_R / max(r, 0.4)) ** 2
    return float(np.clip(length, MIN_LEN, MAX_LEN))


def fade_in_mobject(mob, duration=0.4):
    """Ramp a mobject's stroke+fill opacity from 0 to 1 over `duration`
    seconds via a self-removing updater, so it can be dropped into a scene
    mid-animation (e.g. from inside another updater) without a hard pop —
    no nested self.play() needed.
    """
    mob.set_opacity(0)
    elapsed = [0.0]

    def _fade(m, dt):
        elapsed[0] += dt
        alpha = min(1.0, elapsed[0] / duration)
        m.set_stroke(opacity=alpha)
        m.set_fill(opacity=alpha)
        if alpha >= 1.0:
            m.remove_updater(_fade)

    mob.add_updater(_fade)


# ---------------------------------------------------------------------------
# Retarded-time field construction — shared by Beat 3's live simulation and
# by the frozen snapshot panels that close it, so the panels are the same
# physics rather than a hand-drawn imitation of it.
# ---------------------------------------------------------------------------

def field_samples(r0, R, edges, n_base=110, half_width=0.5, n_cluster=15):
    """Sample positions along a field line, densified around `edges`.

    A uniform sampling fine enough to keep the shell's two boundaries crisp
    would need several hundred points per line per frame. The boundaries
    are the only places the curve turns sharply, so cluster there and stay
    coarse elsewhere — and include each edge exactly, so the colour change
    lands on the boundary instead of on the nearest sample.
    """
    parts = [np.linspace(r0, R, n_base)]
    for e in edges:
        if r0 < e < R:
            parts.append(np.linspace(max(r0, e - half_width),
                                     min(R, e + half_width), n_cluster))
            parts.append(np.array([float(e)]))
    return np.unique(np.concatenate(parts))


def retarded_field(state, T_now, c, r_vals, scale, r_new, r_old,
                   n_lines=N_LINES, stroke_width=3):
    """The field of a moving charge at time `T_now`, drawn as field lines.

    `state(tau)` returns the charge's (position, velocity) at time tau.

    Each sampled point is fixed by two facts, and it matters that they are
    kept separate:

    1. **Where** it is. News emitted at tau has travelled exactly
       `R = c*(T_now - tau)` in every direction, from wherever the charge
       was at tau. So the point lies on that light sphere — same radius
       forwards and backwards, because c is the same forwards and
       backwards.
    2. **Which way** the field points there. Away from the charge's
       *projected present* position: where it would be now if it had
       carried on as it was going at tau. Constant velocity is
       predictable, so the field needs no fresh news to point correctly.

    So the point is the intersection of the ray from the projected
    position along `d` with the light sphere of radius R about the
    retarded position — which is one quadratic:

        u = proj - retarded = v(tau) * (T_now - tau)
        l = -(u.d) + sqrt((u.d)^2 - |u|^2 + R^2)
        P = proj + l*d

    The obvious shortcut, `P = proj + R*d`, is wrong and wrong in a way
    that looks like broken physics: it puts the constant-R surface around
    the *projected* position, so the shell comes out thin in front of the
    charge and thick behind it by a factor (1+tan phi)/(1-tan phi). That
    reads as the news travelling faster downstream than upstream, which is
    exactly what nothing does. With the quadratic the shell is uniform to
    within the distance the charge moved *during* the kick.

    `r_new..r_old` is the light-sphere radius range emitted while the
    charge was accelerating, coloured ACCENT; elsewhere SECOND. A range
    entirely below zero never matches, so the same call serves the
    un-kinked case.
    """
    group = VGroup()
    for i in range(n_lines):
        a = i * TAU / n_lines
        d = np.array([np.cos(a), np.sin(a), 0.0])

        pts = []
        for R in r_vals:
            tau = T_now - R / c
            pos, vel = state(tau)
            u = vel * (T_now - tau)                 # proj - retarded position
            ud = float(u @ d)
            ell = -ud + math.sqrt(max(ud * ud - float(u @ u) + R * R, 0.0))
            pts.append(scale * (pos + u + ell * d))

        runs, run_color, run_pts = [], None, []
        for R, p in zip(r_vals, pts):
            color = ACCENT if r_new <= R <= r_old else SECOND
            if color != run_color and run_pts:
                runs.append((run_color, run_pts))
                run_pts = [run_pts[-1]]  # share the endpoint: no seam
            run_pts.append(p)
            run_color = color
        if run_pts:
            runs.append((run_color, run_pts))

        for color, pts_run in runs:
            if len(pts_run) < 2:
                continue
            curve = VMobject(color=color, stroke_width=stroke_width)
            curve.set_points_smoothly(pts_run)
            group.add(curve)
    return group


def legend_card(rows):
    """A numbered legend on its own card.

    Beat 3's field covers every pixel of the frame, so there is no empty
    corner left to drop text into — the card is what keeps "text never
    overlaps a diagram" true when the diagram is the whole screen.
    """
    entries = VGroup()
    for num, color, text in rows:
        badge = VGroup(
            Circle(radius=0.17, color=color, fill_color=color,
                   fill_opacity=0.16, stroke_width=2),
            Text(num, font=FONT, weight=FONT_WEIGHT_LABEL, color=color).scale(0.36),
        )
        label = Text(text, font=FONT, weight=FONT_WEIGHT_BODY, color=FG).scale(0.4)
        label.next_to(badge, RIGHT, buff=0.22)
        entries.add(VGroup(badge, label))
    entries.arrange(DOWN, aligned_edge=LEFT, buff=0.28)

    card = RoundedRectangle(
        corner_radius=0.16, width=entries.width + 0.8, height=entries.height + 0.66,
        stroke_color=MUTED, stroke_width=1.5, fill_color=BG, fill_opacity=0.93,
    ).move_to(entries)
    return VGroup(card, entries)


def grid_arrow(node, direction, magnitude, length=0.5):
    """One arrow of the probe grid.

    Starts AT the probe and points away from it, rather than straddling it:
    an arrow centred on its own probe reads as a floating vector that
    happens to overlap a dot, not as the force on that dot.

    Magnitude is carried by opacity and stroke width, never by length. A
    faithful 1/r^2 grid spans three orders of magnitude across one frame:
    the near arrows run off screen and the far ones are subpixel. Only
    Beat 1's single test charge keeps true length scaling, because there
    the length is the lesson.
    """
    weight = float(np.clip(magnitude, 0.0, 1.0)) ** 0.42
    arrow = Line(node, node + direction * length,
                 color=SECOND, stroke_width=1.6 + 4.0 * weight)
    arrow.add_tip(tip_length=0.14, tip_width=0.12)
    arrow.set_opacity(0.16 + 0.84 * weight)
    return arrow


def probe_dot(pos, radius=0.055):
    """A bare test charge in the grid — small enough not to compete with Q."""
    return Circle(radius=radius, color=FG, fill_color=FG,
                  fill_opacity=0.55, stroke_width=0).move_to(pos)


def text_card(content, pad_x=0.5, pad_y=0.34):
    """Wrap text in a card so it can sit over a diagram that fills the frame.

    "Text never overlaps a diagram" is easy while there is empty space
    left. Once the field covers every pixel there is none, and a card is
    what keeps the rule true instead of quietly abandoning it.
    """
    card = RoundedRectangle(
        corner_radius=0.14, width=content.width + 2 * pad_x,
        height=content.height + 2 * pad_y, stroke_color=MUTED,
        stroke_width=1.5, fill_color=BG, fill_opacity=0.93,
    ).move_to(content)
    return VGroup(card, content)


def signed_field_key(symbol=r"E_t", scale=0.58):
    """A compact key for raster plots that use hue for the field *sign*.

    The radiation snapshots use ACCENT for one sign of the transverse
    electric field and SECOND for the other. Those colours have different
    jobs elsewhere in the film (the kink and the Coulomb field), so leaving
    this unstated makes a first-time viewer reasonably wonder whether the
    colours name two different fields. Put this key on the image plots,
    where it tells the viewer exactly what the hue encodes without asking
    them to infer it from an equation several shots earlier.
    """
    positive = MathTex(symbol + r" > 0", color=ACCENT).scale(scale)
    negative = MathTex(symbol + r" < 0", color=SECOND).scale(scale)
    entries = VGroup(positive, negative).arrange(DOWN, aligned_edge=LEFT,
                                                  buff=0.20)
    card = RoundedRectangle(
        corner_radius=0.13, width=entries.width + 0.58,
        height=entries.height + 0.44, stroke_color=MUTED,
        stroke_width=1.3, fill_color=BG, fill_opacity=0.93,
    ).move_to(entries)
    return VGroup(card, entries)


# ---------------------------------------------------------------------------
# The kick: rest -> raised-cosine acceleration over `dur` -> constant speed.
# Shared by section 1's Beat 3 and the whole of section 2, so the geometry
# section is measuring the same event the setup section showed.
# ---------------------------------------------------------------------------

def kick_position(elapsed, dur, speed):
    """Raised-cosine acceleration: velocity ramps 0 -> speed over `dur` with
    zero acceleration at both ends, so the field's direction stays
    continuous and the kink is a real smooth bend rather than an artefact
    of a discontinuous velocity."""
    if elapsed <= 0:
        return 0.0
    if elapsed < dur:
        frac = elapsed / dur
        return dur * speed * 0.5 * (frac - np.sin(np.pi * frac) / np.pi)
    return dur * speed * 0.5 + speed * (elapsed - dur)


def kick_velocity(elapsed, dur, speed):
    if elapsed <= 0:
        return 0.0
    if elapsed < dur:
        return speed * 0.5 * (1 - np.cos(np.pi * elapsed / dur))
    return speed


def kick_projection(tau, t, dur, speed):
    """Projected present position: where the charge would be now if it had
    kept doing what it was doing at `tau`. See retarded_field."""
    return np.array([kick_position(tau, dur, speed)
                     + kick_velocity(tau, dur, speed) * (t - tau), 0.0, 0.0])


def kick_state(tau, dur, speed):
    """(position, velocity) of a kicked charge — the pair retarded_field wants."""
    return (np.array([kick_position(tau, dur, speed), 0.0, 0.0]),
            np.array([kick_velocity(tau, dur, speed), 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Shared iconography and diagram pieces
#
# Everything below is used by more than one section, which is the only
# reason it lives here rather than in the scene that first draws it. The
# rule icons in particular are a plant-and-payoff spanning §1 -> §2, so
# both ends have to be literally the same object.
# ---------------------------------------------------------------------------

def maxwell_block(scale=0.52, color=MUTED):
    """The four equations, as one tight 2x2 cluster.

    Texture, not content — the viewer should register *four equations* and
    their compactness, and should not be invited to read them. §0 writes
    them in, §1 flicks them back for a beat and dismisses them, §2 strikes
    them through. Same object each time, so the callback is real.

    The first and third equations are split into `(divergence op, \\vec{E}
    or \\vec{B}, rest)` rather than built as one opaque string. LaTeX still
    typesets all three pieces as one continuous formula — splitting the
    string adds addressable sub-mobjects, it does not change a single pixel
    of the render — but it means `block[0][0][1]` and `block[1][0][1]` are
    the actual `\\vec{E}`/`\\vec{B}` glyphs, not the equation's bounding-box
    centroid. §0's H2 needs exactly that: a floating E/B symbol has to
    launch from where the real glyph sits, not from the middle of `=ρ/ε0`.
    """
    eqs = [
        MathTex(r"\nabla\cdot", r"\vec{E}", r"=\frac{\rho}{\varepsilon_0}"),
        MathTex(r"\nabla\times\vec{E}=-\frac{\partial\vec{B}}{\partial t}"),
        MathTex(r"\nabla\cdot", r"\vec{B}", r"=0"),
        MathTex(r"\nabla\times\vec{B}=\mu_0\vec{J}"
                r"+\mu_0\varepsilon_0\frac{\partial\vec{E}}{\partial t}"),
    ]
    for e in eqs:
        e.set_color(color).scale(scale)
    top = VGroup(eqs[0], eqs[1]).arrange(RIGHT, buff=0.9)
    bottom = VGroup(eqs[2], eqs[3]).arrange(RIGHT, buff=0.9)
    return VGroup(top, bottom).arrange(DOWN, buff=0.62)


def starburst(center=ORIGIN, length=FIELD_LENGTH, color=SECOND,
              n_lines=N_LINES, stroke_width=3, r0=None, fade=False):
    """Q's Coulomb field: `n_lines` straight lines running radially out.

    `fade=True` splits each line into three segments of falling opacity
    instead of one solid stroke. That is for insets and panels, where a
    line physically cannot run past a frame edge: a hard stop would claim
    the field ends at a finite radius, and a tapered tip would claim it
    too. A fade says "continues, we are only drawing this much", which is
    the honest reading for a diagram-within-a-diagram.
    """
    if r0 is None:
        r0 = CHARGE_RADIUS + 0.05
    group = VGroup()
    for i in range(n_lines):
        a = i * TAU / n_lines
        d = np.array([np.cos(a), np.sin(a), 0.0])
        if not fade:
            group.add(Line(center + d * r0, center + d * length,
                           color=color, stroke_width=stroke_width))
            continue
        cuts = np.linspace(r0, length, 4)
        for k, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            seg = Line(center + d * lo, center + d * hi, color=color,
                       stroke_width=stroke_width)
            seg.set_stroke(opacity=(1.0, 0.62, 0.28)[k])
            group.add(seg)
    return group


def chevron_marks(point, direction, n=2, color=ACCENT, size=0.16, gap=0.13,
                  stroke_width=3.5):
    """The standard "these two lines are parallel" double chevron.

    Used exactly twice in the video — §1 2e when rule 2 is stated, and §1
    4d when regions 1 and 3 turn out to obey it — and it has to look
    identical both times or the callback is only a claim.
    """
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0], 0.0])
    marks = VGroup()
    for k in range(n):
        base = np.array(point, dtype=float) + d * (k * gap)
        marks.add(VMobject(color=color, stroke_width=stroke_width)
                  .set_points_as_corners([base - perp * size - d * size * 0.9,
                                          base,
                                          base + perp * size - d * size * 0.9]))
    return marks


def rule_icon(kind, color=MUTED, size=0.30, stroke_width=2.0):
    """One of the three rule glyphs. See `rule_icon_bar`."""
    g = VGroup()
    if kind == 1:                                   # radial lines
        for i in range(8):
            a = i * TAU / 8
            d = np.array([np.cos(a), np.sin(a), 0.0])
            g.add(Line(d * size * 0.24, d * size, color=color,
                       stroke_width=stroke_width))
        g.add(Dot(radius=size * 0.13, color=color))
    elif kind == 2:                                 # two translated bursts
        for shift in (LEFT * size * 0.62, RIGHT * size * 0.62):
            for i in range(8):
                a = i * TAU / 8
                d = np.array([np.cos(a), np.sin(a), 0.0])
                g.add(Line(shift + d * size * 0.2, shift + d * size * 0.62,
                           color=color, stroke_width=stroke_width))
            g.add(Dot(radius=size * 0.11, color=color).move_to(shift))
    else:                                           # the news front
        g.add(Dot(radius=size * 0.13, color=color))
        for k, r in enumerate((0.42, 0.70, 1.0)):
            g.add(Circle(radius=size * r, color=color,
                         stroke_width=stroke_width,
                         stroke_opacity=1.0 - 0.28 * k))
    return g


def rule_icon_bar(corner=DR, buff=0.4):
    """The three rule icons on a card, all invisible until revealed.

    Returns `(bar, icons)`. `icons[i]` starts at zero opacity; play
    `FadeIn(icons[i])` as each rule is stated in §1, and they stay put for
    the rest of the video. §2's payoff lights all three in sequence, which
    is the whole reason they were parked there in the first place.

    They sit on a card because from §1 3d onward the field covers every
    pixel of the frame and there is no empty corner left — same reason
    `text_card` exists.
    """
    icons = VGroup(*[rule_icon(k) for k in (1, 2, 3)])
    icons.arrange(RIGHT, buff=0.42)
    card = RoundedRectangle(
        corner_radius=0.12, width=icons.width + 0.5, height=icons.height + 0.42,
        stroke_color=MUTED, stroke_width=1.2, fill_color=BG, fill_opacity=0.9,
    ).move_to(icons)
    card.set_stroke(opacity=0.4)

    # The icons are positioned but NOT bundled into the card, and their
    # opacity is left alone. `set_opacity(0)` followed by a later
    # `set_opacity(1)` would look like the obvious way to hide them, and it
    # silently destroys them: these glyphs are unfilled rings, so restoring
    # opacity to 1 sets fill_opacity to 1 as well and the news-front icon
    # comes back as three solid discs. Add each icon to the scene when its
    # rule is stated instead.
    layout = VGroup(card, icons).to_corner(corner, buff=buff)
    card.set_z_index(700)
    icons.set_z_index(701)
    layout.remove(icons)
    return card, icons


def retarded_point(state, T_now, c, tau, direction, scale=1.0):
    """The single field-line point carrying news emitted at `tau`.

    Same quadratic as `retarded_field`, exposed one point at a time so a
    beat can point at a specific bend and say *that curve is this instant*
    without re-deriving the geometry (§1 3e/3f) or hand-placing a marker.
    """
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d)
    R = c * (T_now - tau)
    pos, vel = state(tau)
    u = vel * (T_now - tau)
    ud = float(u @ d)
    ell = -ud + math.sqrt(max(ud * ud - float(u @ u) + R * R, 0.0))
    return scale * (pos + u + ell * d)
