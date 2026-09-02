"""Section 3 — the radiation pattern, and the payoff.

Built against storyboard.md §3. Two things it is careful about:

**The radiation is not donut-shaped in space.** `dP/dOmega ∝ sin^2(theta)`
is a plot in *angle* space. What actually travels outward is the spherical
shell from §1, whose brightness varies over its surface — bright round the
equator, dark at the two poles along the kick. Animating a literal torus
flying through space would be showing something false, and it is the usual
way this gets drawn wrong. So 3b is the shell; the donut arrives in 3c and
arrives explicitly labelled as a diagram.

**The coefficient.** At 3a only the electric energy density is on the table,
`u = eps0 E^2 / 2`, which gives `dP/dOmega = q^2 a^2 sin^2 / (32 pi^2 eps0
c^3)`. The familiar `16 pi^2` is the *full* density including B, so it is
what 3f's factor of two produces — and it lands exactly on Larmor. Writing
16 pi^2 at 3a and then doubling it would double an expression that already
counted B once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from manim import *

from theme import (
    ACCENT,
    BEAT_PAUSE,
    BrandScene,
    FG,
    FONT,
    FONT_WEIGHT_BODY,
    FONT_WEIGHT_LABEL,
    FONT_WEIGHT_TITLE,
    LONG_PAUSE,
    MUTED,
    RUN_TIME,
    RUN_TIME_SLOW,
    SECOND,
)
from visuals import (
    CHARGE_RADIUS,
    LINE_STEP,
    N_LINES,
    charge_dot,
    field_samples,
    kick_position,
    kick_state,
    retarded_field,
    text_card,
)

# same event, same numbers as §1 and §2 — the kick is along +x, so theta is
# measured from the kick axis and the pattern's two holes lie left and right
C = 9.0
V = 0.871
DT = 0.1481
SCALE = 0.75
R_MAX = 20.0


def state(tau):
    return kick_state(tau, DT, V)


def caption(*lines, color=MUTED, weight=FONT_WEIGHT_BODY, scale=0.44):
    return VGroup(*[Text(t, font=FONT, weight=weight, color=color) for t in lines]) \
        .scale(scale).arrange(DOWN, aligned_edge=LEFT, buff=0.17)


def lobe(power, scale, color=ACCENT, width=4, n=160):
    """Polar plot of r = sin(theta)^power, as two lobes.

    Drawn as two separate arcs meeting at the origin rather than one closed
    curve: r goes to zero at theta = 0 and pi, and a single smoothed curve
    rounds those points off into a blob — losing the zeros, which are the
    whole content of the shape.
    """
    out = VGroup()
    for lo, hi in ((0.0, PI), (PI, TAU)):
        pts = []
        for t in np.linspace(lo, hi, n):
            r = abs(np.sin(t)) ** power
            pts.append(scale * r * np.array([np.cos(t), np.sin(t), 0.0]))
        arc = VMobject(color=color, stroke_width=width)
        arc.set_points_smoothly(pts)
        out.add(arc)
    return out


def project(p3):
    """Orthographic 3D -> 2D for the donut wireframe.

    The obvious choice — screen_x = X, screen_y = a linear mix of Y and Z —
    puts the view direction inside the Y-Z plane, which is exactly the plane
    the tube circles live in. They then project to line segments and the
    donut reads as a stack of flattened lobes. Depth has to leak into the
    horizontal too, which is what the Z term in screen_x does.
    """
    X, Y, Z = p3
    return np.array([X + 0.42 * Z, 0.88 * Y + 0.30 * Z, 0.0])


def donut_wireframe(power, scale, n_merid=10, n_par=6, n=90):
    """r = sin^theta^power swept about the kick axis, as meridians and
    parallels. Curves nearer the camera are drawn brighter, which is what
    makes a flat wireframe read as a solid of revolution."""
    def pt(theta, phi):
        r = abs(np.sin(theta)) ** power
        return scale * np.array([r * np.cos(theta),
                                 r * np.sin(theta) * np.cos(phi),
                                 r * np.sin(theta) * np.sin(phi)])

    def curve(points3):
        depth = float(np.mean([q[2] for q in points3]))
        m = VMobject(color=SECOND, stroke_width=2)
        m.set_points_smoothly([project(q) for q in points3])
        m.set_stroke(opacity=0.2 + 0.5 * (0.5 + 0.5 * depth / max(scale, 1e-6)))
        return m

    group = VGroup()
    for k in range(n_merid):                      # meridians: the lobe, rotated
        phi = k * TAU / n_merid
        for lo, hi in ((0.0, PI), (PI, TAU)):
            group.add(curve([pt(t, phi) for t in np.linspace(lo, hi, n)]))
    for k in range(1, n_par + 1):                 # parallels: rings of constant theta
        theta = k * PI / (n_par + 1)
        group.add(curve([pt(theta, f) for f in np.linspace(0, TAU, n)]))
    return group


def wave_field(t, omega, c_r, nx=600, ny=338, half_w=7.12, half_h=4.0):
    """The radiated intensity in the lab frame, as an RGBA array.

    Plotted quantity is **power per unit solid angle**, `r^2 * I`, not the
    intensity itself. That is deliberate rather than a cheat to keep the
    outer rings visible: `I` falls as 1/r^2 while the sphere's area grows as
    r^2, and the whole point of the section is that those cancel exactly. So
    a pattern that does not fade with distance is the honest picture of
    "it doesn't decay" — and it is also the only version that stays legible
    to the frame edge.

    Angular part is sin^2(theta) with theta measured from the kick axis (x),
    matching §1 and §2. Radial part is sin^2(omega*(t - r/c)) — the wave —
    and nothing is drawn beyond r = c*t, because the news has not got there.
    """
    xs = np.linspace(-half_w, half_w, nx)
    ys = np.linspace(half_h, -half_h, ny)
    X, Y = np.meshgrid(xs, ys)
    R = np.maximum(np.hypot(X, Y), 1e-6)

    sin2 = (Y / R) ** 2                      # sin^2(theta) about the x-axis
    retarded = t - R / c_r
    wave = np.sin(omega * retarded) ** 2
    val = np.where(retarded > 0, sin2 * wave, 0.0)
    val *= np.clip(R / 0.45, 0, 1)           # hide the singular origin

    rgb = np.array([int(ACCENT[1:3], 16), int(ACCENT[3:5], 16), int(ACCENT[5:7], 16)])
    out = np.zeros((ny, nx, 4), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = (np.clip(val, 0, 1) * 195).astype(np.uint8)
    return out


class Donut(BrandScene):
    def construct(self):
        self.beat_3a_square_it()
        self.beat_3b_the_shell()
        self.beat_3c_the_donut()
        self.beat_3i_the_wave()
        self.beat_3d_payoff()
        self.beat_3f_magnetic_gap()
        self.beat_3g_closing_puzzle()

    # ------------------------------------------------------------------
    # 3a — the angular shape is already there the moment you square
    # ------------------------------------------------------------------
    def beat_3a_square_it(self):
        S = 2.6
        origin = LEFT * 2.6
        theta = ValueTracker(0.001)

        axis = DashedVMobject(Line(origin + LEFT * 3.3, origin + RIGHT * 3.3,
                                   color=MUTED, stroke_width=2), num_dashes=30)
        axis_label = Text("the kick", font=FONT, weight=FONT_WEIGHT_BODY,
                          color=MUTED).scale(0.38)
        axis_label.next_to(axis, RIGHT, buff=0.18)

        def ray():
            a = theta.get_value()
            return Line(origin, origin + S * 1.25 * np.array([np.cos(a), np.sin(a), 0]),
                        color=MUTED, stroke_width=2)

        def bar():
            a = theta.get_value()
            d = np.array([np.cos(a), np.sin(a), 0.0])
            return Line(origin, origin + S * abs(np.sin(a)) * d,
                        color=ACCENT, stroke_width=7)

        live_ray, live_bar = always_redraw(ray), always_redraw(bar)
        traced = TracedPath(lambda: bar().get_end(), stroke_color=ACCENT,
                            stroke_width=3, dissipating_time=None)

        title = Text("how much gets radiated, per direction",
                     font=FONT, weight=FONT_WEIGHT_TITLE, color=FG).scale(0.58)
        title.to_edge(UP, buff=0.55)
        e_lab = MathTex(r"E_t \propto \sin\theta", color=ACCENT).scale(0.8)
        e_lab.move_to(RIGHT * 3.6 + UP * 1.6)

        self.play(FadeIn(title), run_time=0.8)
        self.play(Create(axis), FadeIn(axis_label), FadeIn(charge_dot(origin)),
                  run_time=RUN_TIME)
        self.add(live_ray, live_bar, traced)
        self.play(FadeIn(e_lab), run_time=0.6)
        self.play(theta.animate.set_value(TAU - 0.001), run_time=5.0, rate_func=linear)
        self.wait(BEAT_PAUSE)

        live_ray.clear_updaters()
        live_bar.clear_updaters()
        traced.clear_updaters()
        self.play(FadeOut(live_ray), FadeOut(live_bar), run_time=0.5)

        # squaring is the whole step: the zeros stay pinned and the lobes
        # pull in. Nothing new is claimed — the same fact gets sharper.
        u_eq = MathTex(r"u = \tfrac{1}{2}\varepsilon_0 E_t^{\,2}", color=FG).scale(0.85)
        u_eq.move_to(RIGHT * 3.6 + UP * 0.35)
        squared = lobe(2, S).move_to(origin)

        self.play(FadeIn(u_eq), run_time=0.8)
        self.play(Transform(traced, squared), FadeOut(e_lab),
                  run_time=RUN_TIME_SLOW * 1.3)
        self.wait(BEAT_PAUSE)

        p_eq = MathTex(r"\frac{dP}{d\Omega} = "
                       r"\frac{q^2 a^2}{32\pi^2\varepsilon_0 c^3}\,\sin^2\theta",
                       color=FG).scale(0.72)
        p_eq.move_to(RIGHT * 3.6 + DOWN * 1.3)
        self.play(Write(p_eq), run_time=RUN_TIME_SLOW)
        self.wait(LONG_PAUSE)
        self.play(FadeOut(Group(*self.mobjects[1:])), run_time=RUN_TIME)
        self.wait(BEAT_PAUSE * 0.4)

    # ------------------------------------------------------------------
    # 3b — what actually travels outward is a shell, not a donut
    # ------------------------------------------------------------------
    def beat_3b_the_shell(self):
        # A local scale, smaller than §1's. The bright part of this pattern
        # is broadside — top and bottom — which is exactly where the frame
        # is shortest, so the vertical half-height sets the scale, not the
        # diagonal. At §1's 0.75 the shell leaves frame a third of the way
        # through the expansion and the payoff caption lands on nothing.
        SC = 0.32            # sized so the shell's top clears the title band
        R_DRAW = 34.0        # must still clear the corners: 34*0.32 = 10.9
        T = ValueTracker(0.35)
        N_SEG = 96

        def shell_ring():
            """The shell, drawn as many short arcs whose brightness follows
            sin^2(theta). One VMobject cannot carry per-point opacity, and
            the two dark poles are the entire point of the shot."""
            t = T.get_value()
            r_out = C * t
            r_in = C * (t - DT)
            emit = kick_position(DT, DT, V)
            group = VGroup()
            for i in range(N_SEG):
                a0 = i * TAU / N_SEG
                a1 = (i + 1) * TAU / N_SEG
                mid = 0.5 * (a0 + a1)
                w = np.sin(mid) ** 2
                if w < 0.012:
                    continue
                arc = AnnularSector(
                    inner_radius=SC * r_in, outer_radius=SC * r_out,
                    angle=a1 - a0, start_angle=a0, color=ACCENT,
                    fill_opacity=0.10 + 0.75 * w, stroke_width=0)
                arc.shift(RIGHT * SC * emit * 0.5)
                group.add(arc)
            return group

        def field():
            t = T.get_value()
            r0 = CHARGE_RADIUS + 0.05
            return retarded_field(state, t, C,
                                  field_samples(r0, R_DRAW, (C * (t - DT), C * t)),
                                  SC, -1e9, -1e9,             # no ACCENT: the
                                  stroke_width=2.5)            # shell carries it

        lines = always_redraw(field)
        ring = always_redraw(shell_ring)
        ring.set_z_index(-300)
        charge = always_redraw(
            lambda: charge_dot(SC * kick_position(T.get_value(), DT, V) * RIGHT))

        title = Text("the same shell — now with its brightness",
                     font=FONT, weight=FONT_WEIGHT_TITLE, color=FG).scale(0.56)
        title.to_edge(UP, buff=0.5)

        self.add(ring, lines, charge)
        self.play(FadeIn(title), run_time=0.8)
        self.wait(BEAT_PAUSE)
        self.play(T.animate.set_value(1.05), run_time=6.5, rate_func=linear)
        self.wait(BEAT_PAUSE)

        holes = caption("brightest broadside.",
                        "two dark holes, straight along the kick.",
                        color=FG, weight=FONT_WEIGHT_TITLE, scale=0.5)
        card = text_card(holes).to_corner(DL, buff=0.5).set_z_index(600)
        self.play(FadeIn(card), run_time=0.8)
        self.wait(LONG_PAUSE)

        for m in (lines, ring, charge):
            m.clear_updaters()
        self.play(FadeOut(Group(lines, ring, charge, title, card)), run_time=RUN_TIME)
        self.wait(BEAT_PAUSE * 0.4)

    # ------------------------------------------------------------------
    # 3c — the donut, introduced as a diagram and not as the thing
    # ------------------------------------------------------------------
    def beat_3c_the_donut(self):
        SH = LEFT * 3.7          # the shell, on the left
        PL = RIGHT * 3.3         # the plot, on the right
        R_SHELL, S_PLOT = 2.1, 2.3
        N_SEG = 96

        shell = VGroup()
        for i in range(N_SEG):
            a0, a1 = i * TAU / N_SEG, (i + 1) * TAU / N_SEG
            w = np.sin(0.5 * (a0 + a1)) ** 2
            shell.add(AnnularSector(inner_radius=R_SHELL * 0.86,
                                    outer_radius=R_SHELL, angle=a1 - a0,
                                    start_angle=a0, color=ACCENT,
                                    fill_opacity=0.10 + 0.75 * w, stroke_width=0))
        shell.move_to(SH)
        shell_lab = Text("the thing", font=FONT, weight=FONT_WEIGHT_LABEL,
                         color=MUTED).scale(0.42).next_to(shell, DOWN, buff=0.35)
        plot_lab = Text("a picture of the thing", font=FONT,
                        weight=FONT_WEIGHT_LABEL, color=MUTED).scale(0.42)

        shell_charge = charge_dot(SH)
        self.play(FadeIn(shell), FadeIn(shell_charge), FadeIn(shell_lab),
                  run_time=RUN_TIME)
        self.wait(BEAT_PAUSE)

        # lift the brightness off the surface and replot it as a radius
        movers = VGroup()
        targets = VGroup()
        for k in range(24):
            a = k * TAU / 24
            d = np.array([np.cos(a), np.sin(a), 0.0])
            w = np.sin(a) ** 2
            movers.add(Dot(SH + R_SHELL * 0.93 * d, radius=0.05, color=ACCENT,
                           fill_opacity=0.15 + 0.85 * w))
            targets.add(Dot(PL + S_PLOT * w * d, radius=0.05, color=ACCENT,
                            fill_opacity=0.15 + 0.85 * w))
        axis = DashedVMobject(Line(PL + LEFT * 2.9, PL + RIGHT * 2.9,
                                   color=MUTED, stroke_width=2), num_dashes=26)

        self.play(FadeIn(movers), run_time=0.7)
        self.play(Create(axis), run_time=0.7)
        self.play(Transform(movers, targets), run_time=RUN_TIME_SLOW * 1.4)
        curve = lobe(2, S_PLOT).move_to(PL)
        plot_lab.next_to(axis, DOWN, buff=1.15)
        self.play(Create(curve), FadeOut(movers), FadeIn(plot_lab), run_time=RUN_TIME)
        self.wait(LONG_PAUSE)

        # ...and only now sweep it into three dimensions
        self.play(FadeOut(Group(shell, shell_lab, shell_charge, movers)),
                  Group(curve, axis, plot_lab).animate.move_to(ORIGIN),
                  run_time=RUN_TIME_SLOW)

        # ...and only now sweep it about the kick axis into three dimensions
        donut = donut_wireframe(2, S_PLOT * 1.2).move_to(ORIGIN)
        donut_lab = caption("the donut is the plot.",
                            "what travels is the shell.",
                            color=FG, weight=FONT_WEIGHT_TITLE, scale=0.5)
        donut_lab.to_corner(DL, buff=0.7)

        self.play(FadeOut(axis), FadeOut(plot_lab), run_time=0.5)
        self.play(LaggedStart(*[Create(r) for r in donut], lag_ratio=0.05),
                  run_time=RUN_TIME_SLOW * 1.6)
        self.play(FadeIn(donut_lab), run_time=0.8)
        self.wait(LONG_PAUSE)
        self.play(FadeOut(Group(donut, curve, donut_lab)), run_time=RUN_TIME)
        self.wait(BEAT_PAUSE * 0.4)

    # ------------------------------------------------------------------
    # 3i — one pulse is not a wave
    # ------------------------------------------------------------------
    def beat_3i_the_wave(self):
        """Shake the charge instead of kicking it once.

        Two things this shot must not do. It must not show a donut
        travelling outward — what travels is the wave, and the donut is the
        plot of its angular shape. And it must not show a charge visibly
        swinging while several wavelengths fit on screen: amplitude and
        wavelength are locked by `A/lambda = v/(2*pi*c)`, so that picture
        would be a superluminal charge. At three wavelengths across the
        frame the real amplitude is about six pixels. The charge barely
        moves; the wave is enormous. That is worth saying rather than
        hiding, so the charge is drawn vibrating by its true amount.
        """
        C_R = 4.0                     # rendered units per second, this shot only

        # --- first the field lines, shaken slowly. Slow keeps it cusp-free
        # and keeps the swing visible; the wavelength is then far larger
        # than the frame, which is exactly why no wave is visible yet.
        AMP, OM_SLOW = 0.62, 3.4
        T = ValueTracker(0.0)

        def slow_state(tau):
            if tau <= 0:
                return np.zeros(3), np.zeros(3)
            ramp = min(tau / 1.2, 1.0)
            env = 3 * ramp ** 2 - 2 * ramp ** 3
            return (np.array([AMP * env * np.sin(OM_SLOW * tau), 0.0, 0.0]),
                    np.array([AMP * env * OM_SLOW * np.cos(OM_SLOW * tau), 0.0, 0.0]))

        lines = always_redraw(lambda: retarded_field(
            slow_state, T.get_value(), C, field_samples(0.3, 26.0, ()),
            0.75, -1e9, -1e9, stroke_width=2.5))
        charge = always_redraw(
            lambda: charge_dot(0.75 * slow_state(T.get_value())[0]))

        title = Text("shake it, instead of kicking it once",
                     font=FONT, weight=FONT_WEIGHT_TITLE, color=FG).scale(0.56)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.8)
        self.add(lines, charge)
        self.play(T.animate.set_value(6.0), run_time=6.0, rate_func=linear)
        self.wait(BEAT_PAUSE * 0.6)
        lines.clear_updaters()
        charge.clear_updaters()
        self.play(FadeOut(lines), FadeOut(charge), run_time=RUN_TIME)

        # The probe-grid version of this wave was built and cut. It plots
        # E ∝ sin(theta) while this section's claim is dP/dOmega ∝
        # sin^2(theta), so it needed a whole extra beat explaining that the
        # two shapes differ (120 degrees of beamwidth against 90) purely to
        # stop two of our own shots contradicting each other. "Transverse"
        # is already earned by §2's field triangle and replayed as the first
        # callback in the payoff, and §2's 2i grid already closes the
        # drawing-convention objection where it actually bites — about the
        # kink, which is a drawn line. Here it was a third helping.
        #
        # What the cut costs: the intensity picture pulses but never
        # reverses, so it cannot show that the field flips sign. The video
        # never claims to derive the oscillation, so that belongs with the
        # antenna material or part 2, not here.

        # --- and finally the pattern on its own, at pixel resolution
        W = ValueTracker(0.0)
        omega = {"v": 2 * PI * C_R / 3.4}      # lambda = 3.4 rendered units

        def picture():
            img = ImageMobject(wave_field(W.get_value(), omega["v"], C_R))
            img.stretch_to_fit_width(config.frame_width)
            img.stretch_to_fit_height(config.frame_height)
            img.set_z_index(-600)
            return img

        wave = always_redraw(picture)
        # the true amplitude at this wavelength, drawn rather than exaggerated
        amp_true = (0.10 / TAU) * 3.4
        vib = always_redraw(lambda: charge_dot(
            RIGHT * amp_true * np.sin(omega["v"] * W.get_value())))

        self.add(wave, vib)
        cap = text_card(caption("brightness is power per angle —",
                                "it does not fade with distance.", scale=0.42))
        cap.to_corner(DL, buff=0.5).set_z_index(600)
        self.play(W.animate.set_value(3.2), run_time=3.2, rate_func=linear)
        self.play(FadeIn(cap), run_time=0.7)
        self.play(W.animate.set_value(6.4), run_time=3.2, rate_func=linear)
        self.wait(BEAT_PAUSE)

        note = text_card(caption("the charge moves a few pixels.",
                                 "the wave is the whole frame.",
                                 color=FG, weight=FONT_WEIGHT_TITLE, scale=0.46))
        note.to_corner(DR, buff=0.5).set_z_index(600)
        self.play(FadeOut(cap), FadeIn(note), run_time=0.8)
        self.play(W.animate.set_value(9.6), run_time=3.2, rate_func=linear)
        self.wait(BEAT_PAUSE)

        # --- shake faster: the pattern is untouched, the wavelength halves
        faster = text_card(caption("shake twice as fast:",
                                   "half the wavelength, same pattern.",
                                   color=FG, weight=FONT_WEIGHT_TITLE, scale=0.46))
        faster.to_corner(DR, buff=0.5).set_z_index(600)
        self.play(FadeOut(note), FadeIn(faster), run_time=0.8)
        omega["v"] *= 2
        self.play(W.animate.set_value(13.0), run_time=4.0, rate_func=linear)
        self.wait(LONG_PAUSE)

        wave.clear_updaters()
        vib.clear_updaters()
        self.play(FadeOut(Group(wave, vib, title, faster)), run_time=RUN_TIME)
        self.wait(BEAT_PAUSE * 0.4)

    # ------------------------------------------------------------------
    # 3d/3e — the payoff, and the name that is withheld
    # ------------------------------------------------------------------
    def beat_3d_payoff(self):
        title = Text("what we have, after three rules",
                     font=FONT, weight=FONT_WEIGHT_TITLE, color=FG).scale(0.6)
        title.to_edge(UP, buff=0.6)
        self.play(FadeIn(title), run_time=0.8)

        # each property arrives next to a two-second reminder of the beat
        # that earned it, rather than as a bullet on a card
        rows = VGroup()
        icons = VGroup()

        # 1 — transverse: the field triangle at the kink
        tri = VGroup(
            Line(ORIGIN, UP * 0.62, color=SECOND, stroke_width=4),
            Line(UP * 0.62, UP * 0.62 + LEFT * 0.42, color=ACCENT, stroke_width=4),
            Line(ORIGIN, UP * 0.62 + LEFT * 0.42, color=FG, stroke_width=3),
        )
        # 2 — doesn't decay: two spheres, same count
        flat = VGroup(
            Circle(radius=0.34, color=MUTED, stroke_width=2),
            Circle(radius=0.62, color=MUTED, stroke_width=2),
            *[Line(0.34 * np.array([np.cos(a), np.sin(a), 0]),
                   0.62 * np.array([np.cos(a), np.sin(a), 0]),
                   color=ACCENT, stroke_width=2.5)
              for a in np.linspace(0, TAU, 9)[:-1]],
        )
        # 3 — moves at c: the expanding shell
        shell = VGroup(*[Circle(radius=r, color=ACCENT, stroke_width=2,
                                stroke_opacity=op)
                         for r, op in ((0.28, 0.3), (0.46, 0.55), (0.66, 0.9))])

        for icon, words in ((tri, "transverse"),
                            (flat, "doesn't decay"),
                            (shell, "moves at c")):
            lab = Text(words, font=FONT, weight=FONT_WEIGHT_TITLE, color=FG).scale(0.55)
            row = VGroup(icon, lab).arrange(RIGHT, buff=0.55)
            rows.add(row)
            icons.add(icon)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.62).shift(LEFT * 1.6 + UP * 0.3)

        for row in rows:
            self.play(FadeIn(row[0], scale=0.7), run_time=0.6)
            self.play(FadeIn(row[1], shift=RIGHT * 0.2), run_time=0.5)
            self.wait(BEAT_PAUSE * 0.8)
        self.wait(BEAT_PAUSE)

        # 3e — the name is withheld. The gap is the point, so it gets drawn.
        blank = VGroup(
            Text("so this is", font=FONT, weight=FONT_WEIGHT_TITLE, color=MUTED).scale(0.55),
            Line(ORIGIN, RIGHT * 1.5, color=ACCENT, stroke_width=3),
        ).arrange(RIGHT, buff=0.35)
        blank.next_to(rows, DOWN, buff=0.8).align_to(rows, LEFT)
        self.play(FadeIn(blank[0]), Create(blank[1]), run_time=RUN_TIME)
        self.wait(LONG_PAUSE)

        # and the equations we were never allowed to use
        maxwell = VGroup(
            MathTex(r"\nabla\cdot\vec{E}=\rho/\varepsilon_0"),
            MathTex(r"\nabla\cdot\vec{B}=0"),
            MathTex(r"\nabla\times\vec{E}=-\partial_t\vec{B}"),
            MathTex(r"\nabla\times\vec{B}=\mu_0\vec{J}+\mu_0\varepsilon_0\partial_t\vec{E}"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.62)
        maxwell.set_color(MUTED).to_edge(RIGHT, buff=0.9)
        self.play(FadeIn(maxwell), run_time=RUN_TIME)
        self.wait(BEAT_PAUSE)
        strike = Line(maxwell.get_corner(DL), maxwell.get_corner(UR),
                      color=ACCENT, stroke_width=5)
        self.play(Create(strike), run_time=RUN_TIME)
        never = Text("never used", font=FONT, weight=FONT_WEIGHT_TITLE,
                     color=ACCENT).scale(0.5).next_to(maxwell, DOWN, buff=0.4)
        self.play(FadeIn(never), run_time=0.7)
        self.wait(LONG_PAUSE)
        self.play(FadeOut(Group(title, rows, blank, maxwell, strike, never)),
                  run_time=RUN_TIME)
        self.wait(BEAT_PAUSE * 0.4)

    # ------------------------------------------------------------------
    # 3f — the magnetic gap, drawn as a gap
    # ------------------------------------------------------------------
    def beat_3f_magnetic_gap(self):
        title = Text("what the magnetic field changes",
                     font=FONT, weight=FONT_WEIGHT_TITLE, color=FG).scale(0.58)
        title.to_edge(UP, buff=0.6)

        # B enters in a different register and stays that way for the rest of
        # the video — the viewer should be able to see which half of the
        # picture has been paid for and which is borrowed from part 2.
        e_arrow = Arrow(ORIGIN, UP * 1.5, color=ACCENT, buff=0, stroke_width=6,
                        max_tip_length_to_length_ratio=0.22)
        e_lab = MathTex("E_t", color=ACCENT).scale(0.8).next_to(e_arrow, LEFT, buff=0.2)
        b_arrow = DashedVMobject(
            Arrow(ORIGIN, RIGHT * 1.5, color=MUTED, buff=0, stroke_width=6,
                  max_tip_length_to_length_ratio=0.22), num_dashes=9)
        b_lab = MathTex("B_t", color=MUTED).scale(0.8).next_to(b_arrow, DOWN, buff=0.2)
        pair = VGroup(e_arrow, e_lab, b_arrow, b_lab).move_to(LEFT * 3.6)

        borrowed = caption("dashed: taken on trust from part 2,",
                           "not earned here.", scale=0.42)
        borrowed.next_to(pair, DOWN, buff=0.7)

        self.play(FadeIn(title), run_time=0.8)
        self.play(GrowArrow(e_arrow), FadeIn(e_lab), run_time=RUN_TIME)
        self.play(Create(b_arrow), FadeIn(b_lab), run_time=RUN_TIME)
        self.play(FadeIn(borrowed), run_time=0.7)
        self.wait(BEAT_PAUSE)

        # the factor of two — and the shape does not move at all
        curve = lobe(2, 2.0).move_to(RIGHT * 3.0)
        before = MathTex(r"\frac{dP}{d\Omega} = "
                         r"\frac{q^2 a^2}{32\pi^2\varepsilon_0 c^3}\,\sin^2\theta",
                         color=FG).scale(0.66)
        before.next_to(curve, DOWN, buff=0.75)
        after = MathTex(r"\frac{dP}{d\Omega} = "
                        r"\frac{q^2 a^2}{16\pi^2\varepsilon_0 c^3}\,\sin^2\theta",
                        color=ACCENT).scale(0.66).move_to(before)

        self.play(Create(curve), FadeIn(before), run_time=RUN_TIME)
        self.wait(BEAT_PAUSE)
        self.play(TransformMatchingTex(before, after), run_time=RUN_TIME_SLOW)
        same = caption("the number doubles.", "the shape does not move.",
                       color=FG, weight=FONT_WEIGHT_TITLE, scale=0.46)
        same.next_to(after, DOWN, buff=0.5)
        self.play(FadeIn(same), Flash(curve.get_center(), color=ACCENT,
                                      line_length=0.25, num_lines=14,
                                      flash_radius=2.3), run_time=RUN_TIME)
        self.wait(LONG_PAUSE)
        self.play(FadeOut(Group(title, pair, borrowed, curve, after, same)),
                  run_time=RUN_TIME)
        self.wait(BEAT_PAUSE * 0.4)

    # ------------------------------------------------------------------
    # 3g — the closing puzzle, left open on purpose
    # ------------------------------------------------------------------
    def beat_3g_closing_puzzle(self):
        chain = VGroup(
            MathTex(r"E_t \propto \frac{1}{r}", color=ACCENT).scale(0.8),
            MathTex(r"u \propto \frac{1}{r^2}", color=FG).scale(0.8),
            MathTex(r"A \propto r^2", color=FG).scale(0.8),
            MathTex(r"P = \text{const}", color=ACCENT).scale(0.9),
        ).arrange(RIGHT, buff=0.85)
        chain.shift(UP * 1.35)
        arrows = VGroup(*[
            Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.16,
                  color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.35)
            for i in range(3)])

        self.play(LaggedStart(*[FadeIn(m) for m in chain], lag_ratio=0.35),
                  run_time=RUN_TIME_SLOW)
        self.play(*[GrowArrow(a) for a in arrows], run_time=RUN_TIME)
        self.wait(BEAT_PAUSE)

        note = caption("nowhere in that chain does the magnetic field appear.",
                       color=FG, weight=FONT_WEIGHT_TITLE, scale=0.5)
        note.next_to(chain, DOWN, buff=0.8)
        self.play(FadeIn(note), run_time=0.8)
        self.wait(LONG_PAUSE)

        # the standard story, drawn MUTED and left connected to nothing
        loop = VGroup(
            MathTex("E", color=MUTED).scale(0.9).move_to(LEFT * 1.1),
            MathTex("B", color=MUTED).scale(0.9).move_to(RIGHT * 1.1),
        )
        arc1 = CurvedArrow(LEFT * 0.8 + UP * 0.2, RIGHT * 0.8 + UP * 0.2,
                           angle=-1.1, color=MUTED, stroke_width=3, tip_length=0.2)
        arc2 = CurvedArrow(RIGHT * 0.8 + DOWN * 0.2, LEFT * 0.8 + DOWN * 0.2,
                           angle=-1.1, color=MUTED, stroke_width=3, tip_length=0.2)
        story = VGroup(loop, arc1, arc2).move_to(DOWN * 1.9)
        story_lab = Text("the usual explanation", font=FONT,
                         weight=FONT_WEIGHT_BODY, color=MUTED).scale(0.4)
        story_lab.next_to(story, DOWN, buff=0.35)

        self.play(FadeIn(story), FadeIn(story_lab), run_time=RUN_TIME)
        self.wait(LONG_PAUSE)
        self.wait(BEAT_PAUSE)
