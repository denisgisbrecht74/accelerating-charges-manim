"""Section 2 — Integration, trigonometry. Built against script.md §2.

Four independent scenes, one per movement of the script:

    Derivation      B1  the triangle -> E_t = Q a sin(theta) / 4 pi eps0 c^2 r
    TheWave         B2  what E_t looks like: one kick, then a shake
    PowerLaws       B3  1/r vs 1/r^2, and why that is the whole story
    Identification  B4  the four properties, and the name

Same house style as §1: minimal text, but **here the formulas ARE the
content** — this section is a derivation, so the equations get room and
each symbol arrives only once its length or angle exists on screen.
ACCENT stays locked to the transverse thing.

The event is §1's event, unchanged: C = 9, v/c = 0.097, shell 1.0 units
thick at radius 3.6.

**Script note.** The script writes the final formula with a lower-case q
("E_t = q a sin theta / 4 pi eps_0 c^2 r"), but §1 established Q as the
source charge and q as the *test* charge that gets removed to define E.
The charge in this formula is the source, so it is rendered as Q here.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from manim import *

from theme import (ACCENT, BrandScene, FG, FONT, FONT_WEIGHT_LABEL,
                   MUTED, SECOND)
from visuals import (
    charge_dot,
    maxwell_block,
    field_samples,
    kick_position,
    kick_state,
    retarded_field,
    retarded_point,
    signed_field_key,
    text_card,
)

FIELD_N = 16
FIELD_STEP = TAU / FIELD_N

# §1's event, unchanged
C = 9.0
V = 0.871
SC = 0.75
R_MAX = 20.0
DT = (1.0 / 0.75) / 9.0
T0 = 20.0 / 9.0 + 0.18
T_SHOT = T0 + (3.6 / 0.75) / 9.0
ACC = V / DT                      # the acceleration held fixed below
UP3 = np.array([0.0, 1.0, 0.0])   # broadside: where the effect is largest


# ======================================================================
# B1 — the derivation
# ======================================================================
class Derivation(BrandScene):
    """From §1's frozen kink to E_t = Q a sin(theta) / (4 pi eps0 c^2 r).

    **The kick itself is never rescaled.** An earlier version shrank Delta
    t while holding acceleration fixed so the whole event stayed fold-back
    safe — but that construction is self-similar: `kick_velocity` traces
    the same S-shaped profile at every scale, so the "small Delta t"
    segment never gets any straighter, no matter how far you shrink it.
    Verified: the relative deviation from the chord sat at a flat 3.27%
    from Delta t / 8 all the way down to Delta t / 200.

    A genuine local Taylor expansion needs a FIXED event and a shrinking
    WINDOW around one instant — here, the midpoint of the original kick.
    Zooming a factor of 22 into a window +/- 4.5% of Delta t around that
    point drops the deviation to 0.17%, and it keeps falling as the window
    narrows further, which is what "zoom in and it looks linear" actually
    requires. The original §1 event (T0, DT, V) is never touched.
    """

    EPS_FRAC = 0.045     # zoom window half-width, as a fraction of DT
    ZOOM = 22.0

    def construct(self):
        def st_of():
            return lambda tau: kick_state(tau - T0, DT, V)

        st = st_of()
        mid = T0 + DT / 2
        eps = self.EPS_FRAC * DT

        def build():
            s_new, s_old = C * (T_SHOT - T0 - DT), C * (T_SHOT - T0)
            return retarded_field(st, T_SHOT, C,
                                  field_samples(0.0, R_MAX, (s_new, s_old)),
                                  SC, s_new, s_old,
                                  n_lines=FIELD_N, stroke_width=3.0)

        live = build()
        Q = charge_dot(SC * np.array([kick_position(T_SHOT - T0, DT, V), 0, 0]),
                       radius=0.18)
        self.add(live, Q)
        self.wait(1.8)

        # --- "we will look at a very small timespan of acceleration" -----
        # A bracket on the still-wide shot, marking exactly the sub-window
        # about to be zoomed into — so the cut to the close-up reads as
        # "this piece", not as a new picture.
        pivot = retarded_point(st, T_SHOT, C, mid, UP3, SC)
        p_hi = retarded_point(st, T_SHOT, C, mid + eps, UP3, SC)
        p_lo = retarded_point(st, T_SHOT, C, mid - eps, UP3, SC)
        bracket = Line(p_lo, p_hi, color=ACCENT, stroke_width=7)
        self.play(Create(bracket), run_time=1.0)
        self.wait(1.4)

        # --- so we zoom ---------------------------------------------------
        HOME = np.array([-3.6, 0.0, 0.0])
        frozen = live
        self.remove(bracket)
        # The zoom is about the kink, not about the particle. Remove Q
        # before enlarging the field geometry; otherwise its fade while the
        # line picture races past it reads as the source travelling with the
        # camera and competes with the local straight-line approximation.
        self.remove(Q)
        self.play(frozen.animate.scale(self.ZOOM, about_point=pivot)
                  .shift(HOME - pivot), run_time=2.6, rate_func=smooth)
        self.wait(0.8)

        def zoomed(p):
            return (p - pivot) * self.ZOOM + HOME

        A, B = zoomed(p_hi), zoomed(p_lo)       # A inner/lower, B outer/upper

        # --- and the velocity change is seemingly linear -----------------
        # The one approximation in the whole video, so it is watched rather
        # than asserted: the true curve stays on screen in MUTED and the
        # straight chord is laid over it.
        true_curve = VMobject(color=MUTED, stroke_width=3)
        true_curve.set_points_smoothly(
            [zoomed(retarded_point(st, T_SHOT, C, tau, UP3, SC))
             for tau in np.linspace(mid - eps, mid + eps, 40)])
        chord = Line(A, B, color=ACCENT, stroke_width=5)

        self.play(frozen.animate.set_stroke(opacity=0.28), run_time=0.8)
        self.add(true_curve)
        self.play(Create(chord), run_time=1.4)
        self.wait(1.6)

        # --- split it into components -------------------------------------
        corner = np.array([A[0], B[1], 0.0])
        leg_r = Line(A, corner, color=SECOND, stroke_width=4)
        leg_t = Line(corner, B, color=ACCENT, stroke_width=4)
        sq = RightAngle(Line(corner, A), Line(corner, B), length=0.3,
                        color=MUTED, stroke_width=2)

        lab_r = MathTex("E_r", color=SECOND).scale(0.75).next_to(leg_r, RIGHT, buff=0.2)
        lab_t = MathTex("E_t", color=ACCENT).scale(0.75).next_to(leg_t, UP, buff=0.2)

        self.play(Create(leg_r), Create(leg_t), FadeIn(sq), run_time=1.4)
        self.play(FadeIn(lab_r), FadeIn(lab_t), run_time=0.8)
        self.wait(2.0)

        # --- the same triangle, named the other way -----------------------
        # Not a second triangle beside the first: the geometry is held
        # perfectly still and only the labels cross-fade. If a second
        # triangle appeared the viewer would have to be TOLD they are
        # similar; with one shape wearing two sets of labels there is
        # nothing else it could mean.
        lab_r2 = MathTex(r"c\,\Delta t", color=SECOND).scale(0.7) \
            .next_to(leg_r, RIGHT, buff=0.2)
        lab_t2 = MathTex(r"\Delta v\,t", color=ACCENT).scale(0.7) \
            .next_to(leg_t, UP, buff=0.2)
        self.play(FadeOut(lab_r), FadeOut(lab_t), run_time=0.5)
        self.play(FadeIn(lab_r2), FadeIn(lab_t2), run_time=0.9)
        self.wait(2.2)

        # --- ratios of a triangle are constant ----------------------------
        # Each symbol wears the colour of the leg it measures, so the
        # equation and the triangle are the same object twice.
        eq1 = MathTex(r"\frac{E_t}{E_r} = \frac{\Delta v\,t}{c\,\Delta t}",
                      substrings_to_isolate=["E_t", "E_r"],
                      color=FG).scale(0.85).move_to(np.array([3.1, 1.4, 0]))
        eq1.set_color_by_tex("E_t", ACCENT)
        eq1.set_color_by_tex("E_r", SECOND)
        self.play(Write(eq1), run_time=1.8)
        self.wait(1.8)

        eq2 = MathTex(r"E_t = \frac{\Delta v\,t}{c\,\Delta t}\,E_r",
                      substrings_to_isolate=["E_t", "E_r"],
                      color=FG).scale(0.85).move_to(np.array([3.1, -0.4, 0]))
        eq2.set_color_by_tex("E_t", ACCENT)
        eq2.set_color_by_tex("E_r", SECOND)
        self.play(TransformMatchingTex(eq1.copy(), eq2), run_time=1.6)
        self.wait(2.4)

        # ---- only true broadside: Delta v -> Delta v sin(theta) ---------
        self.play(FadeOut(VGroup(frozen, true_curve, chord, leg_r, leg_t, sq,
                                 lab_r2, lab_t2)),
                  FadeOut(eq1), run_time=1.0)

        O = np.array([-3.9, -1.0, 0.0])
        th = ValueTracker(PI / 2)
        DVL = 2.3

        chg = charge_dot(O, radius=0.15)
        dv_arrow = Arrow(O, O + RIGHT * DVL, color=FG, buff=0, stroke_width=5,
                         max_tip_length_to_length_ratio=0.16)
        dv_lab = MathTex(r"\Delta v", color=FG).scale(0.6) \
            .next_to(dv_arrow, DOWN, buff=0.18)

        def u():
            a = th.get_value()
            return np.array([math.cos(a), math.sin(a), 0.0])

        sight = always_redraw(lambda: Line(O, O + u() * 5.0, color=MUTED,
                                           stroke_width=2.5))
        # the piece of Delta v that survives perpendicular to the line of
        # sight: |Delta v| sin(theta), and it is the whole content of the
        # sine factor
        def perp():
            tip = O + RIGHT * DVL
            foot = O + u() * float((tip - O) @ u())
            return Arrow(foot, tip, color=ACCENT, buff=0, stroke_width=5,
                         max_tip_length_to_length_ratio=0.22)

        perp_arrow = always_redraw(perp)
        ang = always_redraw(lambda: Angle(
            Line(O, O + RIGHT), Line(O, O + u()), radius=0.75,
            color=MUTED, stroke_width=2.5))
        th_lab = always_redraw(lambda: MathTex(r"\theta", color=MUTED).scale(0.6)
                               .move_to(O + 1.08 * np.array(
                                   [math.cos(th.get_value() / 2),
                                    math.sin(th.get_value() / 2), 0.0])))

        self.play(FadeIn(chg), GrowArrow(dv_arrow), FadeIn(dv_lab), run_time=1.0)
        self.wait(0.8)

        # --- "this is only for the perpendicular field lines" -------------
        # At theta = pi/2 the sight line is vertical and the perpendicular
        # component coincides EXACTLY with the full Delta v arrow — so the
        # symbol v_perp is introduced as a relabelling of the arrow already
        # on screen, before sin(theta) exists anywhere.
        self.add(sight, ang, th_lab, perp_arrow)
        self.play(FadeIn(sight), FadeIn(ang), FadeIn(th_lab), run_time=0.8)
        perp_lab = MathTex(r"\Delta v_\perp", color=ACCENT).scale(0.6) \
            .next_to(perp_arrow, UP, buff=0.18)
        self.play(FadeIn(perp_lab), run_time=0.8)
        self.wait(1.0)

        eq2b = MathTex(r"E_t = \frac{\Delta v_\perp\,t}{c\,\Delta t}\,E_r",
                       substrings_to_isolate=["E_t", "E_r", r"\Delta v_\perp"],
                       color=FG).scale(0.85).move_to(np.array([3.1, -0.4, 0]))
        eq2b.set_color_by_tex("E_t", ACCENT)
        eq2b.set_color_by_tex("E_r", SECOND)
        eq2b.set_color_by_tex(r"\Delta v_\perp", ACCENT)
        self.play(TransformMatchingTex(eq2, eq2b), run_time=1.4)
        self.wait(1.6)

        # --- and THEN sin(theta): swing it and watch v_perp shrink --------
        self.play(th.animate.set_value(0.16), run_time=3.4, rate_func=smooth)
        self.wait(1.4)
        self.play(th.animate.set_value(PI / 2), run_time=2.2, rate_func=smooth)
        self.wait(0.8)

        vperp_eq = MathTex(r"\Delta v_\perp = \Delta v\,\sin\theta",
                           substrings_to_isolate=[r"\Delta v_\perp"],
                           color=FG).scale(0.62)
        vperp_eq.set_color_by_tex(r"\Delta v_\perp", ACCENT)
        vperp_eq.next_to(dv_lab, DOWN, buff=0.35).align_to(dv_lab, LEFT)
        self.play(FadeIn(vperp_eq), run_time=1.0)
        self.wait(1.6)

        eq3 = MathTex(r"E_t = \frac{\Delta v\,t\,\sin\theta}{c\,\Delta t}\,E_r",
                      substrings_to_isolate=["E_t", "E_r", r"\sin\theta"],
                      color=FG).scale(0.85).move_to(np.array([3.1, -0.4, 0]))
        eq3.set_color_by_tex("E_t", ACCENT)
        eq3.set_color_by_tex("E_r", SECOND)
        eq3.set_color_by_tex(r"\sin\theta", ACCENT)
        self.play(TransformMatchingTex(eq2b, eq3), FadeOut(vperp_eq), run_time=1.6)
        self.wait(2.2)

        self.play(FadeOut(VGroup(chg, dv_arrow, dv_lab, perp_lab)),
                  FadeOut(sight), FadeOut(ang), FadeOut(th_lab),
                  FadeOut(perp_arrow), run_time=0.9)
        self.remove(sight, ang, th_lab, perp_arrow)

        # ---- the substitutions -----------------------------------------
        self.play(eq3.animate.move_to(np.array([0.0, 2.3, 0])), run_time=1.0)

        # E_r = F_C / q first (§1's own definition of the field), THEN
        # Coulomb's law for F_C — not straight to the Coulomb form, which
        # would skip the step that actually makes it "E" rather than "F".
        top_row = VGroup(
            MathTex(r"\frac{\Delta v}{\Delta t} = a", color=MUTED).scale(0.7),
            MathTex(r"t = \frac{r}{c}", color=MUTED).scale(0.7),
        ).arrange(RIGHT, buff=1.8)
        er_row = MathTex(
            r"E_r = \frac{F_C}{q} = \frac{Q}{4\pi\varepsilon_0 r^2}",
            color=MUTED).scale(0.7)
        subs = VGroup(top_row, er_row).arrange(DOWN, buff=0.35) \
            .move_to(np.array([0.0, 0.55, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in top_row], lag_ratio=0.35),
                  run_time=1.4)
        self.wait(0.6)
        self.play(FadeIn(er_row), run_time=1.0)
        self.wait(2.4)

        final = MathTex(r"E_t = \frac{Q\,a\,\sin\theta}{4\pi\varepsilon_0\,c^2\,r}",
                        substrings_to_isolate=["E_t"],
                        color=FG).scale(1.15).move_to(np.array([0.0, -1.6, 0]))
        final.set_color_by_tex("E_t", ACCENT)
        box = SurroundingRectangle(final, color=MUTED, buff=0.35,
                                   corner_radius=0.12, stroke_width=2)
        self.play(Write(final), run_time=2.2)
        self.play(Create(box), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(eq3), FadeOut(subs), run_time=1.0)
        self.play(VGroup(final, box).animate.move_to(ORIGIN), run_time=1.2)
        self.wait(2.6)


# ======================================================================
# B2 — what E_t actually looks like
# ======================================================================
class TheWave(BrandScene):
    """`E_t` plotted as a signed field over the plane, not as a graph.

    One hue per sign, brightness by magnitude, charge at centre — which is
    the only way the two dead spots along the shake axis are visible as
    *gaps in the pattern* rather than as a claim.

    Everything is the retarded expression the derivation just produced,

        E_t(r, theta, t)  ~  a(t - r/c) * sin(theta) / r

    evaluated on a grid: the rings are at `t - r/c`, so they march out at
    exactly c because that is what the formula says, not because they were
    animated to.
    """

    C = 3.0
    W0 = 5.39              # lambda = 2 pi c / w = 3.5 units -> 4 across frame
    AMP0 = 0.35
    VMAX = 2.2             # knee of the tanh compression, in field units
    RES = (384, 216)

    def construct(self):
        self.grid()
        img = always_redraw(lambda: self.plot(self.t.get_value()))
        self.add(img)

        chg = always_redraw(
            lambda: charge_dot(RIGHT * self.shake(self.t.get_value()),
                               radius=0.16, glow=False)
            .set_opacity(self.charge_dim.get_value()))
        self.add(chg)

        # --- the same single kick as before: one ring, undramatic --------
        self.mode = "kick"
        self.play(self.t.animate.set_value(4.2), run_time=6.0, rate_func=linear)
        self.wait(1.2)

        # --- now shake it ------------------------------------------------
        self.mode = "shake"
        self.t.set_value(0.0)
        # The two hues now encode the *sign* of one field, not two
        # different fields. This needs saying because orange has denoted
        # the kink and blue the radial field everywhere else in the film.
        sign_key = signed_field_key().to_corner(UR, buff=0.48)
        sign_key.set_z_index(600)
        self.play(FadeIn(sign_key), run_time=0.7)
        self.play(self.t.animate.set_value(9.0), run_time=11.0, rate_func=linear)
        self.wait(0.8)

        # The raster already shows the angular zeros and travelling rings.
        # Do not redraw either fact as a figure-eight envelope or a separate
        # sine graph; move directly into the equation that explains the
        # same picture.
        self.play(self.dim.animate.set_value(0.0),
                  self.charge_dim.animate.set_value(0.0),
                  FadeOut(sign_key), run_time=1.0)
        self.remove(img, chg)

        # --- and the wave is already in the equation --------------------
        displacement = MathTex(r"x(t) = A\sin(\omega t)", color=FG).scale(0.85)
        acceleration = MathTex(r"a(t) = -A\omega^2\sin(\omega t)",
                               color=FG).scale(0.85)
        retarded = MathTex(
            r"E_t(r,t) \propto "
            r"\frac{\sin\!\left(\omega(t-r/c)\right)\,\sin\theta}{r}",
            substrings_to_isolate=["E_t"], color=FG).scale(0.78)
        chain = VGroup(displacement, acceleration, retarded) \
            .arrange(DOWN, buff=0.65)
        retarded.set_color_by_tex("E_t", ACCENT)

        wave_form = MathTex(
            r"E_t(r,t) \propto \frac{\sin(\omega t-kr)\,\sin\theta}{r}"
            r"\qquad k=\frac{\omega}{c}",
            substrings_to_isolate=["E_t"], color=FG).scale(0.78)
        wave_form.set_color_by_tex("E_t", ACCENT)
        wave_form.move_to(retarded)

        # The formula is now the only thing on screen. The source and raster
        # stay hidden until the exploratory controls return.
        for m in (displacement, acceleration, retarded):
            self.play(Write(m), run_time=1.5)
            self.wait(0.7)
        self.play(TransformMatchingTex(retarded, wave_form), run_time=1.5)
        self.wait(2.0)
        self.add(img, chg)
        self.play(FadeOut(VGroup(displacement, acceleration, wave_form)),
                  self.dim.animate.set_value(1.0),
                  self.charge_dim.animate.set_value(1.0), run_time=1.2)

        # --- the two knobs ------------------------------------------------
        # Script note: in the shaking picture there is no "acceleration
        # time" — Delta t belonged to the single kick. The knobs here are
        # amplitude and frequency, and frequency changes the WAVELENGTH,
        # which is the more interesting one and is the difference between
        # radio and visible light.
        strength = self.control_card([
            ("a/a₀", lambda: self.amp.get_value() / self.AMP0, ACCENT),
        ]).to_corner(UR, buff=0.48)
        strength.suspend_updating()
        self.play(FadeIn(strength),
                  self.t.animate.set_value(11.5), run_time=2.5, rate_func=linear)
        strength.resume_updating()
        self.play(self.amp.animate.set_value(self.AMP0 * 2.1),
                  self.t.animate.set_value(14.5),
                  run_time=3.0, rate_func=linear)
        self.wait(1.2)
        self.play(self.amp.animate.set_value(self.AMP0),
                  self.t.animate.set_value(16.5),
                  run_time=2.0, rate_func=linear)

        frequency = self.control_card([
            ("ω/ω₀", lambda: self.w.get_value() / self.W0, FG),
            ("λ/λ₀", lambda: self.W0 / self.w.get_value(), MUTED),
        ]).move_to(strength)
        # The live number rebuilds itself on every frame. Freeze it before
        # fading or it restores full opacity while the next card appears.
        # Finish the old card first so the two compact readouts never share
        # the same corner, even during the transition.
        strength.clear_updaters()
        self.play(FadeOut(strength), run_time=0.35)
        frequency.suspend_updating()
        self.play(FadeIn(frequency), run_time=0.35)
        frequency.resume_updating()
        self.play(self.w.animate.set_value(self.W0 * 1.7),
                  self.norm.animate.set_value(1.7 ** 2),
                  self.t.animate.set_value(19.5),
                  run_time=3.0, rate_func=linear)
        self.wait(2.4)
        frequency.clear_updaters()
        self.play(FadeOut(frequency), run_time=0.7)

        # One last variation: keep plotting E_t, but let the direction of
        # acceleration rotate instead of restricting the charge to one axis.
        self.circular_motion(img, chg, sign_key)

    # ------------------------------------------------------------------
    def grid(self):
        w, h = self.RES
        xs = np.linspace(-config.frame_width / 2, config.frame_width / 2, w)
        ys = np.linspace(config.frame_height / 2, -config.frame_height / 2, h)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.R = np.hypot(self.X, self.Y)
        self.SIN = np.abs(self.Y) / np.maximum(self.R, 1e-6)
        self.mode = "kick"
        self.t = ValueTracker(0.0)
        self.amp = ValueTracker(self.AMP0)
        self.w = ValueTracker(self.W0)
        # display gain. Left at 1 the amplitude knob shows as brighter
        # peaks, which is what it is. For the frequency knob it is driven
        # as (w/w0)^2 to cancel the a = -A w^2 growth, so that knob
        # isolates the one thing it is about: the wavelength.
        self.norm = ValueTracker(1.0)
        # Dimming has to happen INSIDE plot(). `img.animate.set_opacity()`
        # on an always_redraw is a no-op: the updater builds a fresh
        # ImageMobject at full strength on the very next frame and throws
        # the animated value away.
        self.dim = ValueTracker(1.0)
        self.charge_dim = ValueTracker(1.0)

    @staticmethod
    def control_card(rows):
        """A live, named control for the final exploratory beat.

        The animation already changes amplitude and frequency, but without
        a readout that action looks like cosmetic motion. Each card states
        the parameter, its reference value, and the live normalised number.
        """
        readouts = VGroup()
        for label, value, color in rows:
            # A live control is a status readout, not derivation notation.
            # Text stays crisp over the dense field raster at video scale;
            # the TeX version was too low-contrast to fulfil that job.
            lhs = Text(label + " =", font=FONT, weight=FONT_WEIGHT_LABEL,
                       color=FG).scale(0.38)
            number = Text(f"{value():.2f}", font=FONT, weight=FONT_WEIGHT_LABEL,
                          color=color).scale(0.40)
            number.add_updater(lambda m, value=value, color=color: m.become(
                Text(f"{value():.2f}", font=FONT, weight=FONT_WEIGHT_LABEL,
                     color=color).scale(0.40).move_to(m.get_center())
                .set_z_index(601)))
            readouts.add(VGroup(lhs, number).arrange(RIGHT, buff=0.16))
        readouts.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        card = text_card(readouts, pad_x=0.34, pad_y=0.26)
        # `text_card` puts the dark backing and the label in one group.
        # Explicitly separate their z-levels: otherwise the card's nearly
        # opaque fill can sit over MathTex after DecimalNumber rebuilds it.
        card[0].set_z_index(600)
        card[1].set_z_index(601, family=True)
        return card

    def circular_motion(self, image, source, sign_key):
        """Show ten seconds of the signed E_t field for circular motion."""
        image.clear_updaters()
        source.clear_updaters()
        self.play(FadeOut(image), FadeOut(source), run_time=0.8)
        self.remove(image, source)

        c_field, radius, omega = 5.0, 1.35, TAU / 5.0
        clock = ValueTracker(0.0)

        def position(now):
            phase = omega * now
            return radius * np.array([math.cos(phase), math.sin(phase), 0.0])

        field = always_redraw(lambda: self.circular_plot(
            clock.get_value(), radius, omega, c_field))
        charge = always_redraw(lambda: charge_dot(
            position(clock.get_value()), radius=0.16,
            glow=False).set_z_index(110))

        self.add(field, charge)
        self.play(FadeIn(sign_key), run_time=0.7)
        self.play(clock.animate.set_value(10.0), run_time=10.0, rate_func=linear)
        self.wait(1.0)

    def circular_plot(self, T, radius, omega, c_field):
        """Nonrelativistic transverse radiation from a circular source."""
        PX, PY = self.X, self.Y

        # Only pixels reached by radiation emitted after the circular motion
        # begins are shown. On that domain the retarded-time root is unique.
        distance_at_start = np.hypot(PX - radius, PY)
        reached = distance_at_start <= c_field * T
        lo = np.zeros(PX.shape)
        hi = np.full(PX.shape, T)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            phase = omega * mid
            sx, sy = radius * np.cos(phase), radius * np.sin(phase)
            distance = np.hypot(PX - sx, PY - sy)
            residual = distance - c_field * (T - mid)
            hi = np.where(residual > 0, mid, hi)
            lo = np.where(residual > 0, lo, mid)
        tau = 0.5 * (lo + hi)

        phase = omega * tau
        sx, sy = radius * np.cos(phase), radius * np.sin(phase)
        dx, dy = PX - sx, PY - sy
        distance = np.maximum(np.hypot(dx, dy), 0.35)
        nx, ny = dx / distance, dy / distance
        ax = -radius * omega ** 2 * np.cos(phase)
        ay = -radius * omega ** 2 * np.sin(phase)

        # Signed projection on the local in-plane transverse direction.
        val = np.where(reached, -(nx * ay - ny * ax) / distance, 0.0)
        v = np.tanh(val / self.VMAX) * 0.86

        def hexrgb(hx):
            hx = hx.lstrip("#")
            return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], float)

        bg, pos, neg = hexrgb("#12131A"), hexrgb("#FF6B4A"), hexrgb("#4AC5FF")
        mag = np.abs(v)[..., None]
        hue = np.where((v > 0)[..., None], pos[None, None, :], neg[None, None, :])
        rgb = bg[None, None, :] + mag * (hue - bg[None, None, :])
        rgba = np.dstack([rgb, np.full(distance.shape, 255.0)]).astype(np.uint8)

        image = ImageMobject(rgba)
        image.stretch_to_fit_width(config.frame_width)
        image.stretch_to_fit_height(config.frame_height)
        image.set_z_index(-900)
        return image

    def shake(self, t):
        if self.mode == "kick":
            return np.array([kick_position(t, 0.5, 0.6), 0.0, 0.0])
        return np.array([self.amp.get_value()
                         * math.sin(self.w.get_value() * t), 0.0, 0.0])

    def accel(self, tr):
        """Acceleration at retarded time, vectorised over the grid."""
        if self.mode == "kick":
            dur, sp = 0.5, 0.6
            a = np.where((tr > 0) & (tr < dur),
                         sp * 0.5 * (np.pi / dur) * np.sin(np.pi * tr / dur),
                         0.0)
            return a
        w = self.w.get_value()
        return -self.amp.get_value() * w ** 2 * np.sin(w * tr)

    def plot(self, t):
        tr = t - self.R / self.C
        val = np.where(tr > 0,
                       self.accel(tr) * self.SIN / np.maximum(self.R, 0.35),
                       0.0)
        # tanh, not a hard clip. E_t goes as 1/r, so a clip turns the
        # whole near field into two solid slabs of colour and the rings
        # only start where it happens to fall below the ceiling. A soft
        # compression keeps the sign and the ring structure readable at
        # every radius and never fully saturates.
        v = np.tanh(val / (self.VMAX * self.norm.get_value())) * 0.86

        def hexrgb(hx):
            hx = hx.lstrip("#")
            return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], float)

        bg, pos, neg = hexrgb("#12131A"), hexrgb("#FF6B4A"), hexrgb("#4AC5FF")
        mag = np.abs(v)[..., None] * self.dim.get_value()
        hue = np.where((v > 0)[..., None], pos[None, None, :], neg[None, None, :])
        rgb = bg[None, None, :] + mag * (hue - bg[None, None, :])
        rgba = np.dstack([rgb, np.full(self.R.shape, 255.0)]).astype(np.uint8)

        im = ImageMobject(rgba)
        im.stretch_to_fit_width(config.frame_width)
        im.stretch_to_fit_height(config.frame_height)
        im.set_z_index(-900)
        return im


def globe(r, color, width=2.4, opacity=1.0):
    """A sphere drawn as wireframe, in 2D.

    Deliberately NOT a `ThreeDScene` with translucent `Sphere`s. Cairo has
    no depth buffer — it sorts whole mobjects — so radial lines threading
    two nested transparent surfaces punch through the wrong side and the
    sorting flickers as anything moves. That is the one 3D case it handles
    worst and it is exactly this shot. A circle plus two latitude ellipses
    reads as a sphere, costs nothing, and cannot flicker.
    """
    g = VGroup(Circle(radius=r, color=color, stroke_width=width))
    for f in (0.34, 0.68):
        e = Ellipse(width=2 * r, height=2 * r * f, color=color,
                    stroke_width=width * 0.7)
        e.set_stroke(opacity=0.45 * opacity)
        g.add(e)
    g[0].set_stroke(opacity=opacity)
    return g


# ======================================================================
# B3 — 1/r, and why that is the whole story
# ======================================================================
class PowerLaws(BrandScene):
    """The turn of the video: 1/r versus 1/r^2.

    The counting argument is the argument, so the puncture points are
    computed at the exact intersection radius and the totals are put on
    screen — "every field line that gets through sphere 1 also goes
    through sphere 2" only lands if the two numbers can be read and
    compared.
    """

    N = 12

    def construct(self):
        # --- it does not get weaker, it gets spread ----------------------
        R1, R2 = 1.85, 3.5
        chg = charge_dot(ORIGIN, radius=0.16)
        rays = VGroup(*[
            Line(ORIGIN, 6.6 * np.array([math.cos(i * TAU / self.N),
                                         math.sin(i * TAU / self.N), 0.0]),
                 color=SECOND, stroke_width=2.6)
            for i in range(self.N)])
        self.play(FadeIn(chg), LaggedStart(*[Create(r) for r in rays],
                                           lag_ratio=0.05), run_time=2.0)
        self.wait(0.8)

        def punctures(r, color):
            return VGroup(*[
                Dot(r * np.array([math.cos(i * TAU / self.N),
                                  math.sin(i * TAU / self.N), 0.0]),
                    radius=0.075, color=color)
                for i in range(self.N)])

        g1, g2 = globe(R1, MUTED), globe(R2, MUTED)
        p1, p2 = punctures(R1, FG), punctures(R2, FG)

        self.play(Create(g1), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d) for d in p1], lag_ratio=0.05),
                  run_time=1.2)
        self.wait(0.6)
        self.play(Create(g2), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d) for d in p2], lag_ratio=0.05),
                  run_time=1.2)

        # the count is the whole argument, so it has to be readable
        # parked at 195 degrees: a gap between two rays, clear of the
        # spacing arrows on the right, and on screen for both radii —
        # directly under the outer globe it fell off the bottom edge.
        def count_at(r):
            a = 195 * DEGREES
            return MathTex(str(self.N), color=FG).scale(0.8).move_to(
                (r + 0.45) * np.array([math.cos(a), math.sin(a), 0.0]))

        n1, n2 = count_at(R1), count_at(R2)
        self.play(FadeIn(n1), FadeIn(n2), run_time=0.9)
        self.wait(2.4)

        # ...but spread further apart on the bigger surface
        def gap(r):
            a0, a1 = 0.0, TAU / self.N
            p, q = (r * np.array([math.cos(a), math.sin(a), 0.0])
                    for a in (a0, a1))
            return DoubleArrow(p, q, color=ACCENT, buff=0.06, stroke_width=3.5,
                               max_tip_length_to_length_ratio=0.22)

        gaps = VGroup(gap(R1), gap(R2))
        self.play(FadeIn(gaps), run_time=1.0)
        self.wait(2.6)
        # NOT FadeOut(gap(R1)): that builds a second pair of arrows and
        # fades those, leaving the ones actually on screen behind.
        self.play(FadeOut(VGroup(g1, g2, p1, p2, n1, n2, rays, chg, gaps)),
                  run_time=1.0)
        self.clear_all()

        # Treat the two fields in the same order as the narration: the
        # Coulomb field first, then the transverse field. Showing both
        # calculations side-by-side made the visual comparison quick, but
        # skipped the logic that turns each field into an energy statement.
        self.energy_case(
            r"E_r = \frac{Q}{4\pi\varepsilon_0 r^2}",
            r"E_r \propto \frac{1}{r^2}",
            r"u_{E,r} = \tfrac12\varepsilon_0 E_r^2 \propto \frac{1}{r^4}",
            r"\frac{dU_{E,r}}{dr} = u_{E,r}\,4\pi r^2 \propto \frac{1}{r^2}",
            SECOND, -2,
        )
        self.clear_all()
        self.energy_case(
            r"E_t = \frac{Q a\sin\theta}{4\pi\varepsilon_0 c^2 r}",
            r"E_t \propto \frac{1}{r}",
            r"u_{E,t} = \tfrac12\varepsilon_0 E_t^2 \propto \frac{1}{r^2}",
            r"\frac{dU_{E,t}}{dr} = u_{E,t}\,4\pi r^2 = \mathrm{constant}",
            ACCENT, 0,
        )

    def clear_all(self):
        doomed = [m for m in self.mobjects if not isinstance(m, ImageMobject)]
        for m in doomed:
            m.clear_updaters()
        self.remove(*doomed)

    def energy_case(self, field_exact, field_power, density, shell_energy,
                    color, net_power):
        """Follow one field all the way to an energy quantity.

        `u_E 4 pi r^2` is *not power*: u_E is energy per volume, so after
        multiplying by a sphere's area the unit is energy per radial metre.
        Equivalently, it is the electric energy in a shell of unit
        thickness. The magnetic partner has intentionally not been invoked
        yet, so calling this a Poynting flux or power would overclaim.
        """
        rt = ValueTracker(1.0)
        exact = MathTex(field_exact, color=color).scale(0.86) \
            .move_to(UP * 2.62)
        field = MathTex(field_power, color=color).scale(0.92).move_to(exact)
        density_eq = MathTex(density, color=color).scale(0.78) \
            .move_to(UP * 1.35)
        quantity = VGroup(
            Text("energy / shell thickness", font=FONT,
                 weight=FONT_WEIGHT_LABEL, color=FG).scale(0.42),
            MathTex(r"[\mathrm{J/m}]", color=MUTED).scale(0.56),
        ).arrange(RIGHT, buff=0.25).move_to(UP * 0.42)
        shell_eq = MathTex(shell_energy, color=color).scale(0.76) \
            .move_to(DOWN * 0.25)

        def shell():
            relative = rt.get_value() ** net_power
            opacity = max(0.04, min(relative, 1.0))
            return globe(0.32 * rt.get_value(), color, width=2.5,
                         opacity=opacity).move_to(DOWN * 2.28)

        ring = always_redraw(shell)
        source = Dot(DOWN * 2.28, radius=0.075, color=FG)

        self.play(Write(exact), run_time=1.4)
        self.wait(0.8)
        self.play(TransformMatchingTex(exact, field), run_time=1.5)
        self.wait(0.8)
        self.play(TransformFromCopy(field, density_eq), run_time=1.5)
        self.wait(1.0)
        self.play(FadeIn(quantity), run_time=0.8)
        self.play(TransformFromCopy(density_eq, shell_eq), run_time=1.6)
        self.wait(1.0)
        self.play(field.animate.set_opacity(0.38),
                  density_eq.animate.set_opacity(0.48), run_time=0.8)
        self.play(FadeIn(ring), FadeIn(source), run_time=1.0)
        self.wait(1.4)
        self.play(rt.animate.set_value(4.4), run_time=8.0, rate_func=linear)
        self.wait(3.4)


# ======================================================================
# B4 — the identification
# ======================================================================
class Identification(BrandScene):
    """Four properties, the name, the receipt, and the one thing missing.

    **Script note.** The first property is the one that does not
    discriminate — every field line travels at c, as the script says
    itself ("as all field lines"). Calling it *the speed of light*
    immediately before concluding "this must be light" is where the
    circularity shows, so it is rendered here as the speed limit of
    information, which §1 also gives us and which costs nothing.
    """

    def construct(self):
        # All four in the same MUTED register — none of them is the
        # "hot" thing yet, they are evidence being assembled. Colour is
        # spent once, on the word they add up to.
        rows = [
            r"\text{at } c",
            r"\text{transverse}",
            r"\text{a wave}",
            r"\text{no loss with distance}",
        ]
        items = VGroup(*[MathTex(t, color=MUTED).scale(0.72) for t in rows])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.62).move_to(LEFT * 0.6 + UP * 0.4)

        for i, m in enumerate(items):
            self.play(FadeIn(m, shift=RIGHT * 0.25), run_time=1.0)
            # "most importantly" — the fourth gets twice the hold
            self.wait(2.2 if i == 3 else 1.1)
        self.wait(1.6)

        # --- combine into the name ------------------------------------------
        # Force one continuous geometric transform. MatchingShapes is free
        # to fade unmatched glyphs, which made the intended morph read as a
        # cross-fade for these two very different pieces of typography.
        word = Text("light", font=FONT, weight=FONT_WEIGHT_LABEL,
                    color=FG).scale(1.5)
        word.move_to(items.get_center())
        self.play(ReplacementTransform(items, word), run_time=1.8)
        self.wait(2.6)

        # --- the receipt: three rules, and Maxwell never used --------------
        self.play(FadeOut(word), run_time=1.0)
        block = maxwell_block(scale=0.62).move_to(ORIGIN)
        self.play(FadeIn(block), run_time=1.2)
        self.wait(1.0)
        # MUTED, not ACCENT — there is nothing here worth making hot, the
        # point is simply that the block was never opened.
        strike = Line(block.get_left() + LEFT * 0.25,
                      block.get_right() + RIGHT * 0.25,
                      color=MUTED, stroke_width=6)
        self.play(Create(strike), run_time=1.2)
        self.wait(2.4)
        self.play(FadeOut(block), FadeOut(strike), run_time=1.0)

        # --- except: the half we never paid for ---------------------------
        # E_t orange, B_t blue: the same two colours the whole video has
        # used for "the thing just derived" and "the field lines", so the
        # viewer reads which half was earned without being told.
        origin = LEFT * 1.2
        e_arr = Arrow(origin, origin + UP * 2.1, color=ACCENT, buff=0,
                      stroke_width=6, max_tip_length_to_length_ratio=0.18)
        e_lab = MathTex("E_t", color=ACCENT).scale(0.9).next_to(e_arr, LEFT, buff=0.25)
        b_arr = DashedVMobject(
            Arrow(origin, origin + RIGHT * 2.1, color=SECOND, buff=0,
                  stroke_width=6, max_tip_length_to_length_ratio=0.18),
            num_dashes=12)
        b_lab = MathTex("B_t", color=SECOND).scale(0.9).next_to(b_arr, DOWN, buff=0.3)
        sq = RightAngle(Line(origin, origin + UP), Line(origin, origin + RIGHT),
                        length=0.34, color=MUTED, stroke_width=2)

        self.play(GrowArrow(e_arr), FadeIn(e_lab), run_time=1.2)
        self.wait(1.4)
        self.play(Create(b_arr), FadeIn(b_lab), FadeIn(sq), run_time=1.4)
        self.wait(3.2)
