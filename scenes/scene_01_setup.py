"""Section 1 — Setup, rules. Built directly against script.md §1.

Five independent scenes, one per movement of the script. They are separate
`Scene` classes on purpose: each renders on its own, so a beat can be
re-cut without re-rendering six minutes of field lines, and each frame
gets composed for itself instead of inheriting whatever the previous beat
left lying around.

    Rule1Coulomb        P1  Coulomb -> the field lines
    Rule2Motion         P2  constant velocity
    Rule3SpeedLimit     P3  news at c, the wiggle, the question
    ThreeRegions        P4a the controlled experiment
    Mechanism           P4b where the transverse component comes from

House style:

* **Text only where the eye needs an anchor.** Formulas yes — they are
  content. Captions no: the voiceover already says it.
* **One thing hot at a time.** ACCENT is locked to the kink from
  `ThreeRegions` on; anything that just needs *focus* uses FG brightness
  against a dimmed field instead, so the kink never has to share.
* **Show one thing, then take it away.** Highlights are removed before the
  next one arrives rather than accumulating into a diagram.
* **Field lines start at the charge**, pass under it, and run past the
  frame edge so the viewport clips them. The charge is added last and its
  BG-filled disc covers the convergence point.

**How relativistic is this?** The controlled experiment — the event §2
actually measures — runs at v/c = 0.097, i.e. gamma - 1 = 0.5%. The two
schematic beats run hotter because their effects scale with v/c and vanish
at wide-shot speed: the wiggle at 0.23 (gamma - 1 = 2.8%) and the
mechanism at 0.22, its steepest knob at 0.30 (4.7%). Every field line is
drawn with `retarded_point`, the exact Lienard-Wiechert construction, so
line *shapes* are right at any speed. What is not modelled is the
relativistic change in line *density* (the Heaviside pancake) — and §1
never argues from density, so nothing here leans on it.
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
    chevron_marks,
    field_samples,
    force_arrow_length,
    grid_arrow,
    kick_position,
    kick_state,
    kick_velocity,
    probe_dot,
    retarded_field,
    retarded_point,
    starburst,
)

# 16 lines, not 20: at 20 they merge into a solid disc near the charge and
# the frame reads as texture rather than as a field.
FIELD_N = 16
FIELD_STEP = TAU / FIELD_N
# Lines start AT the charge and run under it. Holding them off at a radius
# leaves a visible hole with the charge floating in it — the field has to
# look like it belongs to the charge, not like it is orbiting it. The
# BG-filled disc is drawn last and covers the convergence point.
FIELD_R0 = 0.0
FIELD_LEN = 17.0

SIG_ANGLE = 2 * FIELD_STEP
SIG_DIR = np.array([math.cos(SIG_ANGLE), math.sin(SIG_ANGLE), 0.0])


def field_at(center, color=SECOND, width=3.0, length=FIELD_LEN):
    return starburst(center, length, color, n_lines=FIELD_N,
                     stroke_width=width, r0=FIELD_R0)


def trapezoid(t, ramp=0.22):
    """Constant speed through the middle of a leg, eased at the ends.

    `smooth` is never at constant velocity for a single frame, which is the
    one thing P2 is claiming. This holds exact constant speed across the
    middle 56% of every leg and confines the accelerations to short bursts
    at the turns.
    """
    a = ramp
    if t <= a:
        u = t / a
        s = a * (u ** 3 - 0.5 * u ** 4)
    elif t <= 1 - a:
        s = 0.5 * a + (t - a)
    else:
        u = (1 - t) / a
        s = 0.5 * a + (1 - 2 * a) + (0.5 * a - a * (u ** 3 - 0.5 * u ** 4))
    return s / (1.0 - a)


# ======================================================================
# P1 — "a charge Q at rest" -> "this is our first rule"
# ======================================================================
class Rule1Coulomb(BrandScene):
    def construct(self):
        Q = charge_dot(ORIGIN, radius=0.18)
        # Every arrow-to-line transform is created after Q. Keep the source
        # explicitly above those temporary transform mobjects as well as the
        # finished field, so the lines never sweep across the charge.
        Q.set_z_index(100, family=True)
        self.add(Q)
        Q_lab = Text("Q", font=FONT, weight=FONT_WEIGHT_LABEL,
                     color=FG).scale(0.5).next_to(Q, DOWN, buff=0.3)
        self.play(FadeIn(Q_lab), run_time=0.7)
        self.wait(1.2)

        # Split into parts so the q can later be taken out of it on its
        # own — `TransformMatchingTex` matches the pieces that survive and
        # only animates the ones that do not.
        coulomb = MathTex(r"\vec{F}_C", "=", r"\frac{1}{4\pi\varepsilon_0}",
                          r"\frac{Qq}{r^2}", r"\,\hat r",
                          color=MUTED).scale(0.62).to_corner(UL, buff=0.55)
        self.play(Write(coulomb), run_time=1.6)
        self.wait(0.8)

        # --- the test charge, and the arrow on it ----------------------
        # q and the arrow read the SAME two trackers rather than the arrow
        # reading q.get_center(): two updaters can run a frame apart and
        # the arrow visibly lags the dot during the sweep.
        r_tr, a_tr = ValueTracker(3.3), ValueTracker(0.0)

        def q_pos():
            r, a = r_tr.get_value(), a_tr.get_value()
            return r * np.array([math.cos(a), math.sin(a), 0.0])

        q = charge_dot(q_pos(), radius=0.12, glow=False)
        q.add_updater(lambda m: m.move_to(q_pos()))
        q_lab = Text("q", font=FONT, weight=FONT_WEIGHT_LABEL,
                     color=FG).scale(0.45)
        q_lab.add_updater(lambda m: m.next_to(q, DOWN, buff=0.22))
        self.play(FadeIn(q, scale=0.6), FadeIn(q_lab), run_time=1.0)
        self.wait(0.6)

        def arrow():
            r, a = r_tr.get_value(), a_tr.get_value()
            n = np.array([math.cos(a), math.sin(a), 0.0])
            base = q_pos() + n * 0.18
            return Arrow(base, base + n * force_arrow_length(r), color=SECOND,
                         buff=0, stroke_width=6,
                         max_tip_length_to_length_ratio=0.26)

        seed = arrow()
        self.play(GrowArrow(seed), run_time=1.0)
        live = always_redraw(arrow)
        self.remove(seed)
        self.add(live)

        # the arrow gets named, so the symbol in the corner and the thing
        # on screen are visibly the same quantity
        f_lab = always_redraw(
            lambda: MathTex(r"\vec{F}_C", color=SECOND).scale(0.62)
            .next_to(live, UP, buff=0.14))
        self.play(FadeIn(f_lab), run_time=0.7)
        self.wait(1.2)

        # --- 1/r^2: the one place in the video where LENGTH is magnitude
        self.play(r_tr.animate.set_value(1.4), run_time=1.7)
        self.wait(0.7)
        self.play(r_tr.animate.set_value(5.0), run_time=1.7)
        self.wait(0.7)
        self.play(r_tr.animate.set_value(2.9), run_time=1.1)

        # --- direction: along the line joining the centres --------------
        def joiner():
            a = a_tr.get_value()
            n = np.array([math.cos(a), math.sin(a), 0.0])
            return Line(-n * 1.5, n * (r_tr.get_value() + 1.5),
                        color=MUTED, stroke_width=2)

        seed2 = joiner()
        self.play(Create(seed2), run_time=0.9)
        line = always_redraw(joiner)
        self.remove(seed2)
        self.add(line)
        self.play(a_tr.animate.set_value(80 * DEGREES), run_time=2.6,
                  rate_func=smooth)
        self.play(a_tr.animate.set_value(28 * DEGREES), run_time=1.8,
                  rate_func=smooth)
        self.wait(0.5)
        self.play(FadeOut(line), FadeOut(q_lab), FadeOut(Q_lab),
                  FadeOut(f_lab), run_time=0.7)
        self.remove(line, q_lab, f_lab)

        # --- the grid ---------------------------------------------------
        # Arrow LENGTH is clamped from here on and magnitude rides on
        # opacity + stroke weight. A faithful 1/r^2 grid spans three orders
        # of magnitude across one frame: the near arrows leave the screen
        # and the far ones are invisible.
        nodes = []
        for row in range(7):
            y = (row - 3) * 1.06
            off = 0.56 if row % 2 else 0.0
            for col in range(13):
                x = (col - 6) * 1.12 + off
                p = np.array([x, y, 0.0])
                r = float(np.linalg.norm(p))
                if r >= 1.05:
                    nodes.append((p, r))
        nodes.sort(key=lambda nr: float(np.linalg.norm(nr[0] - q_pos())))

        probes, arrows = VGroup(), VGroup()
        for p, r in nodes:
            probes.add(probe_dot(p))
            arrows.add(grid_arrow(p, p / r, (1.45 / r) ** 2))

        q.clear_updaters()
        live.clear_updaters()
        self.play(Transform(q, probes[0]), Transform(live, arrows[0]),
                  run_time=1.1)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(probes[i]), FadeIn(arrows[i]))
                                for i in range(1, len(nodes))],
                              lag_ratio=0.014),
                  run_time=2.4)
        self.wait(1.4)

        # --- take q away, and what is left is the field ------------------
        # The probes leaving and the definition arriving are ONE beat: the
        # field is what the force per unit charge was all along, so the
        # symbol lands exactly as the charges it was defined against go.
        # The law itself loses its q and becomes the field. Written as a
        # second line underneath it would just be another formula to read;
        # transformed in place, the viewer watches the q leave and the
        # thing that is left is what the arrows have been showing all
        # along. Dividing by q IS where E comes from, so it should happen
        # to the equation that is already on screen.
        e_law = MathTex(r"\vec{E}", "=", r"\frac{1}{4\pi\varepsilon_0}",
                        r"\frac{Q}{r^2}", r"\,\hat r",
                        color=MUTED).scale(0.62).to_corner(UL, buff=0.55)
        self.remove(q, live)
        self.add(probes[0], arrows[0])
        self.play(FadeOut(probes), run_time=1.2)
        self.play(TransformMatchingTex(coulomb, e_law), run_time=2.0)
        self.wait(1.6)

        # --- connect them into lines ------------------------------------
        # Heads go first, then every shaft is absorbed into the line it
        # belongs to. A fade-out/fade-in would say "different object"; the
        # viewer has to watch one thing BECOME the other.
        tips = VGroup(*[t for a in arrows for t in a.pop_tips()])
        self.add(tips)
        self.play(FadeOut(tips), run_time=0.7)

        lines = field_at(ORIGIN)
        buckets = [[] for _ in range(FIELD_N)]
        for (p, r), shaft in zip(nodes, arrows):
            k = int(round((math.atan2(p[1], p[0]) % TAU) / FIELD_STEP)) % FIELD_N
            buckets[k].append(shaft)
        merges = [Transform(VGroup(*b), lines[i].copy())
                  for i, b in enumerate(buckets) if b]
        self.play(LaggedStart(*merges, lag_ratio=0.04), run_time=2.6)
        self.remove(*arrows, *[m.mobject for m in merges])
        self.add(lines, Q)
        self.wait(2.4)


# ======================================================================
# P2 — constant velocity
# ======================================================================
class Rule2Motion(BrandScene):
    """One uninterrupted constant-velocity pass.

    The field stays radial and travels rigidly with the charge. A previous
    version repeated the pass and then traced a box; both additions made the
    same claim again and introduced direction changes into a beat whose only
    subject is constant velocity.
    """

    def construct(self):
        x = ValueTracker(-5.6)

        def here():
            return np.array([x.get_value(), 0.0, 0.0])

        field = field_at(ORIGIN)
        field.add_updater(lambda m: m.move_to(here()))
        Q = charge_dot(here(), radius=0.18)
        Q.add_updater(lambda m: m.move_to(here()))
        self.add(field, Q)
        # Linear rate from the first rendered frame to the last: no turn,
        # no second example, and no acceleration cue competing with rule 2.
        self.play(x.animate.set_value(5.6), run_time=24.0, rate_func=linear)
        self.wait(1.2)


# ======================================================================
# P3 — the speed limit, the wiggle, and the question
# ======================================================================
class Rule3SpeedLimit(BrandScene):
    """One continuous retarded-time simulation: the field grows outward at
    c, the charge wiggles, and the beat ends on the question rather than
    on an answer — explaining it is the controlled experiment's job.

    The charge makes two smooth oscillations, then two circular turns, and
    returns to rest at the centre. The orbit itself is never drawn: the
    moving charge and its field carry the motion. The final dashed radial
    reference is the only comparison: it
    makes the transverse departure visible without opening a second visual
    construction before the controlled experiment explains it.
    """

    C = 5.0
    R_MAX = 5.0
    SCALE = 1.8
    DUR = 7.0
    PERIODS = 2
    VMAX = 1.15
    CIRC_DUR = 8.0
    CIRC_PERIODS = 2
    CIRC_RADIUS = 0.55
    CIRC_RAMP = 0.10

    def construct(self):
        C, R_MAX, SC = self.C, self.R_MAX, self.SCALE
        w = TAU * self.PERIODS / self.DUR
        A = self.VMAX / w
        T_OUT = R_MAX / C
        T_SIN = T_OUT + self.DUR
        T_CIRC = T_SIN + self.CIRC_DUR

        def ease(u):
            return 3 * u * u - 2 * u * u * u

        def env(u, ramp=1.4):
            return (ease(min(max(u, 0.0) / ramp, 1.0))
                    * ease(min(max(self.DUR - u, 0.0) / ramp, 1.0)))

        def circular_phase(x, ramp=self.CIRC_RAMP):
            """Mostly uniform phase with smooth, zero-speed endpoints."""
            x = min(max(x, 0.0), 1.0)
            if x < ramp:
                u = x / ramp
                area = ramp * (u ** 3 - 0.5 * u ** 4)
            elif x > 1.0 - ramp:
                u = (1.0 - x) / ramp
                area = (ramp * 0.5 + (1.0 - 2.0 * ramp)
                        + ramp * 0.5
                        - ramp * (u ** 3 - 0.5 * u ** 4))
            else:
                area = ramp * 0.5 + (x - ramp)
            return area / (1.0 - ramp)

        def pos(t):
            if t <= T_OUT:
                return np.zeros(3)
            if t <= T_SIN:
                u = t - T_OUT
                return np.array([A * math.sin(w * u) * env(u), 0.0, 0.0])
            if t <= T_CIRC:
                u = (t - T_SIN) / self.CIRC_DUR
                phase = TAU * self.CIRC_PERIODS * circular_phase(u)
                return self.CIRC_RADIUS * np.array(
                    [1.0 - math.cos(phase), math.sin(phase), 0.0])
            return np.zeros(3)

        def vel(t, h=1e-5):
            return (pos(t + h) - pos(t - h)) / (2 * h)

        def state(tau):
            return pos(tau), vel(tau)

        T = ValueTracker(0.0)

        def build():
            t = T.get_value()
            R = min(C * t, R_MAX)
            if R <= 1e-3:
                return VGroup()
            return retarded_field(state, t, C,
                                  field_samples(0.0, R, ()), SC, -1.0, -1.0,
                                  n_lines=FIELD_N, stroke_width=3.0)

        live = always_redraw(build)
        Q = always_redraw(lambda: charge_dot(SC * pos(T.get_value()), radius=0.18))
        front = always_redraw(
            lambda: Circle(radius=SC * min(C * T.get_value(), R_MAX),
                           color=ACCENT, stroke_width=3.5)
            .set_stroke(opacity=0.9))

        # --- the field goes out at c ------------------------------------
        self.add(live, Q, front)
        self.play(T.animate.set_value(T_OUT), run_time=5.5, rate_func=linear)
        self.remove(front)
        self.wait(1.5)

        # --- one readable disturbance history --------------------------
        self.play(T.animate.set_value(T_SIN), run_time=18.0,
                  rate_func=linear)

        # --- the same field construction under circular motion ---------
        # No path guide: its changing direction is already legible from
        # the moving charge and from the history carried by the field.
        self.play(T.animate.set_value(T_CIRC), run_time=10.0,
                  rate_func=linear)

        # --- hold on the one question this beat creates -----------------
        still = build()
        still_Q = charge_dot(SC * pos(T_CIRC), radius=0.18)
        still_Q.set_z_index(10)   # the dashed reference is added after
                                  # it and would otherwise draw over it
        self.remove(live, Q)
        self.add(still, still_Q)
        self.wait(1.2)

        # what rule 1 on its own would have drawn, laid over what is
        # actually there. Dashed and thin so it reads as the reference
        # rather than as more field — the question is the gap between the
        # two, and the gap is only visible if both are.
        ref = VGroup(*[DashedVMobject(l, num_dashes=40)
                       for l in field_at(SC * pos(T_CIRC), color=MUTED,
                                         width=1.8)])
        self.play(FadeIn(ref), run_time=1.2)
        self.wait(8.0)


# ======================================================================
# P4a — the controlled experiment and the three regions
# ======================================================================
class ThreeRegions(BrandScene):
    """Rest -> accelerate for Delta t -> constant velocity, then take the
    frame apart one claim at a time.

    Highlights are strictly sequential: each appears, is held, and is
    removed before the next arrives, so the frame never accumulates into a
    diagram. Focus is carried by FG brightness against a dimmed field —
    ACCENT stays locked to the kink and never has to share.

    These are the numbers §2 measures: v/c = 0.097, shell 1.0 units thick
    at radius 3.6, no fold-back anywhere drawn.
    """

    C = 9.0
    V = 0.871
    SCALE = 0.75
    R_MAX = 20.0
    DT = (1.0 / 0.75) / 9.0
    T0 = 20.0 / 9.0 + 0.18
    T_SHOT = T0 + (3.6 / 0.75) / 9.0

    def construct(self):
        C, V, SC, DT, T0 = self.C, self.V, self.SCALE, self.DT, self.T0
        T = ValueTracker(T0)

        def st(tau):
            return kick_state(tau - T0, DT, V)

        def cpos(t):
            return np.array([kick_position(t - T0, DT, V), 0.0, 0.0])

        def build():
            t = T.get_value()
            return retarded_field(st, t, C,
                                  field_samples(0.0, self.R_MAX,
                                                (C * (t - T0 - DT), C * (t - T0))),
                                  SC, C * (t - T0 - DT), C * (t - T0),
                                  n_lines=FIELD_N, stroke_width=3.0)

        live = always_redraw(build)
        Q = always_redraw(lambda: charge_dot(SC * cpos(T.get_value()), radius=0.18))
        self.add(live, Q)
        self.wait(1.2)

        # the kick lasts 0.148 sim-seconds; played over 1.6 real ones it is
        # an event rather than a blink
        self.play(T.animate.set_value(T0 + DT), run_time=1.6, rate_func=smooth)
        self.play(T.animate.set_value(self.T_SHOT), run_time=3.6,
                  rate_func=linear)
        self.wait(1.2)

        # --- freeze ------------------------------------------------------
        frozen = build()
        charge_at = SC * cpos(self.T_SHOT)
        Qs = charge_dot(charge_at, radius=0.18)
        self.remove(live, Q)
        self.add(frozen, Qs)

        s_new = C * (self.T_SHOT - T0 - DT)
        s_old = C * (self.T_SHOT - T0)

        rings = VGroup(
            DashedVMobject(Circle(radius=SC * s_new, color=MUTED, stroke_width=2)
                           .move_to(SC * cpos(T0 + DT)), num_dashes=44),
            DashedVMobject(Circle(radius=SC * s_old, color=MUTED, stroke_width=2)
                           .move_to(SC * cpos(T0)), num_dashes=60),
        )
        # FadeIn, never Create: a drawing animation would read as the
        # boundaries emanating from the charge now, when they have in fact
        # been travelling since the kick.
        self.play(frozen.animate.set_stroke(opacity=0.5), FadeIn(rings),
                  run_time=1.2)
        self.wait(1.0)

        gap = np.array([math.cos(1.5 * FIELD_STEP), math.sin(1.5 * FIELD_STEP), 0])
        mid_r = SC * (s_new + s_old) / 2

        def seg(lo, hi, color, width):
            a = retarded_point(st, self.T_SHOT, C,
                               self.T_SHOT - hi / SC / C, SIG_DIR, SC)
            b = retarded_point(st, self.T_SHOT, C,
                               self.T_SHOT - lo / SC / C, SIG_DIR, SC)
            return Line(a, b, color=color, stroke_width=width)

        def numeral(t, r, color):
            return Text(t, font=FONT, weight=FONT_WEIGHT_LABEL,
                        color=color).scale(0.66).move_to(gap * r)

        # --- one region at a time, each cleared before the next ----------
        R1 = (SC * s_old + 0.3, SC * s_old + 2.9)
        R3 = (0.5, SC * s_new - 0.3)

        n1, h1 = numeral("1", 5.9, FG), seg(*R1, FG, 5)
        self.play(FadeIn(n1), Create(h1), run_time=1.2)
        self.wait(2.4)
        self.play(FadeOut(n1), FadeOut(h1), run_time=0.7)

        n2 = numeral("2", mid_r, ACCENT)
        h2 = seg(SC * s_new, SC * s_old, ACCENT, 6.5)
        self.play(FadeIn(n2), Create(h2), run_time=1.2)
        self.wait(2.4)
        self.play(FadeOut(n2), FadeOut(h2), run_time=0.7)

        n3_dir = np.array([math.cos(0.5 * FIELD_STEP),
                           math.sin(0.5 * FIELD_STEP), 0.0])
        n3 = Text("3", font=FONT, weight=FONT_WEIGHT_LABEL,
                  color=FG).scale(0.66).move_to(n3_dir * 1.65)
        h3 = seg(*R3, FG, 5)
        self.play(FadeIn(n3), Create(h3), run_time=1.2)
        self.wait(2.4)
        self.play(FadeOut(n3), FadeOut(h3), run_time=0.7)

        # --- rule 2 in action: 1 and 3 are parallel ----------------------
        p1, p3 = seg(*R1, FG, 5), seg(*R3, FG, 5)
        chevs = VGroup(chevron_marks(p1.get_center(), SIG_DIR, color=FG),
                       chevron_marks(p3.get_center(), SIG_DIR, color=FG))
        self.play(Create(p1), Create(p3), run_time=1.2)
        self.play(FadeIn(chevs), run_time=0.8)
        self.wait(2.6)
        self.play(FadeOut(p1), FadeOut(p3), FadeOut(chevs), run_time=0.7)

        # --- and region 2 refuses to lie along either --------------------
        h2b = seg(SC * s_new, SC * s_old, ACCENT, 6.5)
        label = Text("transverse", font=FONT, weight=FONT_WEIGHT_LABEL,
                     color=ACCENT).scale(0.46)
        label.next_to(h2b.get_center(), RIGHT, buff=0.5)
        self.play(Create(h2b), run_time=1.2)
        self.wait(0.8)
        self.play(FadeIn(label), run_time=0.7)
        self.wait(3.0)


# ======================================================================
# P4b — where the transverse component comes from
# ======================================================================
class Mechanism(BrandScene):
    """The thesis of the section, and the one beat that has to BE the
    script's causal story rather than a picture of its result.

    Five elements leave the charge **radially** at five instants. Each then
    drifts at the velocity the charge had *at its own moment of release* —
    drawn as a velocity vector on every source, so "they inherit a
    different speed" is a thing on screen and not a claim. Different
    arrows, different drift, and the sources separate. The tops of their
    rays are the field line, and it tilts because its sources moved apart,
    not because anything ever curved.

    What falls out is the Purcell "Z": straight above (radial from where
    the charge was), straight below (radial from where it is now), and a
    tilted band joining them.

    Drawn broadside, where the effect is largest, at v/c = 0.22 — the
    schematic of the acceleration phase, since at the wide shot's 0.097
    the whole mechanism is a third of a pixel. `retarded_point` is the
    same call the real field lines use, so the geometry is exact at any
    speed. The ceiling is physics, not taste: the fold parameter works out
    to 2.36*beta, so past beta ~ 0.34 the construction folds back on
    itself; the steepest knob here sits at 0.30.
    """

    C = 6.0
    DT = 0.55
    DV = 1.32
    SC = 0.92
    BASE = np.array([-1.30, -3.00, 0.0])
    N = 5

    def construct(self):
        self.run_case(self.DV, self.DT, measure=True)
        self.clear_field()
        # more acceleration, same Delta t -> sources separate further, the
        # band leans harder, its thickness does not move
        self.run_case(self.DV * 1.35, self.DT, measure=False, hold=2.2)
        self.clear_field()
        # longer Delta t, same acceleration -> a thicker band
        self.run_case(self.DV, self.DT * 1.8, measure=False, hold=2.4)

    def clear_field(self):
        """Hard reset between knob settings.

        Everything in a case is `always_redraw`, so a FadeOut is undone by
        the mobject's own updater on the next frame; clearing the updaters
        first is the only thing that actually removes them.
        """
        doomed = [m for m in self.mobjects if not isinstance(m, ImageMobject)]
        for m in doomed:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in doomed], run_time=0.6)
        self.remove(*doomed)

    def run_case(self, dv, dt, measure, hold=2.6):
        C, SC, BASE, N = self.C, self.SC, self.BASE, self.N
        d = np.array([0.0, 1.0, 0.0])
        taus = np.linspace(0.0, dt, N)
        t_end = 2 * self.DT              # same clock for every knob setting
        VS = 0.78 / dv                   # velocity arrows: longest ~0.78
        T = ValueTracker(0.0)

        def st(tau):
            return kick_state(tau, dt, dv)

        def source(tau, t):
            """Where this element's origin has drifted to: it left from
            where the charge was, and has been moving ever since at the
            velocity the charge had at that instant."""
            p, v = st(tau)
            return BASE + SC * (p + v * (t - tau))

        def element(tau, t):
            return BASE + SC * retarded_point(st, t, C, tau, d)

        def born(t):
            return [tau for tau in taus if t >= tau]

        charge = always_redraw(
            lambda: charge_dot(BASE + SC * st(T.get_value())[0], radius=0.15))
        rays = always_redraw(lambda: VGroup(*[
            Line(source(tau, T.get_value()), element(tau, T.get_value()),
                 color=MUTED, stroke_width=1.8)
            for tau in born(T.get_value())]))
        srcs = always_redraw(lambda: VGroup(*[
            Dot(source(tau, T.get_value()), radius=0.06, color=MUTED)
            for tau in born(T.get_value())]))
        dots = always_redraw(lambda: VGroup(*[
            Dot(element(tau, T.get_value()), radius=0.07, color=ACCENT)
            for tau in born(T.get_value())]))

        # The velocity each element carries, drawn ON the element.
        #
        # Every one of these is racing outward at exactly c — that part
        # they share, and it is why the band keeps a constant thickness.
        # What they do NOT share is the sideways piece each inherited from
        # the charge at its own instant of release, and that is the only
        # component drawn here: horizontal, so perpendicular to the radial
        # direction the element is travelling along. It is the transverse
        # component, and watching five different ones sit on five points
        # of the same line is the entire mechanism.
        #
        # On the elements rather than on the sources because the band is
        # tilted: its points are three quarters of a unit apart
        # vertically, so the arrows separate themselves. Down at the
        # sources they overlapped into one smear.
        def drifts():
            t = T.get_value()
            g = VGroup()
            for tau in born(t):
                v = kick_velocity(tau, dt, dv)
                if v < 0.02:
                    continue
                base = element(tau, t)
                g.add(Arrow(base, base + RIGHT * VS * v, color=ACCENT, buff=0,
                            stroke_width=3.5,
                            max_tip_length_to_length_ratio=0.32))
            return g

        vecs = always_redraw(drifts)

        def line():
            t = T.get_value()
            g = VGroup()
            band = [element(tau, t) for tau in born(t)]
            if len(band) >= 2:
                c = VMobject(color=ACCENT, stroke_width=5)
                c.set_points_smoothly(band)
                g.add(c)
            if t > dt:      # below the band: straight, radial from Q now
                g.add(Line(BASE + SC * st(t)[0], element(dt, t),
                           color=SECOND, stroke_width=4))
            top = element(0.0, t)   # above it: straight, radial from the start
            g.add(Line(top, np.array([top[0], 4.7, 0.0]),
                       color=SECOND, stroke_width=4))
            return g

        curve = always_redraw(line)
        self.add(rays, srcs, curve, dots, vecs, charge)

        # the acceleration phase itself, slowed right down
        self.play(T.animate.set_value(dt), run_time=3.4, rate_func=linear)
        self.wait(0.6)
        # and then the difference in speed does its work
        self.play(T.animate.set_value(t_end), run_time=4.4, rate_func=linear)
        self.wait(hold)

        if measure:
            t = t_end
            lo, hi = element(taus[-1], t), element(taus[0], t)
            # Delta v * t is measured at the BAND, not down at the sources.
            # The two are the same length — the elements sit directly above
            # their own sources — but the bottom of the frame now belongs
            # to the velocity rows, and at the band it doubles as the
            # kink's own lean, which is where the eye already is.
            y_meas = lo[1] - 0.55
            shift = DoubleArrow(np.array([hi[0], y_meas, 0]),
                                np.array([lo[0], y_meas, 0]), color=ACCENT,
                                buff=0, stroke_width=3.5,
                                max_tip_length_to_length_ratio=0.25)
            shift_lab = MathTex(r"\Delta v\,t", color=ACCENT).scale(0.55) \
                .next_to(shift, RIGHT, buff=0.28)
            side = max(hi[0], lo[0]) + 1.7
            thick = DoubleArrow(np.array([side, lo[1], 0]),
                                np.array([side, hi[1], 0]), color=MUTED,
                                buff=0, stroke_width=3.5,
                                max_tip_length_to_length_ratio=0.16)
            thick_lab = MathTex(r"c\,\Delta t", color=MUTED).scale(0.55) \
                .next_to(thick, RIGHT, buff=0.25)
            self.play(FadeIn(shift), FadeIn(shift_lab), run_time=1.0)
            self.wait(1.2)
            self.play(FadeIn(thick), FadeIn(thick_lab), run_time=1.0)
            self.wait(2.8)
