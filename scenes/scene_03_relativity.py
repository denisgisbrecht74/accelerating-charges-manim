"""Section 3 — Special relativity. Built against script.md §3.

A declared detour, and short: three scenes, ~80s.

    Squeeze     the pattern contracts; stronger broadside, weaker fore/aft
    Chasing     shells only — the charge running down its own field
    Headlight   the donut leans forward and closes onto the axis

A live `v/c` readout runs through all three. It is the one piece of text
in the section and it is content rather than decoration: the entire point
is that a quantity which has been ~0.1 for eleven minutes is now allowed
to get big, and the viewer needs to see it move.

**The construction changes here, and it has to.** Everywhere else the
field lines come from `retarded_field`, which gets line *directions*
exactly right at any speed but samples emission directions uniformly —
so the line *densities* are wrong once beta is large. Density is this
entire section, so instead the lines are placed by the exact Lorentz
relation

    tan(theta_lab) = gamma * tan(theta_rest)

i.e. take directions uniform in the REST frame and map them, which is
just the statement that the field pattern is the rest-frame starburst
contracted by 1/gamma along the motion. Verified: at beta = 0.9 that puts
the angular gaps at 10.2 degrees broadside against 43.5 fore-and-aft,
where a naive uniform sampling would hold all of them at 22.5.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from manim import *

from theme import (ACCENT, BrandScene, FG, FONT, FONT_WEIGHT_LABEL,
                   MUTED, SECOND)
from visuals import charge_dot, signed_field_key

N_LINES = 20
LINE_LEN = 17.0


def gamma_of(b):
    return 1.0 / math.sqrt(max(1.0 - b * b, 1e-9))


def beta_readout(tracker, corner=UL):
    """The section's one label: v/c, live."""
    # Keep the equality sign visually separate from the number. At preview
    # resolution `v/c=` read like an unlabelled pair of glyphs; this is a
    # live value, so the intended reading is explicitly “v/c = number”.
    lab = MathTex(r"v/c\;=", color=MUTED).scale(0.7)
    num = DecimalNumber(0.0, num_decimal_places=2, color=MUTED).scale(0.7)
    num.add_updater(lambda m: m.set_value(tracker.get_value()))
    grp = VGroup(lab, num).arrange(RIGHT, buff=0.28).to_corner(corner, buff=0.6)
    return grp


def boosted_field(beta, color=SECOND, width=3.0, n=N_LINES, length=LINE_LEN):
    """The starburst of a uniformly moving charge, with correct spacing.

    Directions are uniform in the charge's REST frame and then contracted
    along the motion by 1/gamma, which is exactly `tan(theta_lab) = gamma
    tan(theta_rest)`. Nothing is created or destroyed as beta rises: the
    same n lines are redistributed, crowding toward broadside and opening
    up fore and aft.
    """
    g = gamma_of(beta)
    out = VGroup()
    for i in range(n):
        tp = i * TAU / n
        d = np.array([math.cos(tp) / g, math.sin(tp), 0.0])
        d /= np.linalg.norm(d)
        out.add(Line(ORIGIN, d * length, color=color, stroke_width=width))
    return out


# ======================================================================
# The squeeze
# ======================================================================
class Squeeze(BrandScene):
    """Camera locked to the charge; only the field changes.

    The two probe arrows carry the payoff, and their lengths are the exact
    Heaviside values at fixed radius: broadside goes as gamma, straight
    ahead as 1 / gamma^2. At beta = 0.9 that is 2.29x and 0.19x — which is
    worth stating precisely, because it is *not* symmetric and the eye
    will not guess it.
    """

    B_END = 0.9
    R_PROBE = 1.9

    def construct(self):
        b = ValueTracker(0.0)
        field = always_redraw(lambda: boosted_field(b.get_value()))
        chg = charge_dot(ORIGIN, radius=0.18)
        read = beta_readout(b)

        self.add(field, chg)
        self.play(FadeIn(read), run_time=0.8)
        self.wait(1.4)

        # Both probes FG, not SECOND: at the field's own colour the
        # forward one — which shrinks to 0.19 of its rest length — vanishes
        # into the lines it is sitting on. A dot at each base keeps it
        # findable once it is that short.
        def probe(direction):
            def make():
                g = gamma_of(b.get_value())
                # E(psi)/E_rest at fixed r: broadside gamma, forward 1/gamma^2
                scale = g if abs(direction[0]) < 0.5 else 1.0 / g ** 2
                base = direction * self.R_PROBE
                return VGroup(
                    Dot(base, radius=0.06, color=FG),
                    Arrow(base, base + direction * 0.85 * scale, color=FG,
                          buff=0, stroke_width=6,
                          max_tip_length_to_length_ratio=0.3))
            return always_redraw(make)

        broad = probe(np.array([0.0, 1.0, 0.0]))
        ahead = probe(np.array([1.0, 0.0, 0.0]))
        # (no dimming here: `field` is an always_redraw, so an animated
        # set_stroke on it is discarded on the next frame. The white
        # probes already separate from the blue lines without it.)
        self.play(FadeIn(broad), FadeIn(ahead), run_time=1.0)
        self.wait(1.2)

        # --- and now let it go relativistic ------------------------------
        self.play(b.animate.set_value(self.B_END), run_time=7.0,
                  rate_func=smooth)
        self.wait(3.4)


# ======================================================================
# Chasing its own field
# ======================================================================
class Chasing(BrandScene):
    """Field lines dropped entirely — shells only.

    Each shell is centred where the charge WAS when it left, so the
    bunching is geometry rather than emphasis: the gap ahead is
    (1 - beta) c dt and behind (1 + beta) c dt. At beta = 0.85 that is a
    factor of 12.3, which needs no exaggeration to read.
    """

    C = 2.2
    BETA = 0.85
    EVERY = 0.55
    T_END = 5.0
    X0 = -5.0

    def construct(self):
        v = self.BETA * self.C
        t = ValueTracker(0.0)
        emits = [k * self.EVERY for k in range(int(self.T_END / self.EVERY) + 1)]

        def x_of(tt):
            return self.X0 + v * tt

        def shells():
            now = t.get_value()
            g = VGroup()
            for te in emits:
                if te > now:
                    continue
                r = self.C * (now - te)
                if r < 0.02:
                    continue
                g.add(Circle(radius=r, color=SECOND, stroke_width=2.4)
                      .move_to(RIGHT * x_of(te))
                      .set_stroke(opacity=0.75))
            return g

        chg = always_redraw(
            lambda: charge_dot(RIGHT * x_of(t.get_value()), radius=0.17))
        read = beta_readout(ValueTracker(self.BETA))

        self.add(always_redraw(shells), chg)
        self.play(FadeIn(read), run_time=0.6)
        self.play(t.animate.set_value(self.T_END), run_time=9.0,
                  rate_func=linear)
        self.wait(3.0)


# ======================================================================
# The headlight
# ======================================================================
class Headlight(BrandScene):
    """The complete headlight scene, followed by circular motion.

    The charge first oscillates while its drift rises from rest to
    v/c = 0.60, making the radiation pattern lean forward. It then makes
    two circular turns in ten seconds. Both beats use the same signed field
    and neither contains a velocity arrow or orbit guide.
    """

    C = 2.4
    W = 4.0
    B = 0.6                     # final drift
    DB = 0.2                    # oscillation amplitude in beta
    T1, T2 = 4.0, 10.0          # hold at rest, then ramp the drift
    T_END = 16.0
    VMAX = 1.2
    RES = (320, 180)

    def initialize_grid(self):
        w, h = self.RES
        xs = np.linspace(-config.frame_width / 2, config.frame_width / 2, w)
        ys = np.linspace(config.frame_height / 2, -config.frame_height / 2, h)
        self.SX, self.SY = np.meshgrid(xs, ys)

    def construct(self):
        self.initialize_grid()
        self.A = self.DB * self.C / self.W

        t = ValueTracker(0.0)
        img = always_redraw(lambda: self.plot(t.get_value()))
        chg = always_redraw(lambda: charge_dot(
            RIGHT * self.A * math.sin(self.W * t.get_value()), radius=0.16))
        beta_now = ValueTracker(0.0)
        beta_now.add_updater(lambda m: m.set_value(self.beta0(t.get_value())))
        read = beta_readout(beta_now)
        sign_key = signed_field_key().to_corner(UR, buff=0.48)
        sign_key.set_z_index(600)
        self.add(img, chg, beta_now)
        self.play(FadeIn(read), FadeIn(sign_key), run_time=0.6)
        self.play(t.animate.set_value(self.T1), run_time=self.T1,
                  rate_func=linear)
        self.play(t.animate.set_value(self.T_END),
                  run_time=self.T_END - self.T1, rate_func=linear)
        self.wait(2.6)
        self.circular_motion(read, sign_key, duration=10.0,
                             previous_image=img, previous_charge=chg,
                             beta_tracker=beta_now)

    def circular_motion(self, read, sign_key, duration,
                        previous_image=None, previous_charge=None,
                        beta_tracker=None):
        """Circular relativistic motion, optionally after the drift beat."""
        if previous_image is not None:
            previous_image.clear_updaters()
            previous_charge.clear_updaters()
            beta_tracker.clear_updaters()
            beta_tracker.set_value(self.B)
            self.play(FadeOut(previous_image), FadeOut(previous_charge),
                      run_time=0.8)
            self.remove(previous_image, previous_charge)

        # beta = r*omega/c. With omega = 2pi/5 this preserves v/c = 0.60
        # and gives one complete turn every five seconds.
        omega = TAU / 5.0
        radius = self.B * self.C / omega
        clock = ValueTracker(0.0)

        def pos(now):
            phase = omega * now
            return radius * np.array([math.cos(phase), math.sin(phase), 0.0])

        first_image = self.circular_plot(0.0, radius, omega)
        first_charge = charge_dot(pos(0.0), radius=0.16).set_z_index(14)
        entrance = [FadeIn(first_image), FadeIn(first_charge)]
        if previous_image is None:
            entrance.extend([FadeIn(read), FadeIn(sign_key)])
        self.play(*entrance, run_time=0.9)

        live_image = always_redraw(
            lambda: self.circular_plot(clock.get_value(), radius, omega))
        live_charge = always_redraw(
            lambda: charge_dot(pos(clock.get_value()), radius=0.16)
            .set_z_index(14))
        self.remove(first_image, first_charge)
        self.add(live_image, live_charge)
        self.play(clock.animate.set_value(duration), run_time=duration,
                  rate_func=linear)
        self.wait(1.0)

    def circular_plot(self, T, radius, omega):
        """Signed in-plane radiation field for uniform circular motion."""
        PX, PY = self.SX, self.SY

        # Solve |P-X(tau)| = c(T-tau). Since |v| < c, the root is unique.
        lo = np.full(PX.shape, T - 30.0 / self.C)
        hi = np.full(PX.shape, T)
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            phase = omega * mid
            sx, sy = radius * np.cos(phase), radius * np.sin(phase)
            distance = np.hypot(PX - sx, PY - sy)
            residual = distance - self.C * (T - mid)
            hi = np.where(residual > 0, mid, hi)
            lo = np.where(residual > 0, lo, mid)
        tau = 0.5 * (lo + hi)

        phase = omega * tau
        sx, sy = radius * np.cos(phase), radius * np.sin(phase)
        dx, dy = PX - sx, PY - sy
        distance = np.maximum(np.hypot(dx, dy), 0.28)
        nx, ny = dx / distance, dy / distance

        bx = -radius * omega * np.sin(phase) / self.C
        by = radius * omega * np.cos(phase) / self.C
        ax = -radius * omega ** 2 * np.cos(phase)
        ay = -radius * omega ** 2 * np.sin(phase)
        kappa = np.maximum(1.0 - nx * bx - ny * by, 0.05)

        # Project n x ((n-beta) x a) onto the in-plane transverse basis.
        # The sign keeps the established orange/blue field convention.
        cross = (nx - bx) * ay - (ny - by) * ax
        val = -cross / (kappa ** 3 * distance)
        v = np.tanh(val / 2.5) * 0.86

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

    # ---- trajectory: drift ramp plus oscillation, all analytic --------
    def beta0(self, t):
        if t <= self.T1:
            return 0.0
        if t >= self.T2:
            return self.B
        return self.B * (t - self.T1) / (self.T2 - self.T1)

    def drift(self, t):
        """Integral of c*beta0 — the part the camera follows."""
        t = np.asarray(t, dtype=float)
        span = self.T2 - self.T1
        mid = self.C * self.B * np.clip(t - self.T1, 0, span) ** 2 / (2 * span)
        late = np.where(t > self.T2,
                        self.C * self.B * (t - self.T2), 0.0)
        return np.where(t <= self.T1, 0.0, mid + late)

    def x_of(self, t):
        return self.drift(t) + self.A * np.sin(self.W * t)

    def a_of(self, t):
        span = self.T2 - self.T1
        ramp = np.where((t > self.T1) & (t < self.T2),
                        self.C * self.B / span, 0.0)
        return ramp - self.A * self.W ** 2 * np.sin(self.W * t)

    def v_of(self, t):
        span = self.T2 - self.T1
        b0 = np.clip((t - self.T1) / span, 0.0, 1.0) * self.B
        return self.C * b0 + self.A * self.W * np.cos(self.W * t)

    # ------------------------------------------------------------------
    def plot(self, T):
        # camera rides the drift, so the charge oscillates about centre
        PX = self.SX + float(self.drift(T))
        PY = self.SY

        # retarded time by bisection. g(tau) = |P - X(tau)| - c(T - tau)
        # is strictly increasing (its slope is at least c - |v| > 0), so
        # bisection is unconditionally safe here.
        lo = np.full(PX.shape, T - 30.0 / self.C)
        hi = np.full(PX.shape, T)
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            R = np.hypot(PX - self.x_of(mid), PY)
            g = R - self.C * (T - mid)
            hi = np.where(g > 0, mid, hi)
            lo = np.where(g > 0, lo, mid)
        tau = 0.5 * (lo + hi)

        dx = PX - self.x_of(tau)
        R = np.maximum(np.hypot(dx, PY), 0.28)
        nx, ny = dx / R, PY / R
        kappa = np.maximum(1.0 - (self.v_of(tau) / self.C) * nx, 0.05)
        val = self.a_of(tau) * ny / (kappa ** 3 * R)
        v = np.tanh(val / self.VMAX) * 0.86

        def hexrgb(hx):
            hx = hx.lstrip("#")
            return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], float)

        bg, pos, neg = hexrgb("#12131A"), hexrgb("#FF6B4A"), hexrgb("#4AC5FF")
        mag = np.abs(v)[..., None]
        hue = np.where((v > 0)[..., None], pos[None, None, :], neg[None, None, :])
        rgb = bg[None, None, :] + mag * (hue - bg[None, None, :])
        rgba = np.dstack([rgb, np.full(R.shape, 255.0)]).astype(np.uint8)

        im = ImageMobject(rgba)
        im.stretch_to_fit_width(config.frame_width)
        im.stretch_to_fit_height(config.frame_height)
        im.set_z_index(-900)
        return im


class HeadlightCircle(Headlight):
    """Standalone extended circular beat for editorial use.

    This is deliberately separate from `Headlight`: three turns over
    fifteen seconds, with the same v/c readout and signed-field key.
    """

    def construct(self):
        self.initialize_grid()
        beta_now = ValueTracker(self.B)
        read = beta_readout(beta_now)
        sign_key = signed_field_key().to_corner(UR, buff=0.48)
        sign_key.set_z_index(600)
        self.add(beta_now)
        self.circular_motion(read, sign_key, duration=15.0)
