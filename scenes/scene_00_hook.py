"""Section 0 — the hook. See storyboard_v2.md §0.

Four renders, ~36s of motion. Voiceover carries S1-S5; the picture carries
geometry. No captions anywhere — the only text on screen is the Maxwell
equations themselves.

**The one structural idea.** Under S3 the phenomenon is absorbed *into* the
equations; under S4/S5 the equations collapse *out of* themselves into the
charge. Same grammar, opposite direction — a loss and then a recovery.
They read as opposites only because they move differently: H3's swallow is
fast and accelerating inward, H4b's collapse slow and easing to rest.

**H1 is boxed, not drawn-and-gathered.** The equations write straight into
their final, larger positions — no spread-and-converge drag — and a
rectangle draws itself around the finished block. "Toolbox" stops being a
metaphor the voiceover has to sell and becomes a literal container the eye
can check off: four items, one box, done. The box travels with the
equations through the rest of the section (dims in H2, is absorbed in H3,
collapses with them in H4b), so it reads as part of the toolbox rather
than as a frame around it.

**H2b is schematic, not a preview.** It used to reuse the exact
retarded-field construction §1 Beat 3 builds properly and earns. Showing
that render here is wrong twice over: it spoils the one payoff the video
has, and it asks a viewer who has seen zero setup to read a precise
physics simulation as if it were self-explanatory. A charge takes a
visible shove and simple wavy lines radiate outward — a child's drawing
of "light comes out," not a claim about the mechanism. The real picture
is earned later; this is only the promise that one exists.

No strikethrough and no rule icons here; both are spent later. §1 opens on
H4b's final frame with no transition, so nothing is held after it.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from manim import *

from theme import BrandScene, FG, MUTED
from visuals import charge_dot, maxwell_block

PAIR_L, PAIR_R = LEFT * 1.15 + UP * 1.95, RIGHT * 1.15 + UP * 1.95


class Hook(BrandScene):
    def construct(self):
        box, cluster = self.h1_toolbox()
        shot = self.h2_e_and_b_then_flash(box, cluster)
        self.h3_absorbed(box, cluster, shot)
        self.h4b_collapse(box, cluster)

    # ------------------------------------------------------------------
    # H1 — the toolbox · S1 · render ~9s
    # ------------------------------------------------------------------
    def h1_toolbox(self):
        """Four equations write in at full size, then get boxed.

        No drag-together: writing straight into the final layout already
        states the arrangement, and the box is what performs "complete" —
        a set with a line drawn around it reads as finished in a way four
        equations sitting near each other doesn't, no matter how tightly
        they're packed.
        """
        cluster = maxwell_block(scale=0.92).move_to(ORIGIN)
        eqs = [cluster[0][0], cluster[0][1], cluster[1][0], cluster[1][1]]

        self.play(LaggedStart(*[Write(e) for e in eqs], lag_ratio=0.28),
                  run_time=4.8)
        self.wait(0.6)

        box = RoundedRectangle(
            corner_radius=0.18, width=cluster.width + 1.1,
            height=cluster.height + 0.85, stroke_color=MUTED,
            stroke_width=2.4, fill_opacity=0.0,
        ).move_to(cluster)
        self.play(Create(box), run_time=1.3)
        self.wait(1.5)
        return box, cluster

    # ------------------------------------------------------------------
    # H2 — what falls out of them · S2 · render ~11s (5 + 6)
    # ------------------------------------------------------------------
    def h2_e_and_b_then_flash(self, box, cluster):
        """(a) E and B lift out and are driven by each other. (b) Hard cut
        to a charge taking a shove and radiating schematically.

        (a) stays symbolic on purpose: the E-B relationship gets notation,
        the radiating charge gets a picture. The two examples in S2 are
        the two videos, and the visual language says which is which
        before the viewer knows there is a part 2.
        """
        # cluster[0][0] is the WHOLE first equation ("∇·E=ρ/ε0"); its
        # .get_center() is the bounding-box centroid, which sits over the
        # "=" sign and fraction, not over the E. maxwell_block splits that
        # equation into (∇·, \vec{E}, =ρ/ε0) for exactly this reason —
        # cluster[0][0][1] is the actual \vec{E} glyph, same for [1] on
        # cluster[1][0] ("∇·B=0"). The floating symbol has to launch from
        # where the real glyph is or the "lifts out of the block" read
        # breaks on the very first frame.
        e_sym = MathTex(r"\vec{E}", color=MUTED).scale(1.15)
        b_sym = MathTex(r"\vec{B}", color=MUTED).scale(1.15)
        e_sym.move_to(cluster[0][0][1].get_center())
        b_sym.move_to(cluster[1][0][1].get_center())
        self.add(e_sym, b_sym)

        tie = DoubleArrow(PAIR_L + RIGHT * 0.42, PAIR_R + LEFT * 0.42,
                          color=MUTED, buff=0, stroke_width=3,
                          max_tip_length_to_length_ratio=0.14)

        # NOT toolbox.animate.set_opacity(): set_opacity touches fill AND
        # stroke on every submobject, and the box was built with
        # fill_opacity=0 specifically to stay an outline. Forcing that to
        # 0.22 turns it into a visible grey slab. Dim the box's stroke and
        # the equations' fill+stroke separately so the box's fill stays at
        # zero throughout.
        self.play(
            cluster.animate.set_opacity(0.22),
            box.animate.set_stroke(opacity=0.22),
            e_sym.animate.move_to(PAIR_L),
            b_sym.animate.move_to(PAIR_R),
            run_time=1.3,
        )
        self.play(GrowFromCenter(tie), run_time=0.5)

        # --- the live coupling --------------------------------------------
        # phase runs 0 -> 2, i.e. two complete exchanges. The pulse sits on
        # E at integer phase and on B at the half-integers; each symbol is
        # largest exactly when the pulse is on it, so B peaks a quarter
        # cycle after E and the lag reads as one driving the other.
        phase = ValueTracker(0.0)
        e_base, b_base = e_sym.copy(), b_sym.copy()
        start, end = PAIR_L + RIGHT * 0.5, PAIR_R + LEFT * 0.5

        # theme colours are plain hex strings; interpolate_color wants
        # ManimColor objects, so convert once rather than per frame
        cool, hot = ManimColor(MUTED), ManimColor(FG)

        def breathe(mob, base, home, at_e):
            def _upd(m):
                p = phase.get_value()
                swing = (1 + math.cos(TAU * p)) / 2 if at_e else \
                        (1 - math.cos(TAU * p)) / 2
                m.become(base.copy().scale(1 + 0.34 * swing)
                         .set_color(interpolate_color(cool, hot, swing))
                         .move_to(home))
            mob.add_updater(_upd)

        breathe(e_sym, e_base, PAIR_L, True)
        breathe(b_sym, b_base, PAIR_R, False)

        pulse = always_redraw(lambda: Dot(
            interpolate(start, end, (1 - math.cos(TAU * phase.get_value())) / 2),
            radius=0.07, color=FG))

        self.add(pulse)
        self.play(phase.animate.set_value(2.0), run_time=3.2, rate_func=linear)
        e_sym.clear_updaters()
        b_sym.clear_updaters()
        self.remove(pulse)

        # --- (b) the schematic shove: hard cut, no fade --------------------
        self.remove(*[m for m in self.mobjects[1:]])   # hard cut

        charge = charge_dot(ORIGIN)
        self.add(charge)
        self.play(
            charge.animate.shift(RIGHT * 0.22),
            Flash(charge, color=FG, flash_radius=0.42, line_length=0.2,
                 num_lines=10, run_time=0.4),
            run_time=0.4, rate_func=rush_into,
        )
        self.play(charge.animate.shift(LEFT * 0.22), run_time=0.4,
                  rate_func=rush_from)

        rays = self.schematic_rays(ORIGIN)
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.07),
                  run_time=2.6)
        shot = Group(rays, charge)
        self.wait(2.2)
        return shot

    # ------------------------------------------------------------------
    @staticmethod
    def schematic_rays(center, n=10, length=2.9, amp=0.13, freq=2.0,
                       r0=0.32):
        """A child's-drawing radiation glyph — deliberately not physics.

        Radiation from a point kick goes outward in every direction, so the
        promise image must do that too. Ten rays leave enough breathing room
        for each wiggle to read independently, while still making the
        isotropic burst obvious. They start at `r0`, rather than on the
        charge, so the sketch never collapses into a knot at its source.
        """
        rays = VGroup()
        for i in range(n):
            a = TAU * i / n
            d = np.array([math.cos(a), math.sin(a), 0.0])
            perp = np.array([-math.sin(a), math.cos(a), 0.0])
            pts = []
            for k in range(48):
                t = k / 47
                ease = min(t / 0.25, 1.0)
                wig = amp * ease * math.sin(freq * TAU * t)
                pts.append(center + d * (r0 + t * length) + perp * wig)
            curve = VMobject(color=FG, stroke_width=3)
            curve.set_points_smoothly(pts)
            rays.add(curve)
        return rays

    # ------------------------------------------------------------------
    # H3 — back to the equations · S3 · render ~10s
    # ------------------------------------------------------------------
    def h3_absorbed(self, box, cluster, shot):
        """The phenomenon goes in; the toolbox remains, unchanged.

        The box does not move, does not brighten and does not react — it
        swallows the picture and looks exactly as it did before it ate it.
        That indifference is "the equations have become the argument for
        these phenomena", made literal.

        Then the space where the charge was is left empty and stays empty.
        The loss is the absence, so this hold is content rather than a
        pause around content, and it is the one hold in the section that
        belongs in the render.
        """
        # same trap as H2a's dim, in reverse: cluster gets full opacity,
        # the box gets its stroke back, and its fill is never touched so
        # it never becomes a filled rectangle.
        cluster.set_opacity(1.0)
        box.set_stroke(opacity=1.0)
        toolbox = VGroup(box, cluster)
        self.add(toolbox)
        # The shrink is anchored just above the ray tips, not at the
        # toolbox's own centre — the light is what is being swallowed, so
        # the vanishing point sits where the light already is rather than
        # pulling it down to the charge first.
        target = shot.get_top() + UP * 0.25
        self.play(
            FadeIn(toolbox, run_time=1.0),
            shot.animate.scale(0.015, about_point=target).set_opacity(0.0),
            run_time=1.6, rate_func=rush_into,
        )
        self.remove(shot)
        self.wait(8.4)

    # ------------------------------------------------------------------
    # H4b — the collapse · S4/S5 · render ~6s · confirmed build
    # ------------------------------------------------------------------
    def h4b_collapse(self, box, cluster):
        """Collapse, not dissolve. All of that, out of this one thing.

        One continuous motion, no cut: the whole toolbox — box and
        equations together — shrinks and fades while it converges on a
        point at frame centre that resolves into the charge. Slow, easing
        to rest, the opposite shape of motion to H3's swallow. §4 plays
        this exact render backwards.
        """
        # NOT set_opacity() on the charge. charge_dot's halo is twelve
        # rings at graded opacities, and set_opacity(1.0) sets every one
        # solid — a plain disc nearly 3x the right size with the halo
        # gone. Growing it from a point leaves those opacities alone, and
        # reads better anyway: the toolbox converges on a point and the
        # charge is what the point turns out to be.
        # target opacity 0.0 is the one case set_opacity on the group is
        # safe: the box's fill is already 0, so animating it toward 0
        # changes nothing, and its stroke plus the equations both fade to
        # nothing right on cue.
        charge = charge_dot(ORIGIN)
        toolbox = VGroup(box, cluster)
        self.play(
            toolbox.animate(rate_func=smooth).scale(0.02, about_point=ORIGIN)
            .set_opacity(0.0),
            GrowFromCenter(charge, rate_func=rush_into),
            run_time=6.0,
        )
        self.remove(toolbox)
        return charge
