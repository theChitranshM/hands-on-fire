"""Showcase mode: one effect per scene, for recording each gesture on its own.

Nothing times out and nothing advances by itself. Inside a scene the gesture can be repeated as often as you like
(every namaste re-lights the lamps, every lift launches another rocket), so each one can be filmed in several takes.
F fires the scene's effect without the gesture. SPACE / B / number keys change scene. Every trigger is logged with
its time, so the clips are easy to find in the footage afterwards.
"""
import time
import numpy as np
from .sequence import Sequence, _ease
from .types import THUMB_TIP, INDEX_TIP, INDEX_MCP, WRIST

# (number, name, gesture, cue for the performer)
SCENES = [
    (0, "Showcase", None, "Pick a scene: 1-9, 0"),
    (1, "Puja", "joined", "Namaste: lamps light, rangoli, petals. Part hands to scatter, join again to repeat"),
    (2, "Diya", "open_palm", "Open palm lights it, close your hand snuffs it, open again relights"),
    (3, "Aarti", "circle", "Circle the diya slowly; a bell on every loop"),
    (4, "Phuljhadi", "pinch", "Pinch to light a sparkler; let go to put it out"),
    (5, "Chakri", "point", "Point your index finger"),
    (6, "Anar", "low_palm", "Push your open palm down"),
    (7, "Rocket", "raise", "Lift your hand up fast: rocket + firework, again and again"),
    (8, "Finale", "arms_wide", "Open your arms wide (or F)"),
    (9, "Lanterns", "raise", "Lift your open palm up: sky lanterns rise from it"),
    (10, "Quiet end", "snap", "Fireworks running: snap your fingers, they vanish, one diya stays. F fades to black"),
]
LAST = len(SCENES) - 1


class Showcase(Sequence):
    beats = SCENES

    def __init__(self, cfg, engine, fx, audio=None, recorder=None):
        super().__init__(cfg, engine, fx, audio, recorder)
        self.k = cfg.showcase
        self.events = []           # (seconds since start, scene, what, how) for finding clips later
        self.t0 = time.perf_counter()
        self._cool = 0.0
        self._state = {}
        if recorder:
            self.take_start = self.t0
            recorder.start_take()   # one continuous recording for the whole session

    # ---- navigation (no take logic: the whole session is one recording) ------------------
    @property
    def name(self):
        return SCENES[self.beat][1]

    def _next_index(self, b):
        return min(b + 1, LAST)

    def cue(self):
        return SCENES[self.beat][3]

    def goto(self, scene, how="manual"):
        scene = int(np.clip(scene, 0, LAST))
        self._clear_scene()
        self.beat, self.entered, self._state, self._cool = scene, time.perf_counter(), {}, 0.0
        self._event("enter", how)
        if scene == 10:
            self.fx["fireworks"].finale(self.sc.finale_s * 3)   # plenty of sky to cut with the snap
        if self.recorder:
            self.recorder.set_status(self.recorder.take_number, f"showcase {scene} {self.name}")

    def advance(self, how="manual"):
        self.goto(self.beat + 1 if self.beat < LAST else 1, how)

    def back(self):
        self.goto(max(self.beat - 1, 1), "manual")

    def reset(self):
        self.goto(self.beat, "manual")

    def finish(self):
        """End of the session: close the recording and log every trigger."""
        if self.recorder and self.take_start is not None:
            self.recorder.end_take([{"beat": s, "name": f"showcase: {SCENES[s][1]} {what}", "how": how, "t": t}
                                    for t, s, what, how in self.events], False)
        self.take_start = None
        return self.events

    def _event(self, what, how):
        t = round(time.perf_counter() - self.t0, 2)
        self.events.append((t, self.beat, what, how))
        if what != "enter":
            print(f"[showcase] {t:7.2f}s  {self.beat} {self.name}: {what} ({how})")

    def _clear_scene(self):
        """A clean stage for every scene, so each clip shows only its own effect."""
        fx = self.fx
        for k in ("particles", "flames", "petals", "lanterns", "fireworks"):
            fx[k].clear()
        if fx["rangoli"].active:
            fx["rangoli"].scatter()
        self.engine.clear_trails()
        fx["diya_row"].set_on(False)
        fx["aarti"].stop()
        self._loops(None)
        self.stab.reset()

    # ---- per-frame --------------------------------------------------------------------------
    def update(self, tf, gestures, now, dt):
        self.tf, self.now, self.dt, self.gestures = tf, now, dt, gestures
        q = self._state.get("quiet")
        self.engine.trail_gain = max(0.0, 1 - (now - q) / self.sc.quiet_trail_fade_s) if q else 1.0
        if self._state.get("fade") is None:
            self.engine.brightness = float(self.cfg.render.brightness)
        self._cool = max(0.0, self._cool - dt)
        self._pick_main_hand(tf, gestures)
        g = SCENES[self.beat][2]
        if g and self.gestures_enabled and gestures[g].just_on and self._cool <= 0:
            self.trigger("gesture")
        s = self.beat
        if s == 1 and self._state.get("joined") and self._state.get("joined_by") == "gesture" \
                and not gestures["joined"].active \
                and now - self._state["joined"] > self.sc.puja_min_s:
            self.fx["rangoli"].scatter()
            self._state["joined"] = None
        if s == 2 and self._state.get("lit") and gestures["fist"].just_on \
                and now - self._state.get("lit_t", 0) > self.k.relight_lock_s:
            self._state["lit"] = False
            if self.last_palm is not None:
                self.fx["flames"].snuff(*self.last_palm)
            self._play("snuff")
            self._state["snuff_t"] = now
            self._event("snuff", "gesture")
        if s == 10 and self._state.get("fade") is not None:
            f = max(0.0, 1 - (now - self._state["fade"]) / self.sc.quiet_fade_s)
            self.engine.brightness = float(self.cfg.render.brightness) * f

    def _pick_main_hand(self, tf, gestures):
        here = self.main_hand in tf.hands and tf.hands[self.main_hand].tracked
        self._main_lost = 0.0 if here else getattr(self, "_main_lost", 0.0) + self.dt
        if self._main_lost < self.sc.hand_switch_s and self.beat in (2, 3, 10):
            return
        if gestures["open_palm"].active and gestures["open_palm"].hand:
            self.main_hand = gestures["open_palm"].hand
        elif not here and tf.hands:
            self.main_hand = next(iter(tf.hands))

    def _play(self, name, v=1.0):
        if self.audio:
            self.audio.play(name, v)

    def trigger(self, how="manual"):
        """The scene's effect, from its gesture or the F key."""
        fx, s, k = self.fx, self.beat, self.k
        W, H = self.engine.size
        now = time.perf_counter()
        h = self._hand(self.gestures[SCENES[s][2]].hand if SCENES[s][2] and how == "gesture" else None)
        palm = h.palm if h is not None else np.array([W * 0.5, H * 0.65])
        what = None
        if s == 1 and how == "manual" and self._state.get("joined") and fx["rangoli"].active:
            fx["rangoli"].scatter()                 # second F: part the hands
            self._state["joined"] = None
            what = "scatter"
        elif s == 1:
            fx["diya_row"].set_on(False)
            fx["diya_row"].light()
            fx["petals"].shower()
            mid = self._mid_palms()
            fx["rangoli"].bloom(tuple(palm if mid is None else mid), self._mean_size())
            self._state["joined"], self._state["joined_by"] = now, how
            self._play("shankh", 0.8)
            self._play("bell_ring", 0.6)
            what = "puja"
        elif s == 2:
            if how == "gesture" and self._state.get("lit"):
                return                          # an open palm on a lit diya is just holding it
            if how == "gesture" and now - self._state.get("snuff_t", -99) < self.sc.min_s[4]:
                return                          # let the smoke curl play before a relight
            if not self._state.get("lit"):
                first = "lit" not in self._state
                self._state["lit"], self._state["lit_t"] = True, now
                self._play("match" if first else "ignite", 0.8)
                if not first and self.last_palm is not None:
                    self._relight_puff()
                what = "light" if first else "relight"
            else:
                self._state["lit"] = False
                if self.last_palm is not None:
                    fx["flames"].snuff(*self.last_palm)
                self._play("snuff")
                what = "snuff"
        elif s == 3:
            what = "circle"
        elif s in (4, 5):
            if how == "manual":
                self._state["force"] = now      # F: light the main hand's sparkler / spin the chakri for a while
            what = "light" if s == 4 else "spin"
        elif s == 6:
            self._state["anar"] = now
            self._state["anar_hand"] = h.label if h is not None else None
            what = "erupt"
        elif s == 7:
            tgt = np.array(self.sc.firework_target_frac) * [W, H] + [np.random.uniform(-1, 1) * W * k.rocket_spread, 0]
            fx["fireworks"].launch(tuple(palm), tuple(tgt), self.sc.launch_flight_s)
            self._cool = k.rocket_cooldown_s
            what = "rocket"
        elif s == 8:
            fx["fireworks"].finale(self.sc.finale_min_s + 1.0)
            self._cool = k.finale_cooldown_s
            what = "finale"
        elif s == 9:
            fx["lanterns"].release(tuple(palm - [0, (h.size if h is not None else 120) * self.sc.lantern_lift]))
            self._play("lantern", 0.8)
            self._cool = k.lantern_cooldown_s
            what = "lanterns"
        elif s == 10 and not self._state.get("quiet"):
            fx["fireworks"].clear()
            self._state["quiet"] = now
            fx["lanterns"].release(tuple(palm - [0, (h.size if h is not None else 120) * self.sc.lantern_lift]))
            self._play("chime", 0.7)
            self._play("lantern", 0.8)
            self.stab.reset()
            what = "snap: quiet end"
        elif s == 10:
            if how == "gesture":
                return
            if self._state.get("fade") is None:
                self._state["fade"] = now
                what = "fade out"
            else:
                self._state["fade"] = None
                self.stab.reset()
                what = "back"
        if what:
            self._event(what, how)

    def _relight_puff(self):
        fc = self.cfg.render.flame
        pos, sz = self.last_palm
        self.fx["particles"].emit(np.array(pos, "f4") + [0, -sz * fc.size_palm_frac * 0.15], int(fc.relight_sparks),
                                  speed=tuple(fc.relight_speed), angle=(-2.9, -0.25), life=(0.45, 1.1), size=(2.2, 3.8),
                                  color_hot=(1.0, 0.92, 0.6), color_cool=(1.0, 0.4, 0.08), gravity=320, drag=1.5,
                                  twinkle=0.7, stretch=1.6, branch=0.3, branch_kind=0)

    # ---- drawing ----------------------------------------------------------------------------
    def _pinch_on(self, h):
        """Per-hand pinch, latched: a sparkler stays lit through short slips and goes out after release_s."""
        st = self._state.setdefault("pinch", {})
        d = np.linalg.norm(h.cam[THUMB_TIP, :2] - h.cam[INDEX_TIP, :2]) / max(h.size_cam, 1e-3)
        if d < self.cfg.gestures.pinch_ratio and h.tracked:
            if h.label not in st or st[h.label] is None:
                self.fx["sparklers"][h.label].ignite()
                self._play("match", 0.5)
                self._event(f"light ({h.label})", "gesture")
            st[h.label] = self.now
        last = st.get(h.label)
        if last is not None and self.now - last > self.k.release_s:
            st[h.label] = None
        return st.get(h.label) is not None

    def draw_trail(self):
        fx, s, dt, now, k = self.fx, self.beat, self.dt, self.now, self.k
        forced = self._state.get("force") is not None and now - self._state["force"] < k.force_s
        any_spark = False
        for label in ("Left", "Right"):
            h = self.tf.hands.get(label)
            on = False
            if s == 4 and h is not None:
                on = self._pinch_on(h) or (forced and label == self.main_hand)
            any_spark |= on
            grip = (h.proj[THUMB_TIP] + h.proj[INDEX_TIP]) / 2 if h is not None else (0, 0)
            fx["sparklers"][label].update(dt, tuple(grip), tuple(h.vel) if h is not None else (0, 0), on,
                                          h.presence if h is not None else 0.0,
                                          direction=(h.proj[INDEX_MCP] - h.proj[WRIST]) if h is not None else None,
                                          length=h.size * self.cfg.render.sparkler.stick_palm_frac if h is not None else None)
        pt = self.gestures["point"]
        hc = self._hand(pt.hand) if s == 5 else None
        spin = s == 5 and hc is not None and (pt.active or forced)
        fx["chakri"].update(dt, tuple(hc.proj[INDEX_TIP]) if hc is not None else (0, 0), spin,
                            hc.presence if hc is not None else 0.0, size=hc.size if hc is not None else None)
        ha = self._hand(self._state.get("anar_hand")) if s == 6 else None
        erupt = s == 6 and ha is not None and self._state.get("anar") is not None and now - self._state["anar"] < k.anar_s
        fx["anar"].update(dt, tuple(self._seat(ha.palm, ha.size)) if ha is not None else (0, 0), erupt,
                          ha.presence if ha is not None else 0.0, size=ha.size if ha is not None else None)
        if self.audio:
            self.audio.loop("sparkler_loop", any_spark)
            self.audio.loop("chakri_loop", spin)
            self.audio.loop("anar_loop", erupt)
        fx["particles"].update(dt)
        fx["fireworks"].update(dt, now)
        fx["particles"].render()
        fx["fireworks"].render()

    def draw_direct(self):
        fx, s, now, fc = self.fx, self.beat, self.now, self.cfg.render.flame
        fx["diya_row"].update(self.dt)
        fx["diya_row"].draw()
        if s == 2:
            lit, t_l = self._state.get("lit"), now - self._state.get("lit_t", now)
            if lit:
                self._palm_diya(_ease(t_l / 0.4), flare=max(0.0, 1 - t_l / 0.6) * fc.relight_flare * 0.7)
            else:
                self._palm_diya(0.0)
        elif s == 3:
            self._palm_diya(_ease((now - self.entered) / 0.6))
            self._aarti_step()
        elif s == 10 and self._state.get("quiet"):
            self._palm_diya(_ease((now - self._state["quiet"]) / 0.8), glow=fc.quiet_glow, calm=1.0, embers=fc.quiet_embers)
        for sp in fx["sparklers"].values():
            if s == 4:
                sp.render_core(fx["flames"])
        if s == 5:
            fx["chakri"].render_core(fx["flames"])
        if s == 6:
            fx["anar"].render_core(fx["flames"])
        fx["flames"].update(self.dt, now)
        fx["flames"].render()
        if fx["rangoli"].active:
            mid = self._mid_palms()
            if mid is not None and s == 1 and self._state.get("joined"):
                fx["rangoli"].set_center(tuple(mid))
            fx["rangoli"].update(self.dt, now)
            fx["rangoli"].render()
        for key in ("aarti", "petals", "lanterns"):
            fx[key].update(self.dt)
            fx[key].render()
