"""The show: what triggers each beat and what gets drawn while it plays.

Beat 0 is idle (black, waiting for joined hands). The last beat is "done" (black, take ends).
Every beat can be forced with the keyboard. If a gesture doesn't come, the beat logs a glitch and, after
sequence.fallback_s, moves on by itself so a take never stalls. The log records how each beat was reached.
"""
import math
import time
import numpy as np
from .types import THUMB_TIP, INDEX_TIP, INDEX_MCP, WRIST
from .tracker import Stabilizer

# (number, name, gesture that starts it; None = automatic)
BEATS = [
    (0, "Idle", None),
    (1, "Puja", "joined"),            # namaste: deepavali row lights up, rangoli blooms, petals, shankh + bell
    (2, "Diya", "open_palm"),         # a diya lights on the palm
    (3, "Aarti", "circle"),           # circle the diya: a light trail, a bell on every loop
    (4, "Snuff", "fist"),
    (5, "Relight", "open_palm"),
    (6, "Phuljhadi", "pinch"),        # the sparkler catches from the diya
    (7, "Chakri", "point"),           # a ground spinner whirls on the fingertip
    (8, "Anar", "low_palm"),          # a flowerpot fountain erupts from the low palm
    (9, "Rocket", "raise"),           # lift fast: a rocket launches from the palm
    (10, "Firework", None),
    (11, "Finale", "arms_wide"),
    (12, "Quiet end", "still"),       # sky lanterns rise, one diya stays on the palm, fade to black
    (13, "Done", None),
]
DONE = len(BEATS) - 1
HINTS = {"joined": "Join your hands in namaste", "open_palm": "Open your palm toward the light",
         "circle": "Circle the diya slowly (aarti)", "fist": "Close your hand slowly over the flame",
         "pinch": "Pinch thumb + index like holding a phuljhadi", "point": "Point your index finger",
         "low_palm": "Lower your open palm", "raise": "Lift your hand up in one steady move",
         "arms_wide": "Open both arms wide", "still": "Hold the diya still"}
MIN_DWELL_S = 0.4
MILESTONES = {1: ("first_puja", "First puja: hands joined, the row of diyas lit up"),
              2: ("first_flame", "First flame on my palm"), 3: ("first_aarti", "First aarti circle"),
              6: ("first_sparkler", "First phuljhadi lit from the diya"), 7: ("first_chakri", "First chakri spinning on my finger"),
              8: ("first_anar", "First anar fountain from my palm"), 10: ("first_firework", "First firework burst on the wall"),
              12: ("first_quiet_end", "First quiet ending: lanterns rising, one diya left on my palm")}
LOOPS = {6: "sparkler_loop", 7: "chakri_loop", 8: "anar_loop"}


def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Sequence:
    def __init__(self, cfg, engine, fx, audio=None, recorder=None, rehearse=False):
        self.cfg, self.sc = cfg, cfg.sequence
        self.engine, self.fx, self.audio, self.recorder = engine, fx, audio, recorder
        self.rehearse = rehearse
        self.gestures_enabled = self.sc.gestures_enabled
        self.beat = 0
        self.entered = time.perf_counter()
        self.take_start = None
        self.log = []                 # [{"beat", "name", "how", "t"}] for the current take
        self.runs = []                # rehearsal summaries
        self.main_hand = "Right"      # the hand holding the diya
        self.act_hand = None          # the hand that made this beat's gesture
        self.last_palm = None         # (pos, size) of the diya, kept through brief tracking loss
        self.missed_flagged = False
        self.on_done = None           # callback(log, completed) when a take ends
        self.stab = Stabilizer(cfg)
        self._aarti = None            # (last heading, accumulated turns)
        self.tf = None

    # ---- navigation -----------------------------------------------------------------------
    @property
    def name(self):
        return BEATS[self.beat][1]

    def _next_index(self, b):
        return min(b + 1, DONE)

    def goto(self, beat, how="manual"):
        now = time.perf_counter()
        beat = int(np.clip(beat, 0, DONE))
        if beat == DONE and self.take_start is None:   # nothing running: don't log a one-beat "take"
            return
        if self.beat == DONE and beat != DONE:          # leaving the end screen starts fresh
            self.reset()
        if 1 <= beat < DONE and self.take_start is None:
            self.take_start = now
            self.log = []
            if self.recorder:
                self.recorder.start_take()
        prev = self.beat
        if how != "gesture":
            self.act_hand = None
        self.beat, self.entered, self.missed_flagged = beat, now, False
        if beat >= 1 and self.take_start is not None:
            self.log.append({"beat": beat, "name": self.name, "how": how, "t": round(now - self.take_start, 2)})
        if self.recorder:
            self.recorder.set_status(self.recorder.take_number, f"{beat} {self.name}")
        self._on_enter(beat, prev)
        if beat == DONE:
            self._end_take(completed=True)

    def advance(self, how="manual"):
        self.goto(1 if self.beat == DONE else self._next_index(self.beat), how)

    def back(self):
        self.goto(max(self.beat - 1, 0), "manual")

    def reset(self):
        if self.take_start is not None and self.beat != DONE:
            self._end_take(completed=False)
        fx = self.fx
        fx["particles"].clear()
        fx["flames"].clear()
        fx["petals"].clear()
        fx["lanterns"].clear()
        fx["diya_row"].set_on(False)
        self.engine.clear_trails()
        self._loops(None)
        self.beat, self.entered = 0, time.perf_counter()

    def _end_take(self, completed):
        log = list(self.log)
        if self.recorder:
            self.recorder.end_take(log, completed)
        manual = sum(1 for e in log if e["how"] == "manual")
        missed = [e for e in log if e["how"] in ("missed", "timeout")]
        self.runs.append({"completed": completed, "manual": manual, "missed": [e["name"] for e in missed],
                          "duration": round(time.perf_counter() - self.take_start, 1)})
        if self.recorder and completed:
            self.recorder.milestone("first_full_run", "First full run, beat 1 to the end, no crash")
            if manual == 0 and not missed:
                self.recorder.milestone("first_clean_take", "First clean take: every beat by gesture, no keyboard")
        if self.on_done:
            self.on_done(log, completed)
        self.take_start = None

    # ---- triggers ---------------------------------------------------------------------------
    def update(self, tf, gestures, now, dt):
        """Check triggers and timers. Drawing happens in draw_trail() / draw_direct()."""
        self.tf, self.now, self.dt, self.gestures = tf, now, dt, gestures
        sc, t_in, b = self.sc, now - self.entered, self.beat
        nxt = self._next_index(b)
        trig = BEATS[nxt][2] if b < DONE else None
        dwell = {10: sc.firework_hold_s, 11: sc.finale_min_s}.get(b, MIN_DWELL_S)
        if b == 9 and t_in >= sc.launch_flight_s:
            self.goto(10, "auto")
        elif b == 12 and t_in >= sc.quiet_hold_s + sc.quiet_fade_s:
            self.goto(DONE, "auto")
        elif trig and self.gestures_enabled and t_in >= dwell and gestures[trig].active:
            self.act_hand = gestures[trig].hand
            self.goto(nxt, "gesture")
        elif trig and b >= 1:
            if t_in > self.cfg.rehearse.beat_timeout_s + dwell and not self.missed_flagged:
                self.missed_flagged = True
                if self.recorder:
                    self.recorder.glitch("missed_gesture", f"beat {nxt} {BEATS[nxt][1]} ({trig})")
                if self.rehearse:
                    self.goto(nxt, "missed")
            elif sc.fallback_s and t_in > sc.fallback_s + dwell and not self.rehearse:
                self.goto(nxt, "timeout")
        if self.beat == 1 and t_in > sc.puja_min_s and not gestures["joined"].active and self.fx["rangoli"].active:
            self.fx["rangoli"].scatter()
        self.engine.trail_gain = max(0.0, 1 - t_in / sc.quiet_trail_fade_s) if self.beat == 12 else 1.0
        # the end fades the whole picture (rising lanterns included) to black, not just the diya
        q = 1.0 - max(0.0, t_in - sc.quiet_hold_s) / sc.quiet_fade_s if self.beat == 12 else 1.0
        self.engine.brightness = float(self.cfg.render.brightness) * max(q, 0.0)
        self._pick_main_hand(tf, gestures)

    def _pick_main_hand(self, tf, gestures):
        if gestures["open_palm"].active and gestures["open_palm"].hand:
            self.main_hand = gestures["open_palm"].hand
        elif self.main_hand not in tf.hands and tf.hands:
            self.main_hand = next(iter(tf.hands))

    # ---- entering a beat --------------------------------------------------------------------
    def _loops(self, beat):
        if self.audio:
            for b, name in LOOPS.items():
                self.audio.loop(name, b == beat)

    def _on_enter(self, beat, prev):
        fx, a, sc = self.fx, self.audio, self.sc
        play = (lambda n, v=1.0: a.play(n, v)) if a else (lambda *x: None)
        W, H = self.engine.size
        self._loops(beat)
        if beat == 1:
            fx["diya_row"].light()
            fx["petals"].shower()
            mid = self._mid_palms()
            mid = np.array([W * 0.5, H * 0.55]) if mid is None else mid
            if sc.rangoli_enabled:
                fx["rangoli"].bloom(tuple(mid), self._mean_size())
            play("shankh", 0.8)
            play("bell_ring", 0.6)
        if prev == 1 and beat != 1 and fx["rangoli"].active:
            fx["rangoli"].scatter()
        if beat == 2:
            play("match", 0.8)
            play("ignite")
            self.stab.reset()
        if beat == 3:
            self._aarti = None
            play("bell", 0.7)
        if prev == 3:
            fx["aarti"].stop()
        if beat == 4 and self.last_palm is not None:
            fx["flames"].snuff(*self.last_palm)
            play("snuff")
        if beat == 5:
            play("ignite")
            if self.last_palm is not None:          # relight: a puff of sparks off the wick
                fc = self.cfg.render.flame
                pos, sz = self.last_palm
                fx["particles"].emit(np.array(pos, "f4") + [0, -sz * fc.size_palm_frac * 0.15], int(fc.relight_sparks),
                                     speed=tuple(fc.relight_speed), angle=(-2.9, -0.25), life=(0.45, 1.1), size=(2.2, 3.8),
                                     color_hot=(1.0, 0.92, 0.6), color_cool=(1.0, 0.4, 0.08), gravity=320, drag=1.5,
                                     twinkle=0.7, stretch=1.6, branch=0.3, branch_kind=0)
        if beat == 6:
            self._lit = set()                       # sparklers catch from the diya as each hand starts pinching
        if beat == 9:
            h = self._hand(self.act_hand)
            start = h.palm if h is not None else np.array([W * 0.5, H * 0.8])
            tgt = np.array(sc.firework_target_frac) * [W, H]
            fx["fireworks"].launch(tuple(start), tuple(tgt), sc.launch_flight_s)
        if beat == 11:
            fx["fireworks"].finale(sc.finale_min_s + 1.0)
        if beat == 12:
            h = self._hand()
            src = h.palm - [0, h.size * sc.lantern_lift] if h is not None else np.array([W * 0.5, H * 0.6])
            fx["lanterns"].release(tuple(src))          # from just above the palm, so the last diya stays clear
            fx["diya_row"].set_on(False)
            play("lantern", 0.8)
            play("chime", 0.7)
            self.stab.reset()
        if beat in (0, DONE):
            fx["flames"].clear()
            fx["diya_row"].set_on(False)
            fx["lanterns"].clear()
            fx["petals"].clear()
        if beat == DONE:
            fx["particles"].clear()
            self.engine.clear_trails()
        if self.recorder and beat in MILESTONES and self.tf_has_hands():
            self.recorder.milestone(*MILESTONES[beat])

    def tf_has_hands(self):
        return bool(self.tf and self.tf.hands)

    # ---- helpers ----------------------------------------------------------------------------
    def _hand(self, label=None):
        if not self.tf:
            return None
        return self.tf.hands.get(label or self.main_hand) or (next(iter(self.tf.hands.values())) if self.tf.hands else None)

    def _mean_size(self):
        hs = list(self.tf.hands.values()) if self.tf else []
        return float(np.mean([h.size for h in hs])) if hs else self.engine.size[1] * 0.12

    def _mid_palms(self):
        hs = list(self.tf.hands.values()) if self.tf else []
        if len(hs) == 2:
            return (hs[0].palm + hs[1].palm) / 2
        return hs[0].palm if hs else None

    def _seat(self, palm, size):
        """Wick position for a diya resting in the palm hollow: the bowl's centre sits on the palm, so the wick
        is lifted by half the bowl's height, straight up (gravity), whatever angle the hand is at."""
        return np.asarray(palm) - np.array([0.0, self.cfg.render.flame.seat_frac * size])

    def _palm_diya(self, intensity, flare=0.0, glow=1.0, calm=0.0, embers=1.0):
        h = self._hand()
        if h is not None:
            seat = self.stab(self._seat(h.palm, h.size), h.size, self.dt)
            self.last_palm = (tuple(seat), h.size)
            intensity *= h.presence
        if self.last_palm is None:
            return
        if h is None:
            intensity = 0.0
        if intensity > 0.01:
            vel = h.vel if h is not None else (0.0, 0.0)
            self.fx["flames"].add(self.last_palm[0], self.last_palm[1], intensity, flare, diya=True, seed=0.0, vel=vel,
                                  glow=glow, calm=calm, embers=embers)

    def _aarti_step(self):
        """Light trail of the diya while circling; a ring pulse + bell on every full loop."""
        h = self._hand()
        if h is None or self.last_palm is None:
            return
        self.fx["aarti"].add(self.last_palm[0], self.now)
        v = h.vel_cam
        if np.linalg.norm(v) / max(h.size_cam, 1e-3) < self.cfg.gestures.circle_min_speed:
            return
        a = math.atan2(v[1], v[0])
        if self._aarti is None:
            self._aarti = (a, 0.0)
            return
        last, turns = self._aarti
        d = (a - last + math.pi) % (2 * math.pi) - math.pi
        new = turns + abs(d) / (2 * math.pi)
        if int(new) > int(turns):
            self.fx["aarti"].pulse(self.last_palm[0])
            if self.audio:
                self.audio.play("bell", 0.6)
        self._aarti = (a, new)

    # ---- drawing ----------------------------------------------------------------------------
    def draw_trail(self):
        """Effects on the trail layer (sparks). Called after engine.begin_trail()."""
        fx, b, t_in, dt = self.fx, self.beat, self.now - self.entered, self.dt
        spark_int = 1.0 if b == 6 else (max(0.0, 1 - t_in / 0.6) if b == 7 else 0.0)
        for label in ("Left", "Right"):
            h = self.tf.hands.get(label)
            sp = fx["sparklers"][label]
            # one hand pinching is enough; a second hand lights its own sparkler when it pinches too
            on = h is not None and spark_int > 0.01 and (label == self.act_hand or label in getattr(self, "_lit", ())
                                                          or self._pinching(h)
                                                          or (self.act_hand is None and label == self.main_hand))
            if on and b == 6 and label not in self._lit:
                self._lit.add(label)
                sp.ignite()
            grip = (h.proj[THUMB_TIP] + h.proj[INDEX_TIP]) / 2 if h is not None else (0, 0)
            sp.update(dt, tuple(grip), tuple(h.vel) if h is not None else (0, 0), on,
                      spark_int * (h.presence if h is not None else 0),
                      direction=(h.proj[INDEX_MCP] - h.proj[WRIST]) if h is not None else None,
                      length=h.size * self.cfg.render.sparkler.stick_palm_frac if h is not None else None)
        h = self._hand(self.act_hand) if b in (7, 8) else None
        fx["chakri"].update(dt, tuple(h.proj[INDEX_TIP]) if (h is not None and b == 7) else (0, 0),
                            b == 7 and h is not None, h.presence if h is not None else 0.0, size=h.size if h is not None else None)
        fx["anar"].update(dt, tuple(self._seat(h.palm, h.size)) if (h is not None and b == 8) else (0, 0),
                          b == 8 and h is not None, h.presence if h is not None else 0.0, size=h.size if h is not None else None)
        fx["particles"].update(dt)
        fx["fireworks"].update(dt, self.now)
        fx["particles"].render()
        fx["fireworks"].render()

    def _pinching(self, h):
        st = self.gestures["pinch"]
        return st.active and st.hand == h.label

    def draw_direct(self):
        """Effects on the direct layer (flames, diyas, rangoli, petals, lanterns). Called after engine.begin_direct()."""
        fx, b, t_in, fc = self.fx, self.beat, self.now - self.entered, self.cfg.render.flame
        fx["diya_row"].update(self.dt)
        fx["diya_row"].draw()
        if b == 2:
            self._palm_diya(_ease(t_in / 0.4), flare=max(0.0, 1 - t_in / 0.5) * 0.5)
        elif b == 3:
            self._palm_diya(1.0)
            self._aarti_step()
        elif b == 4:
            self._palm_diya(max(0.0, 1 - t_in / 0.25))
        elif b == 5:
            self._palm_diya(1.0, flare=max(0.0, 1 - t_in / 0.6) * fc.relight_flare)
        elif b == 6:
            self._palm_diya(max(0.0, 1 - t_in / 0.6))      # the flame passes to the sparkler
        elif b == 12:
            sc = self.sc
            fade = 1.0 if t_in < sc.quiet_hold_s else max(0.0, 1 - (t_in - sc.quiet_hold_s) / sc.quiet_fade_s)
            self._palm_diya(_ease(t_in / 0.8) * fade, glow=fc.quiet_glow, calm=1.0, embers=fc.quiet_embers)
        for s in fx["sparklers"].values():
            if b in (6, 7):
                s.render_core(fx["flames"])
        if b == 7:
            fx["chakri"].render_core(fx["flames"])
        if b == 8:
            fx["anar"].render_core(fx["flames"])
        fx["flames"].update(self.dt, self.now)
        fx["flames"].render()
        if fx["rangoli"].active:
            mid = self._mid_palms()
            if mid is not None and b == 1:
                fx["rangoli"].set_center(tuple(mid))
            fx["rangoli"].update(self.dt, self.now)
            fx["rangoli"].render()
        for k in ("aarti", "petals", "lanterns"):
            fx[k].update(self.dt)
            fx[k].render()
