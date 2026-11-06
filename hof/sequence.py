"""Beats 1-10: what triggers each beat and what gets drawn while it plays.

Beat 0 is idle (black, waiting for an open palm). Beat 11 is "done" (black, take ends).
Every beat can be forced with the keyboard; the log records how each beat was reached.
"""
import time
import numpy as np
from .types import THUMB_TIP, INDEX_TIP

# (number, name, trigger gesture or None for auto/none, minimum seconds in previous beat before it can fire)
BEATS = [
    (0, "Idle", None),
    (1, "Palm diya", "open_palm"),
    (2, "Snuff", "fist"),
    (3, "Relight", "open_palm"),
    (4, "Sparklers", "pinch_both"),
    (5, "Diya catch", "cupped"),
    (6, "Launch", "raise"),
    (7, "Firework", None),
    (8, "Rangoli", "palms_together"),
    (9, "Finale", "arms_wide"),
    (10, "Quiet end", "still"),
    (11, "Done", None),
]
DONE = 11
MIN_DWELL_S = 0.4


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
        self.main_hand = "Right"
        self.last_palm = None          # (pos, size) of the diya, kept through brief tracking loss
        self.missed_flagged = False
        self.on_done = None            # callback(log, completed) when a take ends
        self._quiet_t = None

    # ---- navigation -----------------------------------------------------------------------
    @property
    def name(self):
        return BEATS[self.beat][1]

    def _next_index(self, b):
        n = b + 1
        if n == 8 and not self.sc.rangoli_enabled:
            n = 9
        return min(n, DONE)

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
        b = self.beat - 1
        if b == 8 and not self.sc.rangoli_enabled:
            b = 7
        self.goto(max(b, 0), "manual")

    def reset(self):
        if self.take_start is not None and self.beat != DONE:
            self._end_take(completed=False)
        self.fx["particles"].clear()
        self.engine.clear_trails()
        self.beat, self.entered = 0, time.perf_counter()

    def _end_take(self, completed):
        log = list(self.log)
        if self.recorder:
            self.recorder.end_take(log, completed)
        manual = sum(1 for e in log if e["how"] == "manual")
        missed = [e for e in log if e["how"] == "missed"]
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
        """Check triggers, then queue this frame's effects. Call between engine layer binds via draw()."""
        self.tf, self.now, self.dt = tf, now, dt
        t_in = now - self.entered
        trig = BEATS[self._next_index(self.beat)][2] if self.beat < DONE else None
        nxt = self._next_index(self.beat)
        dwell = {7: self.sc.firework_hold_s, 9: self.sc.finale_min_s}.get(self.beat, MIN_DWELL_S)
        if self.beat == 6 and t_in >= self.sc.launch_flight_s:
            self.goto(7, "auto")
        elif self.beat == 10 and t_in >= self.sc.quiet_hold_s + self.sc.quiet_fade_s:
            self.goto(DONE, "auto")
        elif trig and self.gestures_enabled and t_in >= dwell and gestures[trig].active:
            self.goto(nxt, "gesture")
        elif trig and self.beat >= 1 and t_in > self.cfg.rehearse.beat_timeout_s + dwell and not self.missed_flagged:
            self.missed_flagged = True
            if self.recorder:
                self.recorder.glitch("missed_gesture", f"beat {nxt} {BEATS[nxt][1]} ({trig})")
            if self.rehearse:
                self.goto(nxt, "missed")
        apart = not gestures["palms_together"].active if self.gestures_enabled \
            else t_in > self.cfg.render.rangoli.bloom_s + 1.0
        if self.beat == 8 and t_in > 0.6 and apart and self.fx["rangoli"].active \
                and not getattr(self, "_scattered", False):
            self._scattered = True
            self.fx["rangoli"].scatter()
        self.engine.trail_gain = max(0.0, 1 - t_in / self.sc.quiet_trail_fade_s) if self.beat == 10 else 1.0
        self._pick_main_hand(tf, gestures)

    def _pick_main_hand(self, tf, gestures):
        if gestures["open_palm"].hand:
            self.main_hand = gestures["open_palm"].hand
        elif self.main_hand not in tf.hands and tf.hands:
            self.main_hand = next(iter(tf.hands))

    # ---- entering a beat --------------------------------------------------------------------
    def _on_enter(self, beat, prev):
        fx, a = self.fx, self.audio
        play = (lambda n, v=1.0: a.play(n, v)) if a else (lambda *x: None)
        size = self.engine.size
        if beat in (1, 3):
            play("ignite")
        if beat == 3 and self.last_palm is not None:       # relight: a puff of sparks off the wick
            fc = self.cfg.render.flame
            pos, sz = self.last_palm
            n = int(fc.relight_sparks)
            fx["particles"].emit(np.array(pos, "f4") + [0, -sz * fc.size_palm_frac * 0.15], n, speed=tuple(fc.relight_speed),
                                 angle=(-2.9, -0.25), life=(0.45, 1.1), size=(2.2, 3.8), color_hot=(1.0, 0.92, 0.6), color_cool=(1.0, 0.4, 0.08),
                                 gravity=320, drag=1.5, twinkle=0.7, stretch=1.6, branch=0.3, branch_kind=0)
        if beat == 2 and self.last_palm is not None:
            fx["flames"].snuff(*self.last_palm)
            play("snuff")
        if beat == 6:
            start = self._mid_palms()
            start = np.array(size) * [0.5, 0.8] if start is None else start
            tgt = np.array(self.sc.firework_target_frac) * size
            fx["fireworks"].launch(tuple(start), tuple(tgt), self.sc.launch_flight_s)
        if beat == 8:
            self._scattered = False
            mid = self._mid_palms()
            mid = np.array(size) * [0.5, 0.6] if mid is None else mid
            fx["rangoli"].bloom(tuple(mid), self._mean_size())
            play("bloom")
        if prev == 8 and beat != 8:
            fx["rangoli"].scatter()
        if beat >= 9 and fx["rangoli"].active and fx["rangoli"].state == "bloom":
            fx["rangoli"].scatter()
        if beat == 9:
            fx["fireworks"].finale(self.sc.finale_min_s + 1.0)
        if beat == 10:
            self._quiet_t = time.perf_counter()
            play("chime", 0.7)
        if beat in (0, DONE):
            fx["flames"].clear()
        if beat == DONE:
            fx["particles"].clear()
            self.engine.clear_trails()
        if beat == DONE or beat == 0:
            if a:
                a.loop("sparkler_loop", False)
        if self.recorder:
            m = {1: ("first_flame", "First flame on my palm"), 4: ("first_sparkler", "First sparklers on both hands"),
                 7: ("first_firework", "First firework burst on the wall"), 8: ("first_rangoli", "First rangoli bloom"),
                 10: ("first_quiet_end", "First quiet ending: one diya left on my palm")}.get(beat)
            if m and self.tf_has_hands():
                self.recorder.milestone(*m)

    def tf_has_hands(self):
        tf = getattr(self, "tf", None)
        return bool(tf and tf.hands)

    # ---- drawing ----------------------------------------------------------------------------
    def _hand(self, label=None):
        return self.tf.hands.get(label or self.main_hand) or (next(iter(self.tf.hands.values())) if self.tf.hands else None)

    def _mean_size(self):
        hs = list(self.tf.hands.values()) if getattr(self, "tf", None) else []
        return float(np.mean([h.size for h in hs])) if hs else self.engine.size[1] * 0.12

    def _mid_palms(self):
        hs = list(self.tf.hands.values()) if getattr(self, "tf", None) else []
        if len(hs) == 2:
            return (hs[0].palm + hs[1].palm) / 2
        return hs[0].palm if hs else None

    def _palm_diya(self, intensity, flare=0.0, glow=1.0, calm=0.0, embers=1.0):
        h = self._hand()
        if h is not None:
            # the diya sits on the palm slightly towards the fingers, lifted against tracking jitter
            self.last_palm = (tuple(h.palm), h.size)
            intensity *= h.presence
        if self.last_palm is None:
            return
        if h is None:
            intensity = 0.0
        if intensity > 0.01:
            vel = h.vel if h is not None else (0.0, 0.0)
            self.fx["flames"].add(self.last_palm[0], self.last_palm[1], intensity, flare, diya=True, seed=0.0, vel=vel, glow=glow, calm=calm, embers=embers)

    def draw_trail(self):
        """Effects on the trail layer (sparks). Called after engine.begin_trail()."""
        fx, b, t_in, dt = self.fx, self.beat, self.now - self.entered, self.dt
        spark_int = {4: 1.0, 5: max(0.0, 0.3 * (1 - t_in / 1.5))}.get(b, 0.0)
        for label in ("Left", "Right"):
            h = self.tf.hands.get(label)
            active = h is not None and spark_int > 0.01
            pos = (h.proj[THUMB_TIP] + h.proj[INDEX_TIP]) / 2 if h is not None else (0, 0)
            vel = h.vel if h is not None else (0, 0)
            fx["sparklers"][label].update(dt, tuple(pos), tuple(vel), active,
                                          spark_int * (h.presence if h is not None else 0))
        if self.audio:
            self.audio.loop("sparkler_loop", spark_int > 0.05, volume=spark_int)
        fx["particles"].update(dt)
        fx["fireworks"].update(dt, self.now)
        fx["particles"].render()
        fx["fireworks"].render()

    def draw_direct(self):
        """Effects on the direct layer (flames, diyas, rangoli). Called after engine.begin_direct()."""
        fx, b, t_in = self.fx, self.beat, self.now - self.entered
        if b == 1:
            self._palm_diya(_ease(t_in / 0.4), flare=max(0.0, 1 - t_in / 0.5) * 0.5)
        elif b == 2:
            self._palm_diya(max(0.0, 1 - t_in / 0.25))
        elif b == 3:
            self._palm_diya(1.0, flare=max(0.0, 1 - t_in / 0.6) * self.cfg.render.flame.relight_flare)
        elif b == 4:
            self._palm_diya(max(0.0, 1 - t_in / 0.4))
        elif b in (5, 6):
            mid = self._mid_palms()
            if mid is not None:
                pres = min(h.presence for h in self.tf.hands.values())
                if b == 5:
                    fx["flames"].add(tuple(mid), self._mean_size() * 1.2, _ease(t_in / 0.5) * pres,
                                     flare=max(0.0, 1 - t_in / 0.4) * 0.6, diya=True, seed=3.0)
                else:  # flame has left with the rocket; the empty diya cools down
                    fx["flames"].add(tuple(mid), self._mean_size() * 1.2, 0.0, diya=True, seed=3.0)
        elif b == 10:
            q = self.sc
            fade = 1.0 if t_in < q.quiet_hold_s else max(0.0, 1 - (t_in - q.quiet_hold_s) / q.quiet_fade_s)
            self._palm_diya(_ease(t_in / 0.8) * fade, glow=self.cfg.render.flame.quiet_glow, calm=1.0, embers=self.cfg.render.flame.quiet_embers)
        for s in fx["sparklers"].values():
            if b in (4, 5):
                s.render_core(fx["flames"])
        fx["flames"].update(self.dt, self.now)
        fx["flames"].render()
        if b == 8 or fx["rangoli"].active:
            mid = self._mid_palms()
            if mid is not None and b == 8 and not getattr(self, "_scattered", False):
                fx["rangoli"].set_center(tuple(mid))
            fx["rangoli"].update(self.dt, self.now)
            fx["rangoli"].render()
