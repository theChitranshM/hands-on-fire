"""Gesture classifier: continuous confidences from camera-space landmarks, then hysteresis + minimum hold.

One-hand first: the first real takes showed two hands tracked together in under 3% of frames (one arm reaching
into the light), so every beat except namaste and arms-wide works with a single hand.
All distances are in palm sizes (wrist -> middle MCP) so it works at any distance from the camera.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from .types import WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, TIPS, PIPS

GESTURES = ["open_palm", "fist", "pinch", "point", "circle", "low_palm", "raise", "joined", "arms_wide", "still",
            "pinch_both", "cupped", "palms_together"]


def ramp(x, lo, hi):
    """0 at lo, 1 at hi, linear, clamped. Works for lo > hi (decreasing)."""
    if hi == lo:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class GestureState:
    conf: float = 0.0
    active: bool = False
    just_on: bool = False
    hand: str = None
    since: float = 0.0            # when it last became active
    _above_since: float = None
    _dip_since: float = None
    _below_since: float = None


class GestureClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.g = cfg.gestures
        self.proj_h = cfg.display.max_render_size[1]     # run.py sets the real render height
        self.reset()

    def reset(self):
        self.states = {k: GestureState() for k in GESTURES}
        self._heading = {}            # hand -> deque of (t, heading angle) for circle detection
        self._last_pair = None        # (t, midpoint cam px, palm size) when two hands were last close together

    # ---- per-hand features ----------------------------------------------------------------
    def _extension(self, h):
        x = h.cam[:, :2]
        w = x[WRIST]
        r = [np.linalg.norm(x[t] - w) / max(np.linalg.norm(x[p] - w), 1e-3) for t, p in zip(TIPS, PIPS)]
        k = self.g.finger_extended_ratio
        return np.array([ramp(v, k - 0.12, k + 0.12) for v in r])

    def _hand_feats(self, h, proj_h):
        ext = self._extension(h)
        s = max(h.size_cam, 1e-3)
        pinch_r = np.linalg.norm(h.cam[THUMB_TIP, :2] - h.cam[INDEX_TIP, :2]) / s
        pr = self.g.pinch_ratio
        return {"open": float(ext.min() * 0.5 + ext.mean() * 0.5),
                # a closed hand: judged on the mean curl, because MediaPipe often leaves one finger half-straight
                "fist": ramp(float(ext.mean()), self.g.fist_mean_ext[1], self.g.fist_mean_ext[0]),
                "pinch": ramp(pinch_r, pr * 1.6, pr * 0.8),
                # index out, the rest mostly curled (a relaxed middle finger is normal when pointing)
                "point": float(ext[0] * (1.0 - ext[1:].mean())),
                "up": ramp((h.cam[WRIST, 1] - h.cam[MIDDLE_TIP, 1]) / s, 0.8, 1.5),
                # anar: palm held low, or pushed down (lowering often leaves the camera's view before "low")
                "low": max(ramp(h.palm[1] / max(proj_h, 1), self.g.low_frac - 0.08, self.g.low_frac),
                           ramp(h.vel_cam[1] / s, self.g.push_speed * 0.6, self.g.push_speed)),
                "speed": float(np.linalg.norm(h.vel_cam) / s),
                "vy": float(-h.vel_cam[1] / s)}

    def _circle(self, label, h, now):
        """Turns of the hand's direction of travel in the last circle_window_s (aarti: moving the diya in circles)."""
        g = self.g
        q = self._heading.setdefault(label, deque())
        s = max(h.size_cam, 1e-3)
        if np.linalg.norm(h.vel_cam) / s >= g.circle_min_speed:
            q.append((now, math.atan2(h.vel_cam[1], h.vel_cam[0])))
        while q and now - q[0][0] > g.circle_window_s:
            q.popleft()
        if len(q) < 4:
            return 0.0
        a = np.unwrap(np.array([v for _, v in q]))
        return float(abs(a[-1] - a[0]) / (2 * math.pi))

    # ---- confidences ----------------------------------------------------------------------
    def confidences(self, tf, now):
        g = self.g
        proj_h = self.proj_h
        hands = {k: h for k, h in tf.hands.items() if h.tracked}
        f = {k: self._hand_feats(h, proj_h) for k, h in hands.items()}
        for k in list(self._heading):
            if k not in hands:
                self._heading.pop(k)
        c = {k: (0.0, None) for k in GESTURES}

        def best(feat):
            if not f:
                return 0.0, None
            k = max(f, key=lambda k: f[k][feat])
            return f[k][feat], k

        c["open_palm"] = best("open")
        c["fist"] = best("fist")
        c["pinch"] = best("pinch")
        c["point"] = best("point")
        if f:
            k = max(f, key=lambda k: f[k]["open"] * f[k]["low"])
            c["low_palm"] = (f[k]["open"] * f[k]["low"], k)
            k = max(f, key=lambda k: f[k]["vy"])
            c["raise"] = (ramp(f[k]["vy"], g.raise_speed * 0.6, g.raise_speed), k)
            c["still"] = (min(ramp(v["speed"], g.still_speed * 2, g.still_speed) for v in f.values()), None)
            turns = {k: self._circle(k, h, now) for k, h in hands.items()}
            k = max(turns, key=turns.get)
            c["circle"] = (ramp(turns[k], 0.0, g.circle_turns), k)
        if len(hands) == 2:
            L, R = hands["Left"], hands["Right"]
            fl, fr = f["Left"], f["Right"]
            s = (L.size_cam + R.size_cam) / 2
            d = np.linalg.norm(L.palm_cam - R.palm_cam) / s
            level = ramp(abs(L.palm_cam[1] - R.palm_cam[1]) / s, 1.0, 0.4)
            c["pinch_both"] = (min(fl["pinch"], fr["pinch"]), None)
            close = ramp(d, g.cupped_max_dist, g.cupped_max_dist * 0.6)
            c["cupped"] = (close * level * (1 - 0.5 * max(fl["fist"], fr["fist"])), None)
            c["palms_together"] = (ramp(d, g.palms_together_max_dist, g.palms_together_max_dist * 0.5)
                                   * min(fl["up"], fr["up"]), None)
            c["joined"] = (ramp(d, g.joined_dist * 1.4, g.joined_dist), None)
            if d < g.joined_dist * 1.6:
                self._last_pair = (now, (L.palm_cam + R.palm_cam) / 2, s)
            cam_w = tf.image.shape[1] if tf.image is not None else self.cfg.camera.width
            sep = abs(L.palm_cam[0] - R.palm_cam[0]) / cam_w
            c["arms_wide"] = (ramp(sep, g.arms_wide_frac * 0.8, g.arms_wide_frac), None)
        elif self._last_pair and now - self._last_pair[0] < g.joined_merge_s:
            # pressed-together palms often merge into one detection (or none): two hands that were closing in
            # a moment ago and now read as one hand at their midpoint (or vanish there) are joined
            t0, mid, s = self._last_pair
            if not hands:
                c["joined"] = (0.8, None)
            elif len(hands) == 1:
                h = next(iter(hands.values()))
                c["joined"] = (ramp(np.linalg.norm(h.palm_cam - mid) / s, 1.2, 0.5), None)
        return c

    def update(self, tf, now):
        """Returns {gesture: GestureState} with hysteresis and min hold applied."""
        g = self.g
        for k, (conf, hand) in self.confidences(tf, now).items():
            st = self.states[k]
            st.conf, st.hand, st.just_on = conf, hand, False
            hold = g.min_hold_ms.get(k, 150) / 1000
            if not st.active:
                if conf >= g.on_threshold:
                    st._above_since, st._dip_since = st._above_since or now, None
                    if now - st._above_since >= hold:
                        st.active, st.just_on, st._below_since, st.since = True, True, None, now
                elif st._above_since is not None:
                    # a brief dip (noise, a dropped frame) doesn't restart the hold timer
                    st._dip_since = st._dip_since or now
                    if now - st._dip_since > g.grace_ms / 1000:
                        st._above_since = st._dip_since = None
            else:
                if conf < g.off_threshold:
                    st._below_since = st._below_since or now
                    if now - st._below_since >= min(hold, 0.12):
                        st.active, st._above_since = False, None
                else:
                    st._below_since = None
        return self.states
