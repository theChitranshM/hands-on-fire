"""Gesture classifier: continuous confidences from camera-space landmarks, then hysteresis + minimum hold.

All distances are in palm sizes (wrist -> middle MCP) so it works at any distance from the camera.
"""
from dataclasses import dataclass
import numpy as np
from .types import WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, TIPS, PIPS

GESTURES = ["open_palm", "fist", "pinch_both", "cupped", "raise", "palms_together", "arms_wide", "still"]
_HOLD_KEY = {"pinch_both": "pinch"}


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
    _above_since: float = None
    _below_since: float = None


class GestureClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.g = cfg.gestures
        self.states = {k: GestureState() for k in GESTURES}

    # ---- per-hand features ----------------------------------------------------------------
    def _extension(self, h):
        x, s = h.cam[:, :2], max(h.size_cam, 1e-3)
        w = x[WRIST]
        r = [np.linalg.norm(x[t] - w) / max(np.linalg.norm(x[p] - w), 1e-3) for t, p in zip(TIPS, PIPS)]
        k = self.g.finger_extended_ratio
        return np.array([ramp(v, k - 0.12, k + 0.12) for v in r])

    def _hand_feats(self, h):
        ext = self._extension(h)
        s = max(h.size_cam, 1e-3)
        pinch_r = np.linalg.norm(h.cam[THUMB_TIP, :2] - h.cam[INDEX_TIP, :2]) / s
        pr = self.g.pinch_ratio
        return {"open": float(ext.min() * 0.5 + ext.mean() * 0.5),
                "fist": float(1.0 - ext.max() * 0.5 - ext.mean() * 0.5),
                "pinch": ramp(pinch_r, pr * 1.6, pr * 0.8),
                "up": ramp((h.cam[WRIST, 1] - h.cam[MIDDLE_TIP, 1]) / s, 0.8, 1.5),
                "speed": float(np.linalg.norm(h.vel_cam) / s),
                "vy": float(-h.vel_cam[1] / s)}

    # ---- confidences ----------------------------------------------------------------------
    def confidences(self, tf):
        g = self.g
        hands = {k: h for k, h in tf.hands.items() if h.tracked}
        f = {k: self._hand_feats(h) for k, h in hands.items()}
        c = {k: (0.0, None) for k in GESTURES}

        def best(feat):
            if not f:
                return 0.0, None
            k = max(f, key=lambda k: f[k][feat])
            return f[k][feat], k

        c["open_palm"] = best("open")
        c["fist"] = best("fist")
        if f:
            c["still"] = (min(ramp(v["speed"], g.still_speed * 2, g.still_speed) for v in f.values()), None)
        if len(hands) == 2:
            L, R = hands["Left"], hands["Right"]
            fl, fr = f["Left"], f["Right"]
            s = (L.size_cam + R.size_cam) / 2
            d = np.linalg.norm(L.palm_cam - R.palm_cam) / s
            level = ramp(abs(L.palm_cam[1] - R.palm_cam[1]) / s, 1.0, 0.4)
            c["pinch_both"] = (min(fl["pinch"], fr["pinch"]), None)
            close = ramp(d, g.cupped_max_dist, g.cupped_max_dist * 0.6)
            c["cupped"] = (close * level * (1 - 0.5 * max(fl["fist"], fr["fist"])), None)
            near = ramp(d, g.cupped_max_dist * 1.3, g.cupped_max_dist)
            c["raise"] = (near * ramp(min(fl["vy"], fr["vy"]), g.raise_speed * 0.6, g.raise_speed), None)
            c["palms_together"] = (ramp(d, g.palms_together_max_dist, g.palms_together_max_dist * 0.5)
                                   * min(fl["up"], fr["up"]), None)
            cam_w = tf.image.shape[1] if tf.image is not None else self.cfg.camera.width
            sep = abs(L.palm_cam[0] - R.palm_cam[0]) / cam_w
            c["arms_wide"] = (ramp(sep, g.arms_wide_frac * 0.8, g.arms_wide_frac), None)
        return c

    def update(self, tf, now):
        """Returns {gesture: GestureState} with hysteresis and min hold applied."""
        g = self.g
        for k, (conf, hand) in self.confidences(tf).items():
            st = self.states[k]
            st.conf, st.hand, st.just_on = conf, hand, False
            hold = g.min_hold_ms.get(_HOLD_KEY.get(k, k), 150) / 1000
            if not st.active:
                if conf >= g.on_threshold:
                    st._above_since = st._above_since or now
                    if now - st._above_since >= hold:
                        st.active, st.just_on, st._below_since = True, True, None
                else:
                    st._above_since = None
            else:
                if conf < g.off_threshold:
                    st._below_since = st._below_since or now
                    if now - st._below_since >= min(hold, 0.12):
                        st.active, st._above_since = False, None
                else:
                    st._below_since = None
        return self.states

    def reset(self):
        self.states = {k: GestureState() for k in GESTURES}
