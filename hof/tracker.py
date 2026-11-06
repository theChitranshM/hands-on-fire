"""MediaPipe Hands on its own thread -> One Euro smoothed landmarks in camera px, stable Left/Right slots.

Prediction happens at render time (frame_at): the render thread asks "where will the hand be when this
frame's photons hit the palm?" = capture time + (now - capture) + measured camera-to-photon latency.
"""
import json
import math
import os
import threading
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from .config import path
from .types import Hand, TrackFrame, WRIST, MIDDLE_MCP

PALM_IDS = [0, 5, 9, 13, 17]


def load_latency(cfg):
    """Measured camera-to-photon latency in ms (median), else config default."""
    p = path(cfg.latency.file)
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return float(d.get("median_ms", cfg.latency.default_ms)), d
    return float(cfg.latency.default_ms), None


class OneEuro:
    """Vectorised One Euro filter (Casiez et al.). x is any-shape array; returns (x_hat, dx_hat per s)."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x, self.dx
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        self.x = a * x + (1 - a) * self.x
        return self.x, self.dx


class _Slot:
    def __init__(self, cfg):
        e = cfg.tracker.one_euro
        self.filt = OneEuro(e.min_cutoff, e.beta, e.d_cutoff)
        self.x = self.dx = None
        self.p_right = 0.5
        self.last_seen = 0.0
        self.t = 0.0
        self.score = 0.0

    @property
    def palm(self):
        return self.x[PALM_IDS, :2].mean(0)


class Tracker:
    def __init__(self, cfg, camera, mapping):
        self.cfg, self.camera, self.mapping = cfg, camera, mapping
        tc = cfg.tracker
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path(tc.model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=tc.num_hands,
            min_hand_detection_confidence=tc.min_detection_confidence,
            min_hand_presence_confidence=tc.min_presence_confidence,
            min_tracking_confidence=tc.min_tracking_confidence)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.latency_ms, _ = load_latency(cfg)
        self.slots = [_Slot(cfg), _Slot(cfg)]
        self.lock = threading.Lock()
        self.image = None
        self.frame_t = 0.0
        self.infer_ms = 0.0
        self.track_fps = 0.0
        self.lost_events = 0          # increments when a tracked hand drops out (for the fail catcher)
        self.running = True
        self._t0_ms = None
        self.th = threading.Thread(target=self._loop, daemon=True, name="tracker")
        self.th.start()

    # ---- tracking thread -------------------------------------------------------------------
    def _loop(self):
        seq, last = 0, None
        while self.running:
            img, t, seq = self.camera.wait_new(seq, timeout=0.5)
            if img is None:
                continue
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ts = int(t * 1000)
            if self._t0_ms is not None and ts <= self._t0_ms:
                ts = self._t0_ms + 1
            self._t0_ms = ts
            res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            infer = (time.perf_counter() - t0) * 1000
            dets = []
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], np.float64)
                lab = hd[0].category_name
                if self.cfg.tracker.swap_handedness:
                    lab = "Left" if lab == "Right" else "Right"
                dets.append((pts, lab, hd[0].score))
            with self.lock:
                self._assign(dets, t, w)
                self.image, self.frame_t = img, t
                self.infer_ms = self.infer_ms * 0.9 + infer * 0.1 if self.infer_ms else infer
                if last:
                    f = 1 / max(t - last, 1e-4)
                    self.track_fps = self.track_fps * 0.9 + f * 0.1 if self.track_fps else f
                last = t

    def _assign(self, dets, t, cam_w):
        tc = self.cfg.tracker
        max_jump = tc.slot_match_px * cam_w
        free = [s for s in self.slots]
        for pts, lab, score in sorted(dets, key=lambda d: -d[2]):
            palm = pts[PALM_IDS, :2].mean(0)
            live = [s for s in free if s.x is not None and np.linalg.norm(s.palm - palm) < max_jump]
            if live:
                slot = min(live, key=lambda s: np.linalg.norm(s.palm - palm))
            else:
                empty = [s for s in free if s.x is None] or free
                slot = min(empty, key=lambda s: s.last_seen)
                if slot.x is not None:
                    slot.__init__(self.cfg)
                slot.p_right = 1.0 if lab == "Right" else 0.0
            free.remove(slot)
            slot.x, slot.dx = slot.filt(pts, t)
            k = tc.handedness_ema
            slot.p_right = slot.p_right * (1 - k) + (1.0 if lab == "Right" else 0.0) * k
            slot.last_seen, slot.t, slot.score = t, t, score
        hold = (tc.loss_hold_ms + tc.loss_fade_ms) / 1000
        for s in self.slots:
            if s.x is not None and t - s.last_seen > hold:
                s.__init__(self.cfg)
                self.lost_events += 1

    # ---- render thread ---------------------------------------------------------------------
    def frame_at(self, now, render_size):
        """TrackFrame for a frame about to be shown at perf_counter time `now`, in projector px."""
        tc = self.cfg.tracker
        with self.lock:
            snap = [(s.x.copy(), s.dx.copy(), s.p_right, s.last_seen, s.score)
                    for s in self.slots if s.x is not None]
            img, ft, infer = self.image, self.frame_t, self.infer_ms
        lead_ms = 0.0
        hands = {}
        labeled = self._labels(snap)
        for (x, dx, _, seen, score), label in zip(snap, labeled):
            age = now - seen
            tracked = age * 1000 <= tc.loss_hold_ms + 1000 / max(self.track_fps, 15)
            if tc.predict and tracked:
                lead = (now - seen) + (self.latency_ms + tc.predict_extra_ms) / 1000
                lead = min(lead, tc.predict_max_ms / 1000)
                xp = x + dx * lead * tc.predict_vel_damping
                lead_ms = max(lead_ms, lead * 1000)
            else:
                xp = x
            fade = (age * 1000 - tc.loss_hold_ms) / tc.loss_fade_ms
            presence = float(np.clip(1.0 - fade, 0.0, 1.0)) if age * 1000 > tc.loss_hold_ms else 1.0
            proj = self.mapping.cam_to_proj(xp[:, :2], render_size)
            palm_cam = xp[PALM_IDS, :2].mean(0)
            palm = proj[PALM_IDS].mean(0)
            vel_cam = dx[PALM_IDS, :2].mean(0) if tracked else np.zeros(2)
            p2 = self.mapping.cam_to_proj(np.array([palm_cam, palm_cam + vel_cam * 0.05]), render_size)
            size_cam = float(np.linalg.norm(xp[WRIST, :2] - xp[MIDDLE_MCP, :2]))
            hands[label] = Hand(label=label, cam=xp, proj=proj, palm_cam=palm_cam, palm=palm,
                                vel_cam=vel_cam, vel=(p2[1] - p2[0]) / 0.05, size_cam=size_cam,
                                size=float(np.linalg.norm(proj[WRIST] - proj[MIDDLE_MCP])),
                                presence=presence, tracked=tracked, score=score)
        return TrackFrame(t=ft, hands=hands, image=img, cam_fps=self.camera.fps, infer_ms=infer, lead_ms=lead_ms)

    @staticmethod
    def _labels(snap):
        """Left/Right per slot; two slots never share a label (stronger evidence wins)."""
        if len(snap) == 1:
            return ["Right" if snap[0][2] >= 0.5 else "Left"]
        if len(snap) == 2:
            a, b = snap[0][2], snap[1][2]
            return ["Right", "Left"] if a >= b else ["Left", "Right"]
        return []

    def reload_latency(self):
        self.latency_ms, _ = load_latency(self.cfg)

    def close(self):
        self.running = False
        self.th.join(timeout=1)
        self.landmarker.close()
