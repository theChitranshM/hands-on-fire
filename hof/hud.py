"""Operator view on the laptop: camera feed + landmarks, beat, gesture bars, fps, latency, take, coffee."""
import cv2
import numpy as np
from .gestures import GESTURES
from .sequence import BEATS, DONE, HINTS

_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
          (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
_COL = {"Left": (255, 160, 60), "Right": (40, 140, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _txt(img, s, xy, scale=0.55, col=(230, 230, 230), th=1):
    cv2.putText(img, s, xy, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, scale, col, th, cv2.LINE_AA)


def draw_hands(img, tf, outline=None):
    """Landmarks/skeleton in camera px on img (same size as the camera frame)."""
    if outline is not None:
        cv2.polylines(img, [outline.astype(np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA)
    for label, h in tf.hands.items():
        c = _COL.get(label, (200, 200, 200))
        if not h.tracked:
            c = tuple(int(v * 0.4) for v in c)
        p = h.cam[:, :2].astype(int)
        for a, b in _BONES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), c, 2, cv2.LINE_AA)
        for q in p:
            cv2.circle(img, tuple(q), 3, (255, 255, 255), -1, cv2.LINE_AA)
        pc = h.palm_cam.astype(int)
        cv2.arrowedLine(img, tuple(pc), tuple((pc + h.vel_cam * 0.1).astype(int)), (0, 255, 255), 2)
        _txt(img, f"{label} {h.presence:.2f}", (pc[0] - 30, pc[1] - 12), 0.5, c)


def render_hud(size, tf, gestures, seq, engine, stats, mapping=None):
    """Returns a BGR image of `size` for the HUD window."""
    W, H = size
    hud = np.zeros((H, W, 3), np.uint8)
    cam_w = int(W * 0.68)
    if tf.image is not None:
        img = tf.image.copy()
        draw_hands(img, tf, mapping.proj_outline_in_cam() if mapping is not None and mapping.ok else None)
        ch = int(img.shape[0] * cam_w / img.shape[1])
        hud[:min(ch, H), :cam_w] = cv2.resize(img, (cam_w, ch))[:H]
    else:
        _txt(hud, "NO CAMERA FRAME", (30, 60), 1.0, (0, 0, 255), 2)
    x0 = cam_w + 16
    y = 34
    nb = seq._next_index(seq.beat) if seq.beat < DONE else DONE
    _txt(hud, f"BEAT {seq.beat}: {seq.name}", (x0, y), 0.8, (80, 220, 255), 2)
    y += 26
    trig = BEATS[nb][2] or "auto"
    _txt(hud, f"next: {nb} {BEATS[nb][1]}  [{trig}]", (x0, y), 0.5)
    y += 22
    if trig in HINTS and seq.beat < DONE:
        _txt(hud, HINTS[trig], (12, H - 46), 0.9, (80, 220, 255), 2)     # big performer cue
    _txt(hud, f"gestures {'ON' if seq.gestures_enabled else 'OFF (manual)'}", (x0, y), 0.5,
         (120, 255, 120) if seq.gestures_enabled else (0, 140, 255))
    y += 30
    th_on, th_off = stats["on"], stats["off"]
    bw = W - x0 - 20
    for g in GESTURES:
        st = gestures[g]
        col = (60, 220, 60) if st.active else (150, 150, 150)
        cv2.rectangle(hud, (x0, y - 12), (x0 + bw, y + 2), (50, 50, 50), -1)
        cv2.rectangle(hud, (x0, y - 12), (x0 + int(bw * st.conf), y + 2), col, -1)
        for th, c in ((th_on, (0, 200, 255)), (th_off, (0, 90, 160))):
            cv2.line(hud, (x0 + int(bw * th), y - 14), (x0 + int(bw * th), y + 4), c, 1)
        _txt(hud, f"{g}{' <' if g == trig else ''}", (x0 + 4, y), 0.42, (255, 255, 255))
        y += 22
    y += 10
    lines = [f"projector fps {engine.fps:5.1f}   render {engine.size[0]}x{engine.size[1]}",
             f"camera fps {tf.cam_fps:5.1f}   track fps {stats['track_fps']:5.1f}",
             f"inference {tf.infer_ms:5.1f} ms   lead {tf.lead_ms:5.0f} ms",
             f"latency (measured) {stats['latency_ms']:.0f} ms{'' if stats['latency_measured'] else ' DEFAULT'}",
             f"calibration {'OK' if stats['calibrated'] else 'MISSING - run tools/calibrate.py'}",
             f"take #{stats['take']}   coffee x{stats['coffee']}",
             f"audio {'muted' if stats['muted'] else 'on'}   debug {'on' if stats['debug'] else 'off'}"
             + ("   BLACKOUT" if engine.blackout else ""),
             ]
    for s in lines:
        _txt(hud, s, (x0, y), 0.47)
        y += 21
    if stats.get("rehearse"):
        y += 6
        _txt(hud, "REHEARSAL", (x0, y), 0.6, (0, 200, 255), 2)
        y += 22
        for r in stats["rehearse"][-4:]:
            _txt(hud, r, (x0, y), 0.42)
            y += 18
    keys = "SPACE next  B back  R reset  1-0 jump  G gestures  H hud  M mute  D debug  C calib  K coffee  ESC blackout  Q quit"
    _txt(hud, keys, (12, H - 12), 0.45, (180, 180, 180))
    if stats.get("flash"):
        _txt(hud, stats["flash"], (12, H - 84), 0.7, (0, 220, 255), 2)
    return hud
