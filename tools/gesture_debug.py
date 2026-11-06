"""Live landmark + gesture overlay on the laptop. Run: python -m tools.gesture_debug [--no-projector] [--selftest]

Fingertip dots are also projected (like calibrate verify) unless --no-projector. ESC quits.
"""
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools._common import (Rig, Quit, PhaseEnd, parser, put, draw_hand_cam, draw_fingertips,  # noqa: E402
                           draw_area_outline, hand_color, WHITE, GREY, GREEN, RED, YELLOW)
from hof.config import load_config  # noqa: E402
from hof.mapping import Mapping  # noqa: E402


def load_classifier(cfg):
    try:
        from hof.gestures import GestureClassifier
        return GestureClassifier(cfg)
    except ImportError:
        print("[gestures] hof.gestures not found: showing landmarks only")
        return None


def draw_panel(img, x0, states, cfg, err):
    """Horizontal confidence bar per gesture; on/off thresholds marked; green when active."""
    g, t = cfg.gestures, cfg.tools
    bw, bh = t.gesture_panel_px - 30, t.gesture_bar_h_px
    y = 30
    put(img, "gestures" if states else "gestures: none (hof.gestures missing)", (x0 + 10, y), WHITE, cfg=cfg)
    if err:
        put(img, err[:40], (x0 + 10, y + 24), RED, 0.45)
    for name, st in (states or {}).items():
        y += bh + 34
        bx = x0 + 10
        put(img, f"{name}" + (f" [{st.hand}]" if st.hand else ""), (bx, y - bh - 4), GREEN if st.active else WHITE, 0.5)
        cv2.rectangle(img, (bx, y - bh), (bx + bw, y), (60, 60, 60), 1)
        fill = int(bw * float(np.clip(st.conf, 0, 1)))
        cv2.rectangle(img, (bx, y - bh), (bx + fill, y), GREEN if st.active else GREY, -1)
        for thr, col in ((g.on_threshold, YELLOW), (g.off_threshold, (255, 160, 60))):
            xx = bx + int(bw * thr)
            cv2.line(img, (xx, y - bh - 3), (xx, y + 3), col, 2)
        put(img, f"{st.conf:.2f}", (bx + bw - 40, y - bh - 4), WHITE, 0.45)
    put(img, f"on {g.on_threshold:.2f} (yellow)   off {g.off_threshold:.2f} (orange)", (x0 + 10, img.shape[0] - 110), GREY, 0.45)


def main():
    ap = parser(__doc__)
    ap.add_argument("--no-projector", action="store_true", help="don't open the projector window")
    args = ap.parse_args()
    cfg = load_config()
    rig = Rig(cfg, args, "gesture debug", projector=not args.no_projector, side_px=cfg.tools.gesture_panel_px)
    mapping = Mapping(cfg)
    if not mapping.ok:
        print("[mapping] no calibration yet (plain scaling): run tools.calibrate for fingertip dots that line up")
    from hof.tracker import Tracker
    tracker = Tracker(cfg, rig.cam, mapping)
    gc = load_classifier(cfg)
    err = ""
    print("Gesture debug: hands left=blue right=orange, arrow=velocity, bars=gesture confidence (green = active). ESC quits.")
    rig.phase()
    try:
        while True:
            if "ESCAPE" in rig.pump():
                raise Quit()
            now = time.perf_counter()
            tf = tracker.frame_at(now, rig.size)
            states = None
            if gc:
                try:
                    states = gc.update(tf, now)
                except Exception as e:      # keep the debug view alive while the classifier is being written
                    err = f"{type(e).__name__}: {e}"
            if rig.proj:
                img = rig.black()
                if cfg.debug.calibrated_outline:
                    draw_area_outline(img, cfg)
                draw_fingertips(img, tf, cfg)
                rig.project(img)

            def overlay(im, s):
                for lab, h in tf.hands.items():
                    col = hand_color(cfg, lab)
                    draw_hand_cam(im, h, s, col)
                    p = (h.palm_cam * s).astype(int)
                    q = ((h.palm_cam + h.vel_cam * cfg.tools.vel_arrow_s) * s).astype(int)
                    cv2.arrowedLine(im, tuple(p), tuple(q), YELLOW, 2, cv2.LINE_AA, tipLength=0.3)
                    put(im, f"{lab} palm {h.size_cam:.0f}px pres {h.presence:.2f} {'TRACKED' if h.tracked else 'HELD'}",
                        (p[0] - 60, p[1] + 30), col, 0.5)
                draw_panel(im, rig.rw, states, cfg, err)
            rig.tick(overlay, [
                f"camera {tf.cam_fps:.0f} fps   tracking {tracker.track_fps:.0f} fps   inference {tf.infer_ms:.1f} ms   "
                f"lead {tf.lead_ms:.0f} ms (latency {tracker.latency_ms:.0f} ms)",
                f"hands: {', '.join(f'{k} (presence {v.presence:.2f})' for k, v in tf.hands.items()) or 'none'}",
                "ESC quit"], frame=tf.image)
            if not rig.proj:
                time.sleep(0.005)
    except Quit:
        pass
    finally:
        tracker.close()
        rig.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
