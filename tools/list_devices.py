"""List cameras + monitors and what config would select. Run: python -m tools.list_devices [--selftest]"""
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools._common import parser  # noqa: E402
from hof.camera import list_cameras, resolve_device  # noqa: E402
from hof.config import load_config  # noqa: E402
from hof.render.engine import list_monitors, find_projector_monitor  # noqa: E402
import glfw  # noqa: E402

PROBE_S = 1.0


def probe(cfg, idx):
    """Open camera idx like Camera does for PROBE_S; -> (ok, requested, actual w/h/fps, measured fps)."""
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    c = cfg.camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
    cap.set(cv2.CAP_PROP_FPS, c.fps)
    n, shape, t0 = 0, None, time.perf_counter()
    while time.perf_counter() - t0 < PROBE_S:
        ok, img = cap.read()
        if ok:
            n, shape = n + 1, img.shape
    rep = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return shape, rep, n / PROBE_S


def main():
    parser(__doc__).parse_args()
    cfg = load_config()
    cams = list_cameras()
    print("CAMERAS (AVFoundation order)")
    for i, n in cams or []:
        print(f"  [{i}] {n}")
    if not cams:
        print("  none found (is ffmpeg installed? iPhone near and unlocked?)")
    idx, why = resolve_device(cfg)
    print(f"  config selects: index {idx} ({why}) for camera.device={cfg.camera.device!r}, "
          f"wanting {cfg.camera.width}x{cfg.camera.height}@{cfg.camera.fps}")
    opened = 0
    for i, n in cams:
        r = probe(cfg, i)
        if r is None:
            print(f"  [{i}] {n}: DID NOT OPEN (System Settings > Privacy & Security > Camera: allow your terminal)")
            continue
        shape, rep, fps = r
        opened += shape is not None
        got = f"{shape[1]}x{shape[0]}" if shape is not None else "no frames"
        print(f"  [{i}] {n}: frames {got}, reported {rep[0]:.0f}x{rep[1]:.0f}@{rep[2]:.0f}, measured {fps:.1f} fps")
    print("\nMONITORS")
    mons = list_monitors()
    for m in mons:
        print(f"  [{m['index']}] {m['name']}{' (primary)' if m['primary'] else ''}: current {m['size'][0]}x{m['size'][1]}"
              f"@{m['refresh']}Hz")
        print("      modes: " + ", ".join(f"{a}x{b}" for a, b in m["modes"]))
    sel = find_projector_monitor(cfg.display.projector_monitor)
    if sel is None:
        print(f"  config selects: NO projector (display.projector_monitor={cfg.display.projector_monitor!r}); "
              f"fallback window {'enabled' if cfg.display.fallback_windowed else 'DISABLED -> show will refuse to start'}")
    else:
        mode = glfw.get_video_mode(sel)
        mw, mh = cfg.display.max_render_size
        k = min(1.0, mw / mode.size.width, mh / mode.size.height)
        rs = cfg.display.render_size if cfg.display.render_size != "auto" else \
            (round(mode.size.width * k), round(mode.size.height * k))
        print(f"  config selects: {glfw.get_monitor_name(sel).decode(errors='ignore')} -> render size {rs[0]}x{rs[1]}")
    if cams and not opened:
        print("\nNo camera delivered frames: check camera permission.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
