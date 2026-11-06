"""R1/R2/R5 projector test patterns. Run: python -m tools.brightness_test [--no-camera] [--selftest]

SPACE next pattern, B back, UP/DOWN brightness level, LEFT/RIGHT size scale, ESC quit.
"""
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools._common import Rig, Quit, PhaseEnd, parser, put, WHITE, YELLOW  # noqa: E402
from hof.config import load_config  # noqa: E402
from hof.render.engine import list_monitors  # noqa: E402


def bgr(rgb01):
    return np.array(rgb01[::-1], np.float32) * 255


def chart(size, cfg, mons, real):
    """R2: 1px / 2px gratings both ways, 1px checkerboard, render size + monitor modes."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    bw = cfg.tools.brightness.grating_px
    cell_w, cell_h = w // 4, h // 3
    x0 = 20
    for r, lw in enumerate(bw):                     # r=0 vertical lines, r=1 horizontal, per width
        for c, vertical in enumerate((True, False)):
            x, y = x0 + c * (cell_w + 20), 20 + r * (cell_h + 20)
            g = np.zeros((cell_h, cell_w), np.uint8)
            n = cell_w if vertical else cell_h
            idx = (np.arange(n) // lw) % 2 == 0
            g[:] = (idx[None, :] if vertical else idx[:, None]) * 255
            img[y:y + cell_h, x:x + cell_w] = g[:, :, None]
            put(img, f"{lw}px {'vertical' if vertical else 'horizontal'}", (x + 4, y + 16), (0, 0, 255), 0.5)
    cx, cy = x0 + 2 * (cell_w + 20), 20
    yy, xx = np.mgrid[0:cell_h * 2 + 20, 0:cell_w]
    img[cy:cy + yy.shape[0], cx:cx + cell_w] = (((xx + yy) % 2) * 255).astype(np.uint8)[:, :, None]
    put(img, "1px checker", (cx + 4, cy + 16), (0, 0, 255), 0.5)
    ty = 20 + 2 * (cell_h + 20) + 10
    put(img, f"render size {w}x{h}   ({'projector' if real else 'STAND-IN WINDOW'})", (x0, ty + 20), WHITE, 0.7)
    for i, m in enumerate(mons):
        modes = ", ".join(f"{a}x{b}" for a, b in m["modes"][-4:])
        put(img, f"monitor {m['index']} {m['name']}{' (primary)' if m['primary'] else ''}: now {m['size'][0]}x{m['size'][1]}"
                 f"@{m['refresh']}  top modes: {modes}", (x0, ty + 50 + 24 * i), YELLOW, 0.5)
    put(img, "1px lines blur to grey => panel is not this resolution: set display.render_size in config.yaml",
        (x0, h - 14), (0, 200, 0), 0.55)
    return img


def disc(size, cfg, scale):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    r = int(size[1] * cfg.tools.brightness.disc_frac * scale)
    cv2.circle(img, (size[0] // 2, size[1] // 2), r, tuple(float(v) for v in bgr(cfg.render.flame.body_color)), -1, cv2.LINE_AA)
    return img


def flame(size, cfg, scale):
    """Static teardrop, radial core -> body -> tip -> black at full intensity."""
    w, h = size
    fh = h * cfg.tools.brightness.flame_frac * scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    up = (h / 2 + fh * 0.35 - yy) / fh                           # 0 at base, 1 at tip
    half_w = fh * 0.3 * np.clip(1 - np.clip(up, 0, 1) ** 1.3, 0.02, 1) * (up > -0.2)
    r = np.sqrt(((xx - w / 2) / np.maximum(half_w, 1e-3)) ** 2 * 0.5 + (np.clip(up - 0.4, -1, 2) / 0.65) ** 2)
    fl = cfg.render.flame
    stops = [(0.0, bgr(fl.core_color)), (0.35, bgr(fl.body_color)), (0.75, bgr(fl.tip_color)), (1.0, np.zeros(3, np.float32))]
    out = np.zeros((h, w, 3), np.float32)
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        t = np.clip((r - a) / (b - a), 0, 1)[..., None]
        m = ((r >= a) & (r < b))[..., None]
        out = np.where(m, ca * (1 - t) + cb * t, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def sparks(size, cfg, scale):
    """spark_count hot dots per quadrant, sizes bc.spark_sizes, static random ring."""
    bc = cfg.tools.brightness
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(bc.seed)
    col = tuple(float(v) for v in bgr(cfg.render.flame.core_color))
    quads = [(w * 0.25, h * 0.25), (w * 0.75, h * 0.25), (w * 0.25, h * 0.75), (w * 0.75, h * 0.75)]
    rad = min(w, h) * 0.22 * scale
    for (cx, cy), sz in zip(quads, bc.spark_sizes):
        a = rng.uniform(0, 2 * np.pi, bc.spark_count)
        r = rad * rng.uniform(*bc.spark_ring_frac, bc.spark_count)
        for x, y in zip(cx + r * np.cos(a), cy + r * np.sin(a)):
            if sz == 1:
                img[int(y) % h, int(x) % w] = col
            else:
                cv2.circle(img, (int(x), int(y)), sz // 2 if sz > 2 else 1, col, -1, cv2.LINE_8)
        put(img, f"{sz}px sparks", (int(cx - 40), int(cy + rad + 24)), (0, 0, 200), 0.5)
    return img


def white(size, cfg, scale):
    return np.full((size[1], size[0], 3), 255, np.uint8)


def ramp(size, cfg, scale):
    n = cfg.tools.brightness.ramp_steps
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    for i in range(n):
        v = round(i * 255 / (n - 1))
        x0, x1 = i * w // n, (i + 1) * w // n
        img[:, x0:x1] = v
        put(img, str(v), (x0 + 8, h // 2), (0, 0, 255) if v > 128 else (0, 200, 255), 0.6)
    put(img, "where do the first bands become distinguishable from black? (black crush)", (10, h - 40), (0, 200, 0), 0.6)
    return img


PATTERNS = [("R2 resolution chart", None), ("R1 palm disc (warm orange)", disc), ("flame blob", flame),
            ("thin sparks: 300 hot dots x 1,2,3,5 px", sparks), ("full white (throw brightness)", white),
            ("grey ramp 16 steps (black crush)", ramp)]
ABSOLUTE = ("R2", "grey")      # level multiplier not applied (these test the panel itself)

INSTRUCTIONS = """
Brightness / resolution test patterns (projector). Keys in either window: SPACE next, B back, UP/DOWN brightness,
LEFT/RIGHT size, ESC quit.

R2 (resolution): chart pattern. If the 1px lines blur to grey, the panel is not that resolution:
   set display.render_size in config.yaml to the real native size.
R1 (do thin visuals read?): disc / flame / sparks. Film your PALM and the BACK OF A HAND wearing a DARK SLEEVE with
   your phone, same exposure. Note which surface reads better, and whether 1-2 px sparks are visible at all
   (if not, set sequence.rangoli_enabled: false and enlarge render.fireworks.size).
R5 (throw): full white + ramp from the actual show position. Write the throw distance, picture size and the
   darkest visible ramp step into DECISIONS.md.
"""


def render(idx, level, scale, size, cfg, mons, real):
    name, fn = PATTERNS[idx]
    img = chart(size, cfg, mons, real) if fn is None else fn(size, cfg, scale)
    if not name.startswith(ABSOLUTE):
        img = (img.astype(np.float32) * level).astype(np.uint8)
    put(img, f"{name}  level {level:.1f}  scale {scale:.2f}", (8, size[1] - 36) if idx == 0 else (8, 18), (90, 90, 90), 0.45)
    return img


def main():
    ap = parser(__doc__)
    ap.add_argument("--no-camera", action="store_true", help="don't open the camera (HUD shows status only)")
    args = ap.parse_args()
    cfg = load_config()
    bc = cfg.tools.brightness
    rig = Rig(cfg, args, "brightness test", camera=not args.no_camera)
    mons = list_monitors()
    print(INSTRUCTIONS)
    idx, level, scale, dwell_t = 0, 1.0, 1.0, time.perf_counter()
    dirty, last_show = True, 0.0
    rig.phase(bc.selftest_dwell_s * (len(PATTERNS) + 2))
    try:
        while True:
            for k in rig.pump():
                if k == "ESCAPE":
                    raise Quit()
                elif k == "SPACE":
                    idx = (idx + 1) % len(PATTERNS)
                elif k == "B":
                    idx = (idx - 1) % len(PATTERNS)
                elif k == "UP":
                    level = min(1.0, level + bc.level_step)
                elif k == "DOWN":
                    level = max(bc.level_min, level - bc.level_step)
                elif k == "RIGHT":
                    scale *= bc.scale_step
                elif k == "LEFT":
                    scale /= bc.scale_step
                dirty = True
            now = time.perf_counter()
            if args.selftest and now - dwell_t > bc.selftest_dwell_s:
                idx, dwell_t, dirty = (idx + 1) % len(PATTERNS), now, True
            if dirty or now - last_show > 1.0:
                rig.project(render(idx, level, scale, rig.size, cfg, mons, rig.real))
                dirty, last_show = False, now
            frame = rig.cam.latest()[0] if rig.cam else None
            mean = f"   camera mean {float(frame.mean()):.0f}/255" if frame is not None else ""
            rig.tick(status=[f"[{idx + 1}/{len(PATTERNS)}] {PATTERNS[idx][0]}",
                             f"level {level:.1f}  scale {scale:.2f}{mean}",
                             "SPACE/B pattern   UP/DOWN brightness   LEFT/RIGHT size   ESC quit"])
            time.sleep(0.005)
    except Quit:
        pass
    finally:
        rig.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
