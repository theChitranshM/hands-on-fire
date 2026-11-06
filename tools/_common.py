"""Shared plumbing for the setup/test tools: args, camera + projector + laptop HUD windows, drawing."""
import argparse
import os
import sys
import time
import cv2
import glfw
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hof.camera import Camera  # noqa: E402
from hof.render.engine import ImageWindow, open_projector, poll_keys  # noqa: E402
from hof.types import TIPS, THUMB_TIP, INDEX_TIP  # noqa: E402

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
WHITE, GREY, GREEN, RED, YELLOW = (255, 255, 255), (150, 150, 150), (80, 220, 80), (70, 70, 255), (0, 220, 255)


class Quit(Exception):
    """ESC / window closed."""


class PhaseEnd(Quit):
    """--selftest phase budget used up."""


def parser(desc):
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run a few seconds with no input, then exit 0")
    return ap


def put(img, s, org, color=WHITE, scale=None, thick=None, cfg=None):
    sc = scale or (cfg.tools.font_scale if cfg else 0.6)
    th = thick or (cfg.tools.font_thick if cfg else 1)
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def open_camera(cfg):
    try:
        cam = Camera(cfg)
    except RuntimeError as e:
        print(f"\nCAMERA ERROR: {e}\n")
        sys.exit(1)
    f, _, _ = cam.wait_new(0, timeout=cfg.tools.first_frame_s)
    if f is None:
        cam.close()
        print("\nCAMERA ERROR: opened but no frames arrived. Usually missing permission: System Settings > "
              "Privacy & Security > Camera > allow your terminal, then re-run. (Or the iPhone is asleep.)\n")
        sys.exit(1)
    print(f"[camera] {cam.name} (index {cam.index}) {f.shape[1]}x{f.shape[0]}")
    return cam


def milestone(cfg, name, text, selftest=False):
    """Log to captures/build_log.md through hof.recorder, if that module exists yet."""
    if selftest:
        return
    try:
        from hof.recorder import Recorder
    except ImportError:
        print(f"[recorder] hof.recorder not present; milestone '{name}' not logged")
        return
    try:
        r = Recorder(cfg)
        r.milestone(name, text)
        r.close()
        print(f"[recorder] milestone '{name}' logged")
    except Exception as e:  # never let logging break a tool
        print(f"[recorder] milestone failed: {e}")


class Rig:
    """Camera + projector ImageWindow + laptop HUD ImageWindow, one key queue, event pumping."""

    def __init__(self, cfg, args, title, camera=True, projector=True, side_px=0):
        self.cfg, self.selftest = cfg, args.selftest
        self.keys, self.backlog, self.click = [], [], None
        self._deadline, self._last_hud = None, 0.0
        self.cam = open_camera(cfg) if camera else None
        self.proj, self.real = None, False
        self.size = tuple(cfg.display.max_render_size)
        if projector:
            try:
                self.proj, self.size, self.real = open_projector(cfg, self.keys, image_window=True)
            except RuntimeError as e:
                print(f"\nPROJECTOR ERROR: {e}\n")
                sys.exit(1)
            if not self.real:
                glfw.set_window_pos(self.proj.win, *cfg.tools.fallback_proj_pos)
                print("\n" + "!" * 72 + "\n!! NO PROJECTOR FOUND: using a window on the Mac screen as a stand-in.\n"
                      "!! Plug in HDMI and set System Settings > Displays to EXTEND (not mirror).\n" + "!" * 72 + "\n")
            print(f"[projector] render size {self.size[0]}x{self.size[1]} ({'real' if self.real else 'STAND-IN WINDOW'})")
        cw, ch = cfg.display.hud_size
        self.cw, self.ch, self.side = cw, ch, side_px
        cam_w, cam_h = self.cam.size if self.cam else (16, 9)
        avail_w, avail_h = cw - side_px, ch - cfg.tools.hud_bar_px
        self.s = min(avail_w / cam_w, avail_h / cam_h)
        self.rw, self.rh = int(cam_w * self.s), int(cam_h * self.s)
        self.hud = ImageWindow(f"Hands on Fire - {title}", (cw, ch), key_queue=self.keys, vsync=False)
        self.hud_fps = cfg.display.hud_fps
        self.latency_s = None

    # ---- events -----------------------------------------------------------------------------
    def phase(self, seconds=None):
        """--selftest only: raise PhaseEnd from pump() after `seconds` (default tools.selftest_s)."""
        self._deadline = time.perf_counter() + (seconds or self.cfg.tools.selftest_s) if self.selftest else None

    def _poll(self):
        keys = poll_keys(self.keys)
        if self.hud.should_close or (self.proj and self.proj.should_close):
            keys.append("ESCAPE")
        if self._deadline and time.perf_counter() > self._deadline:
            raise PhaseEnd()
        return keys

    def pump(self):
        keys = self.backlog + self._poll()
        self.backlog = []
        return keys

    def pump_or_quit(self):
        """For waits inside a capture: ESC aborts, other keys are kept for the next pump()."""
        self.backlog += self._poll()
        if "ESCAPE" in self.backlog:
            raise Quit()

    # ---- projector ----------------------------------------------------------------------------
    def black(self):
        return np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def project(self, img):
        """Show on the projector; returns perf_counter right after the swap (t0 for latency maths)."""
        if self.proj is None:
            return time.perf_counter()
        self.proj.show(img)
        return time.perf_counter()

    # ---- laptop HUD ---------------------------------------------------------------------------
    def compose(self, frame=None, overlay=None, status=()):
        img = np.zeros((self.ch, self.cw, 3), np.uint8)
        if frame is not None:
            img[:self.rh, :self.rw] = cv2.resize(frame, (self.rw, self.rh))
        if overlay:
            overlay(img, self.s)
        y0 = self.ch - self.cfg.tools.hud_bar_px + self.cfg.tools.hud_line_px
        for i, line in enumerate(status):
            s, col = line if isinstance(line, tuple) else (line, WHITE)
            put(img, s, (10, y0 + i * self.cfg.tools.hud_line_px), col, cfg=self.cfg)
        return img

    def tick(self, overlay=None, status=(), frame=None, force=False):
        """Refresh the laptop preview at display.hud_fps."""
        now = time.perf_counter()
        if not force and now - self._last_hud < 1.0 / self.hud_fps:
            return
        self._last_hud = now
        if frame is None and self.cam:
            frame = self.cam.latest()[0]
        self.hud.show(self.compose(frame, overlay, status))

    def win_to_cam(self, wx, wy):
        ww, wh = glfw.get_window_size(self.hud.win)
        return wx * self.cw / ww / self.s, wy * self.ch / wh / self.s

    def enable_clicks(self):
        def cb(win, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
                self.click = self.win_to_cam(*glfw.get_cursor_pos(win))
        glfw.set_mouse_button_callback(self.hud.win, cb)

    def latency(self):
        if self.latency_s is None:
            from hof.tracker import load_latency
            self.latency_s = load_latency(self.cfg)[0] / 1000
        return self.latency_s

    # ---- camera capture synchronised to a projection ---------------------------------------------
    def grab_after(self, t_show, status=(), overlay=None):
        """Mean grey (float32) of avg_frames camera frames that were grabbed >= latency + settle_frames after t_show."""
        cc = self.cfg.tools.calibrate
        end = time.perf_counter() + cc.capture_timeout_s
        seq, skip, grays = 0, cc.settle_frames, []
        while len(grays) < cc.avg_frames:
            if time.perf_counter() > end:
                raise RuntimeError("camera stopped delivering frames")
            self.pump_or_quit()
            self.tick(overlay, status)
            f, t, s = self.cam.wait_new(seq, 0.02)
            if f is None:
                continue
            seq = s
            if t < t_show + self.latency():
                continue
            if skip > 0:
                skip -= 1
                continue
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
        return np.mean(grays, axis=0)

    def wait(self, seconds, overlay=None, status=()):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.pump_or_quit()
            self.tick(overlay, status)
            time.sleep(0.002)

    def close(self):
        if self.cam:
            self.cam.close()


# ---- drawing --------------------------------------------------------------------------------------
def hand_color(cfg, label):
    return tuple(int(v) for v in cfg.tools.hand_colors[label])


def draw_hand_cam(img, hand, s, color):
    """Skeleton + 21 landmarks of a Hand on the HUD (cam px * s)."""
    p = (hand.cam[:, :2] * s).astype(int)
    col = tuple(int(c * (0.35 + 0.65 * hand.presence)) for c in color)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(p[a]), tuple(p[b]), col, 2, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, tuple(q), 3, WHITE if hand.tracked else GREY, -1, cv2.LINE_AA)


def draw_area_outline(img, cfg):
    h, w = img.shape[:2]
    mx, my = int(w * cfg.mapping.margin_frac), int(h * cfg.mapping.margin_frac)
    v = cfg.tools.outline_level
    cv2.rectangle(img, (mx, my), (w - mx, h - my), (v, v, v), 1)


def draw_fingertips(img, tf, cfg):
    """Projector content: ring + big dot on right index tip, small dots on every fingertip."""
    t = cfg.tools
    hands = tf.hands
    for lab, h in hands.items():
        k = float(h.presence)
        col = tuple(int(255 * k) for _ in range(3))
        for i in TIPS + (THUMB_TIP,):
            if i == INDEX_TIP and lab == "Right":
                continue
            cv2.circle(img, tuple(int(v) for v in h.proj[i]), t.tip_dot_radius_px, col, -1, cv2.LINE_AA)
    main = hands.get("Right") or next(iter(hands.values()), None)
    if main is not None:
        c = tuple(int(v) for v in main.proj[INDEX_TIP])
        k = float(main.presence)
        cv2.circle(img, c, t.main_dot_radius_px, tuple(int(255 * k) for _ in range(3)), -1, cv2.LINE_AA)
        cv2.circle(img, c, t.ring_radius_px, (0, int(160 * k), int(255 * k)), t.ring_thick_px, cv2.LINE_AA)
