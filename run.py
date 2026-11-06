"""Hands on Fire — show mode.

  python run.py              show
  python run.py --rehearse   loop the sequence, log missed gestures, summary after every run
  python run.py --sim        no camera: a fake pair of hands follows the mouse on the projector window
  python run.py --mute       no sound
"""
import argparse
import time
import numpy as np
import glfw

from hof.config import load_config
from hof.render.engine import Engine, ImageWindow, poll_keys
from hof.render.flame import FlameRenderer
from hof.render.particles import ParticleSystem, Sparkler
from hof.render.fireworks import Fireworks
from hof.render.rangoli import Rangoli
from hof.gestures import GestureClassifier
from hof.sequence import Sequence, DONE, BEATS
from hof.mapping import Mapping
from hof.tracker import load_latency
from hof.types import Hand, TrackFrame, TIPS, THUMB_TIP
from hof.hud import render_hud, draw_hands


class SimTracker:
    """Mouse-driven stand-in for the camera tracker: right hand at the cursor, left hand mirrored."""

    # open-hand landmark template in palm sizes, palm centre at the origin, fingers up
    _T = np.array([[0, 0.9], [-0.45, 0.65], [-0.75, 0.35], [-0.95, 0.05], [-1.1, -0.2], [-0.35, -0.1],
                   [-0.42, -0.6], [-0.46, -0.95], [-0.5, -1.25], [0, -0.15], [0, -0.7], [0, -1.1], [0, -1.4],
                   [0.33, -0.1], [0.38, -0.62], [0.41, -0.98], [0.44, -1.28], [0.62, 0.0], [0.7, -0.4],
                   [0.75, -0.68], [0.8, -0.92]], np.float64)

    def __init__(self, cfg, engine):
        self.engine, self.cfg = engine, cfg
        self.prev, self.prev_t = None, None
        self.latency_ms, self.infer_ms, self.track_fps, self.lost_events = 0.0, 0.0, 60.0, 0

    def frame_at(self, now, render_size):
        win = self.engine.win.win
        mx, my = glfw.get_cursor_pos(win)
        ww, wh = glfw.get_window_size(win)
        p = np.array([mx / max(ww, 1) * render_size[0], my / max(wh, 1) * render_size[1]])
        vel = np.zeros(2) if self.prev is None else (p - self.prev) / max(now - self.prev_t, 1e-3)
        self.prev, self.prev_t = p, now
        size = render_size[1] * 0.11
        hands = {}
        for label, palm in (("Right", p), ("Left", np.array([render_size[0] - p[0], p[1]]))):
            t = self._T * [1 if label == "Right" else -1, 1]
            pts = palm + t * size
            cam = np.c_[pts, np.zeros(21)]
            hands[label] = Hand(label=label, cam=cam, proj=pts, palm_cam=palm, palm=palm, vel_cam=vel, vel=vel,
                                size_cam=size, size=size)
        return TrackFrame(t=now, hands=hands, image=None, cam_fps=0.0)

    def reload_latency(self):
        pass

    def close(self):
        pass


def _place_windows(engine, hud_win):
    """No projector: both windows share the laptop screen, so tile them instead of stacking them."""
    if engine.is_projector:
        return
    x, y, w, h = glfw.get_monitor_workarea(glfw.get_primary_monitor())
    sw = w // 2 - 8
    glfw.set_window_size(engine.win.win, sw, sw * 9 // 16)
    glfw.set_window_pos(engine.win.win, x, y + 28)
    hw = w - sw - 8
    glfw.set_window_size(hud_win.win, hw, min(h - 28, hw * 19 // 32))
    glfw.set_window_pos(hud_win.win, x + sw + 8, y + 28)
    glfw.poll_events()
    engine.win.fb_size = glfw.get_framebuffer_size(engine.win.win)


def debug_overlay(engine, cfg, tf, mapping, calib_check):
    W, H = engine.size
    if cfg.debug.calibrated_outline:
        m = cfg.mapping.margin_frac
        engine.line([(W * m, H * m), (W * (1 - m), H * m), (W * (1 - m), H * (1 - m)), (W * m, H * (1 - m))],
                    (0.25, 0.25, 0.25, 1), closed=True)
    if cfg.debug.safe_zone_9x16:
        sw = H * 9 / 16
        x = (W - sw) / 2
        engine.line([(x, 0), (x + sw, 0), (x + sw, H - 1), (x, H - 1)], (0.0, 0.3, 0.4, 1), closed=True)
    for h in tf.hands.values():
        for i in TIPS + (THUMB_TIP,):
            c = h.proj[i]
            r = 9 if calib_check else 5
            engine.line([(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in np.linspace(0, 6.283, 13)],
                        (1, 1, 1, 1) if calib_check else (0.4, 0.6, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--sim", action="store_true", help="no camera; fake hands follow the mouse")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--hud", action="store_true", help="open the operator view at start (H toggles it)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--record-sim", action="store_true", help="testing only: run the footage logger in --sim")
    ap.add_argument("--seconds", type=float, default=0, help="auto-quit after N seconds (testing)")
    ap.add_argument("--autoplay", type=float, default=0, help="advance one beat every N seconds (demo/testing)")
    ap.add_argument("--shots", default="", help="save a projector frame per beat into this folder (testing)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    keys = []
    engine = Engine(cfg, keys)
    if not engine.is_projector:
        print("[display] No projector found: rendering in a window on the Mac screen. "
              "Plug in HDMI and set System Settings > Displays to Extend (not Mirror).")
    # operator view is optional: H shows/hides it, so normally only the show window is open
    hud_on = bool(args.hud or cfg.display.hud_on_start)
    hud_win = ImageWindow("Hands on Fire — operator", cfg.display.hud_size, key_queue=keys, vsync=False,
                          visible=hud_on)
    _place_windows(engine, hud_win)
    engine.win.use()
    mapping = Mapping(cfg)
    camera = None
    if args.sim:
        tracker = SimTracker(cfg, engine)
    else:
        from hof.camera import Camera
        from hof.tracker import Tracker
        camera = Camera(cfg)
        print(f"[camera] {camera.name} (index {camera.index})")
        tracker = Tracker(cfg, camera, mapping)
    if not mapping.ok and not args.sim:
        print("[mapping] No calibration yet: effects will be scaled, not mapped. Run tools/calibrate.py")

    engine.win.use()
    ctx = engine.ctx
    particles = ParticleSystem(ctx, cfg, engine.size)
    fx = {"particles": particles,
          "flames": FlameRenderer(ctx, cfg, engine.size),
          "sparklers": {"Left": Sparkler(particles, cfg), "Right": Sparkler(particles, cfg)},
          "fireworks": Fireworks(ctx, cfg, engine.size, particles),
          "rangoli": Rangoli(ctx, cfg, engine.size, particles)}

    audio = None
    try:
        from hof.audio import Audio
        audio = Audio(cfg, muted=args.mute or not cfg.audio.enabled)
        fw = fx["fireworks"]
        fw.on_launch = lambda *a: audio.play("whoosh")
        fw.on_burst = lambda *a: audio.play("boom")
        fw.on_crackle = lambda *a: audio.play("crackle", 0.8)
    except Exception as e:  # sound is never allowed to stop the show
        print(f"[audio] disabled: {e}")
    recorder = None
    if cfg.recorder.enabled and not args.no_record and (not args.sim or args.record_sim):  # sim runs are not real takes
        from hof.recorder import Recorder
        recorder = Recorder(cfg)

    seq = Sequence(cfg, engine, fx, audio, recorder, rehearse=args.rehearse)
    if args.sim:
        seq.gestures_enabled = False
    gc = GestureClassifier(cfg)
    rehearse_lines = []
    restart_at = [None]

    def on_done(log, completed):
        how = {e["beat"]: e["how"] for e in log}
        missed = [f"{b} {BEATS[b][1]}" for b, h in how.items() if h == "missed"]
        manual = [f"{b} {BEATS[b][1]}" for b, h in how.items() if h == "manual"]
        line = (f"take {recorder.take_number if recorder else '?'}: {'complete' if completed else 'aborted'}, "
                f"manual {len(manual)}, missed {len(missed)}")
        print("\n=== RUN SUMMARY ===\n" + line)
        for e in log:
            print(f"  {e['t']:6.2f}s  beat {e['beat']:2d} {e['name']:<12} {e['how']}")
        if missed:
            print("  missed gestures: " + ", ".join(missed))
        rehearse_lines.append(line)
        if args.rehearse:
            restart_at[0] = time.perf_counter() + 3.0

    seq.on_done = on_done

    latency_ms, latency_info = load_latency(cfg)
    last = time.perf_counter()
    start = last
    last_hud = 0.0
    last_push = 0.0
    low_fps_since = None
    lost_seen = tracker.lost_events
    calib_check = False
    flash, flash_until = "", 0.0
    plumbing_logged = False
    shot_done = set()

    def say(msg):
        nonlocal flash, flash_until
        flash, flash_until = msg, time.perf_counter() + 2.0
        print(msg)

    try:
        while not engine.win.should_close and not (hud_on and hud_win.should_close):
            for k in poll_keys(keys):
                if k == "SPACE":
                    seq.advance("manual")
                elif k == "B":
                    seq.back()
                elif k == "R":
                    seq.reset()
                elif k.isdigit():
                    seq.goto(int(k) or 10, "manual")
                elif k == "G":
                    seq.gestures_enabled = not seq.gestures_enabled
                    say(f"gesture triggers {'ON' if seq.gestures_enabled else 'OFF'}")
                elif k == "M" and audio:
                    audio.set_muted(not audio.muted)
                elif k == "D":
                    cfg.debug.show_overlay = not cfg.debug.show_overlay
                elif k == "C":
                    calib_check = not calib_check
                    say("calibration check: white rings should sit on your fingertips" if calib_check else "")
                elif k == "K" and recorder:
                    say(f"coffee #{recorder.coffee()} logged")
                elif k == "ESCAPE":
                    engine.blackout = not engine.blackout
                elif k == "H":
                    hud_on = not hud_on
                    (glfw.show_window if hud_on else glfw.hide_window)(hud_win.win)
                    if not hud_on:
                        glfw.focus_window(engine.win.win)
                elif k == "Q":
                    raise KeyboardInterrupt
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            if restart_at[0] and now >= restart_at[0]:
                restart_at[0] = None
                seq.reset()
            if seq.beat == DONE and not args.rehearse and now - seq.entered > cfg.sequence.quiet_fade_s:
                pass  # stays black until R / SPACE / a number

            if args.autoplay and seq.beat not in (6, 10) and now - seq.entered > args.autoplay and seq.beat < DONE:
                seq.advance("manual")
            tf = tracker.frame_at(now + cfg.tracker.predict_extra_ms / 1000, engine.size)
            gestures = gc.update(tf, now)
            seq.update(tf, gestures, now, dt)

            engine.begin_trail()
            seq.draw_trail()
            engine.begin_direct()
            seq.draw_direct()
            if cfg.debug.show_overlay or calib_check:
                debug_overlay(engine, cfg, tf, mapping, calib_check)
            engine.present()
            if args.shots and now - seq.entered > 0.9 * max(args.autoplay, 1) and seq.beat not in shot_done:
                import cv2, os
                shot_done.add(seq.beat)
                os.makedirs(args.shots, exist_ok=True)
                cv2.imwrite(os.path.join(args.shots, f"beat{seq.beat:02d}.png"),
                            engine.grab((640, 360))[:, :, ::-1])

            # ---- footage logger + fail catcher ----
            in_show = 1 <= seq.beat < DONE
            if recorder:
                if now - last_push >= 1.0 / cfg.recorder.record_fps:
                    last_push = now
                    cam = tf.image
                    if cam is not None and tf.hands:      # burn the tracking skeleton into the footage: shows what failed
                        cam = cam.copy()
                        draw_hands(cam, tf)
                    recorder.push(engine.grab(tuple(cfg.recorder.proj_size)), cam, now)
                if in_show and engine.fps < cfg.recorder.fps_glitch_below and now - start > 3:
                    low_fps_since = low_fps_since or now
                    if now - low_fps_since > 0.5:
                        recorder.glitch("low_fps", f"{engine.fps:.0f} fps in beat {seq.beat}")
                        low_fps_since = None
                else:
                    low_fps_since = None
                if tracker.lost_events != lost_seen:
                    lost_seen = tracker.lost_events
                    if in_show and seq.beat not in (7, 9):
                        recorder.glitch("tracking_lost", f"beat {seq.beat} {seq.name}")
                if not plumbing_logged and engine.is_projector and tf.image is not None:
                    plumbing_logged = True
                    recorder.milestone("plumbing", "Projector and camera both live from the Mac")

            if hud_on and now - last_hud >= 1.0 / cfg.display.hud_fps:
                last_hud = now
                stats = {"on": cfg.gestures.on_threshold, "off": cfg.gestures.off_threshold,
                         "track_fps": tracker.track_fps, "latency_ms": tracker.latency_ms,
                         "latency_measured": latency_info is not None, "calibrated": mapping.ok,
                         "take": recorder.take_number if recorder else 0,
                         "coffee": recorder.coffee_count if recorder else 0,
                         "muted": audio.muted if audio else True, "debug": cfg.debug.show_overlay,
                         "rehearse": rehearse_lines if args.rehearse else None,
                         "flash": flash if now < flash_until else ""}
                hud_win.show(render_hud(tuple(cfg.display.hud_size), tf, gestures, seq, engine, stats, mapping))
            if args.seconds and now - start > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if seq.take_start is not None:
            seq.reset()
        if args.rehearse and seq.runs:
            print("\n=== REHEARSAL TOTAL ===")
            done = [r for r in seq.runs if r["completed"]]
            print(f"runs {len(seq.runs)}, completed {len(done)}, "
                  f"<=1 manual trigger: {sum(1 for r in done if r['manual'] <= 1)}")
        tracker.close()
        if camera:
            camera.close()
        if audio:
            audio.close()
        if recorder:
            recorder.close()
            from hof.recorder import write_story_material
            print(f"[story] {write_story_material(cfg)}")


if __name__ == "__main__":
    main()
