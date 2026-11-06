"""Interactive effects preview. Keys: 1 flame, 2 snuff, 3 relight, 4 sparklers, 5 rocket, 6 finale, 7 rangoli, 8 scatter,
9 chakri, 0 anar, P petals, L lanterns, D diya row (light / fade), A aarti trail, I sparkler ignite, ESC quit.
--bench: finale + 2 sparklers for 10s, prints fps, saves frames to --shots.
--bench-diwali: chakri + anar + petals + lanterns + diya row for 10s, prints min/avg fps."""
import argparse
import math
import os
import sys
import time

import cv2
import glfw
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hof.config import load_config
from hof.render.engine import Engine, poll_keys
from hof.render.fireworks import Fireworks
from hof.render.flame import FlameRenderer
from hof.render.particles import ParticleSystem, Sparkler
from hof.render.rangoli import Rangoli
from hof.render.diwali import Anar, AartiTrail, Chakri, DiyaRow, Lanterns, Petals

SHOT_TIMES = (0.8, 1.6, 2.4, 3.4, 4.5, 6.0, 8.0)
PALM = 200.0


def palm_pos(t, W, H):
    """Slow figure-8."""
    return np.array([W * 0.5 + math.sin(t * 0.6) * W * 0.2, H * 0.6 + math.sin(t * 1.2) * H * 0.1], "f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-diwali", action="store_true")
    ap.add_argument("--shots", default="logs/effects_shots")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-vsync", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    bd = a.bench_diwali
    if a.bench or bd or a.no_vsync:
        cfg.display.vsync = False
    keys = []
    eng = Engine(cfg, keys)
    W, H = eng.size
    flames = FlameRenderer(eng.ctx, cfg, eng.size)
    parts = ParticleSystem(eng.ctx, cfg, eng.size)
    fw = Fireworks(eng.ctx, cfg, eng.size, parts)
    rg = Rangoli(eng.ctx, cfg, eng.size, parts)
    sparks = [Sparkler(parts, cfg), Sparkler(parts, cfg)]
    chakri, anar = Chakri(parts, cfg), Anar(parts, cfg)
    petals, lanterns = Petals(eng.ctx, cfg, eng.size), Lanterns(eng.ctx, cfg, eng.size, flames)
    row, aarti = DiyaRow(eng.ctx, cfg, eng.size, flames), AartiTrail(eng.ctx, cfg, eng.size)
    ch_on = an_on = row_on = False
    aarti_t = -1.0
    if bd:
        ch_on = an_on = row_on = True
        row.light()
        petals.shower(n=200, duration=8.0)
        lanterns.release((W * 0.5, H * 0.85), 10)
    lit, inten, flare, spark_on = True, 1.0, 0.0, a.bench
    if a.bench:
        os.makedirs(a.shots, exist_ok=True)
        fw.finale(3.0)
        rg.bloom((W * 0.5, H * 0.62), PALM)
    last = fps_t = time.perf_counter()
    sim, fps_n, maxp = 0.0, 0, 0
    frames, shots, prev = [], list(SHOT_TIMES), [None, None]
    refinale = [3.5, 7.0]
    while not eng.win.should_close:
        now = time.perf_counter()
        dt = min(now - last, 0.05)
        last = now
        sim += dt
        frames.append(now)
        for k in poll_keys(keys):
            if k == "ESCAPE":
                glfw.set_window_should_close(eng.win.win, True)
            elif k == "1":
                lit = True
            elif k == "2" and lit:
                lit = False
                flames.snuff(palm_pos(sim, W, H), PALM)
            elif k == "3":
                lit, flare = True, 1.0
            elif k == "4":
                spark_on = not spark_on
            elif k == "5":
                fw.launch((W / 2, H), (np.random.uniform(0.2, 0.8) * W, H * np.random.uniform(0.2, 0.5)), 0.9)
            elif k == "6":
                fw.finale(3.0)
            elif k == "7":
                rg.bloom(palm_pos(sim, W, H), PALM)
            elif k == "8":
                rg.scatter()
            elif k == "9":
                ch_on = not ch_on
            elif k == "0":
                an_on = not an_on
            elif k == "P":
                petals.shower()
            elif k == "L":
                lanterns.release((W * 0.5, H * 0.85))
            elif k == "D":
                row_on = not row_on
                row.light() if row_on else row.set_on(False)
            elif k == "A":
                aarti_t = 0.0
            elif k == "I":
                sparks[0].ignite()
        if a.bench and not bd and refinale and sim >= refinale[0]:
            refinale.pop(0)
            fw.finale(3.0)
        inten += ((1.0 if lit else 0.0) - inten) * min(1.0, dt * 6)
        flare = max(0.0, flare - dt * 1.6)
        pp = palm_pos(sim, W, H)
        rg.set_center(pp)
        for i, s in enumerate(sparks):
            p = np.array([W * (0.25 + 0.5 * i) + math.sin(sim * 1.3 + i * 2) * W * 0.18,
                          H * 0.62 + math.sin(sim * 2.1 + i) * H * 0.12], "f4")
            v = (p - prev[i]) / max(dt, 1e-3) if prev[i] is not None else np.zeros(2, "f4")
            prev[i] = p
            s.update(dt, p, v, spark_on)
        chakri.update(dt, (W * 0.25, H * 0.62), ch_on, 1.0, PALM)
        anar.update(dt, (W * 0.75, H * 0.8), an_on, 1.0, PALM)
        petals.update(dt)
        lanterns.update(dt)
        row.update(dt)
        if aarti_t >= 0:
            aarti_t += dt
            ap_ = np.array([W * 0.5 + 170 * math.cos(aarti_t * 5.5), H * 0.45 + 170 * math.sin(aarti_t * 5.5)])
            aarti.add(ap_, aarti_t)
            if aarti_t > 3.0:
                aarti.stop()
                aarti_t = -1.0
        aarti.update(dt)
        flames.update(dt, sim)
        fw.update(dt, sim)
        rg.update(dt, sim)
        parts.update(dt)
        maxp = max(maxp, parts.count)
        eng.begin_trail()
        parts.render()
        eng.begin_direct()
        rg.render()
        chakri.render_core(flames)
        anar.render_core(flames)
        row.draw()
        petals.render()
        lanterns.render()
        aarti.render()
        flames.add(pp, PALM, inten, flare, True, 0.0)
        flames.add((W * 0.12, H * 0.82), 170, 1.0, 0.0, True, 3.0)
        for s in sparks:
            s.render_core(flames)
        flames.render()
        eng.present()
        if a.bench and not bd and shots and sim >= shots[0]:
            img = eng.grab((640, 360))
            cv2.imwrite(os.path.join(a.shots, f"shot_{shots.pop(0):.1f}.png"), np.ascontiguousarray(img[:, :, ::-1]))
        fps_n += 1
        if now - fps_t >= 2.0:
            print(f"fps {fps_n / (now - fps_t):.1f}  particles {parts.count}", flush=True)
            fps_t, fps_n = now, 0
        if (a.bench or bd) and sim >= a.seconds:
            break
    if a.bench or bd:
        d = np.diff(frames[10:])
        print(f"{'BENCH-DIWALI' if bd else 'BENCH'} min_fps {1 / d.max():.1f}  avg_fps {len(d) / d.sum():.1f}  max_particles {maxp}")


if __name__ == "__main__":
    main()
