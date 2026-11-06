"""Camera-to-photon latency. Run: python -m tools.latency_test [--selftest]

Projector goes black, then a big white square appears at t0 (right after the swap). Latency = grab time of the
first camera frame whose ROI brightness crosses midway between the dark and bright baselines, minus t0.
The number includes projector processing, camera exposure + transfer + USB/Continuity delay: exactly what
hand-position prediction has to cover. Saved to latency.file, which Tracker reads.
"""
import json
import os
import random
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools._common import Rig, Quit, PhaseEnd, parser, put, milestone, GREEN, RED, YELLOW  # noqa: E402
from hof.config import load_config, path  # noqa: E402
from hof.mapping import Mapping  # noqa: E402


def square_img(size, frac):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    half = int(size[1] * frac / 2)
    cx, cy = size[0] // 2, size[1] // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), -1)
    return img


def roi_mask(rig, mapping, lc):
    """Camera-space mask of the centre of the white square (inverse homography), else whole frame."""
    cw, ch = rig.cam.size
    mask = np.zeros((ch, cw), np.uint8)
    if not mapping.ok:
        mask[:] = 255
        return mask, False
    w, h = mapping.proj_size
    half = h * lc.square_frac * lc.roi_frac / 2
    corners = np.array([[w / 2 - half, h / 2 - half], [w / 2 + half, h / 2 - half],
                        [w / 2 + half, h / 2 + half], [w / 2 - half, h / 2 + half]], np.float64)
    cam = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.linalg.inv(mapping.H)).reshape(-1, 2)
    cv2.fillPoly(mask, [cam.astype(np.int32)], 255)
    return mask, True


class Meter:
    def __init__(self, rig, mask):
        self.rig, self.mask = rig, mask
        self.seq = 0

    def sample(self, frame):
        return float(np.mean(cv2.mean(frame, mask=self.mask)[:3]))

    def level(self, n):
        """Mean ROI brightness over the next n new frames."""
        vals = []
        while len(vals) < n:
            self.rig.pump_or_quit()
            f, t, s = self.rig.cam.wait_new(self.seq, 0.05)
            if f is None:
                continue
            self.seq = s
            vals.append(self.sample(f))
            self.rig.tick(status=self.rig_status)
        return float(np.mean(vals))

    rig_status = ()


def one_trial(rig, meter, white, black, dark, bright, lc, status):
    """-> latency in ms, or None if the brightness never crossed."""
    rig.project(black)
    rig.wait(lc.settle_s + random.uniform(*lc.gap_ms) / 1000, status=status)   # random gap de-syncs from camera phase
    dark_now = meter.level(lc.baseline_frames)
    thr = dark_now + 0.5 * (bright - dark)
    meter.seq = rig.cam.latest()[2]
    t0 = rig.project(white)
    end = t0 + lc.timeout_s
    while time.perf_counter() < end:
        f, t, s = rig.cam.wait_new(meter.seq, 0.02)
        rig.pump_or_quit()
        if f is None:
            continue
        meter.seq = s
        if t > t0 and meter.sample(f) >= thr:
            return (t - t0) * 1000
    return None


def main():
    ap = parser(__doc__)
    args = ap.parse_args()
    cfg = load_config()
    lc, trials = cfg.tools.latency, cfg.latency.trials
    if args.selftest:
        trials = 3
    rig = Rig(cfg, args, "latency test")
    mapping = Mapping(cfg)
    mask, calibrated = roi_mask(rig, mapping, lc)
    meter = Meter(rig, mask)
    print(f"\nLatency test: {trials} trials, ~1.5 s each. Camera must see the projected square "
          f"({'ROI from calibration' if calibrated else 'NOT calibrated: using the whole camera frame'}). "
          "Keep the room dark and don't wave hands in front of the lens. ESC aborts.")
    rig.phase(cfg.tools.selftest_s * 4)
    white, black = square_img(rig.size, lc.square_frac), rig.black()
    out = 0
    samples = []
    try:
        st = ["Measuring baselines ..."]
        rig.project(white)
        rig.wait(lc.settle_s, status=st)
        bright = meter.level(lc.baseline_frames)
        rig.project(black)
        rig.wait(lc.settle_s, status=st)
        dark = meter.level(lc.baseline_frames)
        print(f"  ROI brightness: dark {dark:.1f}, bright {bright:.1f}")
        if bright - dark < lc.min_contrast:
            print(f"\nFAIL: contrast {bright - dark:.1f} < {lc.min_contrast}. The camera doesn't see the white square. "
                  "Aim the camera at the lit area, darken the room, or run tools.calibrate first so the ROI is right.")
            out = 0 if args.selftest else 1
            if not args.selftest:
                return out
        else:
            fails = 0
            for i in range(trials):
                st = [f"trial {i + 1}/{trials}   valid {len(samples)}   fails {fails}",
                      ("median so far: %.0f ms" % np.median(samples)) if samples else "..."]
                ms = one_trial(rig, meter, white, black, dark, bright, lc, st)
                if ms is None or ms < 0:
                    fails += 1
                else:
                    samples.append(ms)
            print(f"  {len(samples)}/{trials} valid trials")
    except PhaseEnd:
        print("(selftest: time budget reached)")
    except Quit:
        print("Aborted.")
        return 1
    finally:
        rig.project(rig.black())
        fps = rig.cam.fps
        rig.close()
    if len(samples) < lc.min_valid:
        if args.selftest:
            print(f"selftest: {len(samples)} valid trials (not enough to save; expected without a camera aimed at the projector)")
            return 0
        print(f"\nFAIL: only {len(samples)} valid trials (< {lc.min_valid}). Not saved.")
        return 1
    a = np.array(samples)
    res = {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
           "max_ms": float(a.max()), "std_ms": float(a.std()), "mean_ms": float(a.mean()), "n_valid": len(a),
           "trials": trials, "camera_fps": float(fps), "camera": rig.cam.name,
           "roi_calibrated": calibrated, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "samples_ms": [round(float(x), 1) for x in a]}
    print(f"\nLatency (ms): median {res['median_ms']:.0f}   p95 {res['p95_ms']:.0f}   min {res['min_ms']:.0f}   "
          f"max {res['max_ms']:.0f}   std {res['std_ms']:.0f}   | camera {fps:.0f} fps")
    print("Note: this is camera-to-photon (includes projector processing, vsync, camera exposure + transfer): "
          "exactly what hand prediction needs, so don't subtract anything.")
    if args.selftest:
        print("selftest: not writing the latency file")
        return 0
    p = path(cfg.latency.file)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {p}. Tracker uses median_ms for prediction.")
    if res["median_ms"] > lc.r4_fail_ms:
        print(f"\nR4 FAIL: above {lc.r4_fail_ms} ms - turn off projector picture processing/keystone/noise reduction, "
              "try Game mode, then re-run; otherwise slow the choreography.")
    else:
        print(f"\nR4 OK: median {res['median_ms']:.0f} ms is within {lc.r4_fail_ms} ms.")
    milestone(cfg, "latency_measured", f"Camera-to-photon latency measured: median {res['median_ms']:.0f} ms, "
                                       f"p95 {res['p95_ms']:.0f} ms", args.selftest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
