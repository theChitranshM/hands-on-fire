"""Camera -> projector calibration (target: under 2 minutes). Run: python -m tools.calibrate [--verify] [--manual]

Step A: project dot grid (one row per step, falls back to one dot per step), difference against black, fit homography.
Step B: verify live: dots follow your fingertips; arrows nudge Mapping.offset; ENTER saves.
"""
import json
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools._common import (Rig, Quit, PhaseEnd, parser, put, milestone, draw_hand_cam, draw_fingertips,  # noqa: E402
                           draw_area_outline, hand_color, WHITE, GREY, GREEN, RED, YELLOW)
from hof.config import load_config  # noqa: E402
from hof.mapping import Mapping  # noqa: E402


def grid_points(cfg, size, cols, rows):
    """(rows, cols, 2) projector px, inset by mapping.margin_frac."""
    mx, my = size[0] * cfg.mapping.margin_frac, size[1] * cfg.mapping.margin_frac
    xs, ys = np.linspace(mx, size[0] - mx, cols), np.linspace(my, size[1] - my, rows)
    return np.array([[(x, y) for x in xs] for y in ys])


def dots_img(size, pts, radius):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for x, y in np.asarray(pts).reshape(-1, 2):
        cv2.circle(img, (int(round(x)), int(round(y))), radius, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def find_blobs(base, lit, cc):
    """[(x, y, area)] camera px of bright round blobs in lit - base."""
    d = cv2.GaussianBlur(cv2.absdiff(lit, base), (cc.blur_px, cc.blur_px), 0)
    peak = float(d.max())
    if peak < cc.diff_min:
        return []
    mask = (d >= max(cc.diff_min, cc.diff_frac * peak)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if per <= 0 or not cc.blob_min_area <= area <= cc.blob_max_area:
            continue
        if 4 * np.pi * area / per ** 2 < cc.blob_min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(d[y:y + h, x:x + w] * mask[y:y + h, x:x + w])
        if m["m00"] > 0:
            out.append((x + m["m10"] / m["m00"], y + m["m01"] / m["m00"], area))
    return out


def observe(rig, pts, cc, status):
    """Black frame, then frame with dots at pts (projector px); returns blobs."""
    base = rig.grab_after(rig.project(rig.black()), status)
    lit = rig.grab_after(rig.project(dots_img(rig.size, pts, rig.cfg.mapping.dot_radius_px)), status)
    return find_blobs(base, lit, cc)


def fit(cam_pts, proj_pts, cc):
    """-> (H, mean_err, max_err, n_inliers, n) with errors in projector px over inliers, or None."""
    cam, proj = np.asarray(cam_pts, np.float64), np.asarray(proj_pts, np.float64)
    if len(cam) < 4:
        return None
    H, m = cv2.findHomography(cam, proj, cv2.RANSAC, cc.ransac_px)
    if H is None:
        return None
    inl = m.ravel().astype(bool)
    e = np.linalg.norm(cv2.perspectiveTransform(cam[inl].reshape(-1, 1, 2), H).reshape(-1, 2) - proj[inl], axis=1)
    return H, float(e.mean()), float(e.max()), int(inl.sum()), len(cam)


def orientation(H, cam_size):
    """+1 if H keeps handedness (a normal, un-mirrored view), -1 if it reflects."""
    c = np.array(cam_size, np.float64) / 2
    q = cv2.perspectiveTransform(np.array([c, c + (1, 0), c + (0, 1)]).reshape(-1, 1, 2), H).reshape(-1, 2)
    return 1 if np.cross(q[1] - q[0], q[2] - q[0]) > 0 else -1


def good(res, cc):
    return res is not None and res[3] >= cc.min_inliers and res[1] <= cc.max_err_px


def rows_pass(rig, grid, cc, st):
    rows, cols = grid.shape[:2]
    cam_rows = []
    for r in range(rows):
        b = observe(rig, grid[r], cc, [f"Step A: row {r + 1}/{rows} ...", "hold the white board still"])
        if len(b) > cols:
            b = sorted(b, key=lambda t: -t[2])[:cols]
        if len(b) == cols:
            cam_rows.append(np.array([[x, y] for x, y, _ in sorted(b)]))
        else:
            cam_rows.append(None)
        print(f"  row {r + 1}/{rows}: saw {len(b)}/{cols}")
    best = None
    for flip in (False, True):      # camera may see the projector mirrored/upside-down: x order flips
        cam, proj = [], []
        for r, row in enumerate(cam_rows):
            if row is not None:
                cam += list(row[::-1] if flip else row)
                proj += list(grid[r])
        res = fit(cam, proj, cc)
        # a symmetric grid fits a mirrored assignment just as well; a real camera doesn't mirror (unless configured)
        if res and orientation(res[0], rig.cam.size) == (-1 if rig.cfg.camera.mirror else 1) \
                and (best is None or (res[3], -res[1]) > (best[0][3], -best[0][1])):
            best = (res, cam)
    return best if best else (None, [])


def dots_pass(rig, grid, cc):
    cam, proj, n = [], [], grid.shape[0] * grid.shape[1]
    for i, p in enumerate(grid.reshape(-1, 2)):
        b = observe(rig, [p], cc, [f"Step A: dot {i + 1}/{n} ...", "hold the white board still"])
        if b:
            x, y, _ = max(b, key=lambda t: t[2])
            cam.append((x, y))
            proj.append(p)
    print(f"  per-dot: saw {len(cam)}/{n}")
    return fit(cam, proj, cc), cam


def auto_calibrate(rig, mapping, st):
    cfg = rig.cfg
    cc = cfg.tools.calibrate
    cols, rows = cfg.mapping.grid
    grid = grid_points(cfg, rig.size, cols, rows)
    print("\nStep A (auto): do not move the board. About 10-30 s.")
    seen = observe(rig, grid, cc, ["Step A: all dots ...", "hold the white board still"])
    print(f"  all-at-once: saw {len(seen)}/{cols * rows} dots")
    if len(seen) < cc.min_dots_seen:
        print("  Camera can barely see the projected dots. Check: board in front of the lens+projector at hand "
              "distance, room dark, projector focused, camera aimed at the lit area, exposure not saturated.")
        return False
    res, cam = (None, [])
    if cc.mode == "rows":
        res, cam = rows_pass(rig, grid, cc, st)
        if not good(res, cc):
            print("  row method didn't fit (camera rotated? merged dots?): falling back to one dot at a time")
    if not good(res, cc):
        res, cam = dots_pass(rig, grid, cc)
    st["detected"] = cam
    return finish(rig, mapping, res, cc, st)


def manual_calibrate(rig, mapping, st):
    cc = rig.cfg.tools.calibrate
    cols, rows = cc.manual_grid
    grid = grid_points(rig.cfg, rig.size, cols, rows).reshape(-1, 2)
    print("\nStep A (manual): a dot is projected; CLICK it in the laptop camera window. SPACE skips a dot, ESC quits.")
    rig.enable_clicks()
    cam, proj = [], []
    for i, p in enumerate(grid):
        rig.project(dots_img(rig.size, [p], rig.cfg.mapping.dot_radius_px))
        rig.click = None
        while rig.click is None:
            keys = rig.pump()
            if "ESCAPE" in keys:
                raise Quit()
            if "SPACE" in keys:
                break
            rig.tick(lambda im, s: [cv2.circle(im, (int(x * s), int(y * s)), 6, GREEN, 2) for x, y in cam],
                     [f"MANUAL {i + 1}/{len(grid)}: click the projected dot in this window", "SPACE skip, ESC quit"])
            time.sleep(0.005)
        if rig.click is not None:
            cam.append(rig.click)
            proj.append(p)
    st["detected"] = cam
    return finish(rig, mapping, fit(cam, proj, cc), cc, st)


def finish(rig, mapping, res, cc, st):
    if not good(res, cc):
        n = res[3] if res else 0
        print(f"  Calibration rejected: {n} inliers (need {cc.min_inliers}), "
              f"mean err {res[1]:.1f} px (max {cc.max_err_px})." if res else "  Not enough points for a fit.")
        return False
    H, mean, mx, n_in, n = res
    if rig.selftest:
        print(f"  (selftest: fit OK, {n_in}/{n} points, err {mean:.2f} px; NOT saving)")
        return False
    mapping.offset = np.zeros(2)         # old nudge belonged to the old homography
    mapping.save(H, rig.cam.size, rig.size, mean)
    st["err"] = mean
    print(f"  OK: {n_in}/{n} points used, reprojection error mean {mean:.2f} px, max {mx:.2f} px -> {mapping.file}")
    return True


def saved_err(mapping):
    try:
        with open(mapping.file) as f:
            return json.load(f).get("reproj_err_px")
    except (OSError, ValueError):
        return None


def persist_offset(mapping, err):
    mapping.save_offset()
    if err is not None:      # Mapping.save_offset() rewrites the file with reproj_err_px=None: restore it
        with open(mapping.file) as f:
            d = json.load(f)
        d["reproj_err_px"] = err
        with open(mapping.file, "w") as f:
            json.dump(d, f, indent=2)


def verify(rig, mapping, tracker, st):
    """Returns 'enter' | 'escape' | 'redo'. Dots follow fingertips on the projector."""
    cfg = rig.cfg
    step = cfg.tools.calibrate.nudge_px
    cols, rows = cfg.mapping.grid
    grid = grid_points(cfg, rig.size, cols, rows).reshape(-1, 2) * (np.array(mapping.proj_size or rig.size) / rig.size)
    print("\nStep B (verify): put your hand in the beam at the depth you'll perform at. The big dot+ring should sit on "
          "your RIGHT index fingertip, small dots on the other tips.\n  Arrows nudge (for parallax at hand depth), "
          "ENTER = save and finish, R = redo calibration, ESC = save nudge and quit.")
    tf = None
    while True:
        keys = rig.pump()
        for k in keys:
            d = {"LEFT": (-step, 0), "RIGHT": (step, 0), "UP": (0, -step), "DOWN": (0, step)}.get(k)
            if d:
                mapping.offset = mapping.offset + np.array(d, np.float64)
            elif k == "ENTER":
                return "enter"
            elif k == "ESCAPE":
                return "escape"
            elif k == "R":
                return "redo"
        now = time.perf_counter()
        tf = tracker.frame_at(now, rig.size)
        img = rig.black()
        if cfg.debug.calibrated_outline:
            draw_area_outline(img, cfg)
        draw_fingertips(img, tf, cfg)
        rig.project(img)

        def overlay(im, s):
            if mapping.ok:
                try:
                    inv = np.linalg.inv(mapping.H)
                    gp = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), inv).reshape(-1, 2)
                    for x, y in gp:
                        if np.isfinite(x) and np.isfinite(y):
                            cv2.drawMarker(im, (int(x * s), int(y * s)), GREY, cv2.MARKER_CROSS, 8, 1)
                    o = mapping.proj_outline_in_cam()
                    if o is not None and np.isfinite(o).all():
                        cv2.polylines(im, [(o * s).astype(np.int32)], True, (60, 120, 60), 1, cv2.LINE_AA)
                except np.linalg.LinAlgError:
                    pass
            for x, y in st.get("detected", []):
                cv2.circle(im, (int(x * s), int(y * s)), 5, GREEN, 1, cv2.LINE_AA)
            for lab, h in tf.hands.items():
                draw_hand_cam(im, h, s, hand_color(cfg, lab))
        err = st.get("err", saved_err(mapping))
        rig.tick(overlay, [
            (f"reproj err: {err:.2f} px" if err is not None else "reproj err: n/a", GREEN if err is not None else YELLOW),
            f"offset ({mapping.offset[0]:+.0f},{mapping.offset[1]:+.0f}) px   hands: {','.join(tf.hands) or 'none'}   "
            f"track {tracker.track_fps:.0f} fps",
            ("dot on finger? arrows to nudge, ENTER save, R redo, ESC quit" if mapping.ok
             else "NOT CALIBRATED (plain scaling shown). Press R to calibrate."), ], frame=tf.image)


def step_a(rig, mapping, args, st):
    """Returns True when a homography was saved."""
    mode = "manual" if args.manual else "auto"
    while True:
        rig.phase()
        try:
            ok = (manual_calibrate if mode == "manual" else auto_calibrate)(rig, mapping, st)
        except PhaseEnd:
            print("  (selftest: phase budget reached)")
            return False
        except RuntimeError as e:
            print(f"  {e}")
            ok = False
        if ok or rig.selftest:
            return ok
        rig.project(rig.black())
        print("  Step A failed. In the laptop window: M = manual click mode, R = retry auto, ESC = quit.")
        while True:
            keys = rig.pump()
            if "ESCAPE" in keys:
                raise Quit()
            if "M" in keys or "R" in keys:
                mode = "manual" if "M" in keys else "auto"
                break
            rig.tick(status=[("Calibration FAILED", RED), "M = manual clicks   R = retry auto   ESC = quit"])
            time.sleep(0.01)


def main():
    ap = parser(__doc__)
    ap.add_argument("--verify", action="store_true", help="skip Step A, check the saved calibration")
    ap.add_argument("--manual", action="store_true", help="click each projected dot instead of auto-detect")
    args = ap.parse_args()
    cfg = load_config()
    rig = Rig(cfg, args, "calibrate")
    mapping, st = Mapping(cfg), {}
    print("\nHOLD A MATTE WHITE BOARD (cardboard/paper) where your hands will be: the homography is only valid for "
          "that plane. Dark room, projector focused, board facing both lens and camera.")
    t_start = time.perf_counter()
    code = 0
    try:
        need_a = not args.verify
        if args.verify and not mapping.ok:
            print("No calibration file yet: running Step A first.")
            need_a = True
        tracker = None
        while True:
            if need_a:
                t_a = time.perf_counter()
                ok = step_a(rig, mapping, args, st)
                if ok:
                    print(f"Step A took {time.perf_counter() - t_a:.1f} s")
                need_a = False
            if tracker is None:
                from hof.tracker import Tracker
                tracker = Tracker(cfg, rig.cam, mapping)
            rig.phase()
            try:
                r = verify(rig, mapping, tracker, st)
            except PhaseEnd:
                print("(selftest: verify ran, exiting without saving)")
                break
            if r == "redo":
                need_a = True
                continue
            if args.selftest:
                break
            persist_offset(mapping, st.get("err", saved_err(mapping)))
            if r == "enter":
                print(f"\nCalibration worked. Total {time.perf_counter() - t_start:.0f} s. Offset "
                      f"{mapping.offset.tolist()} saved.")
                milestone(cfg, "calibration_ok", "Calibration worked: projected dot sat on my fingertip")
            else:
                print("Nudge offset saved; no confirmation recorded.")
            break
        if tracker:
            tracker.close()
    except Quit:
        print("Quit.")
    finally:
        rig.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
