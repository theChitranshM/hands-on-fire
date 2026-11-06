"""Footage logger: ring buffer of combined (camera | projector) frames, glitch clips, take videos, story logs.
All numbers come from real events; nothing is invented. Heavy work runs on background threads."""
import collections
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from .config import ROOT, load_config, path

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _jload(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _jsave(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)


class Recorder:
    def __init__(self, cfg):
        self.cfg, rc = cfg, cfg.recorder
        self.enabled = bool(rc.enabled)
        self.dir = path(rc.dir)
        self.gdir, self.tdir = os.path.join(self.dir, "glitches"), os.path.join(self.dir, "takes")
        for d in (self.dir, self.gdir, self.tdir):
            os.makedirs(d, exist_ok=True)
        self._state_p = os.path.join(self.dir, "state.json")
        self._lock = threading.Lock()
        self._bg, self._bg_lock = [], threading.Lock()
        self._state = _jload(self._state_p, {})
        self._state.setdefault("take", 0)
        self._state.setdefault("milestones", [])
        self._take, self._beat, self._take_t0, self._take_start_iso = self._state["take"], "", None, None
        self._next, self._last_glitch = 0.0, -1e9
        self._period = 1.0 / rc.record_fps
        self.dropped = 0
        self._ring = collections.deque(maxlen=max(2, int(rc.ring_seconds * rc.record_fps)))
        self._enc = [cv2.IMWRITE_JPEG_QUALITY, rc.ring_jpeg_quality]
        self._q = queue.Queue(rc.queue_max)
        self._cmdq = queue.Queue()                # commands are unbounded: they must never be dropped
        self._vw, self._vw_raw, self._vw_final = None, None, None
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True, name="recorder")
        if self.enabled:
            self._th.start()

    # ---- main-thread API -------------------------------------------------
    @property
    def take_number(self):
        return self._take

    @property
    def coffee_count(self):
        try:
            with open(os.path.join(self.dir, "coffee.log")) as f:
                return sum(1 for ln in f if ln.strip())
        except OSError:
            return 0

    def set_status(self, take, beat_name):
        self._take, self._beat = take, beat_name

    def push(self, proj_rgb, cam_bgr, t):
        now = time.perf_counter()
        if not self.enabled or now < self._next - self._period * 0.25:   # accept slightly-early ticks: 60 Hz -> 30 fps
            return
        if now - self._next > self._period:
            self._next = now
        self._next += self._period
        item = (proj_rgb.copy(), None if cam_bgr is None else cam_bgr.copy(), t, self._take, self._beat, time.time())
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def glitch(self, reason, detail=""):
        """Save the last ring_seconds as a clip. Returns the clip path, or None if rate-limited/empty."""
        now = time.perf_counter()
        if not self.enabled or now - self._last_glitch < self.cfg.recorder.glitch_min_gap_s:
            return None
        self._last_glitch = now
        ts = datetime.now()
        p = os.path.join(self.gdir, f"take{self._take:03d}_{ts:%Y%m%d-%H%M%S}_{reason}.mp4")
        self._cmdq.put(("glitch", p, reason, detail, self._take, ts.isoformat(timespec="seconds")))
        return p

    def start_take(self):
        with self._lock:
            if self._take_t0 is not None:
                self.end_take([], False)
            self._state["take"] += 1
            self._take = n = self._state["take"]
            _jsave(self._state_p, self._state)
        ts = datetime.now()
        self._take_t0, self._take_start_iso = time.time(), ts.isoformat(timespec="seconds")
        self._take_path = os.path.join(self.tdir, f"take{n:03d}_{ts:%Y%m%d-%H%M%S}.mp4")
        self._cmdq.put(("start", self._take_path))
        return n

    def end_take(self, beats, completed):
        if self._take_t0 is None:
            return
        dur = round(time.time() - self._take_t0, 2)
        c = collections.Counter(b.get("how") for b in beats)
        row = dict(take=self._take, start=self._take_start_iso, duration_s=dur, completed=bool(completed),
                   gesture_count=c["gesture"], manual_count=c["manual"], missed_count=c["missed"],
                   clean=bool(completed and c["manual"] == 0 and c["missed"] == 0),
                   beats=beats, video=self._take_path)
        self._take_t0 = None
        self._cmdq.put(("end",))
        with self._lock:
            tp = os.path.join(self.dir, "takes.json")
            rows = _jload(tp, [])
            rows.append(row)
            _jsave(tp, rows)

    def milestone(self, key, text):
        with self._lock:
            if key in self._state["milestones"]:
                return False
            self._state["milestones"].append(key)
            _jsave(self._state_p, self._state)
            lp = os.path.join(self.dir, "build_log.md")
            new = not os.path.exists(lp)
            with open(lp, "a") as f:
                if new:
                    f.write("# Build log\n\n")
                f.write(f"- {datetime.now():%Y-%m-%d %H:%M} — {text}\n")
        return True

    def coffee(self):
        with self._lock:
            with open(os.path.join(self.dir, "coffee.log"), "a") as f:
                f.write(datetime.now().isoformat(timespec="seconds") + "\n")
        return self.coffee_count

    def close(self):
        if self._take_t0 is not None:
            self.end_take([], False)
        self._cmdq.put(("quit",))
        if self.enabled:
            self._th.join(timeout=30)
        with self._bg_lock:
            bg = list(self._bg)
        for t in bg:
            t.join(timeout=120)

    # ---- writer thread ---------------------------------------------------
    def _compose(self, proj_rgb, cam_bgr, take, beat, wall):
        rc = self.cfg.recorder
        (cw, ch), (pw, ph) = rc.cam_size, rc.proj_size
        out = np.zeros((max(ch, ph), cw + pw, 3), np.uint8)
        if cam_bgr is not None:
            out[:ch, :cw] = cv2.resize(cam_bgr, (cw, ch), interpolation=cv2.INTER_AREA)
        out[:ph, cw:] = cv2.resize(cv2.cvtColor(proj_rgb, cv2.COLOR_RGB2BGR), (pw, ph), interpolation=cv2.INTER_AREA)
        txt = f"take {take:03d} | {beat or '-'} | {datetime.fromtimestamp(wall):%H:%M:%S}.{int(wall * 1000) % 1000:03d}"
        cv2.putText(out, txt, (9, 21), FONT, rc.font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (9, 21), FONT, rc.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _run(self):
        while True:
            try:
                while True:                       # commands first, in order, then at most one frame
                    cmd = self._cmdq.get_nowait()
                    if cmd[0] == "quit":
                        self._drain_frames()
                        self._close_vw()
                        return
                    self._command(cmd)
            except queue.Empty:
                pass
            try:
                item = self._q.get(timeout=0.02)
            except queue.Empty:
                continue
            self._frame(item)

    def _drain_frames(self):
        while True:
            try:
                self._frame(self._q.get_nowait())
            except queue.Empty:
                return

    def _frame(self, item):
        proj, cam, t, take, beat, wall = item
        img = self._compose(proj, cam, take, beat, wall)
        if self._vw is not None:
            # write by wall-clock time: frames dropped under load are filled with repeats, so takes play in real time
            if self._vw_t0 is None:
                self._vw_t0 = wall
            fps = self.cfg.recorder.record_fps
            target = min(int((wall - self._vw_t0) * fps) + 1, self._vw_n + int(fps * 2))
            while self._vw_n < target:
                self._vw.write(img)
                self._vw_n += 1
        self._ring.append(cv2.imencode(".jpg", img, self._enc)[1])

    def _command(self, cmd):
        if cmd[0] == "glitch":
            self._drain_frames()                  # include everything pushed before the trigger
            frames = list(self._ring)
            if len(frames) >= 2:
                self._spawn(self._dump, frames, *cmd[1:])
        elif cmd[0] == "start":
            self._close_vw()
            self._vw_final = cmd[1]
            self._vw_raw = cmd[1] + ".raw.mp4"
            self._vw = self._open_vw(self._vw_raw)
            self._vw_n, self._vw_t0 = 0, None
        elif cmd[0] == "end":
            self._drain_frames()
            self._close_vw()

    def _open_vw(self, p):
        rc = self.cfg.recorder
        (cw, ch), (pw, ph) = rc.cam_size, rc.proj_size
        return cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*rc.fourcc), rc.record_fps, (cw + pw, max(ch, ph)))

    def _close_vw(self):
        if self._vw is None:
            return
        self._vw.release()
        self._vw = None
        if self._vw_n:
            self._spawn(self._finalize, self._vw_raw, self._vw_final)
        elif os.path.exists(self._vw_raw):
            os.remove(self._vw_raw)               # take with zero frames: no video

    def _spawn(self, fn, *a):
        th = threading.Thread(target=fn, args=a, daemon=True)
        with self._bg_lock:
            self._bg = [t for t in self._bg if t.is_alive()] + [th]
        th.start()

    def _finalize(self, raw, final):
        """mp4v -> H.264 if ffmpeg exists; otherwise keep the mp4v file."""
        if shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf",
                                str(self.cfg.recorder.ffmpeg_crf), "-pix_fmt", "yuv420p", final])
            if r.returncode == 0 and os.path.exists(final):
                os.remove(raw)
                return
        os.replace(raw, final)

    def _dump(self, frames, out, reason, detail, take, when):
        rc = self.cfg.recorder
        raw = out + ".raw.mp4"
        imgs = [cv2.imdecode(f, cv2.IMREAD_COLOR) for f in frames]
        h, w = imgs[0].shape[:2]
        vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*rc.fourcc), rc.record_fps, (w, h))
        for im in imgs:
            vw.write(im)
        vw.release()
        pw = rc.proj_size[0]
        g = [cv2.cvtColor(im[:, w - pw:], cv2.COLOR_BGR2GRAY)[::4, ::4].astype(np.float32) for im in imgs]
        bright = float(np.mean([x.mean() for x in g]))
        motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(g[1:], g[:-1])]))
        self._finalize(raw, out)
        with self._lock:
            gp = os.path.join(self.dir, "glitches.json")
            rows = _jload(gp, [])
            rows.append(dict(path=out, reason=reason, detail=detail, take=take, time=when, frames=len(imgs),
                             brightness=round(bright, 3), motion=round(motion, 3), score=round(bright * motion, 3)))
            _jsave(gp, rows)


def _parse_dt(s):
    try:
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.astimezone()
    except (TypeError, ValueError):
        return None


def _span(a, b):
    s = int((b - a).total_seconds())
    return f"{s // 86400}d {s % 86400 // 3600}h {s % 3600 // 60}m" if s >= 0 else "n/a"


def write_story_material(cfg):
    d = path(cfg.recorder.dir)
    takes, glitches = _jload(os.path.join(d, "takes.json"), []), _jload(os.path.join(d, "glitches.json"), [])
    log = []
    try:
        with open(os.path.join(d, "build_log.md")) as f:
            log = [ln.rstrip("\n") for ln in f if ln.startswith("- ")]
    except OSError:
        pass
    clean = [t for t in takes if t.get("clean")]
    L = ["# Story material (generated from real logs only)", ""]
    L += [f"- Total takes: {len(takes) or 'none yet'}",
          f"- Completed takes: {sum(1 for t in takes if t.get('completed')) or 'none yet'}",
          f"- Clean takes (completed, no manual, no missed): {len(clean) or 'none yet'}"]
    files = len([f for f in os.listdir(os.path.join(d, "glitches")) if f.endswith(".mp4")]) if os.path.isdir(os.path.join(d, "glitches")) else 0
    L.append(f"- Glitch clips: {len(glitches)} logged, {files} files on disk" if glitches or files else "- Glitch clips: none yet")
    rs = collections.Counter(g.get("reason") for g in glitches)
    L.append("- Glitch reasons: " + (", ".join(f"{k}: {v}" for k, v in rs.most_common()) if rs else "none yet"))
    # time from first commit to first clean take
    first = None
    try:
        out = subprocess.run(["git", "log", "--reverse", "--format=%cI"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout
        first = (out.splitlines() or [None])[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    label, t0 = "first git commit", _parse_dt(first)
    if t0 is None:
        label = "first build_log entry (no git history)"
        t0 = _parse_dt(log[0][2:18].replace(" ", "T")) if log else None
    ct = [(_parse_dt(t["start"]), t) for t in clean]
    ct = sorted((a, t) for a, t in ct if a)
    if ct and t0:
        a, t = ct[0]
        end = a.timestamp() + t.get("duration_s", 0)
        L.append(f"- Time from {label} ({t0:%Y-%m-%d %H:%M}) to first clean take (take {t['take']}, {a:%Y-%m-%d %H:%M}): "
                 f"{_span(t0, datetime.fromtimestamp(end).astimezone())}")
    else:
        L.append(f"- Time from {label} to first clean take: none yet" + ("" if t0 else f" (no {label})"))
    try:
        with open(os.path.join(d, "coffee.log")) as f:
            cs = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        cs = []
    L.append(f"- Coffees: {len(cs)} (first {cs[0]}, last {cs[-1]})" if cs else "- Coffees: none yet")
    L += ["", "## Milestones"] + (log or ["none yet"])
    L += ["", "## Per-beat gesture success (all takes)"]
    beats = {}
    for t in takes:
        for b in t.get("beats", []):
            r = beats.setdefault(b.get("beat"), dict(name=b.get("name", ""), n=0, g=0))
            r["n"] += 1
            r["g"] += b.get("how") == "gesture"
    for k in sorted(beats, key=lambda x: (x is None, x)):
        r = beats[k]
        L.append(f"- Beat {k} {r['name']}: {r['g']}/{r['n']} by gesture ({100 * r['g'] / r['n']:.0f}%)")
    if not beats:
        L.append("none yet")
    L += ["", "## Best glitch clips (by visual score = brightness x motion)"]
    pool = sorted((g for g in glitches if os.path.exists(g.get("path", ""))), key=lambda g: -g.get("score", 0))
    pick, seen = [], set()
    for g in pool:
        if g["reason"] not in seen and len(pick) < cfg.recorder.story_best_clips:
            pick.append(g)
            seen.add(g["reason"])
    for g in pool:
        if g not in pick and len(pick) < cfg.recorder.story_best_clips:
            pick.append(g)
    for g in sorted(pick, key=lambda g: -g["score"]):
        L.append(f"- {g['path']} — {g['reason']}, take {g['take']}, {g['time']}, score {g['score']}" + (f", {g['detail']}" if g.get("detail") else ""))
    if not pick:
        L.append("none yet")
    p = os.path.join(d, "story_material.md")
    os.makedirs(d, exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    return p


if __name__ == "__main__":
    if sys.argv[1:] == ["story"]:
        print(write_story_material(load_config()))
    else:
        print("usage: python -m hof.recorder story")
