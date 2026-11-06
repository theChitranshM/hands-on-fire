"""Capture thread: always holds only the newest frame, stamped with perf_counter at grab time."""
import re
import subprocess
import threading
import time
import cv2


def list_cameras():
    """[(index, name)] in AVFoundation order (what OpenCV's AVFoundation backend uses)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                             capture_output=True, text=True, timeout=10).stderr
    except (OSError, subprocess.TimeoutExpired):
        return []
    cams, in_video = [], False
    for line in out.splitlines():
        if "video devices" in line:
            in_video = True
        elif "audio devices" in line:
            break
        elif in_video:
            m = re.search(r"\[(\d+)\] (.+)$", line)
            if m and not m.group(2).startswith("Capture screen"):
                cams.append((int(m.group(1)), m.group(2).strip()))
    return cams


def resolve_device(cfg):
    dev = cfg.camera.device
    if isinstance(dev, int):
        return dev, f"index {dev}"
    for idx, name in list_cameras():
        if str(dev).lower() in name.lower():
            return idx, name
    fb = cfg.camera.fallback_device
    print(f"[camera] '{dev}' not found (is the iPhone near, unlocked, same Apple ID?). Using index {fb}.")
    return fb, f"fallback index {fb}"


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index, self.name = resolve_device(cfg)
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("Camera did not open. System Settings > Privacy & Security > Camera: "
                               "allow your terminal / Python. Then re-run.")
        c = cfg.camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        self.cap.set(cv2.CAP_PROP_FPS, c.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if c.lock_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, c.exposure)
        if c.lock_focus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.lock = threading.Lock()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.fps = 0.0
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True, name="camera")
        self.th.start()

    def _loop(self):
        last = None
        while self.running:
            ok, img = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            if self.cfg.camera.mirror:
                img = cv2.flip(img, 1)
            with self.lock:
                self.frame, self.t, self.seq = img, t, self.seq + 1
            if last:
                self.fps = self.fps * 0.9 + 0.1 / max(t - last, 1e-4) if self.fps else 1 / max(t - last, 1e-4)
            last = t

    def latest(self):
        """(frame, t, seq). frame is None until the first grab."""
        with self.lock:
            return self.frame, self.t, self.seq

    def wait_new(self, seq, timeout=1.0):
        """Block until a frame newer than seq arrives. Returns (frame, t, seq) or (None, 0, seq)."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            f, t, s = self.latest()
            if s != seq and f is not None:
                return f, t, s
            time.sleep(0.001)
        return None, 0.0, seq

    @property
    def size(self):
        f, _, _ = self.latest()
        return (f.shape[1], f.shape[0]) if f is not None else (self.cfg.camera.width, self.cfg.camera.height)

    def close(self):
        self.running = False
        self.th.join(timeout=1)
        self.cap.release()
