"""Camera pixels -> projector pixels via a homography saved by tools/calibrate.py."""
import json
import os
import time
import numpy as np
import cv2
from .config import path


class Mapping:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file = path(cfg.mapping.file)
        self.H = None
        self.cam_size = None
        self.proj_size = None
        self.offset = np.array(cfg.mapping.offset_px, dtype=np.float64)
        self.reproj_err_px = None
        self.load()

    @property
    def ok(self):
        return self.H is not None

    def load(self):
        if os.path.exists(self.file):
            with open(self.file) as f:
                d = json.load(f)
            self.H = np.array(d["H"], dtype=np.float64)
            self.cam_size = tuple(d["cam_size"])
            self.proj_size = tuple(d["proj_size"])
            self.offset = np.array(d.get("offset_px", self.offset), dtype=np.float64)
            self.reproj_err_px = d.get("reproj_err_px")
        return self.ok

    def save(self, H, cam_size, proj_size, reproj_err_px=None):
        self.H, self.cam_size, self.proj_size = np.asarray(H, np.float64), tuple(cam_size), tuple(proj_size)
        self.reproj_err_px = reproj_err_px
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, "w") as f:
            json.dump({"H": self.H.tolist(), "cam_size": list(cam_size), "proj_size": list(proj_size),
                       "offset_px": self.offset.tolist(), "reproj_err_px": reproj_err_px,
                       "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    def save_offset(self):
        if self.ok:
            self.save(self.H, self.cam_size, self.proj_size, self.reproj_err_px)

    def cam_to_proj(self, pts, render_size):
        """pts: (N,2) camera px -> (N,2) px in the engine's render_size. Uncalibrated: plain scaling."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.H is None:
            return pts * (np.array(render_size, np.float64) / self._cam_wh())
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        out = out * (np.array(render_size, np.float64) / np.array(self.proj_size, np.float64))
        return out + self.offset * (render_size[0] / self.proj_size[0])

    def proj_outline_in_cam(self):
        """Projector rectangle corners in camera px (where the camera sees the projected image)."""
        if self.H is None:
            return None
        w, h = self.proj_size
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(c, np.linalg.inv(self.H)).reshape(-1, 2)

    def _cam_wh(self):
        return np.array(self.cam_size or (self.cfg.camera.width, self.cfg.camera.height), np.float64)
