"""Shared data passed between tracking, gestures, sequence and rendering."""
from dataclasses import dataclass, field
import numpy as np

# MediaPipe landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP = 0, 4, 5, 6, 8
MIDDLE_MCP, MIDDLE_TIP, RING_MCP, PINKY_MCP, PINKY_TIP = 9, 12, 13, 17, 20
TIPS = (8, 12, 16, 20)
PIPS = (6, 10, 14, 18)


@dataclass
class Hand:
    label: str                 # "Left" / "Right": the person's actual hand
    cam: np.ndarray            # (21, 3) smoothed landmarks, camera pixels (x, y, z*width)
    proj: np.ndarray           # (21, 2) smoothed + predicted landmarks, projector pixels
    palm_cam: np.ndarray       # (2,) palm centre, camera px
    palm: np.ndarray           # (2,) palm centre, projector px
    vel_cam: np.ndarray        # (2,) camera px / s
    vel: np.ndarray            # (2,) projector px / s
    size_cam: float            # palm size (wrist -> middle MCP), camera px
    size: float                # palm size, projector px
    presence: float = 1.0      # 1 = tracked now, fades to 0 over loss_fade_ms after loss
    tracked: bool = True       # False while holding last pose
    score: float = 1.0


@dataclass
class TrackFrame:
    t: float                               # perf_counter of camera capture
    hands: dict = field(default_factory=dict)   # "Left"/"Right" -> Hand
    image: np.ndarray = None               # BGR camera frame (for HUD/recorder)
    cam_fps: float = 0.0
    infer_ms: float = 0.0
    lead_ms: float = 0.0
