"""Hand light: the projector lights the performer's hands so the camera can track them in a dark room.

A flat, soft-edged warm pool follows each tracked hand (it reads as diya light on the skin). While no hand is
tracked, a dim search light fades up over the performance zone so a hand entering it is lit enough to be found.
Real takes: camera brightness ~28 found a hand in 5% of frames; a lit room (~105) in 33-39%.
"""
import numpy as np
import moderngl
from .engine import fullscreen_quad

_FRAG = """
#version 410
uniform vec2 u_res;
uniform vec4 u_light[3];      // x, y (px), radius (px), gain
uniform vec3 u_color;
uniform float u_edge;         // fraction of the radius that is soft edge
in vec2 uv;
out vec4 f;
void main() {
    vec2 p = vec2(uv.x, 1.0 - uv.y) * u_res;
    float a = 0.0;
    for (int i = 0; i < 3; i++) {
        if (u_light[i].w <= 0.0) continue;
        float r = length(p - u_light[i].xy) / u_light[i].z;
        a = max(a, u_light[i].w * (1.0 - smoothstep(1.0 - u_edge, 1.0, r)));
    }
    f = vec4(u_color * a, 1.0);
}
"""


class HandLight:
    def __init__(self, ctx, cfg, size_px):
        self.c = cfg.render.handlight
        self.size = size_px
        self.prog, self.vao = fullscreen_quad(ctx, _FRAG)
        self.on = bool(self.c.enabled)
        self.gain = float(self.c.gain)
        self._search = 0.0          # 0..1 search-light level
        self._lost = 0.0            # seconds with no tracked hand
        self._hand = {}             # label -> current level (fades in/out)

    def update(self, dt, hands, enabled=True):
        """hands: TrackFrame.hands. enabled=False (e.g. during the firework) fades everything out."""
        c = self.c
        tracked = {k: h for k, h in hands.items() if h.tracked} if enabled and self.on else {}
        self._lost = 0.0 if tracked else self._lost + dt
        want = 1.0 if (enabled and self.on and self._lost > c.search_after_s) else 0.0
        self._search += np.clip(want - self._search, -dt / c.fade_s, dt / c.fade_s)
        for k in ("Left", "Right"):
            w = 1.0 if k in tracked else 0.0
            v = self._hand.get(k, 0.0)
            self._hand[k] = v + np.clip(w - v, -dt / c.fade_s, dt / c.fade_s)
        self._hands = hands

    def render(self):
        """Direct layer (additive). Call after engine.begin_direct()."""
        c, W, H = self.c, self.size[0], self.size[1]
        lights = []
        for k, lvl in self._hand.items():
            h = self._hands.get(k)
            if lvl > 0.01 and h is not None:
                ctr = h.proj.mean(0)                       # whole hand, fingers included, not just the palm
                reach = float(np.linalg.norm(h.proj - ctr, axis=1).max())
                r = max(reach * c.reach_pad, h.size * c.radius_palm)
                lights.append((ctr[0], ctr[1], r, self.gain * lvl * h.presence))
        if self._search > 0.01:
            lights.append((W * c.zone[0], H * c.zone[1], W * c.zone_radius, self.gain * c.search_gain * self._search))
        if not lights:
            return
        lights = (lights + [(0.0, 0.0, 1.0, 0.0)] * 3)[:3]
        self.prog["u_res"] = (float(W), float(H))
        self.prog["u_light"].write(np.array(lights, "f4").tobytes())
        self.prog["u_color"] = tuple(c.color)
        self.prog["u_edge"] = float(c.edge)
        self.vao.render(moderngl.TRIANGLE_STRIP)
