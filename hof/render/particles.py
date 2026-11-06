"""GPU-instanced particle system (numpy sim, one buffer write + one draw) and the Sparkler emitter. Trail layer."""
import numpy as np
import moderngl

from hof.render.engine import GLSL_PX_TO_CLIP

TAU = 6.2831853

# columns: 0 x, 1 y, 2 vx, 3 vy, 4 age (<0 = not born yet), 5 life, 6 size, 7 twinkle, 8 stretch, 9 seed, 10-12 hot, 13-15 cool
# 16 fade-in seconds; CPU only: 17 gravity, 18 drag, 19 branch prob, 20 branch preset, 21 trail rate (1/s), 22 trail life
_NC = 23
_FMT = "2f 2f 1f 1f 1f 1f 1f 1f 3f 3f 1f 6x4/i"
_ATTRS = ("in_pos", "in_vel", "in_age", "in_life", "in_size", "in_twk", "in_str", "in_seed", "in_hot", "in_cool", "in_fade")

_VERT = """
#version 410
in vec2 in_pos; in vec2 in_vel;
in float in_age, in_life, in_size, in_twk, in_str, in_seed, in_fade;
in vec3 in_hot; in vec3 in_cool;
uniform float u_time, u_gain, u_stretch_k;
out vec2 v_uv;
out vec3 v_col;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    float t = max(in_age, 0.0) / in_life;
    float sp = length(in_vel);
    vec2 dir = sp > 1e-3 ? in_vel / sp : vec2(1.0, 0.0);
    float rad = in_size * (1.0 - 0.45 * t);
    float hl = rad + (in_str > 0.0 ? sp * in_str * u_stretch_k : 0.0);
    vec2 off = (dir * c.x * hl + vec2(-dir.y, dir.x) * c.y * rad) * 2.2;
    vec3 col = mix(in_hot, in_cool, smoothstep(0.0, 0.95, t));
    float tw = 1.0 - in_twk * pow(0.5 + 0.5 * sin(u_time * (22.0 + in_seed * 30.0) + in_seed * 40.0), 2.0);
    v_col = col * pow(1.0 - t, 0.85) * tw * u_gain * pow(smoothstep(0.0, in_fade, in_age), 1.5);
    v_uv = c;
    gl_Position = px_to_clip(in_pos + off);
}
"""

_FRAG = """
#version 410
in vec2 v_uv; in vec3 v_col;
out vec4 f;
void main() {
    float r2 = dot(v_uv, v_uv);
    float g = exp(-(v_uv.y * v_uv.y * 7.0 + v_uv.x * v_uv.x * 4.5));
    float core = exp(-(v_uv.y * v_uv.y * 38.0 + v_uv.x * v_uv.x * 14.0));
    float edge = 1.0 - smoothstep(0.7, 1.0, sqrt(r2));
    f = vec4(v_col * (g * 0.8 + core * 1.3) * edge + vec3(core * 0.35) * v_col.r, 1.0);
}
"""


class ParticleSystem:
    def __init__(self, ctx, cfg, size, max_particles=None):
        self.ctx, self.size = ctx, size
        self.c = cfg.render.particles
        self.cap = int(max_particles or self.c.max)
        self._d = np.zeros((self.cap, _NC), dtype="f4")
        self.n = 0
        self.time = 0.0
        self.rng = np.random.default_rng(7)
        self._presets = [self.c.branch, self.c.cross]
        self._prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self._buf = ctx.buffer(reserve=self.cap * _NC * 4, dynamic=True)
        self._vao = ctx.vertex_array(self._prog, [(self._buf, _FMT, *_ATTRS)])
        self._prog["u_res"] = size
        self._prog["u_gain"] = float(self.c.gain)
        self._prog["u_stretch_k"] = float(self.c.stretch_k)

    @property
    def count(self):
        return self.n

    def clear(self):
        self.n = 0

    def emit(self, pos, n, vel=(0, 0), speed=(0, 100), angle=(0, TAU), life=(0.5, 1.0), size=(2, 4),
             color_hot=(1, .9, .6), color_cool=(1, .3, .05), gravity=0.0, drag=0.0, twinkle=0.0,
             stretch=1.0, pos_jitter=0.0, branch=0.0, delay=0.0, branch_kind=0, trail=0.0, trail_life=0.4, fade=None):
        """Spawn particles. pos (2,) or (n,2); ranges are (min,max); speed/angle/life/size/colours may also be (n,..) arrays.
        branch = prob. of splitting on death into preset `branch_kind` (cfg particles.branch / .cross).
        delay = stars are born at random times in [0, delay] (the burst grows instead of popping as a disc).
        fade = seconds a fresh star takes to reach full brightness (default particles.fade_in_s).
        trail = rate/s of fading sparks dropped behind the particle, living `trail_life` s."""
        pos = np.asarray(pos, dtype="f4")
        if pos.ndim == 2:
            n = len(pos)
        n = min(int(n), self.cap - self.n)
        if n <= 0:
            return
        pos = pos[:n] if pos.ndim == 2 else pos
        r = self.rng
        d = self._d[self.n:self.n + n]
        a = np.asarray(angle)[:n] if isinstance(angle, np.ndarray) else r.uniform(angle[0], angle[1], n)
        s = np.asarray(speed)[:n] if isinstance(speed, np.ndarray) else r.uniform(speed[0], speed[1], n)
        d[:, 0:2] = pos
        if pos_jitter:
            d[:, 0:2] += r.normal(0, pos_jitter, (n, 2))
        v = np.asarray(vel, dtype="f4")
        v = v[:n] if v.ndim == 2 else v
        d[:, 2] = np.cos(a) * s + v[..., 0]
        d[:, 3] = np.sin(a) * s + v[..., 1]
        d[:, 4] = -r.uniform(0, delay, n) if delay else 0
        lf = np.asarray(life)[:n] if isinstance(life, np.ndarray) else r.uniform(life[0], life[1], n)
        d[:, 5] = lf
        d[:, 6] = np.asarray(size)[:n] if isinstance(size, np.ndarray) else r.uniform(size[0], size[1], n) if isinstance(size, (tuple, list)) else size
        d[:, 7] = twinkle
        d[:, 8] = stretch
        d[:, 9] = r.random(n)
        d[:, 10:13] = np.asarray(color_hot, dtype="f4")[:n] if np.ndim(color_hot) == 2 else color_hot
        d[:, 13:16] = np.asarray(color_cool, dtype="f4")[:n] if np.ndim(color_cool) == 2 else color_cool
        d[:, 16] = self.c.fade_in_s if fade is None else fade
        d[:, 17] = gravity
        d[:, 18] = drag
        d[:, 19] = branch
        d[:, 20] = branch_kind
        d[:, 21] = trail
        d[:, 22] = trail_life
        self.n += n

    def update(self, dt):
        self.time += dt
        n = self.n
        if not n:
            return
        r = self.rng
        d = self._d[:n]
        d[:, 4] += dt
        d[:, 3] += d[:, 17] * dt
        d[:, 2:4] *= np.exp(-d[:, 18:19] * dt)
        d[:, 0:2] += d[:, 2:4] * dt
        dead = d[:, 4] >= d[:, 5]
        tsel = (d[:, 21] > 0) & (d[:, 4] > 0) & (r.random(n) < d[:, 21] * dt)
        tr = d[tsel].copy() if tsel.any() else None
        br = None
        if dead.any():
            sel = dead & (d[:, 19] > 0) & (r.random(n) < d[:, 19])
            br = d[sel].copy() if sel.any() else None
            keep = ~dead
            m = int(keep.sum())
            d[:m] = d[keep]
            self.n = m
        if br is not None:
            for k, pr in enumerate(self._presets):
                rows = br[br[:, 20] == k]
                if len(rows):
                    self._split(rows, pr)
        if tr is not None:
            t = np.clip(tr[:, 4:5] / tr[:, 5:6], 0, 1)
            hot = tr[:, 10:13] * (1 - t) + tr[:, 13:16] * t
            self.emit(tr[:, 0:2], 0, vel=tr[:, 2:4] * 0.15, speed=(0, 12), life=tr[:, 22] * r.uniform(0.6, 1.0, len(tr)),
                      size=tr[:, 6] * 0.6, color_hot=hot, color_cool=tr[:, 13:16] * 0.55, gravity=tr[:, 17] * 0.4, drag=1.0,
                      twinkle=0.4, stretch=0.0)

    def _split(self, rows, pr):
        """Dying parents spawn child sparks from the preset: random fan (sparkler star) or an even cross (crossette)."""
        r = self.rng
        cnt = r.integers(int(pr.count[0]), int(pr.count[1]) + 1, len(rows))
        par = np.repeat(rows, cnt, axis=0)
        k = len(par)
        if pr.even:
            idx = np.concatenate([np.arange(c) / c for c in cnt])
            ang = (np.repeat(r.uniform(0, TAU, len(rows)), cnt) + idx * TAU) + r.normal(0, 0.06, k)
        else:
            ang = r.uniform(0, TAU, k)
        hot, cool = (par[:, 10:13], par[:, 13:16]) if pr.inherit else (np.asarray(pr.color_hot, "f4"), np.asarray(pr.color_cool, "f4"))
        if pr.inherit:
            hot = np.minimum(hot * pr.brighten, 1.6)
        self.emit(par[:, 0:2], 0, vel=par[:, 2:4] * pr.vel_inherit, speed=r.uniform(*pr.speed, k), angle=ang, life=tuple(pr.life),
                  size=(pr.size * 0.7, pr.size), color_hot=hot, color_cool=cool, gravity=pr.gravity, drag=pr.drag,
                  twinkle=pr.twinkle, stretch=pr.stretch, trail=pr.trail, trail_life=pr.trail_life)

    def render(self):
        if not self.n:
            return
        self._buf.write(self._d[:self.n].tobytes())
        self._prog["u_time"] = self.time
        self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.n)


class Sparkler:
    """A phuljhadi: white-hot burning point, fine gold sparks, bright star sparks that split into 3-5 tiny ones, a glowing wire."""

    def __init__(self, particles, cfg):
        self.p = particles
        self.c = cfg.render.sparkler
        self.pos = None
        self.active = False
        self.intensity = 0.0
        self._acc = 0.0
        self._last = None
        self._t = 0.0
        self._tilt = 0.0

    def update(self, dt, pos, vel, active, intensity=1.0):
        """pos = pinch point px; vel = hand velocity px/s. Emits sparks while active."""
        c = self.c
        self._t += dt
        self.active = bool(active) and intensity > 0.0
        self.intensity = float(intensity)
        pos = np.asarray(pos, dtype="f4")
        self._tilt += (float(np.clip(-vel[0] * c.wire_lean, -c.wire_max_tilt, c.wire_max_tilt)) - self._tilt) * min(1.0, dt * 8)
        if not self.active or dt <= 0:
            self._last = None
            self._acc = 0.0
            self.pos = pos
            return
        prev = pos if self._last is None else self._last
        self._last = pos
        self.pos = pos
        t = self._t
        pulse = 0.8 + 0.35 * np.sin(t * 41.0) * np.sin(t * 13.0 + 1.0) + 0.2 * self.p.rng.random()     # the burn crackles
        self._acc += c.rate * intensity * pulse * dt
        n = int(self._acc)
        self._acc -= n
        if n <= 0:
            return
        r = self.p.rng
        pts = prev + (pos - prev) * r.random((n, 1)).astype("f4")
        hv = np.asarray(vel, dtype="f4") * float(c.inherit_vel)
        nb = int(n * c.big_frac)
        nm = n - nb
        nst = int(nm * c.star_frac)
        base = dict(vel=hv, life=tuple(c.life), gravity=c.gravity, drag=c.drag, stretch=c.stretch, twinkle=c.twinkle)
        sz = (c.size * 0.55, c.size * 1.0)
        hot, cool = tuple(c.color_hot), tuple(c.color_cool)
        if nm - nst > 0:
            self.p.emit(pts[:nm - nst], 0, speed=(c.speed * 0.4, c.speed * 1.1), size=sz, color_hot=hot, color_cool=cool, **base)
        if nst > 0:   # star sparks fly further, burn brighter, then split
            self.p.emit(pts[nm - nst:nm], 0, speed=(c.speed * 0.5, c.speed * 1.25), size=(c.size * 0.9, c.size * 1.4),
                        color_hot=tuple(c.star_color), color_cool=hot, branch=1.0, branch_kind=0,
                        **{**base, "life": (c.life[0] * 0.9, c.life[1] * 1.1)})
        if nb > 0:
            self.p.emit(pts[nm:], 0, speed=(c.speed * 0.3, c.speed * 0.9), size=(c.size * 1.3, c.size * 2.0),
                        color_hot=(1.0, 1.0, 1.0), color_cool=hot, **{**base, "life": (c.life[0], c.life[1] * 0.9)})

    def render_core(self, flame_renderer=None):
        """Queue the white-hot burning point and the glowing wire (needs a FlameRenderer; call before flames.render())."""
        if flame_renderer is None or not self.active or self.pos is None:
            return
        c = self.c
        t = self._t
        fl = 0.85 + 0.15 * np.sin(t * 53.0) + 0.1 * np.sin(t * 31.0) + 0.1 * np.sin(t * 97.0 + 2.0)
        I = self.intensity
        flame_renderer.add_glow(self.pos, c.core_radius * 2.6, color=tuple(c.halo_color), intensity=c.halo_gain * I * fl)   # wide halo
        flame_renderer.add_glow(self.pos, c.core_radius * (0.9 + 0.15 * fl), color=tuple(c.color_cool), intensity=c.core_gain * I * fl)
        flame_renderer.add_glow(self.pos, c.core_radius * 0.55, color=(1.0, 1.0, 1.0), intensity=c.core_gain * 1.2 * I * fl)  # white-hot centre
        ang = np.pi / 2 + self._tilt
        d = np.array([np.cos(ang), np.sin(ang)], "f4")
        m = int(c.wire_dots)
        for i in range(1, m + 1):
            u = i / m
            flame_renderer.add_glow(self.pos + d * c.wire_len * u, c.wire_width * (1.0 - 0.3 * u), color=tuple(c.wire_color),
                                    intensity=c.wire_gain * I * (1.0 - u) ** 1.5 * (0.8 + 0.2 * fl))
