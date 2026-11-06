"""Diwali set pieces: a row of wall diyas, marigold petal showers, the aarti light-trail, chakri and anar fireworks, sky lanterns.
Sparks (Chakri, Anar) go through ParticleSystem (trail layer); everything with its own shape (petals, ribbon, lanterns) is an
instanced draw on the direct layer. All tunables: cfg.render.diwali.*. Projector px, origin top-left, y down."""
import numpy as np
import moderngl

from hof.render.engine import GLSL_PX_TO_CLIP
from hof.render.flame import GLSL_NOISE

TAU = 6.2831853
_STRIP = moderngl.TRIANGLE_STRIP


def _rng(lo_hi, rng, n=None):
    return rng.uniform(lo_hi[0], lo_hi[1], n)


def _quad_vao(ctx, vert, frag, buf, fmt, names, res):
    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
    prog["u_res"] = res
    return prog, ctx.vertex_array(prog, [(buf, fmt, *names)])


# --------------------------------------------------------------------------------------------------------------- diya row
class DiyaRow:
    """Deepavali: a row of small clay lamps on the wall that light one by one, left to right, each with a tiny ignition pop."""

    def __init__(self, ctx, cfg, size_px, flames):
        self.c = cfg.render.diwali.diya_row
        self.flames = flames
        W, H = size_px
        c = self.c
        self.pos = np.stack([np.linspace(c.x_range[0], c.x_range[1], int(c.count)) * W,
                             np.full(int(c.count), c.y_frac * H)], 1)
        self._lit = np.full(int(c.count), -1.0)      # seconds since lit, -1 = dark
        self._t = 0.0
        self._start = None                            # row clock at which light() was called
        self._gap = 0.0
        self._level = 0.0
        self._target = 0.0
        self.t = 0.0

    @property
    def active(self):
        return bool(self._level > 0.003 or self._target > 0)

    def light(self, interval=None):
        """Light the lamps one by one from the left. Also switches the row on."""
        self._gap = float(self.c.interval if interval is None else interval)
        self._lit[:] = -1.0
        self._start = self._t
        self._target = 1.0
        self._level = max(self._level, 0.0)

    def set_on(self, on):
        """Fade the whole row in (lighting it if never lit) or out over fade_s."""
        if on and (self._lit < 0).all():
            self.light(0.0)
        self._target = 1.0 if on else 0.0

    def update(self, dt):
        c = self.c
        self._t += dt
        self.t += dt
        if self._start is not None:      # lamps whose turn has come catch light
            k = np.arange(len(self._lit))
            due = (self._t - self._start >= k * self._gap) & (self._lit < 0)
            self._lit[due] = 0.0
            if self._lit.min() >= 0:
                self._start = None
        self._lit[self._lit >= 0] += dt
        step = dt / max(float(c.fade_s), 1e-3)
        self._level = float(np.clip(self._level + np.clip(self._target - self._level, -step, step), 0, 1))

    def draw(self):
        c = self.c
        if self._level <= 0.003:
            return
        size = float(c.size_px)
        for i, age in enumerate(self._lit):
            if age < 0:
                continue
            ramp = min(1.0, age / c.ramp_s)
            inten = self._level * ramp * ramp * (3 - 2 * ramp)
            fl = c.ignite_flare * max(0.0, 1.0 - age / c.flare_s) ** 2
            self.flames.add(self.pos[i], size, inten, fl, True, c.seed + 1.7 * i, glow=c.glow, calm=c.calm, embers=c.embers)
            if fl > 0.02:     # the pop: a quick white-yellow flash at the wick
                h = size * self.flames.frac
                self.flames.add_glow((self.pos[i][0], self.pos[i][1] - h * 0.3), h * c.pop_radius, (1.0, 0.8, 0.45),
                                     c.pop_gain * fl * self._level)


# ----------------------------------------------------------------------------------------------------------------- petals
_PETAL_VERT = """
#version 410
in vec2 in_pos;
in float in_size, in_rot, in_flip, in_a, in_kind;
in vec3 in_col;
uniform float u_ext;
out vec2 v_p;
out vec3 v_col;
out float v_cf, v_a, v_kind, v_size;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    float cs = cos(in_rot), sn = sin(in_rot);
    v_p = c * u_ext; v_col = in_col; v_cf = cos(in_flip); v_a = in_a; v_kind = in_kind; v_size = in_size;
    vec2 o = c * u_ext * in_size;
    gl_Position = px_to_clip(in_pos + vec2(o.x * cs - o.y * sn, o.x * sn + o.y * cs));
}
"""

_PETAL_FRAG = """
#version 410
in vec2 v_p; in vec3 v_col; in float v_cf, v_a, v_kind, v_size;
uniform float u_gain, u_halo;
out vec4 f;
void main() {
    float fx = max(abs(v_cf), 0.2);          // tumbling: the width foreshortens, edge-on petals are slivers
    vec2 p = vec2(v_p.x / fx, v_p.y);
    float y = clamp(p.y, -1.0, 1.0), t = y * 0.5 + 0.5;
    float body = max(1.0 - y * y, 0.0);
    float w = v_kind < 0.5 ? 0.44 * pow(body, 0.55) * (0.72 + 0.38 * t)           // marigold: long, spatulate, widest toward the tip
                           : 0.72 * pow(body, 0.5) * (0.5 + 0.55 * t);            // rose: broad, cupped
    w *= 1.0 + 0.05 * sin(y * 9.0 + v_col.g * 17.0);                                // ruffled edge
    float aa = 1.1 / (v_size * fx);
    float cov = (1.0 - smoothstep(-aa, aa, abs(p.x) - w)) * (1.0 - smoothstep(0.96, 1.0, abs(p.y)));
    float ex = clamp(abs(p.x) / max(w, 1e-3), 0.0, 1.0);
    float stri = 0.9 + 0.1 * cos(p.x / max(w, 0.06) * 8.0);                         // longitudinal ridges
    float vein = exp(-p.x * p.x / (0.0016 / (fx * fx))) * smoothstep(-0.95, 0.1, y);
    vec3 body_c = v_col * stri * (0.72 + 0.34 * (1.0 - ex * ex)) * mix(0.82, 1.1, t);
    body_c += vec3(1.0, 0.85, 0.55) * vein * 0.32 + v_col * pow(ex, 3.0) * 0.28;   // translucent: bright veins, glowing edge
    float facing = abs(v_cf);
    float light = 0.52 + 0.48 * facing + (1.0 - facing) * 0.45;                     // flat = fully lit, edge-on = thin bright sliver
    float r = length(v_p);
    float halo = exp(-r * r * 1.1) * (1.0 - smoothstep(1.3, 1.9, r)) * u_halo * (0.35 + 0.65 * (0.3 + 0.7 * fx));
    f = vec4((body_c * cov * light * u_gain + v_col * halo) * v_a, 1.0);
}
"""


class Petals:
    """Marigold (genda) and a few rose petals drifting down, lit by the projector: tumble, sway, flutter-braking, soft halo."""

    _FMT = "2f 1f 1f 1f 1f 1f 3f /i"
    _NAMES = ("in_pos", "in_size", "in_rot", "in_flip", "in_a", "in_kind", "in_col")

    def __init__(self, ctx, cfg, size_px):
        self.c = cfg.render.diwali.petals
        self.size = size_px
        self.rng = np.random.default_rng(23)
        n = int(self.c.max)
        self._prog, self._vao = None, None
        self._buf = ctx.buffer(reserve=n * 10 * 4, dynamic=True)
        self._prog, self._vao = _quad_vao(ctx, _PETAL_VERT, _PETAL_FRAG, self._buf, self._FMT, self._NAMES, size_px)
        self._prog["u_ext"] = 1.9
        self._prog["u_gain"] = float(self.c.gain)
        self._prog["u_halo"] = float(self.c.halo)
        self._queue = []        # (t_start, n_left) spawn schedule rows handled below
        self._p = np.zeros((0, 20), "f4")
        self._sched = []        # [t_left_until_start, rate, n_remaining, center, spread]
        self._acc = 0.0
        self.t = 0.0

    # petal columns: 0 x, 1 y, 2 vx_burst, 3 vy_burst, 4 age, 5 life, 6 size, 7 kind, 8 rot0, 9 spin, 10 flip0, 11 tumble, 12 sway amp,
    # 13 sway hz, 14 sway phase, 15 fall, 16 drift, 17-19 colour (CPU keeps x as the sway-free path, drawn x adds the swing)
    @property
    def active(self):
        return bool(len(self._p) or self._sched)

    def clear(self):
        self._p = self._p[:0]
        self._sched.clear()

    def shower(self, center=None, n=None, duration=None, spread=None):
        """Rain petals: from across the whole top of the frame (center None) or from around `center` with an outward puff."""
        c = self.c
        n = int(c.count if n is None else n)
        dur = float(c.duration if duration is None else duration)
        self._sched.append([dur, n, None if center is None else np.asarray(center, "f4"),
                            float(c.spread if spread is None else spread), 0.0])

    def _spawn(self, k, center, spread):
        c, r = self.c, self.rng
        W, H = self.size
        p = np.zeros((k, 20), "f4")
        rose = r.random(k) < c.rose_frac
        mk = np.asarray(c.marigold_colors, "f4")
        rs = np.asarray(c.rose_colors, "f4")
        col = np.where(rose[:, None], rs[r.integers(0, len(rs), k)], mk[r.integers(0, len(mk), k)])
        col = col * r.uniform(0.85, 1.05, (k, 1))
        if center is None:
            p[:, 0] = r.uniform(-0.03, 1.03, k) * W
            p[:, 1] = -r.uniform(20, 60, k)
        else:
            p[:, 0] = center[0] + r.normal(0, spread * 0.5, k)
            p[:, 1] = center[1] + r.normal(0, spread * 0.12, k)
            a = r.uniform(0, TAU, k)
            s = _rng(c.burst_speed, r, k)
            p[:, 2], p[:, 3] = np.cos(a) * s, np.sin(a) * s * 0.7 - 0.4 * s
        p[:, 5] = 60.0
        p[:, 6] = np.where(rose, _rng(c.rose_len, r, k), _rng(c.marigold_len, r, k))
        p[:, 7] = rose
        p[:, 8] = r.uniform(0, TAU, k)
        p[:, 9] = _rng(c.spin, r, k)
        p[:, 10] = r.uniform(0, TAU, k)
        p[:, 11] = _rng(c.tumble, r, k) * r.choice([-1, 1], k)
        p[:, 12] = _rng(c.sway, r, k)
        p[:, 13] = _rng(c.sway_hz, r, k)
        p[:, 14] = r.uniform(0, TAU, k)
        p[:, 15] = _rng(c.fall, r, k)
        p[:, 16] = r.normal(0, 8, k)
        p[:, 17:20] = col
        self._p = np.concatenate([self._p, p])[-int(c.max):]

    def update(self, dt):
        c = self.c
        self.t += dt
        keep = []
        for s in self._sched:      # s = [duration_left, n_left, center, spread, accumulator]
            rate = s[1] / max(s[0], dt)
            s[4] += rate * dt
            k = min(int(s[4]), int(s[1]))
            if k:
                s[4] -= k
                s[1] -= k
                self._spawn(k, s[2], s[3])
            s[0] -= dt
            if s[0] > 0 and s[1] >= 1:
                keep.append(s)
        self._sched = keep
        p = self._p
        if not len(p):
            return
        p[:, 4] += dt
        a = p[:, 4]
        flat = np.abs(np.cos(p[:, 10] + p[:, 11] * a))
        vy_t = p[:, 15] * (1.0 - c.flat_drag * flat * flat * 0.9 + 0.12 * np.sin(a * p[:, 13] * TAU * 2.0 + p[:, 14]))
        # burst velocity bleeds off; the fall speed is approached smoothly (flutter brakes while the face is flat)
        bk = np.exp(-c.burst_drag * dt)
        p[:, 2:4] *= bk
        vy_swing = p[:, 12] * p[:, 13] * TAU * np.cos(a * p[:, 13] * TAU + p[:, 14])
        p[:, 1] += (vy_t * np.clip(a * 1.5, 0, 1) + p[:, 3]) * dt
        p[:, 0] += (p[:, 16] + p[:, 2] + vy_swing) * dt
        H = self.size[1]
        self._p = p[(p[:, 1] < H + 40)]

    def render(self):
        p = self._p
        if not len(p):
            return
        c = self.c
        H = self.size[1]
        a = p[:, 4]
        x = p[:, 0]
        rot = p[:, 8] + p[:, 9] * a + 0.9 * np.cos(a * p[:, 13] * TAU + p[:, 14])      # swings with the sway: pendulum-like
        flip = p[:, 10] + p[:, 11] * a + 0.7 * np.sin(a * 0.9 + p[:, 14])
        vis = np.clip(a / c.fade_in_s, 0, 1) * np.clip((H - p[:, 1]) / (H * c.bottom_fade), 0, 1)
        out = np.empty((len(p), 10), "f4")
        out[:, 0] = x
        out[:, 1] = p[:, 1]
        out[:, 2] = p[:, 6]
        out[:, 3] = rot
        out[:, 4] = flip
        out[:, 5] = vis
        out[:, 6] = p[:, 7]
        out[:, 7:10] = p[:, 17:20]
        self._buf.write(out.tobytes())
        self._vao.render(_STRIP, vertices=4, instances=len(p))


# --------------------------------------------------------------------------------------------------------------- aarti trail
_RIB_VERT = """
#version 410
in vec2 in_pos; in vec2 in_va;
out vec2 v_va;
""" + GLSL_PX_TO_CLIP + """
void main() { v_va = in_va; gl_Position = px_to_clip(in_pos); }
"""
_RIB_FRAG = """
#version 410
in vec2 v_va;
uniform vec3 u_col, u_hot;
uniform float u_gain;
out vec4 f;
void main() {
    float head = 1.0 - v_va.y;
    float s = v_va.x * v_va.x;
    float g = exp(-s * 3.2) * pow(head, 1.5);
    float core = exp(-s * 20.0) * pow(head, 1.1);
    f = vec4((u_col * g * 0.9 + u_hot * core * 1.2) * u_gain, 1.0);
}
"""
_RING_VERT = """
#version 410
in vec2 in_pos; in float in_k;
out vec2 v_q; out float v_k;
uniform float u_R;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    v_q = c * 1.2; v_k = in_k;
    gl_Position = px_to_clip(in_pos + c * u_R * 1.2);
}
"""
_RING_FRAG = """
#version 410
in vec2 v_q; in float v_k;
uniform vec3 u_col, u_hot;
uniform float u_gain, u_w;
out vec4 f;
void main() {
    float r = length(v_q);
    float rho = 1.0 - pow(1.0 - v_k, 2.2);                 // fast out, slowing
    float d = (r - rho) / (u_w * (0.7 + 1.1 * rho));
    float band = exp(-d * d);
    float fade = pow(1.0 - v_k, 1.4) * smoothstep(0.0, 0.05, v_k);
    float wash = exp(-r * r * 4.0) * 0.18 * pow(1.0 - v_k, 3.0);
    f = vec4(((u_col * 0.8 + u_hot * band * 0.6) * band + u_col * wash) * fade * u_gain, 1.0);
}
"""


class AartiTrail:
    """The long-exposure ribbon an aarti plate draws, plus a ripple ring per completed loop."""

    def __init__(self, ctx, cfg, size_px):
        self.c = cfg.render.diwali.aarti
        c = self.c
        self.size = size_px
        self._pts = []                 # (x, y, t)
        self._fade = 1.0
        self._stopping = False
        self._rings = []               # age in [0, 1]
        self._rpos = []
        self._now = 0.0
        self._buf = ctx.buffer(reserve=int(c.max_points) * int(c.subdiv) * 2 * 4 * 4, dynamic=True)
        self._prog, self._vao = _quad_vao(ctx, _RIB_VERT, _RIB_FRAG, self._buf, "2f 2f", ("in_pos", "in_va"), size_px)
        self._prog["u_col"], self._prog["u_hot"], self._prog["u_gain"] = tuple(c.color), tuple(c.hot), float(c.gain)
        self._rbuf = ctx.buffer(reserve=int(c.max_rings) * 3 * 4, dynamic=True)
        self._rprog, self._rvao = _quad_vao(ctx, _RING_VERT, _RING_FRAG, self._rbuf, "2f 1f /i", ("in_pos", "in_k"), size_px)
        self._rprog["u_col"], self._rprog["u_hot"] = tuple(c.color), tuple(c.hot)
        self._rprog["u_gain"], self._rprog["u_w"] = float(c.ring_gain), float(c.ring_width)
        self._rprog["u_R"] = float(c.ring_radius)
        self._nv = 0

    @property
    def active(self):
        return bool(self._pts or self._rings)

    def add(self, pos, now):
        c = self.c
        self._now = now
        self._stopping = False
        self._fade = 1.0
        if self._pts:
            if np.hypot(pos[0] - self._pts[-1][0], pos[1] - self._pts[-1][1]) < c.min_step:
                self._pts[-1] = (self._pts[-1][0], self._pts[-1][1], now)     # held still: keep the head alive
                return
            if now < self._pts[-1][2]:
                self._pts.clear()
        self._pts.append((float(pos[0]), float(pos[1]), float(now)))
        del self._pts[:-int(c.max_points)]

    def pulse(self, pos):
        if len(self._rings) < int(self.c.max_rings):
            self._rings.append(0.0)
            self._rpos.append((float(pos[0]), float(pos[1])))

    def stop(self):
        self._stopping = True

    def update(self, dt):
        c = self.c
        if self._stopping:
            self._fade = max(0.0, self._fade - dt / max(c.fade_s, 1e-3))
            if self._fade <= 0:
                self._pts.clear()
                self._stopping = False
        self._now += dt if self._stopping else 0.0
        self._rings = [a + dt / c.ring_s for a in self._rings]
        keep = [i for i, a in enumerate(self._rings) if a < 1.0]
        self._rpos = [self._rpos[i] for i in keep]
        self._rings = [self._rings[i] for i in keep]
        if not self._stopping and self._pts and self._now - self._pts[-1][2] > c.trail_s:
            self._pts.clear()

    def _ribbon(self):
        c = self.c
        P = np.asarray(self._pts, "f8")
        P = P[P[:, 2] >= self._now - c.trail_s]
        if len(P) < 3:
            return 0
        xy, tt = P[:, :2], P[:, 2]
        for _ in range(int(c.smooth)):          # 1-2-1 smoothing, ends pinned
            xy = np.concatenate([xy[:1], 0.25 * xy[:-2] + 0.5 * xy[1:-1] + 0.25 * xy[2:], xy[-1:]])
        k = int(c.subdiv)
        pp = np.concatenate([xy[:1], xy, xy[-1:]])
        u = (np.arange(k) / k)[None, :, None]
        p0, p1, p2, p3 = (pp[i:len(pp) - 3 + i][:, None, :] for i in range(4))
        pts = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        pts = np.concatenate([pts.reshape(-1, 2), xy[-1:]])
        ts = np.concatenate([(tt[:-1, None] + (tt[1:] - tt[:-1])[:, None] * u[:, :, 0]).reshape(-1), tt[-1:]])
        age = np.clip((self._now - ts) / c.trail_s, 0, 1)
        tan = np.gradient(pts, axis=0)
        ln = np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-4)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1) / ln
        w = (c.tail_width + (c.width - c.tail_width) * (1.0 - age) ** c.taper) * 2.3 * self._fade
        n = len(pts)
        v = np.empty((n * 2, 4), "f4")
        v[0::2, 0:2] = pts + nrm * w[:, None]
        v[1::2, 0:2] = pts - nrm * w[:, None]
        v[0::2, 2], v[1::2, 2] = 1.0, -1.0
        v[:, 3] = np.repeat(age, 2)
        self._buf.write(v.tobytes())
        return n * 2

    def render(self):
        if self._pts:
            nv = self._ribbon()
            if nv:
                self._vao.render(_STRIP, vertices=nv)
        if self._rings:
            rows = np.array([[x, y, a] for (x, y), a in zip(self._rpos, self._rings)], "f4")
            self._rbuf.write(rows.tobytes())
            self._rvao.render(_STRIP, vertices=4, instances=len(rows))


# ---------------------------------------------------------------------------------------------------------------- chakri
class Chakri:
    """Ground spinner: nozzles on the rim throw gold -> green -> red sparks tangentially, drawn as spiral arms."""

    def __init__(self, particles, cfg):
        self.p = particles
        self.c = cfg.render.diwali.chakri
        self.pos = None
        self.size = 150.0
        self.spin = 0.0
        self.theta = 0.0
        self.t = 0.0
        self._acc = [0.0, 0.0]
        self._phase0 = float(self.p.rng.random())

    @property
    def active(self):
        return self.spin > 0.02

    def _colors(self):
        pal = np.asarray(self.c.colors, "f4")            # (phases, 2, 3)
        u = ((self.t / self.c.color_period_s + self._phase0) % 1.0) * len(pal)
        i, f = int(u) % len(pal), u - int(u)
        f = f * f * (3 - 2 * f)
        mix = pal[i] * (1 - f) + pal[(i + 1) % len(pal)] * f
        return mix[0], mix[1]

    def update(self, dt, pos, active, intensity=1.0, size=None):
        c = self.c
        self.pos = np.asarray(pos, "f4")
        if size is not None:
            self.size = float(size)
        self.t += dt
        tgt = float(intensity) if active else 0.0
        rate = dt / (c.spin_up_s if tgt > self.spin else c.spin_down_s)
        self.spin += float(np.clip(tgt - self.spin, -rate, rate))
        self.intensity = float(intensity)
        if self.spin < 0.02 or dt <= 0:
            return
        R = c.radius_frac * self.size
        w = c.omega * self.spin
        th0, self.theta = self.theta, self.theta + w * dt
        r = self.p.rng
        hot, cool = self._colors()
        # jets: sparks born at sub-frame angles along the nozzle's arc, so each arm is a continuous spiral
        self._acc[0] += c.jet_rate * self.spin * dt
        n = int(self._acc[0])
        self._acc[0] -= n
        if n:
            j = r.integers(0, int(c.jets), n)
            th = th0 + (self.theta - th0) * r.random(n) + j * TAU / c.jets
            self._emit(th, R, w, n, c.size, c.life, hot, cool, c.trail, 1.0)
        self._acc[1] += c.hiss_rate * self.spin * dt
        n = int(self._acc[1])
        self._acc[1] -= n
        if n:
            th = r.uniform(0, TAU, n)
            self._emit(th, R, w, n, c.hiss_size, c.hiss_life, np.asarray(c.hiss_hot, "f4"), np.asarray(c.hiss_cool, "f4"), 0.0, 0.55)

    def _emit(self, th, R, w, n, size, life, hot, cool, trail, kv):
        c, r = self.c, self.p.rng
        cs, sn = np.cos(th), np.sin(th)
        tang = np.stack([-sn, cs], 1)
        rad = np.stack([cs, sn], 1)
        vel = tang * (w * R * c.tangent * kv) + rad * c.radial
        pos = self.pos + rad * R
        self.p.emit(pos, n, vel=vel, speed=(0, c.spread), life=tuple(life), size=(size * 0.7, size), color_hot=tuple(hot * c.gain),
                    color_cool=tuple(cool * c.gain), gravity=c.gravity, drag=c.drag, twinkle=c.twinkle, stretch=c.stretch,
                    trail=trail, trail_life=c.trail_life)

    def render_core(self, flames):
        c = self.c
        if self.pos is None or self.spin < 0.02:
            return
        R = c.radius_frac * self.size
        s = self.spin
        hot, cool = self._colors()
        fl = 0.9 + 0.1 * np.sin(self.t * 47.0) * np.sin(self.t * 19.0)
        # rim: with motion blur the discrete nozzles smear into a ring; an arc of dots trails behind each
        for k in range(int(c.rim_dots)):
            a = k * TAU / c.rim_dots + self.theta
            flames.add_glow(self.pos + R * np.array([np.cos(a), np.sin(a)]), R * c.rim_radius * 2, tuple(cool), c.rim_gain * s * fl)
        for j in range(int(c.jets)):
            for m in range(int(c.arc_dots) + 1):
                a = self.theta + j * TAU / c.jets - m * c.arc_step * s
                g = (1.0 - m / (c.arc_dots + 1.0)) ** 1.5
                flames.add_glow(self.pos + R * np.array([np.cos(a), np.sin(a)]), R * c.head_radius * (0.5 + 0.5 * g),
                                tuple(hot), c.head_gain * s * g * fl)
        if c.hub_gain > 0:
            flames.add_glow(self.pos, R * 0.45, tuple(cool), c.hub_gain * s * fl)
        flames.add_glow(self.pos, R * c.halo_radius, tuple(cool), c.halo_gain * s * fl)


# ------------------------------------------------------------------------------------------------------------------ anar
class Anar:
    """Flowerpot fountain: sparks leave the pot mouth in a narrow cone, arc over and fall back; swells, flickers, crackles."""

    def __init__(self, particles, cfg):
        self.p = particles
        self.c = cfg.render.diwali.anar
        self.pos = None
        self.size = 150.0
        self.level = 0.0
        self._on_t = 0.0         # seconds since activation (fizz -> whoosh)
        self._was = False
        self.t = 0.0
        self._acc = [0.0, 0.0]
        self._ph = self.p.rng.uniform(0, TAU, 3)

    @property
    def active(self):
        return self.level > 0.01

    def _swell(self):
        c = self.c
        f = 1.0 + sum(a * np.sin(self.t * hz * TAU + ph) for a, hz, ph in zip(c.swell_amp, c.swell_hz, self._ph))
        return max(0.35, f + 0.1 * (self.p.rng.random() - 0.5))

    def update(self, dt, pos, active, intensity=1.0, size=None):
        c = self.c
        self.pos = np.asarray(pos, "f4")
        if size is not None:
            self.size = float(size)
        self.t += dt
        if active and not self._was:
            self._on_t = 0.0
        self._was = bool(active)
        if active:
            self._on_t += dt
            # fizz at a low level, then a whoosh ramp to full
            u = np.clip((self._on_t - c.fizz_s) / c.whoosh_s, 0, 1)
            tgt = (c.fizz_level + (1 - c.fizz_level) * u * u * (3 - 2 * u)) * float(intensity)
            self.level += (tgt - self.level) * min(1.0, dt * 20.0) if tgt > self.level else (tgt - self.level) * min(1.0, dt * 6.0)
        else:
            self.level = max(0.0, self.level - dt / c.die_s)
        L = self.level
        if L < 0.01 or dt <= 0:
            return
        r = self.p.rng
        sw = self._swell()
        g, h = c.gravity, c.height_palm * self.size
        v_top = np.sqrt(2 * g * h) * c.speed_boost
        v_top *= np.sqrt(L)
        half = np.radians(c.cone_deg) * 0.5
        self._acc[0] += c.rate * L * sw * dt
        n = int(self._acc[0])
        self._acc[0] -= n
        if n:
            ang = -np.pi / 2 + np.clip(r.normal(0, half / 1.5, n), -half, half)
            sp = v_top * (c.speed_lo + (1 - c.speed_lo) * r.random(n) ** 0.7) * (0.9 + 0.2 * sw / 1.2)
            silver = r.random(n) < c.silver_frac
            gold = np.asarray(c.gold, "f4")
            sil = np.asarray(c.silver, "f4")
            hot = np.where(silver[:, None], sil[0], gold[0]) * c.gain
            cool = np.where(silver[:, None], sil[1], gold[1]) * c.gain
            mouth = self.pos + np.array([0.0, -0.06 * self.size], "f4")
            tr = np.where(r.random(n) < c.trail_frac, c.trail, 0.0)
            self.p.emit(mouth + r.normal(0, 0.04 * self.size, (n, 2)) * [1, 0.3], n, speed=sp, angle=ang,
                        life=_rng(c.life, r, n), size=_rng(c.size, r, n) * (0.7 + 0.5 * sw), color_hot=hot, color_cool=cool,
                        gravity=g, drag=c.drag, twinkle=c.twinkle, stretch=c.stretch, trail=tr, trail_life=c.trail_life)
        self._acc[1] += c.crackle_rate * L * dt
        nb = int(self._acc[1])
        self._acc[1] -= nb
        for _ in range(nb):                       # crackle: a little white pop somewhere up the fountain
            hh = h * r.uniform(0.35, 0.95)
            a = -np.pi / 2 + r.normal(0, half / 1.5)
            at = self.pos + np.array([np.cos(a), np.sin(a)], "f4") * hh * 0.9
            k = int(r.integers(*c.crackle_count))
            self.p.emit(at, k, speed=tuple(c.crackle_speed), life=tuple(c.crackle_life), size=c.crackle_size,
                        color_hot=(0.6, 0.6, 0.57), color_cool=(0.6, 0.42, 0.18), gravity=g * 0.5, drag=2.5, twinkle=0.3, stretch=0.6)

    def render_core(self, flames):
        c = self.c
        L = self.level
        if self.pos is None or L < 0.01:
            return
        sw = 0.85 + 0.3 * self._swell() / 1.2
        S = self.size
        flames.add_glow(self.pos, S * c.spill_radius, (1.0, 0.45, 0.1), c.spill_gain * L * sw)
        flames.add_glow(self.pos + [0, -0.05 * S], S * c.mouth_radius * 2.0, (1.0, 0.6, 0.18), c.mouth_gain * 0.5 * L * sw)
        for dx in (-0.2, 0.0, 0.2):               # the pot mouth is a wide ellipse, not a dot
            flames.add_glow(self.pos + [dx * S * 0.6, -0.04 * S], S * c.mouth_radius * 0.7, (1.0, 0.85, 0.5), c.mouth_gain * L * sw)


# ------------------------------------------------------------------------------------------------------------- lanterns
_LAN_VERT = """
#version 410
in vec2 in_pos;
in float in_h, in_tilt, in_a, in_seed;
uniform float u_hx;
out vec2 v_q;
out float v_a, v_seed;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    vec2 q = c * vec2(0.5 * u_hx, 0.5 * u_hx);       // in lantern heights: halo reaches u_hx/2
    v_q = q; v_a = in_a; v_seed = in_seed;
    float cs = cos(in_tilt), sn = sin(in_tilt);
    vec2 o = vec2(q.x, -q.y) * in_h;                    // q.y up
    gl_Position = px_to_clip(in_pos + vec2(o.x * cs - o.y * sn, o.x * sn + o.y * cs));
}
"""

_LAN_FRAG = """
#version 410
in vec2 v_q; in float v_a, v_seed;
uniform float u_time, u_wf, u_gain, u_halo, u_flick, u_hx;
uniform vec3 u_edge, u_mid, u_flame;
out vec4 f;
""" + GLSL_NOISE + """
void main() {
    float fl = 1.0 + u_flick * flick(u_time * 1.4, v_seed * 9.0);
    vec2 p = v_q;                                          // y up, lantern spans y in [-0.5, 0.5], flame near the bottom
    float t = clamp(p.y / 0.5, -1.0, 1.0);
    // paper shell: straight-ish sides bulging low, rounded shoulders, flat open bottom
    float top = t > 0.0 ? sqrt(max(1.0 - pow(t, 2.6), 0.0)) : 1.0;
    float w = 0.5 * u_wf * top * (0.84 + 0.16 * (1.0 - (t + 0.15) * (t + 0.15))) * (t < -0.82 ? 0.97 : 1.0);
    float aa = 0.012;
    float cov = (1.0 - smoothstep(-aa, aa, abs(p.x) - w)) * smoothstep(-0.5 - aa, -0.5 + aa, p.y) * (1.0 - smoothstep(0.5 - aa, 0.5 + aa, p.y));
    float ex = clamp(abs(p.x) / max(w, 1e-3), 0.0, 1.0);
    // lit from within: brightest over the flame, falling toward the shoulders; thicker paper at the edges is deeper red
    vec2 fp = vec2(p.x, p.y + 0.36);
    float near = exp(-dot(fp, fp * vec2(2.4, 1.6)) * 5.0);
    float lum = (0.5 + 0.5 * near) * (0.55 + 0.45 * (1.0 - smoothstep(-0.5, 0.5, p.y) * 0.8)) * fl;
    vec3 paper = mix(u_mid, u_edge, pow(ex, 2.2)) * lum;
    paper = mix(paper, vec3(1.0, 0.8, 0.45) * lum, near * 0.45 * (1.0 - ex * 0.6));
    float seam = 0.93 + 0.07 * cos(p.x / max(w, 0.05) * 6.2832 * 1.5);     // gores in the paper
    paper *= seam * (0.92 + 0.08 * vnoise(p * 30.0 + v_seed * 7.0));       // paper grain
    // bottom rim ring (the open mouth) and the flame
    float rim = exp(-pow((p.y + 0.5) / 0.022, 2.0)) * (1.0 - smoothstep(0.85, 1.0, ex)) * cov;
    paper += u_flame * rim * 0.9 * fl;
    float fy = (p.y + 0.5) / 0.2;
    float fw = 0.09 * (1.0 - fy) * smoothstep(0.0, 0.1, fy) + 0.001;
    float fx = p.x + 0.012 * flick(u_time * 0.9, v_seed * 3.0) * fy;
    float flame = smoothstep(1.0, 0.0, abs(fx) / fw) * (1.0 - smoothstep(0.7, 1.0, fy)) * smoothstep(-0.01, 0.15, fy);
    vec3 col = paper * cov * 0.95 + u_flame * flame * 1.35 * fl;
    float r = length(p);
    float halo = exp(-r * r * 9.0 / (u_hx * u_hx * 0.25 * 2.0)) * (1.0 - smoothstep(0.4 * u_hx, 0.5 * u_hx, r)) * u_halo * fl;
    col += mix(u_edge, u_mid, 0.5) * halo + u_mid * near * 0.12 * (1.0 - cov);
    f = vec4(col * u_gain * v_a, 1.0);
}
"""


class Lanterns:
    """Akash kandil: warm paper lanterns lit from within, rising and swaying, shrinking and fading toward the top."""

    def __init__(self, ctx, cfg, size_px, flames=None):
        self.c = cfg.render.diwali.lanterns
        self.size = size_px
        self.rng = np.random.default_rng(31)
        c = self.c
        self._buf = ctx.buffer(reserve=int(c.max) * 6 * 4, dynamic=True)
        hx = 2.0 * float(c.halo_r)
        self._prog, self._vao = _quad_vao(ctx, _LAN_VERT, _LAN_FRAG, self._buf, "2f 1f 1f 1f 1f /i",
                                          ("in_pos", "in_h", "in_tilt", "in_a", "in_seed"), size_px)
        pr = self._prog
        pr["u_hx"], pr["u_wf"], pr["u_gain"], pr["u_halo"], pr["u_flick"] = hx, float(c.width_frac), float(c.gain), float(c.halo), float(c.flicker)
        pr["u_edge"], pr["u_mid"], pr["u_flame"] = tuple(c.edge_color), tuple(c.mid_color), tuple(c.flame_color)
        self._L = np.zeros((0, 16), "f4")
        self.t = 0.0

    # cols: 0 x0, 1 y, 2 h, 3 age, 4 delay, 5 rise, 6 drift, 7 sway amp, 8 sway hz, 9 phase, 10 tilt phase, 11 seed, 12 x (drawn)
    @property
    def active(self):
        return bool(len(self._L))

    def clear(self):
        self._L = np.zeros((0, 16), "f4")

    def release(self, pos, n=None):
        c, r = self.c, self.rng
        n = int(c.count if n is None else n)
        k = min(n, int(c.max) - len(self._L))
        if k <= 0:
            return
        L = np.zeros((k, 16), "f4")
        L[:, 0] = pos[0] + r.uniform(-1, 1, k) * c.spread
        L[:, 1] = pos[1]
        L[:, 2] = _rng(c.height, r, k)
        L[:, 4] = -np.sort(r.uniform(0, c.stagger_s * n, k))
        L[:, 3] = L[:, 4]
        L[:, 5] = _rng(c.rise, r, k)
        L[:, 6] = _rng(c.drift, r, k)
        L[:, 7] = _rng(c.sway, r, k)
        L[:, 8] = _rng(c.sway_hz, r, k)
        L[:, 9] = r.uniform(0, TAU, k)
        L[:, 10] = r.uniform(0, TAU, k)
        L[:, 11] = r.uniform(0, 50, k)
        L[:, 12] = L[:, 0]
        L[:, 13] = pos[1]
        self._L = np.concatenate([self._L, L])

    def update(self, dt):
        self.t += dt
        L = self._L
        if not len(L):
            return
        L[:, 3] += dt
        a = np.maximum(L[:, 3], 0.0)
        lift = np.clip(a / self.c.lift_s, 0, 1)
        sp = L[:, 5] * lift * lift * (3 - 2 * lift)
        L[:, 1] -= sp * dt
        L[:, 0] += L[:, 6] * lift * dt
        L[:, 12] = L[:, 0] + L[:, 7] * np.sin(a * L[:, 8] * TAU + L[:, 9])
        self._L = L[L[:, 1] > -L[:, 2] * 1.5]

    def render(self):
        L = self._L
        if not len(L):
            return
        c = self.c
        H = self.size[1]
        a = np.maximum(L[:, 3], 0.0)
        alt = np.clip(1.0 - L[:, 1] / H, 0, 1)
        scale = (1.0 - c.shrink * alt) * np.clip(a / c.grow_s, 0.05, 1.0) ** 0.5
        fade = np.clip(L[:, 1] / (H * c.fade_frac), 0, 1) * np.clip(a / (c.grow_s * 0.7), 0, 1)
        tilt = c.tilt * np.sin(a * L[:, 8] * TAU * 1.3 + L[:, 10]) + 0.05 * np.cos(a * L[:, 8] * TAU + L[:, 9])
        out = np.empty((len(L), 6), "f4")
        out[:, 0], out[:, 1] = L[:, 12], L[:, 1]
        out[:, 2] = L[:, 2] * scale
        out[:, 3] = tilt
        out[:, 4] = fade
        out[:, 5] = L[:, 11]
        self._buf.write(out.tobytes())
        self._prog["u_time"] = self.t
        self._vao.render(_STRIP, vertices=4, instances=len(L))
