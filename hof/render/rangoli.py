"""Procedural rangoli / kolam: layered petal rings of different counts, a dot ring and an outer circle that trace themselves on
(strokes first, then the fill floods in), slow counter-rotation, glowing edges with a light chasing round. Scatters into particles. Direct layer."""
import numpy as np
import moderngl

from hof.render.engine import GLSL_PX_TO_CLIP

TAU = 6.2831853
MAXL, MAXD = 6, 3

_VERT = """
#version 410
uniform vec2 u_center; uniform float u_R;
out vec2 v_p;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    v_p = c * u_R * 1.35;
    gl_Position = px_to_clip(u_center + v_p);
}
"""

_FRAG = """
#version 410
#define TAU 6.2831853
in vec2 v_p;
uniform float u_R, u_prog, u_fade, u_rot, u_stagger, u_fill, u_edge_gain, u_time, u_line_px;
uniform int u_nl, u_nd, u_ncol;
uniform vec4 u_layer[6];      // count, r0, r1, colour index   (fractions of R, inner -> outer)
uniform float u_kind[6];      // 0 round petal, 1 pointed
uniform vec4 u_dot[3];        // count, radius, dot radius, colour index
uniform vec3 u_cols[8];
out vec4 f;

vec3 colr(int i) { return u_cols[i % u_ncol]; }
float ease(float x) { x = clamp(x, 0.0, 1.0); return 1.0 - pow(1.0 - x, 3.0); }
float aa(float x, float w) { return 1.0 - smoothstep(0.0, w, x); }

// one petal ring: rgb + coverage. lp = this ring's own 0..1 progress
vec4 petals(int j, float lp) {
    vec4 L = u_layer[j];
    float N = L.x, r0 = L.y * u_R, r1 = L.z * u_R;
    float r = length(v_p);
    if (r < r0 || r > r1) return vec4(0.0);
    float sector = TAU / N;
    float dirn = (j & 1) == 1 ? -1.0 : 1.0;
    float a = atan(v_p.y, v_p.x) + u_rot * dirn * (1.0 + 0.35 * float(j)) + float(j & 1) * sector * 0.5;
    float ap = mod(a, sector) - sector * 0.5;
    float s = (r - r0) / (r1 - r0);
    float pk = u_kind[j];
    float prof = pk > 0.5 ? pow(sin(3.14159 * pow(s, 0.55)), 1.0) : pow(sin(3.14159 * pow(s, 0.7)), 0.65);
    float ha = sector * 0.5 * 0.96 * prof;
    float lat = (ha - abs(ap)) * r;                     // px inside the petal outline
    float e = min(lat, min(r - r0, r1 - r) * 0.9);
    if (e < -u_line_px) return vec4(0.0);
    vec3 base = colr(int(L.w));
    // stroke: traces base -> tip on both sides, then the fill floods in
    float trace = ease(lp * 1.7);
    float stroke = aa(abs(e) - u_line_px * 0.5, u_line_px * 0.7) * step(s, trace);
    float fillk = smoothstep(0.45, 1.0, lp);
    float inside = smoothstep(0.0, 1.5, e) * fillk;
    vec3 col = base * u_fill * (0.55 + 0.6 * s) * inside;
    // inner vein + inlay dot
    float vein = aa(abs(ap) * r - u_line_px * 0.25, 1.2) * (1.0 - s) * inside;
    col += base * vein * 0.7 + vec3(1.0, 0.95, 0.85) * vein * 0.25;
    float rd = r0 + (r1 - r0) * 0.55;
    float dd = length(vec2(r * sin(ap), r * cos(ap) - rd));
    float dr = 0.05 * u_R * (0.7 + 0.3 * float(pk)) * smoothstep(0.6, 1.0, lp);
    col = mix(col, colr(int(L.w) + 1) * 1.1, aa(dd - dr, 1.5) * inside);
    // glowing edge with a light chasing round the ring
    float chase = 0.8 + 0.4 * pow(0.5 + 0.5 * sin(a * N * 0.5 - u_time * 2.2 * dirn + float(j)), 3.0);
    col += mix(base, vec3(1.0), 0.18) * u_edge_gain * chase * stroke;
    return vec4(col, max(max(stroke, inside), 0.0));
}

void main() {
    float r = length(v_p);
    float ang = atan(v_p.y, v_p.x);
    float turn = fract((ang - u_rot * 0.5) / TAU);           // 0..1 around, for angular sweeps
    vec3 acc = vec3(0.0);
    // dot rings and the outer circle first (they sit behind the petals): dots pop in one after another round the ring
    for (int i = 0; i < u_nd; i++) {
        vec4 D = u_dot[i];
        float N = D.x, sec = TAU / N;
        float ra = (i & 1) == 1 ? u_rot * 0.5 : -u_rot * 0.5;
        float a2 = ang + ra;
        float idx = floor(a2 / sec + 0.5);
        float ap = a2 - idx * sec;
        float dd = length(vec2(r * sin(ap), r * cos(ap) - D.y * u_R));
        float vis = smoothstep(0.0, 0.08, u_prog * 1.5 - 0.3 - fract((a2 / TAU) + 8.0) * 0.55 - float(i) * 0.08);
        float tw = 0.8 + 0.2 * sin(u_time * 3.0 + idx * 1.7);
        vec3 dc = colr(int(D.w) + int(mod(idx, 2.0))) * 1.15 * tw;
        float dr = D.z * u_R * vis;
        acc += dc * aa(dd - dr, 1.6) + vec3(1.0, 0.95, 0.85) * aa(dd - dr * 0.4, 1.2) * 0.35 * vis;
    }
    float cr = 0.985 * u_R;
    float cp = ease(u_prog * 1.4 - 0.05);
    acc += colr(1) * 1.1 * aa(abs(r - cr) - u_line_px * 0.55, u_line_px * 0.8) * step(turn, cp);
    float ci = 0.19 * u_R;
    acc += colr(0) * 1.0 * aa(abs(r - ci) - u_line_px * 0.45, u_line_px * 0.7) * step(turn, ease(u_prog * 1.4)) * 0.8;
    // petal rings, outer to inner, each overlapping the last
    float L = float(u_nl);
    for (int j = u_nl - 1; j >= 0; j--) {
        float lp = clamp(u_prog * (1.0 + (L - 1.0) * u_stagger) - float(u_nl - 1 - j) * u_stagger * 0.6 - float(j) * 0.0, 0.0, 1.0);
        lp = clamp(u_prog * (1.0 + (L - 1.0) * u_stagger) - float(j) * u_stagger, 0.0, 1.0);
        vec4 rc = petals(j, lp);
        acc = mix(acc, rc.rgb, clamp(rc.a, 0.0, 1.0));
    }
    // heart: a white-hot bead in a ring of marigold
    float pc = ease(u_prog * 2.0);
    float rc = 0.07 * u_R * pc;
    acc = mix(acc, mix(colr(1) * 1.5, vec3(1.0, 0.97, 0.85) * 1.6, 1.0 - smoothstep(0.0, rc, r)), aa(r - rc, 1.5));
    f = vec4(acc * u_fade * (0.93 + 0.07 * sin(u_time * 3.0)), 1.0);
}
"""


class Rangoli:
    def __init__(self, ctx, cfg, size, particles):
        self.p = particles
        self.c = cfg.render.rangoli
        self.scale = float(cfg.render.scale)
        self.rng = np.random.default_rng(3)
        self.center = np.array([size[0] / 2, size[1] / 2], "f4")
        self.R = 100.0
        self.t = 0.0
        self.clock = 0.0
        self.fade = 1.0
        self.state = "off"       # off | bloom | scatter
        self._prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self._vao = ctx.vertex_array(self._prog, [])
        c = self.c
        self.cols = [tuple(x) for x in c.colors][:8]
        self.layers = [dict(l) for l in c.layers][:MAXL]
        pr = self._prog
        pr["u_res"] = size
        pr["u_ncol"] = len(self.cols)
        pr["u_cols"].value = self.cols + [self.cols[0]] * (8 - len(self.cols))
        pr["u_nl"] = len(self.layers)
        L = [(l["n"], l["r0"], l["r1"], l["col"]) for l in self.layers]
        pr["u_layer"].value = L + [(1, 0, 0, 0)] * (MAXL - len(L))
        pr["u_kind"].value = [float(l.get("kind", 0)) for l in self.layers] + [0.0] * (MAXL - len(L))
        dots = [dict(d) for d in c.dots][:MAXD]
        pr["u_nd"] = len(dots)
        D = [(d["n"], d["r"], d["size"], d["col"]) for d in dots]
        pr["u_dot"].value = D + [(1, 0, 0, 0)] * (MAXD - len(D))
        pr["u_stagger"] = float(c.ring_stagger)
        pr["u_fill"] = float(c.fill_gain)
        pr["u_edge_gain"] = float(c.edge_gain)

    @property
    def active(self):
        return self.state != "off"

    def bloom(self, center, scale):
        self.center = np.array(center, "f4")
        self.R = float(scale) * float(self.c.radius_palm_frac) * self.scale
        self.t, self.fade, self.state = 0.0, 1.0, "bloom"

    def set_center(self, center):
        self.center = np.array(center, "f4")

    def scatter(self):
        """Burst the petals into coloured particles (each ring in its own colour, a few white sparks) and fade the shape."""
        if self.state != "bloom":
            return
        c, rng = self.c, self.rng
        n = int(c.scatter_count)
        prog = min(self.t / c.bloom_s, 1.0)
        w = np.array([(l["r1"] ** 2 - l["r0"] ** 2) for l in self.layers])
        w = w / w.sum()
        for j, l in enumerate(self.layers):
            m = max(1, int(n * w[j]))
            r = self.R * rng.uniform(l["r0"], l["r0"] + (l["r1"] - l["r0"]) * min(prog * 1.2, 1.0), m)
            a = rng.uniform(0, TAU, m)
            d = np.stack([np.cos(a), np.sin(a)], 1)
            pos = (self.center + d * r[:, None]).astype("f4")
            sp = rng.uniform(*c.scatter_speed, m)
            sw = np.stack([-d[:, 1], d[:, 0]], 1) * rng.uniform(-110, 110, (m, 1))       # a swirl so the petals tumble apart
            col = np.array(self.cols[int(l["col"]) % len(self.cols)])
            self.p.emit(pos, 0, vel=d * sp[:, None] + sw, speed=(0, 0), life=tuple(c.scatter_life), size=(3.0, 5.5),
                        color_hot=tuple(np.minimum(col * 0.8 + 0.2, 1.0) * c.scatter_gain), color_cool=tuple(col * 0.5 * c.scatter_gain),
                        gravity=140, drag=1.6, twinkle=0.4, stretch=0.5)
        k = int(n * 0.08)
        a = rng.uniform(0, TAU, k)
        d = np.stack([np.cos(a), np.sin(a)], 1)
        self.p.emit((self.center + d * self.R * rng.uniform(0.1, 1.0, (k, 1))).astype("f4"), 0, vel=d * rng.uniform(150, 600, (k, 1)),
                    speed=(0, 0), life=(0.5, 1.2), size=(2.0, 3.5), color_hot=(0.7, 0.65, 0.55), color_cool=(0.5, 0.3, 0.15),
                    gravity=100, drag=1.8, twinkle=0.9, stretch=0.4)
        self.state, self.t = "scatter", 0.0

    def update(self, dt, now):
        self.clock += dt
        if self.state == "bloom":
            self.t += dt
        elif self.state == "scatter":
            self.t += dt
            self.fade = max(0.0, 1.0 - self.t / self.c.scatter_fade_s)
            if self.fade <= 0.0:
                self.state = "off"

    def render(self):
        if self.state == "off":
            return
        pr = self._prog
        pr["u_center"] = tuple(self.center)
        pr["u_R"] = self.R
        pr["u_prog"] = 1.0 if self.state == "scatter" else min(self.t / self.c.bloom_s, 1.0)
        pr["u_fade"] = self.fade
        pr["u_rot"] = self.clock * float(self.c.spin)
        pr["u_time"] = self.clock
        pr["u_line_px"] = max(float(self.c.line_px), float(self.c.line_frac) * self.R)
        self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
