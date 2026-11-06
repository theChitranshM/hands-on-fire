"""Procedural flames (fbm, domain-warped, rising), clay diyas, palm light pools, smoke, embers and glow dots. Direct layer."""
import numpy as np
import moderngl

from hof.render.engine import GLSL_PX_TO_CLIP

GLSL_NOISE = """
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 3.7); a *= 0.5; }
    return s;
}
// never-repeating flicker in about [-1, 1]: three noise octaves at unrelated rates
float flick(float t, float s) {
    return (vnoise(vec2(t * 2.3, s)) - 0.5) * 1.2 + (vnoise(vec2(t * 7.1 + 3.0, s * 1.7)) - 0.5) * 0.9
         + (vnoise(vec2(t * 17.3, s * 0.6 + 9.0)) - 0.5) * 0.5;
}
"""

# Flame, diya, pool and smoke shaders share: instanced quad from gl_VertexID, v_q = local coords in flame-height units (y up).
# Per-instance row: x y h intensity flare seed lean stretch glow calm
_VERT = """
#version 410
in vec2 in_pos;
in float in_h, in_int, in_flare, in_seed, in_lean, in_stretch, in_glow, in_calm;
uniform vec4 u_box;          // x0 x1 y0 y1 in units of H (y up)
uniform float u_flare_scale;
out vec2 v_q;
out float v_int, v_flare, v_seed, v_H, v_lean, v_stretch, v_glow, v_calm;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    float H = in_h * (1.0 + u_flare_scale * in_flare);
    vec2 q = vec2(mix(u_box.x, u_box.y, c.x), mix(u_box.z, u_box.w, c.y));
    v_q = q; v_int = in_int; v_flare = in_flare; v_seed = in_seed; v_H = H;
    v_lean = in_lean; v_stretch = in_stretch; v_glow = in_glow; v_calm = in_calm;
    gl_Position = px_to_clip(in_pos + vec2(q.x, -q.y) * H);
}
"""
_VARY = "in vec2 v_q; in float v_int, v_flare, v_seed, v_H, v_lean, v_stretch, v_glow, v_calm;\n"

_FLAME_FRAG = """
#version 410
""" + _VARY + """
uniform float u_time, u_speed, u_fh, u_sway, u_halo_r, u_halo_gain, u_glow, u_gust, u_flame_gain;
uniform vec3 u_core, u_body, u_tip;
out vec4 f;
""" + GLSL_NOISE + """
void main() {
    float t = u_time * u_speed + v_seed * 31.7;
    float fl = v_flare, calm = v_calm;
    float fk = flick(t, v_seed);
    float gust = smoothstep(0.7, 0.95, vnoise(vec2(t * 0.6, v_seed + 4.0))) * (1.0 - calm);   // now and then the flame ducks and leans
    float hs = 1.0 + u_fh * (1.0 - 0.6 * calm) * (fk * 1.5 - 0.4 * gust * u_gust) + v_stretch;
    hs *= 1.0 + 0.14 * fl;
    float sway = u_sway * (1.0 - 0.5 * calm) * (flick(t * 0.7 + 11.0, v_seed * 5.0) * 1.8 + gust * u_gust * 0.8) + 0.03 * sin(t * 1.9) * calm;
    vec2 q = v_q;
    float y = q.y / hs;
    float x = q.x - (sway + v_lean) * y * y;
    vec2 np = vec2(x * 2.6, y * 1.9 - t * 2.3);
    float w1 = fbm(np);
    float w2 = fbm(np * 1.8 + vec2(w1 * 2.2, -t * 0.8) + 5.2);
    x += (w2 - 0.45) * 0.42 * (0.1 + y * 0.9);
    float prof = 0.27 * pow(max(1.0 - y, 0.0), 0.85) * smoothstep(-0.10, 0.06, y);
    prof *= 1.0 + 0.3 * exp(-pow((y - 0.14) / 0.16, 2.0));
    prof *= 1.0 + 0.25 * fl;
    float ap = abs(x) / max(prof, 1e-3);
    float d = 1.0 - ap;
    d -= (w1 - 0.42) * 0.45 * smoothstep(0.3, 1.0, y);
    d = clamp(d, 0.0, 1.0) * smoothstep(0.0, 0.03, prof);
    float mask = smoothstep(0.0, 0.22, d);
    float heat = pow(d, 1.3) * (1.0 - 0.5 * y) + 0.12 * fl;
    vec3 col = mix(u_tip, u_body, smoothstep(0.08, 0.45, heat));
    col = mix(col, u_core, smoothstep(0.6, 0.95, heat));
    // outer sheath: wider, softer, deep orange-red, reaching a little above the body
    float ds = clamp(1.0 - abs(x) / max(prof * 1.45, 1e-3) - (y - 0.05) * 0.35, 0.0, 1.0) * smoothstep(0.0, 0.03, prof);
    vec3 flame = col * mask * (0.62 + 0.28 * heat + fl * 0.3) + u_tip * vec3(1.0, 0.7, 0.5) * pow(ds, 2.2) * 0.32 * (1.0 - mask);
    // inner flame: narrow, short, yellow-white teardrop with its own flicker
    float d2 = clamp(1.0 - abs(x) / max(prof * 0.42, 1e-3) - y * (0.9 + 0.12 * fk) + 0.1, 0.0, 1.0) * smoothstep(0.0, 0.03, prof);
    flame = mix(flame, vec3(1.0, 0.93, 0.7) * 1.05, smoothstep(0.2, 0.85, d2) * 0.8);
    // blue base at the wick
    float blue = exp(-pow((y + 0.01) / 0.08, 2.0)) * smoothstep(0.35, 0.0, ap) * 0.45;
    flame += vec3(0.2, 0.38, 1.0) * blue * max(mask, ds);
    // tight halo + wide soft glow (the palm pool is a separate pass)
    vec2 hq = vec2(q.x, q.y - 0.4) / u_halo_r;
    float r = length(hq);
    float halo = pow(max(1.0 - r, 0.0), 2.0) * u_halo_gain * (0.85 + 0.2 * fk);
    halo *= 1.0 + fl * 0.5;
    vec3 hc = vec3(1.0, 0.5, 0.12) * halo + vec3(1.0, 0.8, 0.35) * pow(max(1.0 - r * 1.7, 0.0), 2.5) * 0.6;
    hc += fl * vec3(1.0, 0.85, 0.55) * pow(max(1.0 - r * 1.3, 0.0), 2.0) * 0.45;
    f = vec4((flame * u_flame_gain + hc * u_glow * v_glow) * v_int, 1.0);
}
"""

_POOL_FRAG = """
#version 410
""" + _VARY + """
uniform float u_time, u_speed, u_pool_gain, u_pool_r, u_breath;
uniform vec3 u_pool_col;
out vec4 f;
""" + GLSL_NOISE + """
void main() {
    float t = u_time * u_speed + v_seed * 31.7;
    float fk = 0.9 + 0.12 * flick(t, v_seed) * (1.0 - v_calm * 0.6);
    fk *= 1.0 + v_calm * u_breath * sin(u_time * 1.3 + v_seed);           // slow breathing in the quiet end
    vec2 e = vec2(v_q.x / u_pool_r, (v_q.y + 0.1) / (u_pool_r * 0.62));
    float r = length(e);
    float g = pow(max(1.0 - r, 0.0), 1.8);
    vec3 c = u_pool_col * g * 0.8 + vec3(1.0, 0.62, 0.22) * pow(max(1.0 - r * 1.7, 0.0), 2.2) * 0.55;
    f = vec4(c * u_pool_gain * v_glow * fk * (1.0 + 0.6 * v_flare) * v_int, 1.0);
}
"""

_DIYA_FRAG = """
#version 410
""" + _VARY + """
uniform float u_time, u_speed, u_gain;
uniform vec3 u_clay, u_rim, u_oil;
out vec4 f;
""" + GLSL_NOISE + """
float cov(vec2 p, vec2 c, vec2 ab, float aa) { return 1.0 - smoothstep(1.0 - aa / min(ab.x, ab.y), 1.0, length((p - c) / ab)); }
void main() {
    vec2 p = vec2(v_q.x, -v_q.y);          // y down, 0 = wick / oil level
    float aa = 1.6 / v_H;
    float t = u_time * u_speed + v_seed * 31.7;
    float fk = (0.92 + 0.1 * flick(t, v_seed)) * (1.0 + 0.7 * v_flare);   // the bowl is lit by the flame: it flickers with it
    vec2 bc = vec2(0.0, 0.02), ab = vec2(0.52, 0.34);
    float down = smoothstep(0.0, aa * 2.0, p.y - 0.02);
    float body = cov(p, bc, ab, aa) * down;
    float spout = cov(p, vec2(0.56, 0.035), vec2(0.15, 0.065), aa) + cov(p, vec2(0.50, 0.09), vec2(0.12, 0.07), aa) * 0.0;
    float foot = cov(p, vec2(0.0, 0.345), vec2(0.21, 0.055), aa);
    float rimo = cov(p, vec2(0.0, 0.02), vec2(0.545, 0.095), aa);
    float rimi = cov(p, vec2(0.0, 0.022), vec2(0.455, 0.062), aa);
    // terracotta body lit from above by the flame, warm bounce from the palm along the bottom edge
    float e = length((p - bc) / ab);
    float lit = 0.16 + 0.8 * exp(-max(p.y - 0.02, 0.0) / 0.13) * (1.0 - 0.45 * pow(abs(p.x) / 0.52, 2.0)) + 0.35 * pow(clamp(e, 0.0, 1.0), 5.0) * smoothstep(0.1, 0.3, p.y);
    vec3 col = u_clay * lit * fk * (0.9 + 0.2 * vnoise(p * 40.0));          // faint clay grain
    // painted dots: two rows following the bowl curve, cream on terracotta
    float dots = 0.0;
    for (int k = -4; k <= 4; k++) {
        float a = float(k) * 0.36;
        vec2 c1 = bc + 0.74 * vec2(ab.x * sin(a), ab.y * cos(a));
        dots = max(dots, 1.0 - smoothstep(0.0, aa * 2.0 + 0.012, length(p - c1) - 0.026 * (0.5 + 0.5 * cos(a))));
        float a2 = (float(k) + 0.5) * 0.36;
        vec2 c2 = bc + 0.46 * vec2(ab.x * sin(a2), ab.y * cos(a2));
        dots = max(dots, 1.0 - smoothstep(0.0, aa * 2.0 + 0.01, length(p - c2) - 0.017 * (0.5 + 0.5 * cos(a2))));
    }
    col = mix(col, vec3(1.0, 0.86, 0.55) * lit * fk * 1.15, dots * body);
    // a thin gold band under the rim
    float band = (1.0 - smoothstep(0.0, aa * 2.0, abs(e - 0.9) * ab.y - 0.008)) * body;
    col = mix(col, u_rim * lit * fk * 0.9, band * 0.6);
    col = mix(col, u_clay * 0.7 * lit * fk, foot * (1.0 - body));
    col = mix(col, u_clay * 0.9 * fk * (0.4 + lit * 0.5), spout);
    // lit rim: a gold roll with a bright front edge
    float rimband = clamp(rimo - rimi, 0.0, 1.0);
    col = mix(col, u_rim * (0.85 + 0.6 * smoothstep(-0.02, 0.1, p.y)) * fk * 1.25, rimband);
    // glinting oil: warm gradient, a bright reflection streak of the flame, a hot pinprick at the wick
    float oilg = 0.55 + 0.7 * (1.0 - clamp(length(vec2(p.x / 0.45, (p.y - 0.022) / 0.06)), 0.0, 1.0));
    vec3 oil = u_oil * oilg * fk * 1.15;
    float streak = exp(-pow(p.x / (0.05 + 0.03 * abs(p.y)), 2.0)) * exp(-pow((p.y - 0.02) / 0.04, 2.0));
    float glint = exp(-pow((p.x - 0.19 + 0.04 * sin(t * 1.3)) / 0.03, 2.0)) * exp(-pow((p.y - 0.03) / 0.012, 2.0));
    oil += vec3(1.0, 0.9, 0.55) * (streak * 0.9 + glint * 1.1) * fk;
    col = mix(col, oil, rimi);
    float a = max(max(body, spout), max(foot, rimo));
    f = vec4(col * a * u_gain * v_int, 1.0);
}
"""

_SMOKE_FRAG = """
#version 410
""" + _VARY + """
uniform vec3 u_col;
uniform float u_gain;
out vec4 f;
""" + GLSL_NOISE + """
void main() {
    float a = v_flare;                       // smoke age 0..1 carried in the flare slot
    float rise = 0.3 + 2.4 * a;
    float yy = v_q.y;
    float s = clamp(yy / rise, 0.0, 1.0);
    // a curl: lazy S-shaped wave that drifts and widens as it rises
    float cx = sin(yy * 4.2 - a * 6.0 + v_seed * 6.0) * 0.25 * (0.2 + s) + 0.18 * a * sin(v_seed * 11.0);
    float w = 0.09 + 0.2 * yy;
    float n = fbm(vec2((v_q.x - cx) * 3.4, yy * 1.9 - a * 3.0) + v_seed * 10.0);
    float dens = exp(-pow((v_q.x - cx) / w, 2.0)) * smoothstep(0.15, 0.6, n + 0.28);
    dens *= smoothstep(0.0, 0.1, yy) * (1.0 - smoothstep(0.6, 1.0, s)) * step(yy, rise);
    float fade = pow(max(1.0 - a, 0.0), 0.9) * smoothstep(0.0, 0.06, a);
    // the lower smoke is still lit warm by the dying wick
    vec3 c = mix(u_col, vec3(1.0, 0.55, 0.22), exp(-yy * 3.0) * (1.0 - a));
    f = vec4(c * dens * fade * u_gain * v_int, 1.0);
}
"""

# glow dots: per-instance colour. Used for sparkler cores and wires, embers, the dying wick.
_GVERT = """
#version 410
in vec2 in_pos;
in float in_r, in_int;
in vec3 in_col;
out vec2 v_q;
out float v_int;
out vec3 v_col;
""" + GLSL_PX_TO_CLIP + """
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    v_q = c; v_int = in_int; v_col = in_col;
    gl_Position = px_to_clip(in_pos + c * in_r);
}
"""
_GLOW_FRAG = """
#version 410
in vec2 v_q; in float v_int; in vec3 v_col;
out vec4 f;
void main() {
    float r = length(v_q);
    float g = pow(max(1.0 - r, 0.0), 2.2) * 0.8 + exp(-r * r * 26.0) * 1.4;
    f = vec4(mix(v_col, vec3(1.0), exp(-r * r * 30.0)) * g * v_int, 1.0);
}
"""

_STRIDE = 10


class FlameRenderer:
    def __init__(self, ctx, cfg, size):
        self.ctx, self.size = ctx, size
        self.c = cfg.render.flame
        self.scale = float(cfg.render.scale)
        self.frac = float(self.c.size_palm_frac)
        self.t = 0.0
        self.dt = 1 / 60
        self.rng = np.random.default_rng(5)
        self._flames, self._diyas, self._pools, self._glows = [], [], [], []
        self._smoke = []                          # [x, y, h, age_s, seed]
        self._wicks = []                          # [x, y, h, age_s]: glowing wick dying after a snuff
        self._emb = []                            # [x, y, vx, vy, age, life, r, seed, h]
        self._lean = {}                           # seed -> [lean, stretch] smoothed
        c = self.c

        def mk(frag, n):
            prog = ctx.program(vertex_shader=_VERT, fragment_shader=frag)
            buf = ctx.buffer(reserve=n * _STRIDE * 4, dynamic=True)
            vao = ctx.vertex_array(prog, [(buf, "2f " + "1f " * 8 + "/i", "in_pos", "in_h", "in_int", "in_flare", "in_seed",
                                           "in_lean", "in_stretch", "in_glow", "in_calm")])
            prog["u_res"] = size
            prog["u_flare_scale"] = float(c.flare_scale)
            return prog, buf, vao

        self._fp, self._fb, self._fv = mk(_FLAME_FRAG, c.max)
        self._dp, self._db, self._dv = mk(_DIYA_FRAG, c.max)
        self._pp, self._pb, self._pv = mk(_POOL_FRAG, c.max)
        self._sp, self._sb, self._sv = mk(_SMOKE_FRAG, c.max_smoke)
        self._gp = ctx.program(vertex_shader=_GVERT, fragment_shader=_GLOW_FRAG)
        self._gcap = int(c.max_glow)
        self._gb = ctx.buffer(reserve=self._gcap * 7 * 4, dynamic=True)
        self._gv = ctx.vertex_array(self._gp, [(self._gb, "2f 1f 1f 3f /i", "in_pos", "in_r", "in_int", "in_col")])
        self._gp["u_res"] = size
        self._sp["u_flare_scale"] = self._pp["u_flare_scale"] = 0.0
        hr = float(c.halo_radius)
        pr = float(c.pool_radius)
        self._fp["u_box"] = (-hr - 0.05, hr + 0.05, -hr + 0.4 - 0.05, hr + 0.4 + 0.05)
        self._dp["u_box"] = (-0.8, 0.8, -0.62, 0.15)
        self._pp["u_box"] = (-pr, pr, -0.1 - pr * 0.62, -0.1 + pr * 0.62)
        self._sp["u_box"] = (-1.0, 1.0, -0.1, float(c.smoke_height))
        for k, v in (("u_speed", c.flicker_speed), ("u_fh", c.flicker_height), ("u_sway", c.flicker_sway),
                     ("u_halo_r", hr), ("u_halo_gain", c.halo_gain), ("u_glow", c.glow), ("u_gust", c.gust),
                     ("u_flame_gain", c.flame_gain)):
            self._fp[k] = float(v)
        self._fp["u_core"], self._fp["u_body"], self._fp["u_tip"] = (tuple(c[k]) for k in ("core_color", "body_color", "tip_color"))
        self._dp["u_speed"] = self._pp["u_speed"] = float(c.flicker_speed)
        self._dp["u_clay"], self._dp["u_rim"], self._dp["u_oil"] = (tuple(c[k]) for k in ("diya_color", "rim_color", "oil_color"))
        self._dp["u_gain"] = float(c.diya_gain)
        self._pp["u_pool_gain"], self._pp["u_pool_r"], self._pp["u_breath"] = float(c.pool_gain), pr, float(c.breath)
        self._pp["u_pool_col"] = tuple(c.pool_color)
        self._sp["u_col"] = tuple(c.smoke_color)
        self._sp["u_gain"] = float(c.smoke_gain)

    def _h(self, size):
        return float(size) * self.frac * self.scale

    def add(self, pos, size, intensity=1.0, flare=0.0, diya=True, seed=0.0, vel=(0.0, 0.0), glow=1.0, calm=0.0, embers=1.0):
        """Queue one flame (base at pos). Flame height = size * size_palm_frac * scale.
        vel = hand velocity px/s: the flame leans against it. glow scales pool + halo. calm 0..1 = steadier flame, slow breathing pool.
        embers scales the rate of tiny sparks lifting off the tip."""
        if intensity <= 0.003 or len(self._flames) >= self.c.max:
            return
        c = self.c
        h = self._h(size)
        st = self._lean.setdefault(round(seed), [0.0, 0.0])
        k = min(1.0, self.dt * float(c.lean_rate))
        st[0] += (float(np.clip(-vel[0] / h * c.lean_k, -c.lean_max, c.lean_max)) - st[0]) * k
        st[1] += (float(np.clip(vel[1] / h * c.stretch_k, -c.stretch_max, c.stretch_max)) - st[1]) * k
        row = (float(pos[0]), float(pos[1]), h, float(intensity), float(flare), float(seed), st[0], st[1], float(glow), float(calm))
        self._flames.append(row)
        if diya:
            self._diyas.append((row[0], row[1], h * float(c.diya_size_frac), *row[3:]))
            if c.pool_gain > 0:
                self._pools.append((row[0], row[1] + h * 0.1, h, *row[3:]))
        rate = float(c.ember_rate) * embers * intensity * (1.0 - 0.4 * calm)
        if rate > 0 and self.rng.random() < rate * self.dt and len(self._emb) < c.max_embers:
            r = self.rng
            sp = r.uniform(*c.ember_speed) * (1.0 - 0.45 * calm)
            life = r.uniform(*c.ember_life) * (1.0 + 0.8 * calm)
            self._emb.append([row[0] + st[0] * h * 0.5 + r.normal(0, h * 0.05), row[1] - h * 0.85, r.normal(0, 8) + st[0] * 60, -sp * h,
                              0.0, life, h * float(c.ember_size), r.random() * 6.28, h])

    def add_glow(self, pos, radius, color=(1.0, 0.8, 0.4), intensity=1.0):
        """Queue a soft white-hot glow dot (sparkler burning point)."""
        if len(self._glows) < self._gcap:
            self._glows.append((float(pos[0]), float(pos[1]), float(radius), float(intensity), *(float(x) for x in color)))

    def snuff(self, pos, size):
        """Spawn a smoke curl rising from pos and leave the wick glowing red for a second."""
        h = self._h(size)
        if len(self._smoke) < self.c.max_smoke:
            self._smoke.append([float(pos[0]), float(pos[1]), h, 0.0, float(self.rng.random())])
        self._wicks.append([float(pos[0]), float(pos[1]), h, 0.0])

    def clear(self):
        """Drop smoke, embers and dying wicks (the sequence calls this when the show ends)."""
        self._smoke.clear(); self._wicks.clear(); self._emb.clear()

    def update(self, dt, now):
        self.t = now
        self.dt = max(dt, 1e-3)
        for s in self._smoke:
            s[3] += dt
        self._smoke = [s for s in self._smoke if s[3] < self.c.smoke_s]
        for w in self._wicks:
            w[3] += dt
        self._wicks = [w for w in self._wicks if w[3] < self.c.wick_s]
        for e in self._emb:
            e[4] += dt
            e[0] += (e[2] + 14.0 * np.sin(e[4] * 2.2 + e[7]) * e[8] / 100.0) * dt
            e[1] += e[3] * dt
            e[3] *= 1.0 - 0.35 * dt
        self._emb = [e for e in self._emb if e[4] < e[5]]

    def _draw(self, prog, buf, vao, rows):
        if not rows:
            return
        buf.write(np.array(rows, dtype="f4").tobytes())
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(rows))

    def render(self):
        """Draw queued flames, diyas, pools, smoke, embers, glows into the current (additive) framebuffer; clear the queue."""
        c = self.c
        self._dp["u_time"] = self._fp["u_time"] = self._pp["u_time"] = self.t
        self._draw(self._pp, self._pb, self._pv, self._pools)
        self._draw(self._dp, self._db, self._dv, self._diyas)
        self._draw(self._fp, self._fb, self._fv, self._flames)
        if self._smoke:
            rows = [(s[0], s[1], s[2] * 0.9, 1.0, s[3] / c.smoke_s, s[4], 0, 0, 1, 0) for s in self._smoke]
            self._draw(self._sp, self._sb, self._sv, rows)
        g = list(self._glows)
        for w in self._wicks:     # the wick glows white-orange, then red, then gone
            a = w[3] / c.wick_s
            k = (1.0 - a) ** 1.6 * (0.85 + 0.15 * np.sin(self.t * 23.0 + w[0]))
            col = (1.0, 0.75 - 0.5 * a, 0.35 - 0.3 * a)
            g.append((w[0], w[1] - w[2] * 0.02, w[2] * 0.16, k * 1.5, *col))
            g.append((w[0], w[1] - w[2] * 0.02, w[2] * 0.4, k * 0.35, 1.0, 0.4, 0.1))
        for e in self._emb:
            a = e[4] / e[5]
            tw = 0.7 + 0.3 * np.sin(e[4] * 17.0 + e[7] * 5.0)
            g.append((e[0], e[1], e[6] * (1.0 - 0.4 * a), (1.0 - a) ** 0.8 * min(1.0, e[4] * 6.0) * tw * c.ember_gain,
                      1.0, 0.55 + 0.25 * (1 - a), 0.15))
        if g:
            g = g[:self._gcap]
            self._gb.write(np.array(g, dtype="f4").tobytes())
            self._gv.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(g))
        self._flames.clear(); self._diyas.clear(); self._pools.clear(); self._glows.clear()
