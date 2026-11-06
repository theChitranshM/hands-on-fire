"""Diwali fireworks: eased rockets with a flickering head and crackling tail, six burst types (peony, chrysanthemum, willow, ring,
crossette, glitter), secondary crackle, drifting embers, a crescendo finale ending in a gold willow curtain. All via ParticleSystem."""
import numpy as np

TAU = 6.2831853


class Fireworks:
    def __init__(self, ctx, cfg, size, particles):
        self.p = particles
        self.size = size
        self.c = cfg.render.fireworks
        self.rng = np.random.default_rng(11)
        self.t = 0.0
        self._rockets = []      # dict(start, target, t0, flight, last, palette, scale, kind)
        self._events = []       # (time, fn)
        self.on_burst = None
        self.on_crackle = None
        self.on_launch = None

    @property
    def active(self):
        return bool(self._rockets or self._events)

    def clear(self):
        """Drop rockets in flight and pending bursts/crackles."""
        self._rockets, self._events = [], []

    def _pal(self, palette):
        pals = self.c.palettes
        if palette is None:
            palette = self.rng.choice(list(pals.keys()))
        if isinstance(palette, str):
            palette = pals[palette]
        return [tuple(c) for c in palette]

    def _kind(self, kind):
        if kind is None:
            names = list(self.c.kind_weights.keys())
            w = np.array([self.c.kind_weights[k] for k in names], float)
            kind = names[self.rng.choice(len(names), p=w / w.sum())]
        return kind

    def launch(self, start, target, flight_s, kind=None, palette=None, scale=1.0):
        """Rocket from start to target in flight_s seconds (fast then slowing); bursts on arrival. kind None = random type."""
        self._rockets.append(dict(start=np.array(start, "f4"), target=np.array(target, "f4"), t0=self.t,
                                  flight=max(float(flight_s), 0.05), last=np.array(start, "f4"), palette=palette,
                                  scale=scale, kind=kind))
        if self.on_launch:
            self.on_launch(tuple(start))

    def burst(self, pos, palette=None, scale=1.0, kind=None):
        """Burst now (kind: peony|chrysanthemum|willow|ring|crossette|glitter, None = random); crackle follows; embers drift down."""
        c, r, p = self.c, self.rng, self.p
        kind = self._kind(kind)
        k = c.kinds[kind]
        pos = np.asarray(pos, "f4")
        if k.get("palette"):
            palette = k["palette"]
        hot, cool = (np.asarray(x) * c.intensity for x in self._pal(palette))
        alt = np.asarray(self._pal(None)[0]) * c.intensity
        if k.get("shift"):                       # colour-changing stars: fade from the shell colour into a second hue
            cool = alt if np.abs(alt - hot).sum() > 0.6 else cool[::-1]
        S = c.speed * scale * k["speed"]
        n = int(c.burst_count * scale * k["count"])
        drag = k.get("drag", c.drag)
        kw = dict(gravity=c.gravity * scale * k["gravity"], drag=drag, twinkle=k["twinkle"], stretch=k["stretch"], life=tuple(k["life"]),
                  trail=k["trail"], trail_life=k["trail_life"], delay=c.start_stagger, fade=c.burst_fade)
        sz = (c.size * 0.8 * k["size"], c.size * 1.3 * k["size"])
        sp = c.speed_spread
        pts = np.repeat(pos[None], n, 0)
        hotc, coolc = tuple(hot), tuple(cool)
        if kind == "ring":
            a = r.uniform(0, TAU, n)
            tilt = r.uniform(0.35, 1.0)
            rot = r.uniform(0, TAU)
            sp_ = S * r.uniform(1 - sp * 0.25, 1.0, n)
            vx, vy = np.cos(a) * sp_, np.sin(a) * sp_ * tilt
            v = np.stack([vx * np.cos(rot) - vy * np.sin(rot), vx * np.sin(rot) + vy * np.cos(rot)], 1)
            p.emit(pts, 0, vel=v, speed=(0, 0), size=sz, color_hot=hotc, color_cool=coolc, **kw)
            m = int(n * 0.3)                      # a small inner burst in a contrasting colour fills the ring's centre
            p.emit(np.repeat(pos[None], m, 0), 0, speed=(S * 0.1, S * 0.4), size=(c.size, c.size * 1.4),
                   color_hot=(0.9 * c.intensity,) * 3, color_cool=tuple(alt), **{**kw, "trail": 0})
        else:
            n_in = int(n * c.ring_frac) if kind in ("peony", "chrysanthemum") else 0
            p.emit(pts[:n - n_in], 0, speed=(S * (1 - sp), S), size=sz, color_hot=hotc, color_cool=coolc,
                   branch=1.0 if kind == "crossette" else 0.0, branch_kind=1, **kw)
            if n_in:    # slower inner shell in a contrasting colour
                p.emit(pts[:n_in], 0, speed=(S * 0.35, S * 0.62), size=(c.size * 0.9, c.size * 1.4),
                       color_hot=(0.9 * c.intensity,) * 3, color_cool=tuple(alt), **{**kw, "trail": 0})
        # flash: a tight soft glow that is gone fast (the engine's bloom does the rest)
        f = c.flash_gain * k["flash"]
        p.emit(pos, 1, speed=(0, 0), size=(c.flash_size * scale,) * 2, life=(0.14, 0.18),
               color_hot=tuple(np.array((1.0, 0.95, 0.8)) * f), color_cool=hotc, stretch=0.0)
        p.emit(pos, 1, speed=(0, 0), size=(c.flash_size * scale * 0.5,) * 2, life=(0.25, 0.3),
               color_hot=(f,) * 3, color_cool=hotc, stretch=0.0)
        # embers drift down towards the hands
        p.emit(np.repeat(pos[None], int(c.embers * scale * k["embers"]), 0), 0, speed=(10, S * 0.35), size=(c.size * 0.7, c.size * 1.1),
               life=tuple(c.ember_life), color_hot=tuple(c.ember_color[0]), color_cool=tuple(c.ember_color[1]),
               gravity=c.ember_gravity, drag=0.6, twinkle=0.9, stretch=0.0, vel=(0, 20), delay=0.3)
        if self.on_burst:
            self.on_burst(tuple(pos))
        if k["crackle"]:
            delay = r.uniform(*c.crackle_delay)
            chunks = int(c.crackle_chunks)
            for i in range(chunks):
                self._events.append((self.t + delay + i * 0.06 + r.random() * 0.04,
                                     lambda pos=pos, scale=scale, i=i: self._crackle(pos, scale, i == 0)))

    def _crackle(self, pos, scale, first):
        c, p = self.c, self.p
        if first and self.on_crackle:
            self.on_crackle(tuple(pos))
        rad = c.speed * scale / c.drag * c.crackle_radius
        n = max(1, int(c.crackle_count * scale / c.crackle_chunks))
        ang = self.rng.uniform(0, TAU, n)
        rr = rad * np.sqrt(self.rng.random(n)) * 0.9
        sag = 0.5 * c.gravity * scale * 0.5          # burst has sagged by the time crackle fires
        pts = pos + np.stack([np.cos(ang) * rr, np.sin(ang) * rr + sag * 0.35], 1)
        ch, cc = [tuple(x) for x in c.crackle_color]
        p.emit(pts.astype("f4"), 0, speed=(0, 40), life=tuple(c.crackle_life), size=(c.crackle_size * 0.9, c.crackle_size * 1.6),
               color_hot=ch, color_cool=cc, twinkle=1.0, stretch=0.0, gravity=60, drag=1.0)

    def finale(self, duration=3.0):
        """Crescendo: staggered rockets of mixed types, sparse at first and tightening, then a gold willow curtain across the sky."""
        c, r = self.c, self.rng
        w, h = self.size
        x0, x1, y0, y1 = c.finale_area
        n = int(c.finale_bursts)
        kinds, names = list(c.finale_kinds), list(c.palettes.keys())
        for i in range(n):
            u = (i + r.random() * 0.5) / n
            tt = self.t + duration * u ** c.finale_ramp
            tgt = (w * r.uniform(x0, x1), h * r.uniform(y0, y1))
            start = (np.clip(tgt[0] + r.normal(0, w * 0.08), 0, w), h * 1.02)
            kind = kinds[i % len(kinds)]
            pal = None if kind == "willow" else names[i % len(names)] if r.random() < 0.8 else None
            sc = (0.8 + 0.4 * u) * r.uniform(0.9, 1.1)
            fl = r.uniform(*c.finale_flight)
            self._events.append((tt, lambda s=start, g=tgt, f=fl, k=kind, p=pal, sc=sc: self.launch(s, g, f, k, p, sc)))
        m = int(c.curtain_count)
        for i in range(m):
            tgt = (w * (0.1 + 0.8 * (i + r.uniform(-0.15, 0.15)) / max(m - 1, 1)), h * r.uniform(*c.curtain_y))
            order = (i * 2) % m                      # stagger outward-in instead of left to right
            self._events.append((self.t + duration + 0.15 + order * c.curtain_gap,
                                 lambda g=tgt: self.launch((g[0], h * 1.02), g, r.uniform(0.5, 0.7), "willow", None, c.curtain_scale)))

    def update(self, dt, now):
        self.t += dt
        c, r, p = self.c, self.rng, self.p
        for ev in [e for e in self._events if e[0] <= self.t]:
            self._events.remove(ev)
            ev[1]()
        keep = []
        for k in self._rockets:
            u = (self.t - k["t0"]) / k["flight"]
            e = 1.0 - (1.0 - min(u, 1.0)) ** 2.6
            pos = k["start"] + (k["target"] - k["start"]) * e
            mv = pos - k["last"]
            ln = float(np.hypot(*mv))
            back = -mv / ln if ln > 1e-3 else np.array([0.0, 1.0], "f4")
            ba = float(np.arctan2(back[1], back[0]))
            # crackling tail: twinkling sparks thrown in a narrow cone behind the head, some of which pop into tiny crackles
            n = max(1, int(c.rocket_trail_rate * dt))
            pts = k["last"] + mv * r.random((n, 1)).astype("f4")
            p.emit(pts, 0, speed=(10, 110), angle=(ba - 0.5, ba + 0.5), life=tuple(c.rocket_trail_life), size=(1.6, 3.4),
                   color_hot=(1.0, 0.96, 0.75), color_cool=(1.0, 0.4, 0.08), gravity=160, drag=2.0, twinkle=1.0, stretch=0.5,
                   branch=c.rocket_crackle, branch_kind=0)
            # head: hot core + flickering halo, size jittering every frame
            hs = c.rocket_head_size * (0.75 + 0.6 * r.random())
            hp = np.stack([k["last"], pos])
            p.emit(hp, 0, speed=(0, 0), life=(0.05, 0.07), size=(hs, hs * 1.2), color_hot=(1.0, 1.0, 0.9), color_cool=(1.0, 0.8, 0.4), stretch=0.0)
            p.emit(pos, 0, speed=(0, 0), life=(0.04, 0.05), size=(hs * 1.9, hs * 2.3), color_hot=(0.5, 0.35, 0.12),
                   color_cool=(0.5, 0.2, 0.05), stretch=0.0)
            k["last"] = pos
            if u >= 1.0:
                self.burst(k["target"], k["palette"], k["scale"], k["kind"])
            else:
                keep.append(k)
        self._rockets = keep

    def render(self):
        """Rocket heads/trails are particles; nothing extra to draw."""
