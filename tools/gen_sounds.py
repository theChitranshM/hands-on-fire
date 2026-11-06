"""Synthesize the show's sounds (numpy only, deterministic) into cfg.audio.dir. Run: python -m tools.gen_sounds"""
import os
import wave
import numpy as np
from hof.config import load_config, path


def _rng(cfg, k):
    return np.random.default_rng(cfg.audio.seed + k)


def _noise(rng, n, lo=None, hi=None, sr=44100):
    """White noise, optionally band-limited with a smooth FFT mask (circular, so loops cleanly)."""
    x = rng.standard_normal(n)
    if lo is None and hi is None:
        return x
    f = np.fft.rfftfreq(n, 1 / sr)
    m = np.ones_like(f)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(f, 1e-9)) ** 4)
    if hi:
        m *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(np.fft.rfft(x) * m, n)


def _lp(x, cut):
    """One-pole lowpass with per-sample cutoff in 0..1 (fraction coefficient)."""
    a = np.broadcast_to(np.asarray(cut, float), x.shape)
    y = np.empty_like(x)
    s = 0.0
    for i in range(len(x)):
        s += a[i] * (x[i] - s)
        y[i] = s
    return y


def _env(n, sr, a, d):
    """Attack/exponential-decay envelope (seconds)."""
    t = np.arange(n) / sr
    return np.minimum(t / max(a, 1e-4), 1) * np.exp(-t / d)


def _reverb(x, sr, mix=0.3, tail=0.6, delays=(0.0297, 0.0371, 0.0411, 0.0437), fb=0.78):
    """Cheap Schroeder-ish reverb: parallel feedback combs with damping, input padded for the tail."""
    n = len(x) + int(tail * sr)
    xin = np.concatenate([x, np.zeros(n - len(x))])
    out = np.zeros(n)
    for d in delays:
        D = int(d * sr)
        y = xin.copy()
        for s in range(D, n, D):
            blk = y[s - D:s]
            y[s:s + D] += fb * blk[:len(y[s:s + D])]
        out += _lp(y, 0.35)
    return xin * (1 - mix) + out / len(delays) * mix * 2


def _norm(x, db):
    return x / (np.max(np.abs(x)) + 1e-12) * 10 ** (db / 20)


def ignite(cfg, sr):
    r = _rng(cfg, 1)
    n = int(1.4 * sr)
    t = np.arange(n) / sr
    cut = 0.02 + 0.25 * np.exp(-t / 0.12)               # filter opens then closes: "whoomp"
    body = _lp(r.standard_normal(n), cut) * _env(n, sr, 0.06, 0.28)
    f = 55 + 40 * np.exp(-t / 0.1)
    thump = np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, sr, 0.03, 0.18)
    flick = _lp(r.standard_normal(n), 0.5) * _env(n, sr, 0.15, 0.5) * (0.5 + 0.5 * np.sin(2 * np.pi * 11 * t)) * 0.15
    return _reverb(body * 1.4 + thump * 0.8 + flick, sr, 0.2, 0.4)


def snuff(cfg, sr):
    r = _rng(cfg, 2)
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    x = _lp(r.standard_normal(n), 0.04 + 0.4 * np.exp(-t / 0.05)) * _env(n, sr, 0.01, 0.07)
    x += 0.15 * _noise(r, n, 3000, 7000, sr) * _env(n, sr, 0.005, 0.03)
    return _reverb(x, sr, 0.15, 0.25)


def sparkler_loop(cfg, sr):
    r = _rng(cfg, 3)
    n = 2 * sr
    hiss = _noise(r, n, 3500, 11000, sr) * 0.35
    hiss *= 0.75 + 0.25 * _noise(r, n, None, 30, sr) / 0.2  # slow shimmer, circular
    out = hiss
    for _ in range(170):                                   # tiny crackles, wrapping around the loop end
        i = int(r.integers(0, n))
        L = int(r.uniform(0.0008, 0.004) * sr)
        pop = r.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * r.uniform(0.2, 1.0)
        pop = np.diff(np.concatenate([[0], pop]))
        idx = (i + np.arange(L)) % n
        out[idx] += pop * 1.5
    out -= np.mean(out)
    return np.tanh(out / np.std(out) * 0.45)           # soft-limit pops so the sizzle sits louder


def whoosh(cfg, sr):
    r = _rng(cfg, 4)
    n = int(1.1 * sr)
    t = np.arange(n) / sr
    u = t / t[-1]
    cut = 0.01 + 0.5 * u ** 2                               # rising sweep
    x = _lp(r.standard_normal(n), cut) * (u ** 1.5) * (1 - u ** 8)
    x += 0.2 * _noise(r, n, 800, 3000, sr) * u ** 2 * (1 - u ** 6)
    return _reverb(x, sr, 0.15, 0.3)


def boom(cfg, sr):
    r = _rng(cfg, 5)
    n = int(3.0 * sr)
    t = np.arange(n) / sr
    f = 38 + 90 * np.exp(-t / 0.08)
    sub = np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, sr, 0.004, 0.45)
    thud = _lp(r.standard_normal(n), 0.03 + 0.4 * np.exp(-t / 0.05)) * _env(n, sr, 0.002, 0.2)
    crack = _noise(r, n, 1500, 9000, sr) * _env(n, sr, 0.001, 0.05) * 0.5
    rumble = _lp(r.standard_normal(n), 0.015) * _env(n, sr, 0.05, 0.9) * 3
    return _reverb(sub + thud * 1.6 + crack + rumble, sr, 0.35, 1.2)


def crackle(cfg, sr):
    r = _rng(cfg, 6)
    n = int(1.9 * sr)
    out = np.zeros(n)
    times = np.sort(np.concatenate([r.uniform(0, 1.5, 120) ** 1.15 * 1.5 / 1.5 ** 1.15, r.uniform(0, 0.4, 40)]))
    for ti in times:
        i = int(ti * sr)
        L = int(r.uniform(0.001, 0.009) * sr)
        pop = _noise(r, L, 1200, r.uniform(4000, 10000), sr) * np.exp(-np.arange(L) / (L / 4))
        out[i:i + L] += pop[:max(0, min(L, n - i))] * r.uniform(0.1, 1.0) * (1.2 - ti / 2)
    return _reverb(np.tanh(out / np.std(out[:int(1.5 * sr)]) * 0.5), sr, 0.25, 0.3)


def chime(cfg, sr):
    n = int(5.0 * sr)
    t = np.arange(n) / sr
    base = 523.25  # C5 warm bell: inharmonic partials, slow decays
    out = np.zeros(n)
    for ratio, amp, dec in [(1, 1.0, 2.2), (2.0, 0.4, 1.6), (2.76, 0.18, 1.0), (4.07, 0.07, 0.6), (0.5, 0.35, 2.6)]:
        out += amp * np.sin(2 * np.pi * base * ratio * t + 0.3 * ratio) * _env(n, sr, 0.008, dec)
    out += 0.4 * np.sin(2 * np.pi * base * 1.5 * t) * _env(n, sr, 0.15, 1.8) * 0.3   # soft fifth swell
    return _reverb(out, sr, 0.4, 1.5)


def bloom(cfg, sr):
    r = _rng(cfg, 8)
    n = int(1.8 * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k, f in enumerate([1046.5, 1318.5, 1568.0, 2093.0, 2637.0]):
        on = k * 0.09
        e = np.where(t > on, np.exp(-(t - on) / 0.5) * np.minimum((t - on) / 0.04, 1), 0)
        out += np.sin(2 * np.pi * f * t) * e * (0.6 / (1 + k * 0.3)) * (0.8 + 0.2 * np.sin(2 * np.pi * (5 + k) * t))
    out += 0.08 * _noise(r, n, 5000, 12000, sr) * np.minimum(t / 0.3, 1) * np.exp(-t / 0.6)
    return _reverb(out, sr, 0.4, 0.8)


# brass hand-bell partial ratios (inharmonic) with relative amps and decay multipliers
_BELL = [(1.0, 1.0, 1.0), (2.01, 0.55, 0.8), (2.76, 0.6, 0.7), (4.18, 0.4, 0.5),
         (5.43, 0.3, 0.38), (6.8, 0.2, 0.28), (8.9, 0.12, 0.2), (11.3, 0.07, 0.14)]


def _bell_strike(r, n, sr, base, decay, bright=1.0):
    """One brass bell strike: detuned partial pairs (slow beating), quick attack, metallic click."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for ratio, amp, dm in _BELL:
        f = base * ratio * (1 + r.uniform(-0.004, 0.004))
        e = _env(n, sr, 0.0015, decay * dm)
        beat = r.uniform(1.5, 4.5)                          # Hz offset between the twin partials
        a = amp * (bright if ratio > 3 else 1.0)
        out += a * e * (np.sin(2 * np.pi * f * t + r.uniform(0, 6.28)) + 0.8 * np.sin(2 * np.pi * (f + beat) * t + r.uniform(0, 6.28)))
    out += 0.5 * _noise(r, n, 3000, 9000, sr) * _env(n, sr, 0.0005, 0.008)   # clapper click
    return out


def bell(cfg, sr):
    r = _rng(cfg, 9)
    n = int(3.0 * sr)
    return _reverb(_bell_strike(r, n, sr, 1650.0, 1.1), sr, 0.25, 0.6)


def bell_ring(cfg, sr):
    r = _rng(cfg, 10)
    dur = 2.5
    n = int(dur * sr)
    out = np.zeros(n)
    t, k = 0.0, 0
    while t < dur - 0.1:
        L = int(0.9 * sr)
        base = 1650.0 * (1 + r.uniform(-0.01, 0.01))
        strike = _bell_strike(r, L, sr, base, 0.35, bright=1.1) * r.uniform(0.55, 1.0)
        i = int(t * sr)
        m = min(L, n - i)
        out[i:i + m] += strike[:m]
        t += r.uniform(1 / 8.0, 1 / 6.0)                    # 6-8 strikes per second, uneven
        k += 1
    u = np.arange(n) / n
    out *= np.minimum(np.arange(n) / (0.03 * sr), 1) * (1 - u ** 2.5)    # swing in, fade out
    return _reverb(out, sr, 0.25, 0.6)


def match(cfg, sr):
    r = _rng(cfg, 11)
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    ns = int(0.17 * sr)                                    # scratch: rough, stuttering friction
    ts = np.arange(ns) / sr
    rough = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 70 * ts + 3 * _noise(r, ns, None, 40, sr) / 6))
    scratch = _noise(r, ns, 1800, 9000, sr) * rough * np.minimum(ts / 0.01, 1) * (1 - (ts / ts[-1]) ** 3) * 0.7
    x = np.zeros(n)
    x[:ns] += scratch
    on = 0.15                                              # flare: soft whoosh as the head catches
    u = np.clip((t - on) / 0.45, 0, None)
    fl = _lp(r.standard_normal(n), 0.03 + 0.25 * np.exp(-u / 0.12)) * np.minimum(u / 0.04, 1) * np.exp(-u / 0.17) * (t > on)
    x += fl * 3.0
    x += 0.25 * _noise(r, n, 3500, 9000, sr) * np.minimum(u / 0.02, 1) * np.exp(-u / 0.08) * (t > on)
    for _ in range(9):                                     # sparse little pops
        i = int(r.uniform(0.16, 0.45) * sr)
        L = int(r.uniform(0.001, 0.003) * sr)
        x[i:i + L] += r.standard_normal(L) * np.exp(-np.arange(L) / (L / 3)) * r.uniform(0.2, 0.5)
    return _reverb(x, sr, 0.12, 0.2)


def _pops(r, out, n, count, lo, hi, gain):
    """Circular crackle pops (wrap around the end so loops stay seamless)."""
    for _ in range(count):
        i = int(r.integers(0, n))
        L = int(r.uniform(lo, hi) * 44100)
        pop = np.diff(np.concatenate([[0], r.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * r.uniform(0.2, 1.0)]))
        out[(i + np.arange(L)) % n] += pop * gain


def anar_loop(cfg, sr):
    r = _rng(cfg, 12)
    n = 2 * sr
    sl = _noise(r, n, None, 12, sr)
    sl /= np.std(sl)
    roar = _noise(r, n, 700, 8000, sr) * (0.8 + 0.2 * np.tanh(sl))
    air = _noise(r, n, 5000, 14000, sr) * (0.7 + 0.25 * np.tanh(sl * 0.8))
    low = _noise(r, n, 120, 500, sr) * 0.5
    out = roar / np.std(roar) * 0.55 + air / np.std(air) * 0.5 + low / np.std(low) * 0.18
    _pops(r, out, n, 420, 0.0006, 0.003, 1.3)
    out -= np.mean(out)
    return np.tanh(out / np.std(out) * 0.5)


def chakri_loop(cfg, sr):
    r = _rng(cfg, 13)
    n = 2 * sr
    t = np.arange(n) / sr
    wob = _noise(r, n, None, 3, sr)
    wob = wob / np.std(wob) * 0.4                          # circular, so phase returns to start
    whirr = 0.5 + 0.5 * np.cos(2 * np.pi * 15 * t + wob)   # 15 Hz = 30 whole cycles in 2 s
    hiss = _noise(r, n, 2500, 11000, sr)
    body = _noise(r, n, 300, 1800, sr)
    out = hiss / np.std(hiss) * (0.25 + 0.75 * whirr ** 1.5) * 0.5
    out += body / np.std(body) * (0.2 + 0.8 * whirr) * 0.3
    out += 0.12 * np.sin(2 * np.pi * 240 * t + 4 * np.sin(2 * np.pi * 15 * t)) * whirr   # thin spin buzz
    _pops(r, out, n, 200, 0.0008, 0.004, 1.2)
    out -= np.mean(out)
    return np.tanh(out / np.std(out) * 0.45)


def lantern(cfg, sr):
    r = _rng(cfg, 14)
    n = int(2.5 * sr)
    t = np.arange(n) / sr
    sw = np.sin(np.pi * t / t[-1]) ** 2                    # slow in, slow out
    air = _noise(r, n, 250, 1400, sr) * sw
    air += 0.25 * _noise(r, n, 1800, 5000, sr) * sw ** 2
    pad = np.zeros(n)
    for f, a in [(110.0, 1.0), (164.8, 0.6), (220.0, 0.5), (277.2, 0.25)]:
        for d in (-0.5, 0.5):
            pad += a * np.sin(2 * np.pi * (f + d) * t + r.uniform(0, 6.28))
    pad *= np.sin(np.pi * t / t[-1]) ** 1.5 * (0.85 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
    x = air / np.std(air) * 0.5 + pad / np.max(np.abs(pad)) * 0.9
    x = _lp(x, 0.12) * sw ** 0.5
    return _reverb(x, sr, 0.3, 0.6)


def shankh(cfg, sr):
    r = _rng(cfg, 15)
    n = int(2.5 * sr)
    t = np.arange(n) / sr
    u = t / t[-1]
    f0 = 345 * (1 - 0.1 * np.exp(-t / 0.12)) * (1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(t / 0.8, 1))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    amp = np.minimum(t / 0.4, 1) ** 1.5 * (1 + 0.15 * np.sin(np.pi * u)) * np.exp(-np.maximum(t - 1.5, 0) / 0.35)
    tone = np.zeros(n)
    for k in range(1, 10):
        tone += np.sin(k * ph + r.uniform(0, 6.28)) * (1.0 / k ** 1.05) * (1.4 if k in (2, 3) else 1.0)
    breath = _noise(r, n, 500, 3500, sr) * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * t) ** 2 * 0.2) * 0.25
    x = (tone + breath * 1.2) * amp
    x = _lp(x, 0.45)
    return _reverb(x, sr, 0.3, 0.9)


SOUNDS = dict(ignite=ignite, snuff=snuff, sparkler_loop=sparkler_loop, whoosh=whoosh,
              boom=boom, crackle=crackle, chime=chime, bloom=bloom, bell=bell, bell_ring=bell_ring,
              match=match, anar_loop=anar_loop, chakri_loop=chakri_loop, lantern=lantern, shankh=shankh)

LEVEL_DB = dict(lantern=-9.0, bell_ring=-1.0)           # per-sound trim after peak normalisation


def write_wav(p, x, sr):
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def generate(cfg, only=None, force=False):
    """Generate any missing (or all if force) wavs. Returns list of names written."""
    d = path(cfg.audio.dir)
    os.makedirs(d, exist_ok=True)
    sr, done = cfg.audio.sample_rate, []
    for name, fn in SOUNDS.items():
        p = os.path.join(d, name + ".wav")
        if (only and name not in only) or (os.path.exists(p) and not force):
            continue
        x = fn(cfg, sr)
        if not name.endswith("_loop"):                         # loops stay circular; fade ends of one-shots
            f = int(0.01 * sr)
            x[-f:] *= np.linspace(1, 0, f)
        write_wav(p, _norm(x, cfg.audio.peak_dbfs + LEVEL_DB.get(name, 0.0)), sr)
        done.append(name)
    return done


if __name__ == "__main__":
    print("wrote:", generate(load_config(), force=True))
