"""Fire-and-forget sound via a sounddevice (PortAudio) mixing callback. Never blocks the render thread;
degrades to a no-op if no output device is available. (Not pygame: its SDL clashes with the SDL inside OpenCV.)"""
import os
import threading
import wave
import numpy as np
from .config import path


def _read_wav(p):
    with wave.open(p, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    return x


class Audio:
    def __init__(self, cfg, muted=False):
        self.cfg, self._muted, self.ok = cfg, muted, False
        self.snd = {}
        self._voices = []                 # [sound, pos, gain]
        self._loops = {}                  # name -> [sound, pos, gain, target]
        self._lock = threading.Lock()
        self._stream = None
        try:
            import sounddevice as sd
            from tools.gen_sounds import SOUNDS, generate
            d = path(cfg.audio.dir)
            if any(not os.path.exists(os.path.join(d, n + ".wav")) for n in SOUNDS):
                generate(cfg)
            for n in SOUNDS:
                self.snd[n] = _read_wav(os.path.join(d, n + ".wav"))
            self._fade_step = cfg.audio.buffer / (cfg.audio.sample_rate * cfg.audio.fade_ms / 1000)
            self._stream = sd.OutputStream(samplerate=cfg.audio.sample_rate, channels=1, dtype="float32",
                                           blocksize=cfg.audio.buffer, latency="low", callback=self._cb)
            self._stream.start()
            self.ok = bool(cfg.audio.enabled)
        except Exception as e:
            print(f"[audio] disabled ({e})")

    def _cb(self, out, frames, time_info, status):
        mix = np.zeros(frames, np.float32)
        with self._lock:
            keep = []
            for v in self._voices[-self.cfg.audio.num_channels:]:
                s, p, g = v
                n = min(frames, len(s) - p)
                mix[:n] += s[p:p + n] * g
                v[1] = p + n
                if v[1] < len(s):
                    keep.append(v)
            self._voices = keep
            for name, v in list(self._loops.items()):
                s, p, g, tgt = v
                g = min(tgt, g + self._fade_step) if g < tgt else max(tgt, g - self._fade_step)
                idx = (np.arange(frames) + p) % len(s)
                mix += s[idx] * g
                v[1], v[2] = (p + frames) % len(s), g
                if tgt == 0 and g <= 0:
                    del self._loops[name]
        out[:, 0] = np.clip(mix * self.cfg.audio.volume, -1, 1)

    @property
    def muted(self):
        return self._muted

    def set_muted(self, m):
        self._muted = bool(m)
        if self._muted:
            with self._lock:
                self._voices.clear()
                for v in self._loops.values():
                    v[3] = 0.0

    def play(self, name, volume=1.0):
        s = self.snd.get(name)
        if not (self.ok and s is not None) or self._muted:
            return
        with self._lock:
            self._voices.append([s, 0, float(volume)])

    def loop(self, name, on, volume=1.0):
        """Start/stop a looping sound idempotently with a short fade; volume can be updated every frame."""
        s = self.snd.get(name)
        if not (self.ok and s is not None):
            return
        with self._lock:
            v = self._loops.get(name)
            if on and not self._muted:
                if v is None:
                    self._loops[name] = [s, 0, 0.0, float(volume)]
                else:
                    v[3] = float(volume)
            elif v is not None:
                v[3] = 0.0

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self.ok = False
