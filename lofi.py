"""Procedural lo-fi loop generator. Writes a seamless stereo WAV loop.
Usage: python lofi.py out.wav [seed]
"""
import sys, wave
import numpy as np
 
SR = 44100
BPM = 78
BEAT = 60.0 / BPM
BAR = BEAT * 4

rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 7)

def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)

# Progressions (root midi + chord tones), each 4 bars, played twice per section
PROGS = [
    [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]],  # Fmaj7 Em7 Dm7 Cmaj7
    [[50, 53, 57, 60], [55, 59, 62, 65], [48, 52, 55, 59], [45, 48, 52, 55]],  # Dm7 G7 Cmaj7 Am7
    [[57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 65], [52, 55, 59, 62]],  # Am7 Fmaj7 G7 Em7
    [[48, 52, 55, 59], [57, 60, 64, 67], [50, 53, 57, 60], [55, 59, 62, 66]],  # Cmaj7 Am7 Dm7 G
]
BARS_PER_SECTION = 16
N_BARS = BARS_PER_SECTION * len(PROGS)
N = int(round(N_BARS * BAR * SR))
L = np.zeros(N); R = np.zeros(N)

def env_adsr(n, a, d, s, r):
    t = np.arange(n) / SR
    e = np.ones(n) * s
    na = max(1, int(a * SR)); nd = max(1, int(d * SR)); nr = max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na)[: min(na, n)]
    if na < n:
        k = min(nd, n - na)
        e[na:na + k] = np.linspace(1, s, nd)[:k]
    if nr < n:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e

def add(buf, start, sig):
    s = int(start * SR) % N
    n = len(sig)
    end = s + n
    if end <= N:
        buf[s:end] += sig
    else:
        k = N - s
        buf[s:] += sig[:k]
        buf[: n - k] += sig[k:]  # wrap -> seamless loop

def epiano(f, dur, vel):
    n = int(dur * SR); t = np.arange(n) / SR
    detune = 1 + rng.uniform(-0.002, 0.002)
    s = (np.sin(2 * np.pi * f * detune * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 3)
         + 0.12 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 6))
    s *= np.exp(-t * 0.9) * env_adsr(n, 0.01, 0.3, 0.8, 0.4)
    s *= 1 + 0.15 * np.sin(2 * np.pi * 4.5 * t)  # tremolo
    return s * vel

def bass(f, dur, vel):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2 * f * t)
    return s * env_adsr(n, 0.01, 0.2, 0.7, 0.15) * vel

def kick(vel=1.0):
    n = int(0.35 * SR); t = np.arange(n) / SR
    f = 50 + 70 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * vel

def snare(vel=1.0):
    n = int(0.25 * SR); t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(6) / 6, mode="same")
    return (0.6 * noise * np.exp(-t * 18) + 0.3 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)) * vel

def hat(vel=1.0):
    n = int(0.06 * SR); t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(3) / 3, mode="same")
    return noise * np.exp(-t * 70) * vel

SWING = 0.06 * BEAT
for bar in range(N_BARS):
    sec = bar // BARS_PER_SECTION
    chord = PROGS[sec][bar % 4]
    t0 = bar * BAR
    # chord stab: slightly strummed, held most of the bar
    for i, n_ in enumerate(chord):
        sig = epiano(midi(n_), BAR * 1.05, 0.16)
        add(L, t0 + i * 0.012, sig * (0.8 + 0.1 * i)); add(R, t0 + i * 0.012, sig * (1.1 - 0.1 * i))
    # little melody notes on top (pentatonic-ish from chord)
    for b in range(8):
        if rng.random() < 0.28:
            n_ = chord[rng.integers(1, 4)] + 12 + rng.choice([0, 2, 0, -1 if False else 0])
            st = t0 + b * BEAT / 2 + (SWING if b % 2 else 0)
            sig = epiano(midi(n_), BEAT * 0.9, 0.09)
            pan = rng.uniform(0.6, 1.0)
            add(L, st, sig * pan); add(R, st, sig * (1.6 - pan))
    # bass
    root = chord[0] - 12
    for st, d in [(0, 1.5), (1.5, 0.5), (2.5, 1.5)]:
        sig = bass(midi(root if st != 1.5 else root + 7), d * BEAT, 0.32)
        add(L, t0 + st * BEAT, sig); add(R, t0 + st * BEAT, sig)
    # drums (drop out on first bar of every 4th section for breathing)
    if not (sec == 2 and bar % BARS_PER_SECTION < 2):
        for st in [0, 2.5]:
            k = kick(0.55); add(L, t0 + st * BEAT, k); add(R, t0 + st * BEAT, k)
        for st in [1, 3]:
            s = snare(0.22); add(L, t0 + st * BEAT, s * 0.9); add(R, t0 + st * BEAT, s)
        for e8 in range(8):
            st = e8 * 0.5 + (0.06 if e8 % 2 else 0)
            h = hat(0.07 if e8 % 2 else 0.1); add(L, t0 + st * BEAT, h * 0.7); add(R, t0 + st * BEAT, h)

# warm low-pass (one-pole, circular so loop stays seamless: run twice)
def lowpass(x, a=0.35):
    y = np.empty_like(x); prev = 0.0
    for _ in range(2):
        # vectorised-ish via scipy-less filter: use lfilter-like loop in chunks with numpy
        out = np.empty_like(x)
        acc = prev
        # simple implementation (fast enough for a few minutes of audio)
        for i in range(0, len(x), 4096):
            seg = x[i:i + 4096]
            o = np.empty_like(seg)
            for j in range(len(seg)):
                acc += a * (seg[j] - acc)
                o[j] = acc
            out[i:i + 4096] = o
        prev = acc
        y = out
    return y

try:
    from numba import njit
    @njit(cache=True)
    def _lp(x, a, prev):
        y = np.empty_like(x)
        acc = prev
        for i in range(len(x)):
            acc += a * (x[i] - acc)
            y[i] = acc
        return y, acc
    def lowpass(x, a=0.35):
        _, p = _lp(x, a, 0.0)
        y, _ = _lp(x, a, p)
        return y
except Exception:
    pass

L = lowpass(L); R = lowpass(R)
# vinyl crackle + hiss
crackle = np.zeros(N)
idx = rng.integers(0, N, size=int(N_BARS * BAR * 6))
crackle[idx] = rng.uniform(-1, 1, size=len(idx)) * 0.08
hiss = rng.standard_normal(N) * 0.004
L += crackle + hiss; R += np.roll(crackle, 37) + hiss
mx = max(np.abs(L).max(), np.abs(R).max())
L = np.tanh(L / mx * 1.2) * 0.8; R = np.tanh(R / mx * 1.2) * 0.8
out = (np.stack([L, R], 1) * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(out.tobytes())
print(f"wrote {sys.argv[1]} {N/SR:.1f}s")
