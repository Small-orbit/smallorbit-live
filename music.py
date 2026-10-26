"""Real-time procedural lo-fi synced to the ant.
Each ant step can trigger a melodic pluck whose pitch comes from the colour the
ant just read and the way it turned. When the ant gets fast, steps are folded
onto a 16th-note grid so it stays musical. Pads, bass, soft drums and vinyl
crackle sit underneath. Produces int16 stereo at 48 kHz, one video frame at a time.
"""
import numpy as np

SR = 48000

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

PROGS = [
    [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]],   # Fmaj7 Em7 Dm7 Cmaj7
    [[45, 48, 52, 55], [50, 53, 57, 60], [43, 47, 50, 53], [48, 52, 55, 59]],   # Am7 Dm7 G7 Cmaj7
    [[50, 53, 57, 60], [43, 47, 50, 53], [48, 52, 55, 59], [45, 48, 52, 55]],   # Dm7 G7 Cmaj7 Am7
]

class Music:
    def __init__(self, rng, fps=30):
        self.rng = rng
        self.fps = fps
        self.spf = SR // fps                     # samples per video frame
        self.n = 0                               # global sample clock
        self.voices = []                         # [start, freq, amp, decay, kind, pan, dur]
        self.bpm = 82.0
        self.six = int(SR * 60 / self.bpm / 4)   # samples per 16th
        self.next16 = 0
        self.i16 = 0
        self.drums = 0.0                         # 0..1 drum level (fades in)
        self.drum_target = 0.0
        self.quant = False
        self.pending = None                      # (state, turn) waiting for next 16th
        self.last_note = 72
        self.new_episode()

    def new_episode(self):
        self.prog = PROGS[int(self.rng.integers(len(PROGS)))]
        self.key = int(self.rng.integers(-3, 4))
        self.drum_target = 0.0
        self.quant = False
        self.pending = None

    def chord(self):
        bar = (self.i16 // 16) % 4
        return [m + self.key for m in self.prog[bar]]

    def scale_note(self, state, turn, ncol):
        ch = self.chord()
        pool = sorted(set([c + 12 for c in ch] + [c + 24 for c in ch] + [ch[0] + 14, ch[0] + 26]))
        pool = [p for p in pool if 62 <= p <= 90]
        # colour picks the degree, turn direction moves up/down an octave-ish
        idx = (state * 3 + (1 if turn > 0 else 0)) % len(pool)
        note = pool[idx]
        # keep the line smooth
        while note - self.last_note > 7: note -= 12
        while self.last_note - note > 7: note += 12
        note = min(max(note, 64), 91)
        self.last_note = note
        return note

    def pluck(self, at, note, amp=0.22):
        pan = 0.5 + 0.35 * np.sin(note * 1.7)
        self.voices.append([at, mtof(note), amp, 2.6, 0, pan, SR * 2])

    # --- events from the simulation ---------------------------------------
    def steps(self, events, ncol, fast):
        """events: list of (frac_in_frame, state, turn). fast: steps/sec."""
        self.quant = fast > 5.5
        self.drum_target = 0.0 if fast < 1.0 else (0.6 if fast < 40 else 1.0)
        if not events:
            return
        if not self.quant:
            for frac, st, tu in events:
                at = self.n + int(frac * self.spf)
                self.pluck(at, self.scale_note(st, tu, ncol))
        else:
            st, tu = events[-1][1], events[-1][2]
            self.pending = (st, tu, ncol)

    def finale(self):
        at = self.n
        for k, m in enumerate(self.chord()):
            self.pluck(at + k * 2200, m + 24, 0.18)
        self.drum_target = 0.0

    # --- synthesis ----------------------------------------------------------
    def _schedule_grid(self, end):
        while self.next16 < end:
            at = self.next16
            s = self.i16 % 16
            swing = int(self.six * 0.12) if s % 2 else 0
            at += swing
            if s == 0:
                ch = self.chord()
                for k, m in enumerate(ch):
                    self.voices.append([at, mtof(m + 12), 0.045, 0.0, 1, 0.3 + 0.13 * k, self.six * 16])
                self.voices.append([at, mtof(ch[0] - 12), 0.20, 1.6, 2, 0.5, int(SR * 1.2)])
            if s == 8:
                self.voices.append([at, mtof(self.chord()[0] - 12), 0.15, 2.2, 2, 0.5, int(SR * 0.9)])
            d = self.drums
            if d > 0.02:
                if s in (0, 10):
                    self.voices.append([at, 55.0, 0.55 * d, 9.0, 3, 0.5, int(SR * 0.35)])
                if s in (4, 12):
                    self.voices.append([at, 180.0, 0.22 * d, 14.0, 4, 0.55, int(SR * 0.25)])
                if s % 2 == 0:
                    self.voices.append([at, 0.0, (0.05 if s % 4 else 0.035) * d, 40.0, 5, 0.62, int(SR * 0.06)])
            if self.quant and self.pending is not None:
                st, tu, nc = self.pending
                self.pluck(at, self.scale_note(st, tu, nc), 0.18)
                if self.drums > 0.8 and s % 4 == 2:
                    self.pluck(at + self.six // 2, self.last_note + 7 if self.last_note < 84 else self.last_note - 5, 0.10)
                self.pending = None
            self.next16 += self.six
            self.i16 += 1

    def render(self):
        n0, N = self.n, self.spf
        self.drums += (self.drum_target - self.drums) * 0.02
        self._schedule_grid(n0 + N)
        L = np.zeros(N); R = np.zeros(N)
        t_idx = np.arange(N)
        keep = []
        for v in self.voices:
            st, f, a, dec, kind, pan, dur = v
            if st >= n0 + N:
                keep.append(v); continue
            if st + dur + SR * 0.9 < n0:
                continue
            tt = (n0 + t_idx - st) / SR
            on = tt >= 0
            tt = np.where(on, tt, 0)
            if kind == 0:      # pluck: soft e-piano / marimba
                env = np.exp(-dec * tt) * np.minimum(1, tt * 400)
                ph = 2 * np.pi * f * tt
                s = (np.sin(ph) + 0.35 * np.sin(2 * ph) * np.exp(-6 * tt) + 0.12 * np.sin(3 * ph + 0.3)) * env
            elif kind == 1:    # pad
                d = dur / SR
                env = np.minimum(1, tt / 0.7) * np.clip((d + 0.8 - tt) / 0.8, 0, 1)
                ph = 2 * np.pi * f * tt
                s = (np.sin(ph) + 0.25 * np.sin(2 * ph * 1.002) + 0.1 * np.sin(3 * ph)) * env * (0.85 + 0.15 * np.sin(2 * np.pi * 0.3 * tt))
            elif kind == 2:    # bass
                env = np.exp(-dec * tt) * np.minimum(1, tt * 200)
                s = np.sin(2 * np.pi * f * tt) * env
            elif kind == 3:    # kick
                fr = f + 90 * np.exp(-30 * tt)
                s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-dec * tt)
            elif kind == 4:    # snare (soft)
                s = (0.6 * self.rng.standard_normal(N) + 0.5 * np.sin(2 * np.pi * f * tt)) * np.exp(-dec * tt)
            else:              # hat
                nz = self.rng.standard_normal(N)
                nz = nz - np.concatenate(([0], nz[:-1]))
                s = nz * np.exp(-dec * tt)
            s = s * on * a
            L += s * (1 - pan); R += s * pan
            if st + dur + SR * 0.9 >= n0 + N:
                keep.append(v)
        self.voices = keep
        # vinyl crackle + hiss
        cr = np.zeros(N)
        k = self.rng.poisson(1.2)
        if k:
            cr[self.rng.integers(0, N, k)] = self.rng.uniform(-0.25, 0.25, k)
        hiss = self.rng.standard_normal(N) * 0.004
        L += cr + hiss; R += cr * 0.8 + hiss
        out = np.stack([L, R], 1)
        out = np.tanh(out * 1.1) * 0.85
        self.n += N
        return (out * 32767).astype(np.int16).tobytes()
