"""Small Orbit endless turmite livestream renderer (v2: real 3D cubes, dark theme).
Writes raw BGR frames to stdout at W x H @ 30 fps, forever or for --seconds.
Each episode: random Langton's-ant/turmite rule, slow start, exponential
speed-up, hold on the result, fade, next rule.
Renderer: numba-parallel per-pixel raycast of extruded cubes over the grid,
with a mip pyramid for far-away/zoomed-out views.
"""
import os, sys, argparse, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numba import njit, prange, set_num_threads
from music import Music

W = int(os.environ.get("W", 1280)); H = int(os.environ.get("H", 720)); FPS = 30
SS = int(os.environ.get("SS", 1))          # supersamples per axis in 3D zone
G = 2048                                    # grid size (wraps), power of 2
LEVELS = 12

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=0)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--font", default="Anton-Regular.ttf")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--audio", default="")
args = ap.parse_args()
if args.threads: set_num_threads(args.threads)
rng = np.random.default_rng(args.seed)

CURATED = ["RL", "RLR", "LLRR", "LRRRRRLLR", "LLRRRLRLRLLR", "RRLLLRLLLRRR",
           "RRLRR", "LRRL", "RLLR", "LLRRRLRRRRR", "RRLLLRRRLRRR", "LRRRRLLLRRR"]

def pick_rule():
    if rng.random() < 0.55:
        return str(rng.choice(CURATED))
    while True:
        n = int(rng.integers(2, 9))
        r = "".join(rng.choice(["L", "R"], n))
        if "L" in r and "R" in r:
            return r

@njit(cache=True)
def run(grid, turns, ncol, x, y, d, nsteps, bb):
    dx = (0, 1, 0, -1); dy = (-1, 0, 1, 0)
    m = grid.shape[0] - 1
    for _ in range(nsteps):
        c = grid[y, x]
        d = (d + turns[c]) & 3
        grid[y, x] = (c + 1) % ncol
        x = (x + dx[d]) & m
        y = (y + dy[d]) & m
        if x < bb[0]: bb[0] = x
        if x > bb[1]: bb[1] = x
        if y < bb[2]: bb[2] = y
        if y > bb[3]: bb[3] = y
    return x, y, d

@njit(parallel=True, cache=True)
def build_mip(grid, bright, mips, offs):
    # level 0: per-cell top brightness (0 = floor)
    n = grid.shape[0]
    for y in prange(n):
        for x in range(n):
            mips[y * n + x] = bright[grid[y, x]]
    for l in range(1, offs.shape[0]):
        s = n >> l; ps = s * 2; po = offs[l - 1]; o = offs[l]
        for y in prange(s):
            for x in range(s):
                a = po + (2 * y) * ps + 2 * x
                mips[o + y * s + x] = 0.25 * (mips[a] + mips[a + 1] + mips[a + ps] + mips[a + ps + 1])

@njit(cache=True)
def sample_mip(mips, offs, lev, u, v, n):
    # bilinear in level `lev`, coords u,v in cell units (wrapping)
    s = n >> lev; o = offs[lev]
    fu = u / (1 << lev) - 0.5; fv = v / (1 << lev) - 0.5
    x0 = int(math.floor(fu)); y0 = int(math.floor(fv))
    ax = fu - x0; ay = fv - y0
    m = s - 1
    x0 &= m; y0 &= m; x1 = (x0 + 1) & m; y1 = (y0 + 1) & m
    a = mips[o + y0 * s + x0]; b = mips[o + y0 * s + x1]
    c = mips[o + y1 * s + x0]; d = mips[o + y1 * s + x1]
    return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay

@njit(cache=True)
def shade_ray(grid, heights, bright, mips, offs, ox, oy, oz, dx, dy, dz,
              hmax, antx, anty, show_ant, grid_k, lod_px, out):
    """Trace one ray. Writes linear-ish BGR into out[0..2]."""
    n = grid.shape[0]; m = n - 1
    # floor + top-plane intersections (dz < 0)
    t0 = -oz / dz
    fx = ox + dx * t0; fy = oy + dy * t0
    # pixel footprint on floor, in cells
    fp = t0 * lod_px
    # colours (BGR, 0..255)
    FLB, FLG, FLR = 30.0, 26.0, 24.0       # floor
    if fp > 0.55:
        # far: mip average of top brightness, flat
        lev = math.log2(fp / 0.55)
        l0 = int(lev)
        if l0 >= offs.shape[0] - 1:
            b = sample_mip(mips, offs, offs.shape[0] - 1, fx, fy, n)
        else:
            fr = lev - l0
            b = sample_mip(mips, offs, l0, fx, fy, n) * (1 - fr) + sample_mip(mips, offs, l0 + 1, fx, fy, n) * fr
        k = b * 0.82  # cubes partly hidden sides: a bit darker than tops
        out[0] = FLB + (236.0 - FLB) * k
        out[1] = FLG + (232.0 - FLG) * k
        out[2] = FLR + (228.0 - FLR) * k
        if show_ant:
            ddx = fx - (antx + 0.5); ddy = fy - (anty + 0.5)
            rr = max(0.9, fp * 2.2)
            if ddx * ddx + ddy * ddy < rr * rr:
                out[0] = 40.0; out[1] = 45.0; out[2] = 235.0
        return t0
    t1 = (hmax - oz) / dz
    px = ox + dx * t1; py = oy + dy * t1
    cx = int(math.floor(px)); cy = int(math.floor(py))
    sx = 1 if dx > 0 else -1; sy = 1 if dy > 0 else -1
    idx = 1.0 / abs(dx) if dx != 0 else 1e30
    idy = 1.0 / abs(dy) if dy != 0 else 1e30
    nbx = (cx + 1 - px) if dx > 0 else (px - cx)
    nby = (cy + 1 - py) if dy > 0 else (py - cy)
    tmx = t1 + nbx * idx; tmy = t1 + nby * idy
    tin = t1; face = 0  # 0 top-entry, 1 x-side, 2 y-side
    for _ in range(8):
        tout = min(tmx, tmy, t0)
        gx = cx & m; gy = cy & m
        c = grid[gy, gx]
        isant = show_ant and gx == antx and gy == anty
        h = heights[c]
        if isant and h < 0.35: h = 0.35
        if h > 0.0:
            zin = oz + dz * tin; zout = oz + dz * tout
            if h >= zin - 1e-6 and face != 0:
                # side face hit
                if face == 1:
                    lit = 0.62 if sx < 0 else 0.38
                else:
                    lit = 0.50 if sy < 0 else 0.30
                if isant:
                    out[0] = 30.0 * lit * 1.6; out[1] = 35.0 * lit * 1.6; out[2] = 235.0 * lit * 1.6
                else:
                    b = bright[c]
                    out[0] = 236.0 * b * lit + 10.0; out[1] = 232.0 * b * lit + 10.0; out[2] = 228.0 * b * lit + 10.0
                return tin
            if h > zout:
                th = (h - oz) / dz
                if face == 0 or th >= tin:
                    hx = ox + dx * th - cx; hy = oy + dy * th - cy
                    e = min(hx, 1 - hx, hy, 1 - hy)
                    edge = 0.86 if e < 0.035 else 1.0
                    if isant:
                        out[0] = 40.0 * edge; out[1] = 45.0 * edge; out[2] = 240.0 * edge
                    else:
                        b = bright[c] * edge
                        out[0] = 236.0 * b; out[1] = 232.0 * b; out[2] = 228.0 * b
                    return th
        if tout >= t0:
            break
        if tmx < tmy:
            cx += sx; tin = tmx; tmx += idx; face = 1
        else:
            cy += sy; tin = tmy; tmy += idy; face = 2
    # floor with soft grid lines
    ux = fx - math.floor(fx); uy = fy - math.floor(fy)
    e = min(ux, 1 - ux, uy, 1 - uy)
    lw = 0.03 + fp * 0.5
    g = grid_k * max(0.0, 1.0 - e / lw)
    # ambient-occlusion-ish darkening next to tall neighbours
    out[0] = FLB + 22.0 * g; out[1] = FLG + 22.0 * g; out[2] = FLR + 24.0 * g
    return t0

@njit(parallel=True, cache=True)
def render(frame, grid, heights, bright, mips, offs, cam, antx, anty, show_ant, ss, fade):
    Hh, Ww = frame.shape[0], frame.shape[1]
    ox, oy, oz = cam[0], cam[1], cam[2]
    fx_, fy_, fz_ = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    F = cam[12]; hmax = cam[13]; dist = cam[14]; grid_k = cam[15]
    BGB, BGG, BGR_ = 18.0, 15.0, 14.0
    lod_px = 1.0 / F
    for j in prange(Hh):
        buf = np.empty(3)
        for i in range(Ww):
            # centre ray first: decides LOD / supersampling
            sxp = (i + 0.5 - Ww * 0.5) / F; syp = (j + 0.5 - Hh * 0.5) / F
            dx = fx_ + sxp * rx - syp * ux; dy = fy_ + sxp * ry - syp * uy; dz = fz_ + sxp * rz - syp * uz
            dn = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dn; dy /= dn; dz /= dn
            if dz >= -1e-4:
                b0 = BGB; b1 = BGG; b2 = BGR_; tt = 1e9
            else:
                t0 = -oz / dz
                fp = t0 / F
                if fp > 0.55 or ss == 1:
                    tt = shade_ray(grid, heights, bright, mips, offs, ox, oy, oz, dx, dy, dz, hmax,
                                   antx, anty, show_ant, grid_k, lod_px, buf)
                    b0 = buf[0]; b1 = buf[1]; b2 = buf[2]
                else:
                    b0 = 0.0; b1 = 0.0; b2 = 0.0; tt = 0.0
                    inv = 1.0 / (ss * ss)
                    for q in range(ss):
                        for p in range(ss):
                            a = (i + (p + 0.5) / ss - Ww * 0.5) / F
                            c = (j + (q + 0.5) / ss - Hh * 0.5) / F
                            ex = fx_ + a * rx - c * ux; ey = fy_ + a * ry - c * uy; ez = fz_ + a * rz - c * uz
                            en = math.sqrt(ex * ex + ey * ey + ez * ez)
                            tt += shade_ray(grid, heights, bright, mips, offs, ox, oy, oz, ex / en, ey / en, ez / en,
                                            hmax, antx, anty, show_ant, grid_k, lod_px, buf) * inv
                            b0 += buf[0] * inv; b1 += buf[1] * inv; b2 += buf[2] * inv
            # distance fog toward background
            fg = (tt - dist) / (dist * 1.6)
            fg = 0.0 if fg < 0 else (0.85 if fg > 0.85 else fg)
            b0 = b0 + (BGB - b0) * fg; b1 = b1 + (BGG - b1) * fg; b2 = b2 + (BGR_ - b2) * fg
            # vignette + fade
            vx = (i - Ww * 0.5) / (Ww * 0.5); vy = (j - Hh * 0.5) / (Hh * 0.5)
            v = 1.0 - 0.30 * (vx * vx + vy * vy)
            if v < 0.55: v = 0.55
            v *= fade
            b0 *= v; b1 *= v; b2 *= v
            frame[j, i, 0] = min(255.0, max(0.0, b0))
            frame[j, i, 1] = min(255.0, max(0.0, b1))
            frame[j, i, 2] = min(255.0, max(0.0, b2))

def camera(cx, cy, half, yaw, pitch):
    fov = math.radians(42)
    F = (H / 2) / math.tan(fov / 2)
    dist = half / math.tan(fov / 2) * 1.15
    sy_, cyw = math.sin(yaw), math.cos(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    f = np.array([cp * cyw, cp * sy_, -sp])
    r = np.array([-sy_, cyw, 0.0])
    u = np.cross(r, f)
    o = np.array([cx, cy, 0.0]) - f * dist
    return o, f, r, u, F, dist

def project(pt, o, f, r, u, F):
    v = np.asarray(pt) - o
    z = v @ f
    return W / 2 + F * (v @ r) / z, H / 2 - F * (v @ u) / z

try:
    FONT = ImageFont.truetype(args.font, int(34 * H / 720))
    FONT_S = ImageFont.truetype(args.font, int(21 * H / 720))
except Exception:
    FONT = ImageFont.load_default(); FONT_S = FONT
SH = int(70 * H / 720)

SH = int(48 * H / 720)
def text_strip(s1, font):
    img = Image.new("RGBA", (W, SH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    tw = dr.textlength(s1, font=font)
    dr.text(((W - tw) / 2, 2), s1, font=font, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    a = np.asarray(img)
    return a[..., [2, 1, 0]].astype(np.uint16), a[..., 3:4].astype(np.uint16)

offs = np.zeros(LEVELS, np.int64)
o_ = 0
for l in range(LEVELS):
    offs[l] = o_; o_ += (G >> l) ** 2
mips = np.zeros(o_, np.float32)

out = sys.stdout.buffer
frame = np.zeros((H, W, 3), np.uint8)
frame_i = 0
t_end = args.seconds * FPS if args.seconds > 0 else float("inf")
yaw = rng.uniform(0, 2 * math.pi)
camv = np.zeros(16)
ty0 = H - SH - int(14 * H / 720)
ty1 = int(16 * H / 720)
import os as _os, fcntl
music = Music(rng, FPS)
aud = None
if args.audio:
    fd = _os.open(args.audio, _os.O_RDWR)
    try: fcntl.fcntl(fd, 1031, 1 << 20)
    except Exception: pass
    aud = _os.fdopen(fd, 'wb', buffering=0)

while frame_i < t_end:
    rule = pick_rule()
    ncol = len(rule)
    turns = np.array([1 if c == "R" else -1 for c in rule], np.int64)
    grid = np.zeros((G, G), np.uint8)
    heights = np.zeros(ncol, np.float64); bright = np.zeros(ncol, np.float32)
    for k in range(1, ncol):
        q = (k - 1) / max(1, ncol - 2) if ncol > 2 else 0.0
        heights[k] = 0.55 - 0.30 * q
        bright[k] = 0.95 - 0.42 * q
    hmax = 0.6
    x = y = G // 2; d = 0
    bb = np.array([x, x, y, y], np.int64)
    steps = 0
    cx, cy, half = float(x) + 0.5, float(y) + 0.5, 6.0
    acc = 0.0
    INTRO, MAXT = 7.0, 55.0
    hold_frames = 0; ep_f = 0; done = False; fade_out = 0
    yaw_speed = rng.uniform(0.10, 0.18) * rng.choice([-1, 1])   # rad per second
    last_mip = -99
    music.new_episode(); fin = False
    rule_txt = text_strip(f"RULE {rule}", FONT)
    while frame_i < t_end:
        t = ep_f / FPS
        if not done:
            spf = 0.1 if t < INTRO else min(0.1 * 1.45 ** (t - INTRO), 40000.0)
            acc += spf
            n = int(acc); acc -= n
            ev = []
            if 0 < n <= 4:
                for q in range(n):
                    st = int(grid[y, x]); ev.append(((q + 0.5) / n, st, int(turns[st])))
                    x, y, d = run(grid, turns, ncol, x, y, d, 1, bb)
                steps += n
            elif n:
                x, y, d = run(grid, turns, ncol, x, y, d, n, bb)
                steps += n
                st = int(grid[y, x]); ev.append((1.0, st, int(turns[st])))
            music.steps(ev, ncol, spf * FPS)
            ext = max(bb[1] - bb[0], bb[3] - bb[2])
            if t > MAXT or ext > G * 0.8 or steps > 6_000_000:
                done = True
        else:
            if not fin:
                music.finale(); fin = True
            hold_frames += 1
            if hold_frames > 5 * FPS:
                fade_out += 1
                if fade_out > FPS:
                    break
        bcx, bcy = (bb[0] + bb[1]) / 2 + 0.5, (bb[2] + bb[3]) / 2 + 0.5
        ext = max(bb[1] - bb[0], bb[3] - bb[2]) + 1
        tgt_half = max(6.0, ext * 0.6 + 3)
        w_ant = 1.0 if t < INTRO else max(0.0, 1 - (t - INTRO) / 6)
        tcx = w_ant * (x + 0.5) + (1 - w_ant) * bcx
        tcy = w_ant * (y + 0.5) + (1 - w_ant) * bcy
        k = 0.04 if not done else 0.06
        cx += (tcx - cx) * k; cy += (tcy - cy) * k; half += (tgt_half - half) * k
        yaw += yaw_speed / FPS
        pitch = math.radians(56)
        o, f, r, u, F, dist = camera(cx, cy, half, yaw, pitch)
        # mip needed once cells get smaller than ~2 px at the far edge
        if half > 40 and (ep_f - last_mip) >= (1 if not done else 15):
            build_mip(grid, bright, mips, offs); last_mip = ep_f
        camv[:] = (*o, *f, *r, *u, F, hmax, dist, 1.0 if half < 60 else max(0.0, 1 - (half - 60) / 60), 0)[:16]
        a = min(1.0, ep_f / FPS)
        if fade_out: a = 1 - fade_out / FPS
        show_ant = (not done) or hold_frames < FPS
        render(frame, grid, heights, bright, mips, offs, camv, x, y, show_ant, SS, a)
        for (rgb, al), yy0 in ((text_strip(f"STEP {steps:,}", FONT), ty0), (rule_txt, ty1)):
            reg = frame[yy0:yy0 + SH].astype(np.uint16)
            frame[yy0:yy0 + SH] = ((reg * (255 - al) + rgb * al) // 255).astype(np.uint8)
        out.write(frame.tobytes())
        ab = music.render()
        if aud is not None: aud.write(ab)
        frame_i += 1; ep_f += 1
