"""Small Orbit endless turmite livestream renderer.
Writes raw BGR frames to stdout (1280x720 @ 30 fps) forever, or for --seconds.
Each episode: random Langton's-ant rule, slow start, exponential speed-up,
hold on the result, fade, next rule.
"""
import sys, time, argparse, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from numba import njit

W, H, FPS = 1280, 720, 30
G = 2048                       # grid size (wraps)
BG = np.array([236, 236, 236], np.float32)
RED = (40, 40, 220)            # BGR accent

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=0)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--font", default="Anton-Regular.ttf")
args = ap.parse_args()
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
    g = grid.shape[0]
    for _ in range(nsteps):
        c = grid[y, x]
        d = (d + turns[c]) & 3
        grid[y, x] = (c + 1) % ncol
        x = (x + dx[d]) % g
        y = (y + dy[d]) % g
        if x < bb[0]: bb[0] = x
        if x > bb[1]: bb[1] = x
        if y < bb[2]: bb[2] = y
        if y > bb[3]: bb[3] = y
    return x, y, d

def palette(ncol):
    # colour 0 = floor; others = greys from mid to near-black (one accent: red ant only)
    p = np.zeros((ncol, 3), np.uint8)
    p[0] = 236
    for i in range(1, ncol):
        v = int(150 - 130 * (i - 1) / max(1, ncol - 2)) if ncol > 2 else 22
        p[i] = v
    return p

try:
    FONT = ImageFont.truetype(args.font, 34)
    FONT_S = ImageFont.truetype(args.font, 22)
except Exception:
    FONT = ImageFont.load_default(); FONT_S = FONT

def text_strip(s1, s2):
    img = Image.new("RGBA", (W, 70), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    tw = dr.textlength(s1, font=FONT)
    dr.text(((W - tw) / 2, 4), s1, font=FONT, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    tw2 = dr.textlength(s2, font=FONT_S)
    dr.text(((W - tw2) / 2, 44), s2, font=FONT_S, fill=(255, 255, 255, 230), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    a = np.asarray(img).astype(np.float32)
    return a[..., [2, 1, 0]], a[..., 3:4] / 255.0

# precomputed vignette + fog
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
vig = 1 - 0.22 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2
vig = np.clip(vig, 0.6, 1)[..., None]
fog = np.clip(1 - yy / (H * 0.55), 0, 1)[..., None] ** 1.6 * 0.75

def homography(cx, cy, half, yaw, pitch, x0, y0, wcells, tex_px):
    """Map texture pixels -> screen pixels for a perspective camera looking at (cx,cy)."""
    fov = math.radians(42)
    f = (H / 2) / math.tan(fov / 2)
    dist = half / math.tan(fov / 2) * 1.05
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)
    def proj(wx, wy):
        # world relative to target, rotated by yaw
        rx = (wx - cx) * cyw - (wy - cy) * syw
        ry = (wx - cx) * syw + (wy - cy) * cyw
        # camera: ry axis is "up the screen"; tilt by pitch (pitch=90 -> top-down)
        zc = dist + ry * cp          # depth
        yc = ry * sp                 # screen vertical (world units)
        zc = max(zc, half * 0.05)
        return (W / 2 + f * rx / zc, H / 2 - f * yc / zc)
    src = np.float32([[0, 0], [tex_px, 0], [tex_px, tex_px], [0, tex_px]])
    dst = np.float32([proj(x0, y0 + wcells), proj(x0 + wcells, y0 + wcells),
                      proj(x0 + wcells, y0), proj(x0, y0)])
    # texture row 0 is top (max y in world as seen "far"); flip handled by dst order
    return cv2.getPerspectiveTransform(src, dst), proj

out = sys.stdout.buffer
frame_i = 0
t_end = args.seconds * FPS if args.seconds > 0 else float("inf")
yaw = rng.uniform(0, 2 * math.pi)

while frame_i < t_end:
    rule = pick_rule()
    ncol = len(rule)
    turns = np.array([1 if c == "R" else -1 for c in rule], np.int64)
    grid = np.zeros((G, G), np.uint8)
    pal = palette(ncol)
    x = y = G // 2; d = 0
    bb = np.array([x, x, y, y], np.int64)
    steps = 0
    cx, cy, half = float(x), float(y), 7.0
    acc = 0.0
    INTRO, MAXT = 6.0, 52.0
    hold_frames = 0
    ep_f = 0
    done = False
    fade_out = 0
    yaw_speed = rng.uniform(0.012, 0.025) * rng.choice([-1, 1])
    while frame_i < t_end:
        t = ep_f / FPS
        if not done:
            if t < INTRO:
                spf = 0.2
            else:
                spf = min(0.2 * 1.42 ** (t - INTRO), 40000.0)
            acc += spf
            n = int(acc); acc -= n
            if n:
                x, y, d = run(grid, turns, ncol, x, y, d, n, bb)
                steps += n
            ext = max(bb[1] - bb[0], bb[3] - bb[2])
            if t > MAXT or ext > G * 0.8 or steps > 6_000_000:
                done = True
        else:
            hold_frames += 1
            if hold_frames > 4 * FPS:
                fade_out += 1
                if fade_out > FPS:
                    break
        # camera targets
        bcx, bcy = (bb[0] + bb[1]) / 2 + 0.5, (bb[2] + bb[3]) / 2 + 0.5
        ext = max(bb[1] - bb[0], bb[3] - bb[2]) + 1
        tgt_half = max(7.0, ext * 0.62 + 4)
        w_ant = 1.0 if t < INTRO else max(0.0, 1 - (t - INTRO) / 6)
        tcx = w_ant * (x + 0.5) + (1 - w_ant) * bcx
        tcy = w_ant * (y + 0.5) + (1 - w_ant) * bcy
        k = 0.04 if not done else 0.06
        cx += (tcx - cx) * k; cy += (tcy - cy) * k; half += (tgt_half - half) * k
        yaw += yaw_speed / FPS * 2
        pitch = math.radians(58)
        # texture window (covers more than visible so screen is filled)
        span = half * 3.2
        wcells = int(min(G, max(16, math.ceil(span))))
        x0 = int(math.floor(cx - wcells / 2)); y0 = int(math.floor(cy - wcells / 2))
        xs = np.arange(x0, x0 + wcells) % G; ys = np.arange(y0, y0 + wcells) % G
        crop = grid[np.ix_(ys, xs)]
        tex = pal[crop]
        cellpx = max(1, int(1600 // wcells))
        if cellpx > 1:
            tex = np.repeat(np.repeat(tex, cellpx, 0), cellpx, 1)
            if cellpx >= 6:
                tex[::cellpx, :, :] = (tex[::cellpx, :, :] * 0.86).astype(np.uint8)
                tex[:, ::cellpx, :] = (tex[:, ::cellpx, :] * 0.86).astype(np.uint8)
            tex_px = wcells * cellpx
        else:
            tex_px = 1600
            tex = cv2.resize(tex, (tex_px, tex_px), interpolation=cv2.INTER_AREA)
        tex = tex[::-1]  # row 0 = far edge
        Hm, proj = homography(cx, cy, half, yaw, pitch, x0, y0, wcells, tex_px)
        frame = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(236, 236, 236))
        # ant marker
        if not done or hold_frames < FPS:
            px, py = proj(x + 0.5, y + 0.5)
            r = max(4, int(0.45 * (H / 2) / (half * 0.8) * 0.9))
            r = min(r, 22)
            cv2.circle(frame, (int(px), int(py)), r, RED, -1, cv2.LINE_AA)
        f = frame.astype(np.float32)
        f = f * (1 - fog) + BG * fog
        f *= vig
        # fade in / out
        a = min(1.0, ep_f / (FPS * 1.0))
        if fade_out:
            a = 1 - fade_out / FPS
        if a < 1:
            f = f * a + BG * vig * (1 - a)
        rgb, al = text_strip(f"STEP {steps:,}", f"RULE {rule}")
        f[H - 92:H - 22] = f[H - 92:H - 22] * (1 - al) + rgb * al
        out.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
        frame_i += 1; ep_f += 1
