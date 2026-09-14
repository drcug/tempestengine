#!/usr/bin/env python3
"""Process 2-row walk-cycle sheets into a 4-direction (down/up/left/right) grid."""
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image

SRC = Path("/opt/cursor/artifacts/assets")
DST = Path("/workspace/assets")
DST.mkdir(parents=True, exist_ok=True)


def dilate(mask, n=2):
    m = mask.copy()
    for _ in range(n):
        p = np.pad(m, 1, constant_values=False)
        m = (
            m
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return m


def is_chroma(rgb):
    """Hot pink / magenta screen, not red clothing or purple cloth."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    mag = np.sqrt((r - 255.0) ** 2 + (g - 0.0) ** 2 + (b - 255.0) ** 2) < 110
    # hot pink / rose used by the image model (e.g. 218,7,120)
    pink = (r > 160) & (g < 90) & (b > 70) & (b > g + 25) & (r > g + 70) & (r + b > 2 * g + 90)
    # near-sampled fuchsia with low green
    fuch = (r > 190) & (g < 60) & (b > 90)
    return mag | pink | fuch


def flood_chroma(arr):
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3]
    cand = is_chroma(rgb)
    # also anything close to the median border color
    border = np.concatenate(
        [
            rgb[0].reshape(-1, 3),
            rgb[-1].reshape(-1, 3),
            rgb[1:-1, 0].reshape(-1, 3),
            rgb[1:-1, -1].reshape(-1, 3),
        ]
    )
    bg = np.median(border.astype(np.float32), axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    cand |= dist < 62

    mask = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if cand[y, x] and not mask[y, x]:
                mask[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if cand[y, x] and not mask[y, x]:
                mask[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and cand[ny, nx]:
                mask[ny, nx] = True
                q.append((ny, nx))
    # catch anti-aliased pink fringes next to already-keyed pixels
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    fringe = (r > 140) & (g < 110) & (b > 60) & (b > g) & (r > g + 40)
    mask |= dilate(mask, 1) & fringe
    return dilate(mask, 2)


def key_image(im):
    im = im.convert("RGBA")
    arr = np.array(im)
    mask = flood_chroma(arr)
    arr[:, :, 3] = np.where(mask, 0, arr[:, :, 3]).astype(np.uint8)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    leftover = (a > 0) & is_chroma(arr[:, :, :3])
    arr[leftover, 3] = 0
    return Image.fromarray(arr)


def crop_alpha(img, pad=2):
    arr = np.array(img)
    ys, xs = np.where(arr[:, :, 3] > 20)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def outline(img, color=(8, 6, 16, 255)):
    arr = np.array(img)
    a = arr[:, :, 3] > 40
    ring = dilate(a, 1) & ~a
    out = arr.copy()
    out[ring] = color
    return Image.fromarray(out)


def fit_on(canvas_w, canvas_h, spr):
    sw, sh = spr.size
    scale = min(canvas_w / max(1, sw), canvas_h / max(1, sh))
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    spr = spr.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    out.paste(spr, ((canvas_w - nw) // 2, canvas_h - nh), spr)
    return out


def split_rows(name, cols=4, rows=2, fw=36, fh=48):
    raw = Image.open(SRC / name).convert("RGBA")
    w, h = raw.size
    cw, ch = w // cols, h // rows
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell = raw.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            cell = key_image(cell)
            cell = crop_alpha(cell, pad=3)
            cell = outline(cell)
            row.append(fit_on(fw, fh, cell))
        out.append(row)
    return out


def assemble(rows, fw, fh):
    sheet = Image.new("RGBA", (fw * 4, fh * len(rows)), (0, 0, 0, 0))
    for r, row in enumerate(rows):
        for c, fr in enumerate(row):
            sheet.paste(fr, (c * fw, r * fh), fr)
    return sheet


CHARS = [
    "prospero",
    "miranda",
    "ariel",
    "caliban",
    "ferdinando",
    "antonio",
    "trinculo",
    "stefano",
    "beast",
    "wisp",
    "mutin",
]


def save(im, name):
    out = DST / name
    im.save(out, "PNG", optimize=True)
    print(f"{name:24s} {im.size}  {out.stat().st_size // 1024}KB")


def main():
    fw, fh = 36, 48
    for k in CHARS:
        ns = split_rows(f"wc_{k}_ns.png", fw=fw, fh=fh)
        ew = split_rows(f"wc_{k}_ew.png", fw=fw, fh=fh)
        down, up = ns[0], ns[1]
        right, left = ew[0], ew[1]
        save(assemble([down, up, left, right], fw, fh), f"wc_{k}.png")
    print("done")


if __name__ == "__main__":
    main()
