#!/usr/bin/env python3
"""Process top-down walk strips and bust portraits."""
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

def magenta_mask(arr):
    r = arr[:, :, 0].astype(np.int32)
    g = arr[:, :, 1].astype(np.int32)
    b = arr[:, :, 2].astype(np.int32)
    dist = np.sqrt((r - 255.0) ** 2 + (g - 0.0) ** 2 + (b - 255.0) ** 2)
    mag = dist < 95
    mag |= (r > 150) & (b > 130) & (g < 150) & ((r + b) > g * 2 + 40)
    mag |= (r > 180) & (b > 180) & (g < 120)
    return mag, dist

def flood_bg(arr):
    h, w = arr.shape[:2]
    mag, dist = magenta_mask(arr)
    mask = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if mag[y, x] and not mask[y, x]:
                mask[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if mag[y, x] and not mask[y, x]:
                mask[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and mag[ny, nx]:
                mask[ny, nx] = True
                q.append((ny, nx))
    mask |= dist < 42
    return dilate(mask, 3)

def key_image(im):
    im = im.convert("RGBA")
    arr = np.array(im)
    mask = flood_bg(arr)
    arr[:, :, 3] = np.where(mask, 0, 255).astype(np.uint8)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    leftover = (a > 0) & (r > 160) & (b > 160) & (g < 90)
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
    scale = min(canvas_w / sw, canvas_h / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    spr = spr.resize((nw, nh), Image.Resampling.NEAREST)
    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    out.paste(spr, ((canvas_w - nw) // 2, canvas_h - nh), spr)
    return out

def process_portrait(name, size=72):
    im = key_image(Image.open(SRC / name))
    arr = np.array(im)
    r, g, b, a = arr[:, :, 0].astype(np.int32), arr[:, :, 1].astype(np.int32), arr[:, :, 2].astype(np.int32), arr[:, :, 3]
    mag = (a > 0) & (r > 120) & (b > 100) & (g < 160) & ((r + b) > (g * 9 // 5 + 30))
    arr[mag, 3] = 0
    im = crop_alpha(Image.fromarray(arr), pad=2)
    w, h = im.size
    sc = size / max(w, h)
    im = im.resize((max(1, int(round(w * sc))), max(1, int(round(h * sc)))), Image.Resampling.LANCZOS)
    arr = np.array(im)
    r, g, b, a = arr[:, :, 0].astype(np.int32), arr[:, :, 1].astype(np.int32), arr[:, :, 2].astype(np.int32), arr[:, :, 3]
    mag = (a > 0) & (r > 130) & (b > 110) & (g < 150) & ((r + b) > g * 2)
    arr[mag, 3] = 0
    im = Image.fromarray(arr)
    return outline(im)


def process_strip(name, cols=4, fw=36, fh=32):
    im = key_image(Image.open(SRC / name))
    w, h = im.size
    cw = w // cols
    frames = []
    for i in range(cols):
        cell = im.crop((i * cw, 0, (i + 1) * cw, h))
        cell = crop_alpha(cell, pad=2)
        cell = outline(cell)
        frames.append(fit_on(fw, fh, cell))
    sheet = Image.new("RGBA", (fw * cols, fh), (0, 0, 0, 0))
    for i, fr in enumerate(frames):
        sheet.paste(fr, (i * fw, 0), fr)
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
]
FOES = ["beast", "wisp", "mutin"]

def save(im, name):
    out = DST / name
    im.save(out, "PNG", optimize=True)
    print(f"{name:28s} {im.size}  {out.stat().st_size // 1024}KB")

def main():
    for k in CHARS:
        save(process_portrait(f"port_{k}.png", 72), f"port_{k}.png")
        save(process_strip(f"td_{k}.png", fw=36, fh=30), f"td_{k}.png")
        back = SRC / f"td_{k}_back.png"
        if back.exists():
            save(process_strip(f"td_{k}_back.png", fw=36, fh=30), f"td_{k}_back.png")
    for k in FOES:
        save(process_strip(f"td_{k}.png", fw=38, fh=28), f"td_{k}.png")
    print("done")

if __name__ == "__main__":
    main()
