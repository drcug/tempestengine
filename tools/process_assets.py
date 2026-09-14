#!/usr/bin/env python3
"""Chroma-key, crop, pixelate and pack AI-generated game assets."""
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageFilter

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

def erode(mask, n=1):
    return ~dilate(~mask, n)

def magenta_mask(arr):
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    dist = np.sqrt((r - 255) ** 2 + (g - 0) ** 2 + (b - 255) ** 2)
    # JPEG-fringed magenta / hot pink
    mag = dist < 95
    mag |= (r > 150) & (b > 130) & (g < 150) & ((r.astype(np.int32) + b) > g * 2 + 40)
    mag |= (r > 180) & (b > 180) & (g < 120)
    return mag, dist

def flood_bg(arr):
    h, w = arr.shape[:2]
    mag, dist = magenta_mask(arr)
    mask = np.zeros((h, w), dtype=bool)
    q = deque()
    edge = []
    for x in range(w):
        edge.append((0, x))
        edge.append((h - 1, x))
    for y in range(h):
        edge.append((y, 0))
        edge.append((y, w - 1))
    for y, x in edge:
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
    # eat magenta halo
    mask = dilate(mask, 3)
    return mask

def crop_alpha(img, pad=2):
    arr = np.array(img)
    a = arr[:, :, 3]
    ys, xs = np.where(a > 20)
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

def process_sprite(name, height, extra_dilate=0):
    im = Image.open(SRC / name).convert("RGBA")
    arr = np.array(im)
    mask = flood_bg(arr)
    if extra_dilate:
        mask = dilate(mask, extra_dilate)
    arr[:, :, 3] = np.where(mask, 0, 255).astype(np.uint8)
    # kill leftover near-magenta inside residual
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    leftover = (a > 0) & (r > 160) & (b > 160) & (g < 90)
    arr[leftover, 3] = 0
    im = Image.fromarray(arr)
    im = crop_alpha(im, pad=4)
    w, h = im.size
    scale = height / h
    nw, nh = max(1, int(round(w * scale))), height
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    # slight pixel crunch
    im = im.resize((max(1, nw // 1), nh), Image.Resampling.NEAREST)
    im = outline(im)
    return im

def process_tile(name, size=256):
    im = Image.open(SRC / name).convert("RGB")
    # center-crop to square then downscale (tiles have painted borders)
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im.convert("RGBA")

def process_title(name, size=(896, 504)):
    im = Image.open(SRC / name).convert("RGB")
    im = im.resize(size, Image.Resampling.LANCZOS)
    return im.convert("RGBA")

SPRITES = {
    "spr_prospero.png": 52,
    "spr_miranda.png": 50,
    "spr_ariel.png": 50,
    "spr_caliban.png": 48,
    "spr_ferdinando.png": 50,
    "spr_antonio.png": 50,
    "spr_trinculo.png": 50,
    "spr_stefano.png": 50,
    "spr_beast.png": 40,
    "spr_wisp.png": 44,
    "spr_mutin.png": 50,
    "spr_portal.png": 56,
    "spr_relic.png": 28,
    "tile_decor_jungle.png": 32,
    "tile_decor_stone.png": 32,
}

TILES = [
    "tile_sand.png",
    "tile_jungle.png",
    "tile_stone.png",
    "tile_boss.png",
    "tile_cave.png",
    "tile_ruins.png",
    "tile_wall.png",
    "tile_water.png",
    "tile_water2.png",
]

def save(im, name):
    out = DST / name
    im.save(out, "PNG", optimize=True)
    print(f"{name:28s} {im.size}  {out.stat().st_size//1024}KB")

def main():
    for name, h in SPRITES.items():
        extra = 1 if name.startswith("tile_decor") else 0
        im = process_sprite(name, h, extra_dilate=extra)
        save(im, name)
    for name in TILES:
        save(process_tile(name, 256 if "water" not in name else 128), name)
    save(process_title("bg_title.png"), "bg_title.png")
    print("done")

if __name__ == "__main__":
    main()
