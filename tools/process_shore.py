#!/usr/bin/env python3
"""Chroma-key AI shoreline overlays down to 32x32 tiles."""
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image

SRC = Path("/opt/cursor/artifacts/assets")
DST = Path("/workspace/assets")
DST.mkdir(parents=True, exist_ok=True)
TS = 32


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
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    mag = np.sqrt((r - 255.0) ** 2 + (g - 0.0) ** 2 + (b - 255.0) ** 2) < 110
    pink = (r > 160) & (g < 90) & (b > 70) & (b > g + 25) & (r > g + 70) & (r + b > 2 * g + 90)
    fuch = (r > 190) & (g < 60) & (b > 90)
    return mag | pink | fuch


def flood_chroma(arr):
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3]
    cand = is_chroma(rgb)
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
    arr[:, :, 3] = np.where(mask, 0, 255).astype(np.uint8)
    leftover = (arr[:, :, 3] > 0) & is_chroma(arr[:, :, :3])
    arr[leftover, 3] = 0
    return Image.fromarray(arr)


def crop_alpha(img, pad=0):
    arr = np.array(img)
    ys, xs = np.where(arr[:, :, 3] > 20)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, int(xs.min()) - pad), min(arr.shape[1], int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(arr.shape[0], int(ys.max()) + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def to_tile(im, anchor="top"):
    im = key_image(im)
    im = crop_alpha(im, pad=2)
    sw, sh = im.size
    if anchor == "top":
        # north edge: full width, band at top (~12px of foam+sand)
        nh = max(10, min(16, int(round(TS * sh / max(1, sw)))))
        band = im.resize((TS, nh), Image.Resampling.LANCZOS)
        out = Image.new("RGBA", (TS, TS), (0, 0, 0, 0))
        out.paste(band, (0, 0), band)
        return out
    if anchor == "bottom":
        nh = max(8, min(12, int(round(TS * sh / max(1, sw)))))
        band = im.resize((TS, nh), Image.Resampling.LANCZOS)
        out = Image.new("RGBA", (TS, TS), (0, 0, 0, 0))
        out.paste(band, (0, TS - nh), band)
        return out
    # outer corner: keep aspect, pin to top-left, about half the tile
    side = 18
    piece = im.resize((side, side), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (TS, TS), (0, 0, 0, 0))
    out.paste(piece, (0, 0), piece)
    return out


def save(im, name):
    out = DST / name
    im.save(out, "PNG", optimize=True)
    print(f"{name:24s} {im.size}  {out.stat().st_size}B")


def main():
    save(to_tile(Image.open(SRC / "shore_edge_n.png"), "top"), "shore_edge.png")
    save(to_tile(Image.open(SRC / "shore_outer_nw.png"), "corner"), "shore_outer.png")
    save(to_tile(Image.open(SRC / "wet_edge_s.png"), "bottom"), "wet_edge.png")
    print("done")


if __name__ == "__main__":
    main()
