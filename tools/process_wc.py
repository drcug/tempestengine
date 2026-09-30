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
    mag = np.sqrt((r - 255.0) ** 2 + (g - 0.0) ** 2 + (b - 255.0) ** 2) < 125
    # hot pink / rose used by the image model (e.g. 218,7,120)
    pink = (r > 160) & (g < 90) & (b > 70) & (b > g + 25) & (r > g + 70) & (r + b > 2 * g + 90)
    # near-sampled fuchsia with low green
    fuch = (r > 190) & (g < 60) & (b > 90)
    # JPEG-compressed magenta (#FF00AA-ish)
    jpeg_mag = (r > 200) & (g < 55) & (b > 140) & (r > g + 100)
    # lime / green screens (used when magenta would eat purple wisps)
    lime = (g > 170) & (r < 140) & (b < 140) & (g > r + 50) & (g > b + 40)
    return mag | pink | fuch | jpeg_mag | lime


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
    cand |= dist < 52

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
    return dilate(mask, 1)


def largest_blob(arr):
    """Keep only the biggest opaque island so leftover grid specks don't inflate the crop."""
    a = arr[:, :, 3] > 20
    try:
        from scipy.ndimage import label
        lab, n = label(a)
        if n == 0:
            return arr
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = lab == counts.argmax()
        out = arr.copy()
        out[:, :, 3] = np.where(keep, arr[:, :, 3], 0)
        return out
    except Exception:
        pass
    h, w = a.shape
    vis = np.zeros((h, w), dtype=np.uint16)
    best, best_n, lab = [], 0, 1
    for y in range(h):
        for x in range(w):
            if not a[y, x] or vis[y, x]:
                continue
            q = deque([(y, x)])
            vis[y, x] = lab
            cells = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = lab
                        q.append((ny, nx))
                        cells.append((ny, nx))
            if len(cells) > best_n:
                best, best_n = cells, len(cells)
            lab += 1
            if lab > 4000:
                break
    if best_n < 40:
        return arr
    keep = np.zeros((h, w), dtype=bool)
    for y, x in best:
        keep[y, x] = True
    out = arr.copy()
    out[:, :, 3] = np.where(keep, arr[:, :, 3], 0)
    return out


def key_image(im):
    im = im.convert("RGBA")
    arr = np.array(im)
    mask = flood_chroma(arr)
    arr[:, :, 3] = np.where(mask, 0, arr[:, :, 3]).astype(np.uint8)
    # leftover grid / JPEG fringe on the cell frame
    arr[:5, :, 3] = 0
    arr[-5:, :, 3] = 0
    arr[:, :5, 3] = 0
    arr[:, -5:, 3] = 0
    return Image.fromarray(largest_blob(arr))


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


def load_sheet(stem):
    """Prefer the newest bird's-eye JPEG over leftover 3/4 PNG sheets."""
    found = []
    for ext in (".jpg", ".jpeg", ".png"):
        p = SRC / f"{stem}{ext}"
        if p.exists():
            found.append(p)
    if not found:
        raise FileNotFoundError(stem)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return Image.open(found[0]).convert("RGBA")


def split_rows(stem, cols=4, rows=2, fw=40, fh=52):
    raw = load_sheet(stem)
    w, h = raw.size
    cw, ch = w // cols, h // rows
    # inset drops 1px black grid lines that some sheets include
    inset = 8
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell = raw.crop(
                (
                    c * cw + inset,
                    r * ch + inset,
                    (c + 1) * cw - inset,
                    (r + 1) * ch - inset,
                )
            )
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
    fw, fh = 40, 52
    for k in CHARS:
        ns = split_rows(f"wc_{k}_ns", fw=fw, fh=fh)
        ew = split_rows(f"wc_{k}_ew", fw=fw, fh=fh)
        down, up = ns[0], ns[1]
        right, left = ew[0], ew[1]
        save(assemble([down, up, left, right], fw, fh), f"wc_{k}.png")
    print("done")


if __name__ == "__main__":
    main()
