# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numpy twin of ``glsl/field.py``: vector fields and Line Integral Convolution (LIC).

Pixel centres are at (x + 0.5, y + 0.5) in "pixel coordinates"; row 0 is the bottom. Fields are
(H, W, 2) float32 arrays sampled bilinearly with clamp-to-edge. LIC integrates every pixel's
streamline in both directions with a midpoint (RK2) step of fixed length and accumulates a
bilinear sample of the source per step.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import np_noise

F32 = np.float32
EPS2 = F32(1e-12)          # squared field length below which a streamline stops
IMPULSE_SEED_STEP = 7919   # impulse noise seed = source noise seed + this
_CHUNK = 1 << 16

KERNELS = ("BOX", "TRIANGLE")


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

def luminance(img):
    return (img[..., 0] * F32(0.2126) + img[..., 1] * F32(0.7152) +
            img[..., 2] * F32(0.0722)).astype(F32)


def sobel(lum):
    """Sobel gradient of an (H, W) array (clamp-to-edge), scaled by 1/8. Returns (gx, gy):
    d/dx along columns, d/dy along rows (row 0 = bottom, so +y is up)."""
    p = np.pad(lum, 1, mode="edge")
    h, w = lum.shape

    def at(dx, dy):
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    gx = ((at(1, 1) + F32(2.0) * at(1, 0) + at(1, -1)) -
          (at(-1, 1) + F32(2.0) * at(-1, 0) + at(-1, -1))) * F32(0.125)
    gy = ((at(-1, 1) + F32(2.0) * at(0, 1) + at(1, 1)) -
          (at(-1, -1) + F32(2.0) * at(0, -1) + at(1, -1))) * F32(0.125)
    return gx.astype(F32), gy.astype(F32)


def rotate(vx, vy, c, s):
    c, s = F32(c), F32(s)
    return (vx * c - vy * s).astype(F32), (vx * s + vy * c).astype(F32)


def unit_field(field):
    """Normalise an (H, W, 2) field; zero vectors stay zero. Returns (unit, length)."""
    ln = np.sqrt(field[..., 0] * field[..., 0] + field[..., 1] * field[..., 1])
    inv = np.where(ln * ln < EPS2, F32(0.0), F32(1.0) / np.maximum(ln, F32(1e-30)))
    return (field * inv[..., None]).astype(F32), ln.astype(F32)


# ---------------------------------------------------------------------------
# LIC
# ---------------------------------------------------------------------------

def noise_source(w, h, seed, density):
    """Default LIC source: grey white noise (H, W, 4) and sparse impulses (H, W) (0 / 1)."""
    ix = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    iy = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))
    n = np_noise.rand(ix, iy, seed)
    src = np.stack([n, n, n, np.ones_like(n)], axis=-1).astype(F32)
    imp = impulses(w, h, seed, density)
    return src, imp


def impulses(w, h, seed, density):
    ix = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    iy = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))
    r = np_noise.rand(ix, iy, (int(seed) + IMPULSE_SEED_STEP) & 0xFFFFFFFF)
    return (r < F32(density)).astype(F32)


def _pad(arr, h, w):
    """(H*W, C) -> (H+2, W+2, C) edge-replicated and flattened to rows, so taps never clip."""
    a = arr.reshape(h, w, -1)
    return np.ascontiguousarray(np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")).reshape(
        (h + 2) * (w + 2), -1)


def _taps(px, py, w, h):
    """Bilinear taps into a padded array (see _pad) for pixel-coordinate positions."""
    ux = np.clip(px - F32(0.5), F32(-0.5), F32(w) - F32(0.5))
    uy = np.clip(py - F32(0.5), F32(-0.5), F32(h) - F32(0.5))
    fx = np.floor(ux)
    fy = np.floor(uy)
    tx = (ux - fx).astype(F32)
    ty = (uy - fy).astype(F32)
    i00 = ((fy + F32(1.0)) * F32(w + 2) + (fx + F32(1.0))).astype(np.intp)
    return i00, i00 + (w + 2), tx, ty


def _bilerp(arr, taps):
    i0, i1, tx, ty = taps
    one = F32(1.0)
    if arr.ndim > 1:
        tx = tx[:, None]
        ty = ty[:, None]
    top = arr[i0] * (one - tx) + arr[i0 + 1] * tx
    bot = arr[i1] * (one - tx) + arr[i1 + 1] * tx
    return top * (one - ty) + bot * ty


def _lic_chunk(fld, pay, w, h, p0x, p0y, length, step, triangle, winv):
    half = F32(0.5)
    acc = _bilerp(pay, _taps(p0x, p0y, w, h))
    n = p0x.shape[0]
    wsum = np.ones(n, F32)
    for dirn in (F32(1.0), F32(-1.0)):
        px, py = p0x, p0y
        alive = np.ones(n, bool)
        for i in range(1, length + 1):
            v1 = _bilerp(fld, _taps(px, py, w, h))
            v1x, v1y = v1.real, v1.imag
            l1 = v1x * v1x + v1y * v1y
            alive &= l1 >= EPS2
            if not alive.any():
                break
            n1 = F32(1.0) / np.sqrt(np.maximum(l1, EPS2))
            v1x, v1y = v1x * n1, v1y * n1
            hs = half * dirn * step
            v2 = _bilerp(fld, _taps(px + hs * v1x, py + hs * v1y, w, h))
            v2x, v2y = v2.real, v2.imag
            l2 = v2x * v2x + v2y * v2y
            n2 = F32(1.0) / np.sqrt(np.maximum(l2, EPS2))
            ok = l2 >= EPS2
            v2x = np.where(ok, v2x * n2, v1x)
            v2y = np.where(ok, v2y * n2, v1y)
            ds = dirn * step
            px = (px + ds * v2x).astype(F32)
            py = (py + ds * v2y).astype(F32)
            wt = (F32(1.0) - F32(i) * winv) if triangle else F32(1.0)
            s = _bilerp(pay, _taps(px, py, w, h))
            aw = alive.astype(F32) * wt
            acc += s * aw[:, None]
            wsum += aw
    return acc / wsum[:, None]


def lic(field, src, imp, length, step=1.0, kernel="BOX"):
    """Line integral convolution.

    field: (H, W, 2) float32 vector field (any magnitude; only the direction is used).
    src: (H, W, C) float32 image to smear; imp: (H, W) float32 sparse impulses (smeared the
    same way, for the streamline rendering). length: steps per direction; step: pixels per step.
    kernel: "BOX" (equal weights) or "TRIANGLE" (weight 1 - i / (length + 1) at step i).
    Returns (colour (H, W, C), impulse average (H, W)). Samples beyond a stopped streak (a zero
    field) are skipped and the weights renormalised, so a constant source stays constant.
    Chunks of pixels are processed on a thread pool (numpy releases the GIL)."""
    h, w = field.shape[:2]
    c = src.shape[2]
    length = int(length)
    step = F32(step)
    # The field is stored as complex64 (one gather per tap instead of two).
    fc = np.ascontiguousarray(field, dtype=F32)
    fld = _pad(fc.view(np.complex64).reshape(h * w, 1), h, w)[:, 0]
    pay = _pad(np.concatenate([src.reshape(h * w, c), imp.reshape(h * w, 1)], axis=1)
               .astype(F32), h, w)
    winv = F32(1.0) / F32(length + 1)
    triangle = kernel == "TRIANGLE"
    out = np.empty((h * w, c + 1), F32)
    xs = np.broadcast_to(np.arange(w, dtype=F32)[None, :] + F32(0.5), (h, w)).reshape(-1)
    ys = np.broadcast_to(np.arange(h, dtype=F32)[:, None] + F32(0.5), (h, w)).reshape(-1)

    def work(a):
        b = min(h * w, a + _CHUNK)
        out[a:b] = _lic_chunk(fld, pay, w, h, xs[a:b], ys[a:b], length, step, triangle, winv)

    starts = list(range(0, h * w, _CHUNK))
    nthreads = min(len(starts), os.cpu_count() or 1)
    if nthreads > 1:
        with ThreadPoolExecutor(nthreads) as ex:
            list(ex.map(work, starts))
    else:
        for a in starts:
            work(a)
    out = out.reshape(h, w, c + 1)
    return out[..., :c], out[..., c]
