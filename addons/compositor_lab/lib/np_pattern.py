# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numpy twin of ``glsl/pattern.py``: Voronoi / Mosaic cells and the Pattern node's tilings.

Coordinates are float32 arrays; results are float32. Operation order mirrors the GLSL so the
only CPU/GPU differences are float rounding (fma contraction, fast math). Hashes come from
``np_noise`` (bit-exact with the GLSL ones).
"""

import numpy as np

from . import np_noise

F32 = np.float32
I32 = np.int32

PATTERNS = ("STRIPES", "CHECKER", "DOTS", "HEX", "TRUCHET_ARC", "TRUCHET_DIAG", "MOIRE", "RINGS")
METRICS = ("EUCLIDEAN", "MANHATTAN", "CHEBYSHEV")

_BIG = F32(1e10)


# ---------------------------------------------------------------------------
# Voronoi
# ---------------------------------------------------------------------------

def _vor_dist(dx, dy, metric):
    if metric == 1:
        return np.abs(dx) + np.abs(dy)
    if metric == 2:
        return np.maximum(np.abs(dx), np.abs(dy))
    return dx * dx + dy * dy


def site_offset(ix, iy, ss, jitter, cphi, sphi, anim):
    """Jittered feature point of cell (ix, iy) relative to the cell centre (jitter applied):
    ``jitter * (ra * cos(phi) + rb * sin(phi))`` with ra, rb in [-0.5, 0.5) random per cell."""
    rx, ry, _ = np_noise.hash3_u01(ix, iy, 0, ss)
    rx = rx - F32(0.5)
    ry = ry - F32(0.5)
    if anim:
        bx, by, _ = np_noise.hash3_u01(ix, iy, 1, ss)
        rx = rx * cphi + (bx - F32(0.5)) * sphi
        ry = ry * cphi + (by - F32(0.5)) * sphi
    return rx * jitter, ry * jitter


def voronoi(x, y, metric, jitter, cphi, sphi, anim, ss):
    """Nearest / second nearest feature points over the 3x3 cell neighbourhood.
    x, y: float32 cell-space coordinates (same shape). Returns (f1, f2, cell_x, cell_y,
    site_x, site_y): distances in the metric, the winning cell (int32) and its site position
    (float32, cell space)."""
    jitter, cphi, sphi = F32(jitter), F32(cphi), F32(sphi)
    fl_x, fl_y = np.floor(x), np.floor(y)
    fx, fy = x - fl_x, y - fl_y
    ix, iy = fl_x.astype(I32), fl_y.astype(I32)
    d1 = np.full(x.shape, _BIG, F32)
    d2 = np.full(x.shape, _BIG, F32)
    wx = np.zeros(x.shape, I32)
    wy = np.zeros(x.shape, I32)
    sx = np.zeros(x.shape, F32)
    sy = np.zeros(x.shape, F32)
    half = F32(0.5)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ox, oy = site_offset(ix + I32(dx), iy + I32(dy), ss, jitter, cphi, sphi, anim)
            px = F32(dx) + half + ox
            py = F32(dy) + half + oy
            d = _vor_dist(px - fx, py - fy, metric).astype(F32)
            lt1 = d < d1
            lt2 = (~lt1) & (d < d2)
            d2 = np.where(lt1, d1, np.where(lt2, d, d2)).astype(F32)
            d1 = np.where(lt1, d, d1).astype(F32)
            wx = np.where(lt1, I32(dx), wx)
            wy = np.where(lt1, I32(dy), wy)
            sx = np.where(lt1, px, sx).astype(F32)
            sy = np.where(lt1, py, sy).astype(F32)
    if metric == 0:
        d1, d2 = np.sqrt(d1), np.sqrt(d2)
    return d1, d2, ix + wx, iy + wy, fl_x + sx, fl_y + sy


def cell_rgb(cx, cy, ss):
    """Random colour of a cell (3 float32 arrays in [0, 1))."""
    return np_noise.hash3_u01(cx, cy, 2, ss)


def cell_id(cx, cy, ss):
    """Random scalar id of a cell in [0, 1)."""
    return np_noise.hash3_u01(cx, cy, 3, ss)[0]


def border_mask(f1, f2, width, k):
    """Anti-aliased border coverage from F2 - F1 (``k`` = cell units per pixel; the gradient of
    F2 - F1 is at most 2, so a ramp of total width 2k is about one pixel)."""
    if width <= 0.0:
        return np.zeros(f1.shape, F32)
    inv = F32(1.0) / (F32(2.0) * F32(k))
    return np.clip(F32(0.5) + (F32(width) - (f2 - f1)) * inv, F32(0.0), F32(1.0)).astype(F32)


# ---------------------------------------------------------------------------
# Pattern tilings. Each returns coverage of colour B in [0, 1].
# ---------------------------------------------------------------------------

def aa(s, inv_w):
    """Coverage from a signed distance (positive inside) with a linear ramp of width 1/inv_w."""
    return np.clip(F32(0.5) + s * inv_w, F32(0.0), F32(1.0)).astype(F32)


def stripes_s(u, duty):
    duty = F32(duty)
    if duty <= 0.0:
        return np.full(u.shape, F32(-1.0), F32)
    if duty >= 1.0:
        return np.full(u.shape, F32(1.0), F32)
    c = (u - np.floor(u)) - F32(0.5) * duty
    c = c - np.floor(c + F32(0.5))
    return (F32(0.5) * duty - np.abs(c)).astype(F32)


def _checker_s(u, v):
    fu, fv = np.floor(u), np.floor(v)
    n = fu.astype(I32) + fv.astype(I32)
    e = F32(0.5) - np.maximum(np.abs((u - fu) - F32(0.5)), np.abs((v - fv) - F32(0.5)))
    return np.where((n & I32(1)) == 1, e, -e).astype(F32)


def _dots_s(u, v, duty):
    duty = F32(duty)
    if duty <= 0.0:
        return np.full(u.shape, F32(-1.0), F32)
    dx = (u - np.floor(u)) - F32(0.5)
    dy = (v - np.floor(v)) - F32(0.5)
    return (F32(0.5) * duty - np.sqrt(dx * dx + dy * dy)).astype(F32)


_HEX_H = F32(1.7320508)
_HEX_HALF_H = F32(0.8660254)
_HEX_INV_H = F32(0.57735027)


def _hex_s(u, v, duty):
    duty = F32(duty)
    half = F32(0.5)
    ax = np.floor(u) + half
    ay = (np.floor(v * _HEX_INV_H) + half) * _HEX_H
    qx = u - half
    qy = v - _HEX_HALF_H
    bx = np.floor(qx) + half + half
    by = (np.floor(qy * _HEX_INV_H) + half) * _HEX_H + _HEX_HALF_H
    h1x, h1y = u - ax, v - ay
    h2x, h2y = u - bx, v - by
    first = (h1x * h1x + h1y * h1y) < (h2x * h2x + h2y * h2y)
    hx = np.where(first, h1x, h2x)
    hy = np.where(first, h1y, h2y)
    ax_, ay_ = np.abs(hx), np.abs(hy)
    e = half - np.maximum(ax_, half * ax_ + _HEX_HALF_H * ay_)
    return (e - (F32(1.0) - duty) * half).astype(F32)


def _truchet_flip(cu, cv, ss):
    r = np_noise.hash3_u01(cu.astype(I32), cv.astype(I32), 0, ss)[0]
    return r < F32(0.5)


def _truchet_s(mode, u, v, duty, ss):
    cu, cv = np.floor(u), np.floor(v)
    fx, fy = u - cu, v - cv
    lw = F32(duty) * F32(0.25)
    if mode == "TRUCHET_ARC":
        fx = np.where(_truchet_flip(cu, cv, ss), F32(1.0) - fx, fx).astype(F32)
        d1 = np.abs(np.sqrt(fx * fx + fy * fy) - F32(0.5))
        gx, gy = fx - F32(1.0), fy - F32(1.0)
        d2 = np.abs(np.sqrt(gx * gx + gy * gy) - F32(0.5))
        return (lw - np.minimum(d1, d2)).astype(F32)
    # Diagonals: round-capped segments, so strokes join across the corners of neighbouring tiles.
    best = np.full(u.shape, F32(-1e10), F32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px = fx - F32(dx)
            py = fy - F32(dy)
            px = np.where(_truchet_flip(cu + F32(dx), cv + F32(dy), ss), F32(1.0) - px, px)
            t = np.clip((px + py) * F32(0.5), F32(0.0), F32(1.0))
            qx, qy = px - t, py - t
            best = np.maximum(best, lw - np.sqrt(qx * qx + qy * qy))
    return best.astype(F32)


def pattern_mask(mode, u, v, duty, inv_w, ss, moire_c=1.0, moire_s=0.0):
    """Coverage of colour B for pattern ``mode`` (a name in PATTERNS) at pattern coordinates
    (u, v) (one unit = one period / cell), with edge ramp 1/inv_w."""
    inv_w = F32(inv_w)
    if mode == "STRIPES":
        return aa(stripes_s(u, duty), inv_w)
    if mode == "CHECKER":
        return aa(_checker_s(u, v), inv_w)
    if mode == "DOTS":
        return aa(_dots_s(u, v, duty), inv_w)
    if mode == "HEX":
        return aa(_hex_s(u, v, duty), inv_w)
    if mode in ("TRUCHET_ARC", "TRUCHET_DIAG"):
        return aa(_truchet_s(mode, u, v, duty, ss), inv_w)
    if mode == "MOIRE":
        u2 = u * F32(moire_c) + v * F32(moire_s)
        return (aa(stripes_s(u, duty), inv_w) * aa(stripes_s(u2, duty), inv_w)).astype(F32)
    if mode == "RINGS":
        return aa(stripes_s(np.sqrt(u * u + v * v), duty), inv_w)
    raise ValueError(mode)


def to_pattern_space(w, h, k, rot_c, rot_s, ox, oy):
    """Pattern coordinates (u, v) of every pixel centre: centred on the image, rotated, then
    offset. Returns two (h, w) float32 arrays."""
    k, c, s = F32(k), F32(rot_c), F32(rot_s)
    qx = ((np.arange(w, dtype=F32) + F32(0.5)) - F32(0.5) * F32(w)) * k
    qy = ((np.arange(h, dtype=F32) + F32(0.5)) - F32(0.5) * F32(h)) * k
    qx = np.broadcast_to(qx[None, :], (h, w))
    qy = np.broadcast_to(qy[:, None], (h, w))
    u = qx * c + qy * s + F32(ox)
    v = qy * c - qx * s + F32(oy)
    return u.astype(F32), v.astype(F32)
