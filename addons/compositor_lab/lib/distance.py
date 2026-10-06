# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Exact Euclidean distance transforms: numpy (CPU) and compute shader (GPU) with identical results.

``edt_sq(feature, cap)``: squared distance (float32, integer valued) from every pixel to the
nearest True pixel of ``feature`` (0 on feature pixels), clamped to ``cap``. CPU: Felzenszwalb &
Huttenlocher's linear-time lower-envelope algorithm run on all rows (then columns) at once, one
vectorised numpy step per column. Exact for any distance.

``gpu_edt_sq(src, dst, radius, ...)``: the same on the GPU with two separable passes (see
``glsl/distance.py``), exact for distances <= ``radius`` and clamped to ``cap_for(radius)`` beyond.
Use ``cap_for(radius)`` as the ``cap`` of the CPU version for bit-identical results.

``d_ge / d_gt / d_le / d_lt(d2, a)``: compare squared distances with a float32 threshold ``a``
(``d >= a`` ...) without a square root, mirroring ``lab_d_*`` in the GLSL.

``scratch(w, h, role, fmt)``: cached (per thread) scratch GPUTexture for multi-pass nodes.
"""

import numpy as np

from . import gpu as lab_gpu
from .glsl import distance as _glsl

F32 = np.float32
_BIG = 1e20


def cap_for(radius):
    """Clamp value (float) of the squared distance for a search radius."""
    return float((int(radius) + 1) ** 2)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def _edt1d(f):
    """Lower envelope of parabolas along axis 1 for every row of ``f`` (float64, (L, n)):
    out[l, q] = min_p f[l, p] + (q - p)^2."""
    n_lines, n = f.shape
    if n == 1:
        return f.copy()
    rows = np.arange(n_lines)
    v = np.zeros((n_lines, n), np.int64)
    z = np.empty((n_lines, n + 1))
    z[:, 0] = -np.inf
    z[:, 1] = np.inf
    k = np.zeros(n_lines, np.int64)
    for q in range(1, n):
        fq = f[:, q] + float(q * q)
        while True:
            vk = v[rows, k]
            s = (fq - (f[rows, vk] + (vk * vk).astype(np.float64))) / (2.0 * (q - vk))
            pop = s <= z[rows, k]
            if not pop.any():
                break
            k = k - pop
        k = k + 1
        v[rows, k] = q
        z[rows, k] = s
        z[rows, k + 1] = np.inf
    out = np.empty_like(f)
    k[:] = 0
    for q in range(n):
        while True:
            adv = z[rows, k + 1] < q
            if not adv.any():
                break
            k = k + adv
        vk = v[rows, k]
        out[:, q] = ((q - vk) ** 2).astype(np.float64) + f[rows, vk]
    return out


def edt_sq(feature, cap=None):
    """Squared Euclidean distance to the nearest True pixel of ``feature`` ((H, W) bool), float32.

    Pixels with no feature at all (or beyond ``cap``) get ``cap`` (default 1e18)."""
    feature = np.asarray(feature, bool)
    f = np.where(feature, 0.0, _BIG)
    g = _edt1d(np.ascontiguousarray(f.T)).T          # along y (columns)
    d = _edt1d(np.ascontiguousarray(g))              # along x (rows)
    if cap is None:
        cap = 1e18
    return np.minimum(d, float(cap)).astype(F32)


def d_ge(d2, a):
    a = F32(a)
    return (a <= 0) | (d2 >= a * a)


def d_gt(d2, a):
    a = F32(a)
    return (a < 0) | (d2 > a * a)


def d_le(d2, b):
    b = F32(b)
    return (b >= 0) & (d2 <= b * b)


def d_lt(d2, b):
    b = F32(b)
    return (b > 0) & (d2 < b * b)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

scratch = lab_gpu.scratch      # kept for the nodes using ``distance.scratch(w, h, role, fmt)``
clear_cache = lab_gpu.clear_cache


def gpu_edt_sq(src, dst, radius, polarity=1, threshold=0.5):
    """Squared distance transform on the GPU.

    ``src``: GPUTexture whose value >= ``threshold`` marks feature pixels (``polarity`` +1) or
    < ``threshold`` (``polarity`` -1). ``dst``: R32F GPUTexture of the same size, receives
    ``min(d2, cap_for(radius))``."""
    radius = max(int(radius), 1)
    w, h = int(dst.width), int(dst.height)
    g = scratch(w, h, "dt_g", "R32F")
    uniforms = {"d_r": ("int", radius), "d_pol": ("int", 1 if polarity > 0 else -1),
                "d_thr": ("float", float(threshold)), "d_cap": ("float", cap_for(radius))}
    lab_gpu.pointwise(_glsl.PASS1_BODY, {"G": g}, inputs={"F": ("float", src)},
                      uniforms=uniforms, libs=("distance",))
    lab_gpu.pointwise(_glsl.PASS2_BODY, {"D": dst}, inputs={"G": ("float", g)},
                      uniforms=uniforms, libs=("distance",))
