# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numpy twin of ``glsl/dither.py``: per-pixel dither thresholds in [0, 1).

``threshold(mode, shape, seed)`` -> (H, W) float32. Use as
``floor(v * (levels - 1) + t) / (levels - 1)``; mode 0 gives 0.5 (plain rounding).
Modes: 0 none, 1/2/3 Bayer 2x2/4x4/8x8, 4 R2 low-discrepancy (blue-noise like), 5 white hash.
Pixel coordinates are texel indices, row 0 = bottom.
"""

import numpy as np

from . import np_noise

F32 = np.float32
U32 = np.uint32

MODES = ("NONE", "BAYER2", "BAYER4", "BAYER8", "BLUE", "RANDOM")

# R2 sequence constants (Roberts): 1/phi2 and 1/phi2^2 as 32 bit fixed point.
R2_A = int(0.7548776662466927 * 4294967296.0)
R2_B = int(0.5698402909980532 * 4294967296.0)


def bayer_index(x, y, levels):
    """Bayer matrix entry (0 .. 4**levels - 1) for integer arrays x, y (any shape)."""
    x = np.asarray(x).astype(U32)
    y = np.asarray(y).astype(U32)
    mask = U32((1 << levels) - 1)
    x = x & mask
    y = y & mask
    v = np.zeros(np.broadcast(x, y).shape, U32)
    for i in range(levels):
        xb = (x >> U32(i)) & U32(1)
        yb = ((x ^ y) >> U32(i)) & U32(1)
        v = (v << U32(2)) | (yb << U32(1)) | xb
    return v


def threshold_at(mode, x, y, seed=0):
    """Threshold for integer pixel coordinate arrays x, y (broadcastable)."""
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    if 1 <= mode <= 3:
        idx = bayer_index(x, y, mode)
        return ((idx.astype(F32) + F32(0.5)) * F32(1.0 / (1 << (2 * mode)))).astype(F32)
    if mode == 4:
        ss = np_noise.pcg(np.array([int(seed) & 0xFFFFFFFF], dtype=U32))[0]
        u = x.astype(U32) * U32(R2_A) + y.astype(U32) * U32(R2_B) + ss
        return np_noise.u01(u)
    if mode == 5:
        return np_noise.rand(x, y, seed)
    return np.full(np.broadcast(x, y).shape, 0.5, F32)


def threshold(mode, shape, seed=0):
    h, w = int(shape[0]), int(shape[1])
    y, x = np.mgrid[0:h, 0:w]
    return threshold_at(int(mode), x, y, seed)
