# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numpy twin of ``glsl/hash.py`` and ``glsl/noise.py``: same hash, gradients and formulas.

Positions are float32 arrays (any broadcastable shape); results are float32. Operation order
mirrors the GLSL so the only CPU/GPU differences are float rounding (fma contraction, fast math).
"""

import numpy as np

F32 = np.float32
U32 = np.uint32
I32 = np.int32

NOISE_TYPES = ("VALUE", "PERLIN", "SIMPLEX", "WORLEY_F1", "WORLEY_F2_F1")


def f32(x):
    return np.asarray(x, dtype=F32)


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

def pcg(v):
    """PCG hash of a uint32 array."""
    v = np.asarray(v, dtype=U32)
    state = v * U32(747796405) + U32(2891336453)
    word = ((state >> ((state >> U32(28)) + U32(4))) ^ state) * U32(277803737)
    return (word >> U32(22)) ^ word


def scramble_seed(seed):
    """lab_pcg(seed) for a python int (pure integer maths, no overflow warnings)."""
    v = int(seed) & 0xFFFFFFFF
    state = (v * 747796405 + 2891336453) & 0xFFFFFFFF
    word = ((((state >> ((state >> 28) + 4)) ^ state)) * 277803737) & 0xFFFFFFFF
    return ((word >> 22) ^ word) & 0xFFFFFFFF


def pcg3d(x, y, z):
    x = x * U32(1664525) + U32(1013904223)
    y = y * U32(1664525) + U32(1013904223)
    z = z * U32(1664525) + U32(1013904223)
    x = x + y * z
    y = y + z * x
    z = z + x * y
    x = x ^ (x >> U32(16))
    y = y ^ (y >> U32(16))
    z = z ^ (z >> U32(16))
    x = x + y * z
    y = y + z * x
    z = z + x * y
    return x, y, z


def hash3(ix, iy, iz, ss):
    """ix, iy, iz: int32 arrays (or python ints); ss: scrambled seed (python int)."""
    ss = int(ss) & 0xFFFFFFFF
    # Arrays only (0-d values would be numpy scalars, which warn on integer overflow).
    ix, iy, iz = np.broadcast_arrays(np.asarray(ix, dtype=I32), np.asarray(iy, dtype=I32),
                                     np.asarray(iz, dtype=I32))
    x = ix.astype(U32) + U32(ss)
    y = iy.astype(U32) + U32(ss ^ 0x68E31DA4)
    z = iz.astype(U32) + U32(ss ^ 0xB5297A4D)
    return pcg3d(x, y, z)


def u01(h):
    return (h >> U32(8)).astype(F32) * F32(1.0 / 16777216.0)


def hash3_u01(ix, iy, iz, ss):
    hx, hy, hz = hash3(ix, iy, iz, ss)
    return u01(hx), u01(hy), u01(hz)


def rand(ix, iy, seed):
    """lab_rand: white noise per pixel in [0, 1)."""
    return u01(hash3(ix, iy, 0, scramble_seed(seed))[0])


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _fade(t):
    return t * t * t * (t * (t * F32(6.0) - F32(15.0)) + F32(10.0))


def _floor3(x, y, z):
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    return fx, fy, fz, fx.astype(I32), fy.astype(I32), fz.astype(I32)


def _lerp(a, b, t):
    return a + (b - a) * t


def _trilerp(c, ux, uy, uz):
    x00 = _lerp(c[0], c[1], ux)
    x10 = _lerp(c[2], c[3], ux)
    x01 = _lerp(c[4], c[5], ux)
    x11 = _lerp(c[6], c[7], ux)
    y0 = _lerp(x00, x10, uy)
    y1 = _lerp(x01, x11, uy)
    return _lerp(y0, y1, uz)


_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))


def value3(x, y, z, seed):
    ss = scramble_seed(seed)
    fx, fy, fz, ix, iy, iz = _floor3(x, y, z)
    ux, uy, uz = _fade(x - fx), _fade(y - fy), _fade(z - fz)
    c = [u01(hash3(ix + dx, iy + dy, iz + dz, ss)[0]) * F32(2.0) - F32(1.0)
         for dx, dy, dz in _CORNERS]
    return _trilerp(c, ux, uy, uz)


def grad3(h, dx, dy, dz):
    k = (h >> U32(16)) & U32(15)
    u = np.where(k < 8, dx, dy)
    v = np.where(k < 4, dy, np.where((k == 12) | (k == 14), dx, dz))
    return np.where((k & U32(1)) == 0, u, -u) + np.where((k & U32(2)) == 0, v, -v)


def perlin3(x, y, z, seed):
    ss = scramble_seed(seed)
    fx, fy, fz, ix, iy, iz = _floor3(x, y, z)
    px, py, pz = x - fx, y - fy, z - fz
    ux, uy, uz = _fade(px), _fade(py), _fade(pz)
    c = []
    for dx, dy, dz in _CORNERS:
        h = hash3(ix + dx, iy + dy, iz + dz, ss)[0]
        c.append(grad3(h, px - F32(dx), py - F32(dy), pz - F32(dz)))
    return _trilerp(c, ux, uy, uz)


def _scorner3(ix, iy, iz, ss, dx, dy, dz):
    t = F32(0.6) - (dx * dx + dy * dy + dz * dz)
    pos = t > 0
    t = np.where(pos, t, F32(0.0)).astype(F32)
    t = t * t
    g = grad3(hash3(ix, iy, iz, ss)[0], dx, dy, dz)
    return np.where(pos, t * t * g, F32(0.0)).astype(F32)


def simplex3(x, y, z, seed):
    F3 = F32(1.0 / 3.0)
    G3 = F32(1.0 / 6.0)
    ss = scramble_seed(seed)
    s = (x + y + z) * F3
    fx, fy, fz = np.floor(x + s), np.floor(y + s), np.floor(z + s)
    ix, iy, iz = fx.astype(I32), fy.astype(I32), fz.astype(I32)
    t = (fx + fy + fz) * G3
    x0, y0, z0 = x - (fx - t), y - (fy - t), z - (fz - t)
    one = F32(1.0)
    gx = np.where(x0 >= y0, one, F32(0.0)).astype(F32)
    gy = np.where(y0 >= z0, one, F32(0.0)).astype(F32)
    gz = np.where(z0 >= x0, one, F32(0.0)).astype(F32)
    i1 = (np.minimum(gx, one - gz), np.minimum(gy, one - gx), np.minimum(gz, one - gy))
    i2 = (np.maximum(gx, one - gz), np.maximum(gy, one - gx), np.maximum(gz, one - gy))
    n = _scorner3(ix, iy, iz, ss, x0, y0, z0)
    n = n + _scorner3(ix + i1[0].astype(I32), iy + i1[1].astype(I32), iz + i1[2].astype(I32), ss,
                      x0 - i1[0] + G3, y0 - i1[1] + G3, z0 - i1[2] + G3)
    n = n + _scorner3(ix + i2[0].astype(I32), iy + i2[1].astype(I32), iz + i2[2].astype(I32), ss,
                      x0 - i2[0] + F32(2.0) * G3, y0 - i2[1] + F32(2.0) * G3,
                      z0 - i2[2] + F32(2.0) * G3)
    n = n + _scorner3(ix + 1, iy + 1, iz + 1, ss,
                      x0 - one + F32(3.0) * G3, y0 - one + F32(3.0) * G3,
                      z0 - one + F32(3.0) * G3)
    return n * F32(32.0)


def grad2(h, dx, dy):
    k = (h >> U32(16)) & U32(7)
    odd = (k & U32(1)) != 0
    diag = np.where((k & U32(1)) == 0, dx, -dx) + np.where((k & U32(2)) == 0, dy, -dy)
    axis_x = np.where(odd, -dx, dx)
    axis_y = np.where(odd, -dy, dy)
    return np.where(k < 4, diag, np.where(k < 6, axis_x, axis_y))


def _scorner2(ix, iy, ss, dx, dy):
    t = F32(0.5) - (dx * dx + dy * dy)
    pos = t > 0
    t = np.where(pos, t, F32(0.0)).astype(F32)
    t = t * t
    g = grad2(hash3(ix, iy, 0, ss)[0], dx, dy)
    return np.where(pos, t * t * g, F32(0.0)).astype(F32)


def simplex2(x, y, seed):
    F2 = F32(0.36602540378)
    G2 = F32(0.21132486540)
    ss = scramble_seed(seed)
    s = (x + y) * F2
    fx, fy = np.floor(x + s), np.floor(y + s)
    ix, iy = fx.astype(I32), fy.astype(I32)
    t = (fx + fy) * G2
    x0, y0 = x - (fx - t), y - (fy - t)
    i1x = (x0 > y0).astype(I32)
    i1y = I32(1) - i1x
    n = _scorner2(ix, iy, ss, x0, y0)
    n = n + _scorner2(ix + i1x, iy + i1y, ss, x0 - i1x.astype(F32) + G2, y0 - i1y.astype(F32) + G2)
    n = n + _scorner2(ix + 1, iy + 1, ss, x0 - F32(1.0) + F32(2.0) * G2, y0 - F32(1.0) + F32(2.0) * G2)
    return n * F32(70.0)


def worley3(x, y, z, seed, jitter):
    """Returns (F1, F2) distances."""
    ss = scramble_seed(seed)
    fx, fy, fz, ix, iy, iz = _floor3(x, y, z)
    px, py, pz = x - fx, y - fy, z - fz
    jitter = F32(jitter)
    half = F32(0.5)
    f1 = np.full(np.broadcast(x, y, z).shape, F32(8.0), dtype=F32)
    f2 = f1.copy()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rx, ry, rz = hash3_u01(ix + dx, iy + dy, iz + dz, ss)
                qx = F32(dx) + half + (rx - half) * jitter - px
                qy = F32(dy) + half + (ry - half) * jitter - py
                qz = F32(dz) + half + (rz - half) * jitter - pz
                d2 = qx * qx + qy * qy + qz * qz
                lt1 = d2 < f1
                lt2 = (~lt1) & (d2 < f2)
                f2 = np.where(lt1, f1, np.where(lt2, d2, f2)).astype(F32)
                f1 = np.where(lt1, d2, f1).astype(F32)
    return np.sqrt(f1), np.sqrt(f2)


def basis(ntype, x, y, z, seed, jitter=1.0):
    """Signed noise basis. ntype: 0..4 or a name in NOISE_TYPES."""
    if isinstance(ntype, str):
        ntype = NOISE_TYPES.index(ntype)
    if ntype == 0:
        return value3(x, y, z, seed)
    if ntype == 1:
        return perlin3(x, y, z, seed)
    if ntype == 2:
        return simplex3(x, y, z, seed)
    f1, f2 = worley3(x, y, z, seed, jitter)
    if ntype == 3:
        return np.clip(F32(2.0) * f1 - F32(1.0), F32(-1.0), F32(1.0)).astype(F32)
    return np.clip(F32(2.0) * (f2 - f1) - F32(1.0), F32(-1.0), F32(1.0)).astype(F32)


def fbm(ntype, x, y, z, seed, jitter=1.0, octaves=4, lacunarity=2.0, gain=0.5, ridged=False):
    total = F32(0.0)
    amp = F32(1.0)
    norm = F32(0.0)
    freq = F32(1.0)
    lacunarity, gain = F32(lacunarity), F32(gain)
    for i in range(int(octaves)):
        fi = F32(i)
        n = basis(ntype, x * freq + F32(1.7) * fi, y * freq + F32(9.2) * fi,
                  z * freq + F32(4.3) * fi, seed, jitter)
        if ridged:
            n = F32(1.0) - np.abs(n)
            n = n * n * F32(2.0) - F32(1.0)
        total = total + n * amp
        norm = norm + amp
        amp = amp * gain
        freq = freq * lacunarity
    return (total / norm).astype(F32)


def warp(ntype, x, y, z, seed, jitter=1.0, amount=0.0):
    if amount == 0.0:
        return x, y, z
    amount = F32(amount)
    seed = int(seed)
    wx = basis(ntype, x, y, z, (seed + 101) & 0xFFFFFFFF, jitter)
    wy = basis(ntype, x + F32(5.2), y + F32(1.3), z + F32(3.7), (seed + 202) & 0xFFFFFFFF, jitter)
    wz = basis(ntype, x + F32(2.9), y + F32(7.1), z + F32(6.3), (seed + 303) & 0xFFFFFFFF, jitter)
    return x + wx * amount, y + wy * amount, z + wz * amount


def curl2(ntype, x, y, z, seed, jitter=1.0, eps=1e-2):
    eps = F32(eps)
    dx = basis(ntype, x + eps, y, z, seed, jitter) - basis(ntype, x - eps, y, z, seed, jitter)
    dy = basis(ntype, x, y + eps, z, seed, jitter) - basis(ntype, x, y - eps, z, seed, jitter)
    k = F32(0.5) / eps
    return dy * k, -dx * k


def noise_field(ntype, x, y, z, seed, jitter=1.0, octaves=4, lacunarity=2.0, gain=0.5,
                ridged=False, warp_amount=0.0):
    """Warp + fBm mapped to [0, 1] (the Noise node's field). Float32 array."""
    wx, wy, wz = warp(ntype, x, y, z, seed, jitter, warp_amount)
    n = fbm(ntype, wx, wy, wz, seed, jitter, octaves, lacunarity, gain, ridged)
    return np.clip(F32(0.5) + F32(0.5) * n, F32(0.0), F32(1.0)).astype(F32)
