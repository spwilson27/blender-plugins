# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numpy twin of ``glsl/color.py``. Arrays are (..., 3) float32 (last axis = channels)."""

import numpy as np

F32 = np.float32

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F32)

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], dtype=F32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], dtype=F32)
_M2_INV = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]], dtype=F32)
_M1_INV = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=F32)


def lab_mod(x, y):
    return x - y * np.floor(x / y)


def luma(c):
    c = np.asarray(c, dtype=F32)
    return F32(0.2126) * c[..., 0] + F32(0.7152) * c[..., 1] + F32(0.0722) * c[..., 2]


def srgb_to_linear(c):
    c = np.asarray(c, dtype=F32)
    a = np.abs(c)
    r = np.where(a <= F32(0.04045), a / F32(12.92),
                 np.power((a + F32(0.055)) / F32(1.055), F32(2.4)))
    return np.where(c < 0, -r, r).astype(F32)


def linear_to_srgb(c):
    c = np.asarray(c, dtype=F32)
    a = np.abs(c)
    r = np.where(a <= F32(0.0031308), a * F32(12.92),
                 F32(1.055) * np.power(a, F32(1.0 / 2.4)) - F32(0.055))
    return np.where(c < 0, -r, r).astype(F32)


def rgb_to_hsv(c):
    c = np.asarray(c, dtype=F32)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    d = mx - mn
    dd = np.where(d > 0, d, F32(1.0))
    h = np.where(mx == r, (g - b) / dd,
                 np.where(mx == g, (b - r) / dd + F32(2.0), (r - g) / dd + F32(4.0)))
    h = np.where((mx == r) & (h < 0), h + F32(6.0), h)
    h = np.where(d > 0, h / F32(6.0), F32(0.0))
    s = np.where(mx > 0, d / np.where(mx > 0, mx, F32(1.0)), F32(0.0))
    return np.stack([h, s, mx], axis=-1).astype(F32)


def _hsv_chan(h6, off):
    return np.clip(np.abs(lab_mod(h6 + F32(off), F32(6.0)) - F32(3.0)) - F32(1.0), 0, 1)


def hsv_to_rgb(hsv):
    hsv = np.asarray(hsv, dtype=F32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = (h - np.floor(h)) * F32(6.0)
    k = np.stack([_hsv_chan(h6, 0.0), _hsv_chan(h6, 4.0), _hsv_chan(h6, 2.0)], axis=-1)
    return (v[..., None] * (F32(1.0) + (k - F32(1.0)) * s[..., None])).astype(F32)


def linear_to_oklab(c):
    c = np.asarray(c, dtype=F32)
    lms = np.stack([_M1[i, 0] * c[..., 0] + _M1[i, 1] * c[..., 1] + _M1[i, 2] * c[..., 2]
                    for i in range(3)], axis=-1)
    lms = np.cbrt(lms)
    return np.stack([_M2[i, 0] * lms[..., 0] + _M2[i, 1] * lms[..., 1] + _M2[i, 2] * lms[..., 2]
                     for i in range(3)], axis=-1).astype(F32)


def oklab_to_linear(lab):
    lab = np.asarray(lab, dtype=F32)
    lms = np.stack([lab[..., 0] * _M2_INV[i, 0] + lab[..., 1] * _M2_INV[i, 1] +
                    lab[..., 2] * _M2_INV[i, 2] for i in range(3)], axis=-1)
    lms = lms * lms * lms
    return np.stack([_M1_INV[i, 0] * lms[..., 0] + _M1_INV[i, 1] * lms[..., 1] +
                     _M1_INV[i, 2] * lms[..., 2] for i in range(3)], axis=-1).astype(F32)


def oklab_to_oklch(lab):
    lab = np.asarray(lab, dtype=F32)
    return np.stack([lab[..., 0], np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2),
                     np.arctan2(lab[..., 2], lab[..., 1])], axis=-1).astype(F32)


def oklch_to_oklab(lch):
    lch = np.asarray(lch, dtype=F32)
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(lch[..., 2]),
                     lch[..., 1] * np.sin(lch[..., 2])], axis=-1).astype(F32)


def linear_to_oklch(c):
    return oklab_to_oklch(linear_to_oklab(c))


def oklch_to_linear(lch):
    return oklab_to_linear(oklch_to_oklab(lch))
