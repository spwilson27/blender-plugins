# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Independent scalar / matrix reference formulas for the blend tests (no lib code used)."""
import math

import numpy as np

BOUNDED = {"SCREEN", "OVERLAY", "SOFT_LIGHT", "SOFT_LIGHT_PEGTOP", "HARD_LIGHT", "VIVID_LIGHT",
           "LINEAR_LIGHT", "PIN_LIGHT", "HARD_MIX", "COLOR_DODGE", "COLOR_BURN", "EXCLUSION"}


def _clamp(x):
    return min(1.0, max(0.0, x))


def ref_sep(mode, cb, cs):
    """Scalar python blend of one channel, with the node's documented clamping."""
    if mode in BOUNDED:
        cb, cs = _clamp(cb), _clamp(cs)
    if mode == "NORMAL":
        r = cs
    elif mode == "MULTIPLY":
        r = cb * cs
    elif mode == "SCREEN":
        r = 1 - (1 - cb) * (1 - cs)
    elif mode == "OVERLAY":
        r = 2 * cb * cs if cb <= 0.5 else 1 - 2 * (1 - cb) * (1 - cs)
    elif mode == "HARD_LIGHT":
        r = 2 * cb * cs if cs <= 0.5 else 1 - 2 * (1 - cb) * (1 - cs)
    elif mode == "SOFT_LIGHT":
        if cs <= 0.5:
            r = cb - (1 - 2 * cs) * cb * (1 - cb)
        else:
            d = ((16 * cb - 12) * cb + 4) * cb if cb <= 0.25 else math.sqrt(cb)
            r = cb + (2 * cs - 1) * (d - cb)
    elif mode == "SOFT_LIGHT_PEGTOP":
        r = (1 - 2 * cs) * cb * cb + 2 * cs * cb
    elif mode in ("COLOR_DODGE", "COLOR_BURN", "VIVID_LIGHT"):
        def dodge(b, s):
            return 0.0 if b <= 0 else 1.0 if s >= 1 else min(1.0, b / (1 - s))

        def burn(b, s):
            return 1.0 if b >= 1 else 0.0 if s <= 0 else 1 - min(1.0, (1 - b) / s)
        if mode == "COLOR_DODGE":
            r = dodge(cb, cs)
        elif mode == "COLOR_BURN":
            r = burn(cb, cs)
        else:
            r = burn(cb, 2 * cs) if cs <= 0.5 else dodge(cb, 2 * cs - 1)
    elif mode == "LINEAR_LIGHT":
        r = cb + 2 * cs - 1
    elif mode == "PIN_LIGHT":
        r = min(cb, 2 * cs) if cs <= 0.5 else max(cb, 2 * cs - 1)
    elif mode == "HARD_MIX":
        r = 1.0 if cb + cs >= 1 else 0.0
    elif mode == "LINEAR_DODGE":
        r = cb + cs
    elif mode == "LINEAR_BURN":
        r = cb + cs - 1
    elif mode == "SUBTRACT":
        r = cb - cs
    elif mode == "DIVIDE":
        r = cb / cs if cs > 0 else (1.0 if cb > 0 else 0.0)
    elif mode == "DIFFERENCE":
        r = abs(cb - cs)
    elif mode == "EXCLUSION":
        r = cb + cs - 2 * cb * cs
    elif mode == "DARKEN":
        r = min(cb, cs)
    elif mode == "LIGHTEN":
        r = max(cb, cs)
    else:
        raise ValueError(mode)
    return _clamp(r) if mode in BOUNDED else r


# OKLab via numpy matrices; the inverse is computed, not copied from the library.
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(rgb):
    lms = np.cbrt(rgb @ _M1.T)
    return lms @ _M2.T


def from_oklab(lab):
    lms = lab @ np.linalg.inv(_M2).T
    return (lms ** 3) @ np.linalg.inv(_M1).T


def ref_nonsep(mode, cb, cs):
    """cb, cs: (n, 3) float64. Hue/Sat/Color/Lum via OKLCh polar components (atan2 / cos / sin)."""
    luma = lambda c: c @ np.array([0.2126, 0.7152, 0.0722])
    if mode == "DARKER_COLOR":
        return np.where((luma(cs) < luma(cb))[:, None], cs, cb)
    if mode == "LIGHTER_COLOR":
        return np.where((luma(cs) > luma(cb))[:, None], cs, cb)
    lb, ls = to_oklab(cb), to_oklab(cs)
    Lb, Cb, hb = lb[:, 0], np.hypot(lb[:, 1], lb[:, 2]), np.arctan2(lb[:, 2], lb[:, 1])
    Ls, Cs, hs = ls[:, 0], np.hypot(ls[:, 1], ls[:, 2]), np.arctan2(ls[:, 2], ls[:, 1])
    if mode == "HUE":
        L, C, h = Lb, np.where(Cs > 1e-5, Cb, 0.0), hs
    elif mode == "SATURATION":
        L, C, h = Lb, np.where(Cb > 1e-5, Cs, 0.0), hb
    elif mode == "COLOR":
        L, C, h = Lb, Cs, hs
    else:
        L, C, h = Ls, Cb, hb
    out = from_oklab(np.stack([L, C * np.cos(h), C * np.sin(h)], axis=1))
    return np.maximum(out, 0.0)
