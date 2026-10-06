# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numpy twin of ``glsl/blend.py``. See that file for terminology (cb = backdrop A, cs = source B).

``blend_premul(mode, a, b, fac)`` works on premultiplied RGBA (..., 4) float32 arrays, exactly like
the compositor's colours: unpremultiply, blend, composite with W3C source-over, premultiply.
"""

import numpy as np

from . import np_color
from .glsl.blend import BOUNDED, CHROMA_EPS, MODES

F32 = np.float32


def _dodge(cb, cs):
    safe = np.where(cs >= 1, F32(0.5), cs)
    r = np.minimum(F32(1.0), cb / (F32(1.0) - safe))
    return np.where(cb <= 0, F32(0.0), np.where(cs >= 1, F32(1.0), r)).astype(F32)


def _burn(cb, cs):
    safe = np.where(cs <= 0, F32(0.5), cs)
    r = F32(1.0) - np.minimum(F32(1.0), (F32(1.0) - cb) / safe)
    return np.where(cb >= 1, F32(1.0), np.where(cs <= 0, F32(0.0), r)).astype(F32)


def separable(mode, cb, cs):
    """Per-channel blend of float32 arrays (mode: name from MODES)."""
    one, two, half = F32(1.0), F32(2.0), F32(0.5)
    if mode == "NORMAL":
        return cs
    if mode == "MULTIPLY":
        return cb * cs
    if mode == "SCREEN":
        return cb + cs - cb * cs
    if mode == "OVERLAY":
        return np.where(cb <= half, two * cb * cs, one - two * (one - cb) * (one - cs))
    if mode == "SOFT_LIGHT":
        lo = cb - (one - two * cs) * cb * (one - cb)
        d = np.where(cb <= F32(0.25), ((F32(16.0) * cb - F32(12.0)) * cb + F32(4.0)) * cb,
                     np.sqrt(np.maximum(cb, F32(0.0))))
        hi = cb + (two * cs - one) * (d - cb)
        return np.where(cs <= half, lo, hi)
    if mode == "SOFT_LIGHT_PEGTOP":
        return (one - two * cs) * cb * cb + two * cs * cb
    if mode == "HARD_LIGHT":
        return np.where(cs <= half, two * cb * cs, one - two * (one - cb) * (one - cs))
    if mode == "VIVID_LIGHT":
        return np.where(cs <= half, _burn(cb, two * cs), _dodge(cb, two * cs - one))
    if mode == "LINEAR_LIGHT":
        return cb + two * cs - one
    if mode == "PIN_LIGHT":
        return np.where(cs <= half, np.minimum(cb, two * cs), np.maximum(cb, two * cs - one))
    if mode == "HARD_MIX":
        return np.where(cb + cs >= one, one, F32(0.0))
    if mode == "COLOR_DODGE":
        return _dodge(cb, cs)
    if mode == "COLOR_BURN":
        return _burn(cb, cs)
    if mode == "LINEAR_DODGE":
        return cb + cs
    if mode == "LINEAR_BURN":
        return cb + cs - one
    if mode == "SUBTRACT":
        return cb - cs
    if mode == "DIVIDE":
        safe = np.where(cs > 0, cs, one)
        return np.where(cs > 0, cb / safe, np.where(cb > 0, one, F32(0.0)))
    if mode == "DIFFERENCE":
        return np.abs(cb - cs)
    if mode == "EXCLUSION":
        return cb + cs - two * cb * cs
    if mode == "DARKEN":
        return np.minimum(cb, cs)
    if mode == "LIGHTEN":
        return np.maximum(cb, cs)
    raise ValueError(mode)


def nonseparable(mode, cb, cs):
    """cb, cs: (..., 3) float32."""
    if mode == "DARKER_COLOR":
        return np.where((np_color.luma(cs) < np_color.luma(cb))[..., None], cs, cb)
    if mode == "LIGHTER_COLOR":
        return np.where((np_color.luma(cs) > np_color.luma(cb))[..., None], cs, cb)
    ob = np_color.linear_to_oklab(cb)
    os_ = np_color.linear_to_oklab(cs)
    chb = np.sqrt(ob[..., 1] * ob[..., 1] + ob[..., 2] * ob[..., 2])
    chs = np.sqrt(os_[..., 1] * os_[..., 1] + os_[..., 2] * os_[..., 2])
    eps = F32(CHROMA_EPS)
    if mode == "HUE":
        k = np.where(chs > eps, chb / np.where(chs > eps, chs, F32(1.0)), F32(0.0))
        r = np.stack([ob[..., 0], os_[..., 1] * k, os_[..., 2] * k], axis=-1)
    elif mode == "SATURATION":
        k = np.where(chb > eps, chs / np.where(chb > eps, chb, F32(1.0)), F32(0.0))
        r = np.stack([ob[..., 0], ob[..., 1] * k, ob[..., 2] * k], axis=-1)
    elif mode == "COLOR":
        r = np.stack([ob[..., 0], os_[..., 1], os_[..., 2]], axis=-1)
    elif mode == "LUMINOSITY":
        r = np.stack([os_[..., 0], ob[..., 1], ob[..., 2]], axis=-1)
    else:
        raise ValueError(mode)
    return np.maximum(np_color.oklab_to_linear(r), F32(0.0))


def blend_rgb(mode, cb, cs):
    """Blend straight RGB arrays (..., 3)."""
    cb = np.asarray(cb, dtype=F32)
    cs = np.asarray(cs, dtype=F32)
    if MODES.index(mode) >= 21:
        return nonseparable(mode, cb, cs).astype(F32)
    bounded = mode in BOUNDED
    if bounded:
        cb = np.clip(cb, F32(0.0), F32(1.0))
        cs = np.clip(cs, F32(0.0), F32(1.0))
    r = separable(mode, cb, cs).astype(F32)
    if bounded:
        r = np.clip(r, F32(0.0), F32(1.0))
    return r


def blend_premul(mode, a, b, fac):
    """a, b: (..., 4) premultiplied; fac: scalar or (..., 1)/(...) array. Returns (..., 4)."""
    a = np.asarray(a, dtype=F32)
    b = np.asarray(b, dtype=F32)
    fac = np.asarray(fac, dtype=F32)
    if fac.ndim == a.ndim - 1:
        fac = fac[..., None]
    aa = a[..., 3:4]
    ab = b[..., 3:4]
    cb = np.where(aa > 0, a[..., :3] / np.where(aa > 0, aa, F32(1.0)), F32(0.0)).astype(F32)
    cs = np.where(ab > 0, b[..., :3] / np.where(ab > 0, ab, F32(1.0)), F32(0.0)).astype(F32)
    bl = blend_rgb(mode, cb, cs)
    one = F32(1.0)
    rgb = a[..., :3] * (one - ab) + b[..., :3] * (one - aa) + bl * (aa * ab)
    full = np.concatenate([rgb, aa + ab - aa * ab], axis=-1)
    return (a + (full - a) * fac).astype(F32)
