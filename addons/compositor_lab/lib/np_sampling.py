# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Image sampling, gradients and Gaussian blur on the CPU (numpy). GPU twin: ``glsl/sampling.py``.

Conventions
-----------
Images are float32 ``(H, W, C)`` arrays, row 0 = bottom. Continuous pixel coordinates put the
centre of pixel ``i`` at ``i + 0.5`` (so sampling at ``(x + 0.5, y + 0.5)`` returns pixel ``(x, y)``
exactly).

Edge modes (``mode`` is an index or a name of ``EDGE_MODES``):

* ``CLAMP``   indices outside the image repeat the border pixel
* ``REPEAT``  the image tiles
* ``MIRROR``  the image tiles mirrored, border pixel duplicated (``... 1 0 | 0 1 2 .. n-1 | n-1 ..``)

``np.pad`` modes ``edge`` / ``wrap`` / ``symmetric`` implement exactly these.
"""

import math

import numpy as np

EDGE_MODES = ("CLAMP", "REPEAT", "MIRROR")
_PAD_MODE = {0: "edge", 1: "wrap", 2: "symmetric"}
_COORD_LIMIT = 1.0e6   # coordinates are clamped to +-1e6 pixels (keeps int32 index math safe)


def edge_mode_index(mode):
    if isinstance(mode, str):
        return EDGE_MODES.index(mode)
    return int(mode)


def edge_index(i, n, mode):
    """Map integer indices ``i`` (array or int) into ``[0, n)`` according to the edge mode."""
    mode = edge_mode_index(mode)
    i = np.asarray(i)
    if mode == 0:
        return np.clip(i, 0, n - 1)
    if mode == 1:
        return np.mod(i, n)
    m = np.mod(i, 2 * n)
    return np.where(m >= n, 2 * n - 1 - m, m)


def pad(img, before, after, axis, mode):
    """Pad one axis of ``img`` by ``before`` / ``after`` pixels with the edge mode."""
    widths = [(0, 0)] * img.ndim
    widths[axis] = (int(before), int(after))
    return np.pad(img, widths, mode=_PAD_MODE[edge_mode_index(mode)])


def _prep_coords(x, y):
    f32 = np.float32
    x = np.nan_to_num(np.asarray(x, f32), nan=0.0, posinf=_COORD_LIMIT, neginf=-_COORD_LIMIT)
    y = np.nan_to_num(np.asarray(y, f32), nan=0.0, posinf=_COORD_LIMIT, neginf=-_COORD_LIMIT)
    return (np.clip(x, -_COORD_LIMIT, _COORD_LIMIT) - f32(0.5),
            np.clip(y, -_COORD_LIMIT, _COORD_LIMIT) - f32(0.5))


def sample_bilinear(img, x, y, mode="CLAMP"):
    """Bilinear sample of ``img`` (H, W, C) at pixel coordinates ``x``, ``y`` (arrays of one
    shape; pixel centres at i + 0.5). Returns ``x.shape + (C,)`` float32."""
    f32 = np.float32
    h, w = img.shape[:2]
    fx, fy = _prep_coords(x, y)
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x0 = x0.astype(np.int32)
    y0 = y0.astype(np.int32)
    xa, xb = edge_index(x0, w, mode), edge_index(x0 + 1, w, mode)
    ya, yb = edge_index(y0, h, mode), edge_index(y0 + 1, h, mode)
    c00, c10 = img[ya, xa], img[ya, xb]
    c01, c11 = img[yb, xa], img[yb, xb]
    bot = c00 + (c10 - c00) * tx
    top = c01 + (c11 - c01) * tx
    return (bot + (top - bot) * ty).astype(f32, copy=False)


def cubic_weights(t):
    """Catmull-Rom weights for the 4 taps at offsets -1, 0, 1, 2 (t = fractional position)."""
    t2 = t * t
    t3 = t2 * t
    half = np.float32(0.5)
    return (half * (-t3 + t2 + t2 - t),
            half * (3.0 * t3 - 5.0 * t2 + 2.0),
            half * (-3.0 * t3 + 4.0 * t2 + t),
            half * (t3 - t2))


def sample_bicubic(img, x, y, mode="CLAMP"):
    """Bicubic (Catmull-Rom) sample, same conventions as ``sample_bilinear``. May overshoot the
    input range slightly near sharp edges."""
    h, w = img.shape[:2]
    fx, fy = _prep_coords(x, y)
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x0 = x0.astype(np.int32)
    y0 = y0.astype(np.int32)
    wx = cubic_weights(tx)
    wy = cubic_weights(ty)
    xi = [edge_index(x0 + k, w, mode) for k in (-1, 0, 1, 2)]
    out = 0.0
    for j, yk in enumerate((-1, 0, 1, 2)):
        yi = edge_index(y0 + yk, h, mode)
        row = 0.0
        for i in range(4):
            row = row + img[yi, xi[i]] * wx[i]
        out = out + row * wy[j]
    return np.asarray(out, np.float32)


def sample(img, x, y, mode="CLAMP", cubic=False):
    return (sample_bicubic if cubic else sample_bilinear)(img, x, y, mode)


def sobel(img, mode="CLAMP"):
    """Sobel gradients (gx, gy) of every channel, normalised by 1/8 so they are the derivative per
    pixel of a linear ramp. x grows with the column, y with the row (up in compositor space)."""
    img = np.asarray(img, np.float32)
    h, w = img.shape[:2]
    p = pad(pad(img, 1, 1, 0, mode), 1, 1, 1, mode)

    def col(dx, dy):   # img shifted so that result[y, x] = img[y + dy, x + dx]
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    gx = ((col(1, -1) + np.float32(2.0) * col(1, 0) + col(1, 1)) -
          (col(-1, -1) + np.float32(2.0) * col(-1, 0) + col(-1, 1))) * np.float32(0.125)
    gy = ((col(-1, 1) + np.float32(2.0) * col(0, 1) + col(1, 1)) -
          (col(-1, -1) + np.float32(2.0) * col(0, -1) + col(1, -1))) * np.float32(0.125)
    return gx, gy


def gaussian_radius(sigma):
    """Kernel half-width used by the blur: ceil(3 sigma), at least 1."""
    return max(1, int(math.ceil(3.0 * float(sigma))))


def gaussian_weights(sigma):
    """Normalised 1-D Gaussian kernel of ``2 * gaussian_radius + 1`` float32 taps."""
    r = gaussian_radius(sigma)
    i = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-i * i / (2.0 * float(sigma) ** 2))
    return (k / k.sum()).astype(np.float32)


def blur_axis(img, sigma, axis, mode="CLAMP"):
    """1-D Gaussian blur along ``axis`` (0 = rows / y, 1 = columns / x). sigma <= 0: unchanged."""
    img = np.asarray(img, np.float32)
    if sigma <= 0.0:
        return img
    r = gaussian_radius(sigma)
    wts = gaussian_weights(sigma)
    n = img.shape[axis]
    p = pad(img, r, r, axis, mode)
    out = None
    for k in range(2 * r + 1):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(k, k + n)
        term = p[tuple(sl)] * wts[k]
        out = term if out is None else out + term
    return out


def blur_gaussian(img, sigma, mode="CLAMP"):
    """Separable Gaussian blur (horizontal then vertical), kernel radius ceil(3 sigma)."""
    return blur_axis(blur_axis(img, sigma, 1, mode), sigma, 0, mode)
