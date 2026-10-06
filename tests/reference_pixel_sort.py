# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Reference pixel sort in plain numpy, used by the tests to check the node's output.

Kept independent of the add-on so the add-on's CPU path is compared against separate code.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Core algorithm (pure numpy, no bpy dependency)
# ---------------------------------------------------------------------------

KEYS = ('LUMA', 'HUE', 'SATURATION', 'VALUE', 'RED', 'GREEN', 'BLUE')


def compute_key(rgb, key):
    """rgb: (..., 3) float array -> (...) float key array."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if key == 'LUMA':
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if key == 'RED':
        return r
    if key == 'GREEN':
        return g
    if key == 'BLUE':
        return b
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    if key == 'VALUE':
        return mx
    delta = mx - mn
    if key == 'SATURATION':
        return np.where(mx > 0, delta / np.where(mx > 0, mx, 1), 0.0)
    # HUE in [0, 1)
    d = np.where(delta > 0, delta, 1)
    h = np.where(
        mx == r, ((g - b) / d) % 6,
        np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    h = np.where(delta > 0, h / 6.0, 0.0)
    return h % 1.0


def pixel_sort(pixels, mask_key='LUMA', lo=0.25, hi=0.8, sort_key='LUMA',
               vertical=False, reverse=False, invert_mask=False, mask=None):
    """Sort runs of in-threshold pixels.

    pixels: (H, W, C) float array, C >= 3 (row 0 = top or bottom, irrelevant).
    Runs are contiguous pixels along a row (or column if vertical) whose
    mask key lies in [lo, hi]. Each run is sorted by sort_key.
    mask: optional (H, W) bool array (same orientation as pixels); a pixel is
    only sorted where it is True (in addition to the threshold test, which is
    the only one affected by invert_mask). Runs are split where either fails.
    Returns a new array of the same shape.
    """
    img = np.asarray(pixels)
    if vertical:
        img = np.swapaxes(img, 0, 1)
    h, w, c = img.shape
    flat = img.reshape(h * w, c)
    rgb = flat[:, :3]

    mkey = compute_key(rgb, mask_key)
    sel = (mkey >= lo) & (mkey <= hi)
    if invert_mask:
        sel = ~sel
    if mask is not None:
        user = np.asarray(mask, dtype=bool)
        if vertical:
            user = np.swapaxes(user, 0, 1)
        sel &= user.reshape(h * w)
    mask = sel

    idx = np.flatnonzero(mask)
    out = flat.copy()
    if idx.size:
        # A run starts at a masked pixel whose predecessor is unmasked or
        # is the last pixel of the previous row.
        prev = np.empty_like(mask)
        prev[0] = False
        prev[1:] = mask[:-1]
        prev[::w] = False
        label = np.cumsum(mask & ~prev)[idx]

        skey = compute_key(rgb[idx], sort_key)
        if reverse:
            skey = -skey
        # label is non-decreasing over idx, so sorting by (label, key)
        # permutes pixels only within their own run.
        perm = np.lexsort((skey, label))
        out[idx] = flat[idx[perm]]

    out = out.reshape(h, w, c)
    if vertical:
        out = np.swapaxes(out, 0, 1)
    return np.ascontiguousarray(out)
