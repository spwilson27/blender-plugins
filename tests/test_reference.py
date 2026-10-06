# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Check the vectorised numpy reference against a naive per-row loop.

Runs in Blender's Python (for numpy): Blender -b --python-exit-code 1 --python test_reference.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_pixel_sort as ref  # noqa: E402


def naive(px, mask_key, lo, hi, sort_key, vertical, reverse, invert, mask=None):
    img = np.swapaxes(px, 0, 1) if vertical else px
    msk = None if mask is None else (np.swapaxes(mask, 0, 1) if vertical else mask)
    out = img.copy()
    for y in range(img.shape[0]):
        row = img[y]
        m = ref.compute_key(row[:, :3], mask_key)
        m = (m >= lo) & (m <= hi)
        if invert:
            m = ~m
        if msk is not None:
            m &= msk[y]
        x = 0
        while x < len(row):
            if not m[x]:
                x += 1
                continue
            e = x
            while e < len(row) and m[e]:
                e += 1
            seg = row[x:e]
            k = ref.compute_key(seg[:, :3], sort_key)
            order = np.argsort(-k if reverse else k, kind='stable')
            out[y, x:e] = seg[order]
            x = e
    return np.swapaxes(out, 0, 1) if vertical else out


rng = np.random.default_rng(1)
n = 0
for mk in ref.KEYS:
    for sk in ref.KEYS:
        for vertical in (False, True):
            for reverse in (False, True):
                for invert in (False, True):
                    px = rng.random((13, 17, 4)).astype(np.float32)
                    a = ref.pixel_sort(px, mk, 0.2, 0.9, sk, vertical, reverse, invert)
                    b = naive(px, mk, 0.2, 0.9, sk, vertical, reverse, invert)
                    assert np.array_equal(a, b), (mk, sk, vertical, reverse, invert)
                    n += 1
print(f"PASS: {n} option combinations match the naive loop")

# Runs must not wrap from the end of one row to the start of the next.
px = np.zeros((2, 3, 4), np.float32)
px[..., 3] = 1
px[0, :, :3] = np.array([0.9, 0.5, 0.3])[:, None]
px[1, :, :3] = np.array([0.2, 0.8, 0.4])[:, None]
r = ref.pixel_sort(px, 'LUMA', 0.0, 1.0, 'LUMA')
assert list(r[0, :, 0]) == sorted(px[0, :, 0]) and list(r[1, :, 0]) == sorted(px[1, :, 0])
print("PASS: runs do not wrap rows")
