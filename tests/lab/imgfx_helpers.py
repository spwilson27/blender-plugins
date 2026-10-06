# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Helpers shared by the Filters B tests (Kuwahara, Displace, Liquify, Edge Stylise): test images
and independent float64 reference implementations (sampling, Sobel, Gaussian blur)."""

import numpy as np

import harness as H

LUMA = (0.2126, 0.7152, 0.0722)


def smooth_image(w, h, seed=1, alpha=False):
    """Smooth RGBA float32 test image (low-frequency sines): sampling differences between CPU and
    GPU stay far below the pixel-to-pixel contrast, so parity can use tight tolerances."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    px = np.empty((h, w, 4), np.float32)
    for c in range(3):
        f = rng.uniform(1.0, 3.0, 2)
        ph = rng.uniform(0, 6.28, 2)
        px[..., c] = 0.5 + 0.4 * np.sin(2 * np.pi * f[0] * x / w + ph[0]) * np.cos(
            2 * np.pi * f[1] * y / h + ph[1])
    px[..., 3] = 1.0
    if alpha:
        a = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * x / w * 1.5 + y / h * 3.0))
        px[..., :3] *= a[..., None].astype(np.float32)
        px[..., 3] = a
    return px


def disc_image(w, h, radius, center=None, color=(1.0, 0.5, 0.25)):
    """Opaque-disc-on-transparent image (hard edge; pixel centres inside the radius are set).
    Returns (image (h, w, 4) premultiplied, boolean mask)."""
    cx, cy = center if center is not None else (w / 2.0, h / 2.0)
    y, x = np.mgrid[0:h, 0:w]
    mask = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= radius * radius
    img = np.zeros((h, w, 4), np.float32)
    img[mask, :3] = color
    img[mask, 3] = 1.0
    return img, mask


def luma(rgb):
    return rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]


def ref_index(i, n, mode):
    """Edge-mode index remap (independent form: mirror = reflect of the residue mod 2n)."""
    i = np.asarray(i, np.int64)
    if mode == "CLAMP":
        return np.minimum(np.maximum(i, 0), n - 1)
    if mode == "REPEAT":
        return i % n
    k = i % (2 * n)
    return np.minimum(k, 2 * n - 1 - k)


def ref_bilinear(img, x, y, mode="CLAMP"):
    """float64 bilinear sample at pixel coordinates (centres at i + 0.5)."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    fx = np.asarray(x, np.float64) - 0.5
    fy = np.asarray(y, np.float64) - 0.5
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    out = 0.0
    for dy, wy in ((0, 1.0 - ty), (1, ty)):
        for dx, wx in ((0, 1.0 - tx), (1, tx)):
            out = out + img[ref_index(y0 + dy, h, mode), ref_index(x0 + dx, w, mode)] * wx * wy
    return out


def ref_sobel(img, mode="CLAMP"):
    """float64 Sobel / 8 gradients (gx, gy) of an (H, W, C) image, clamped edges."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    ys = np.arange(h)
    xs = np.arange(w)

    def at(dx, dy):
        return img[ref_index(ys + dy, h, mode)[:, None], ref_index(xs + dx, w, mode)[None, :]]

    gx = (at(1, -1) + 2 * at(1, 0) + at(1, 1) - at(-1, -1) - 2 * at(-1, 0) - at(-1, 1)) / 8.0
    gy = (at(-1, 1) + 2 * at(0, 1) + at(1, 1) - at(-1, -1) - 2 * at(0, -1) - at(1, -1)) / 8.0
    return gx, gy


def ref_gaussian(img, sigma, mode="CLAMP"):
    """float64 separable Gaussian blur, radius ceil(3 sigma), by explicit index gathering."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    r = max(1, int(np.ceil(3.0 * sigma)))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2.0 * sigma * sigma))
    k /= k.sum()
    tmp = sum(k[j] * img[:, ref_index(np.arange(w) + j - r, w, mode)] for j in range(2 * r + 1))
    return sum(k[j] * tmp[ref_index(np.arange(h) + j - r, h, mode)] for j in range(2 * r + 1))


def render_both(node, size, props=None, inputs=None, images=None):
    """(cpu, gpu) renders of a node."""
    return (H.render_node(node, "CPU", size, props, inputs, images),
            H.render_node(node, "GPU", size, props, inputs, images))
