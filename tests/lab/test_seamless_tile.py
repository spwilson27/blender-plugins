# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Seamless Tile node: wrap-around continuity, centre identity, mirror symmetry, independent
reference formula, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_seamless_tile.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

NODE = "CompositorNodeLabSeamlessTile"
SIZE = (96, 64)
F32 = np.float32

# CPU and GPU evaluate the same two-step lerp in float32; only fma contraction differs.
GPU_ATOL = 2e-6


def smooth_image(w, h, alpha=False, seed=0):
    """Smooth, clearly non-tileable image: ramps + low frequency waves (+ optional alpha)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = x / w, y / h
    px = np.empty((h, w, 4), F32)
    px[..., 0] = u
    px[..., 1] = v
    px[..., 2] = 0.5 + 0.4 * np.sin(6.0 * u + 4.0 * v + seed)
    px[..., 3] = 1.0
    if alpha:
        a = 0.3 + 0.7 * (0.5 + 0.5 * np.cos(3 * u - 2 * v))
        px[..., :3] *= a[..., None].astype(F32)
        px[..., 3] = a
    return px


def seam_error(img):
    """Mean abs difference across the wrap-around seams (left|right, bottom|top), and the mean
    abs difference between neighbouring pixels inside the image (the natural step size)."""
    sx = np.abs(img[:, 0] - img[:, -1]).mean()
    sy = np.abs(img[0] - img[-1]).mean()
    ix = np.abs(np.diff(img, axis=1)).mean()
    iy = np.abs(np.diff(img, axis=0)).mean()
    return sx, sy, ix, iy


def ref_offset(a, bw, axes="BOTH"):
    """Independent float64 reference (np.roll based)."""
    a = a.astype(np.float64)
    h, w = a.shape[:2]

    def mask(n):
        c = np.arange(n) + 0.5
        e = np.minimum(c, n - c) / n
        t = np.clip(e * 2.0 / bw, 0, 1)
        return t * t * (3 - 2 * t)

    mx, my = mask(w)[None, :, None], mask(h)[:, None, None]
    c = a
    if axes in ("BOTH", "X"):
        c = np.roll(a, -(w // 2), axis=1) * (1 - mx) + a * mx
    d = c
    if axes in ("BOTH", "Y"):
        d = np.roll(c, -(h // 2), axis=0) * (1 - my) + c * my
    return d


@H.guard("offset")
def test_offset():
    a = smooth_image(*SIZE)
    for bw in (0.25, 0.5, 1.0):
        for axes in ("BOTH", "X", "Y"):
            out = H.render_node(NODE, "CPU", SIZE, props={"mode": "OFFSET", "blend_width": bw, "axes": axes},
                                images={"Image": a})
            H.compare("offset bw=%g axes=%s: CPU == np.roll reference" % (bw, axes), out,
                      ref_offset(a, bw, axes), 3e-6)

    out = H.render_node(NODE, "CPU", SIZE, props={"mode": "OFFSET", "blend_width": 0.5},
                        images={"Image": a})
    sx, sy, ix, iy = seam_error(out)
    sxa, sya, ixa, iya = seam_error(a)
    print("  input seams (%.4f, %.4f) steps (%.4f, %.4f); output seams (%.4f, %.4f) steps (%.4f, %.4f)"
          % (sxa, sya, ixa, iya, sx, sy, ix, iy))
    H.check(sxa > 20 * ixa and sya > 20 * iya, "the test image is not seamless (seam >> neighbour step)")
    H.check(sx < 1.5 * ix + 1e-3 and sy < 1.5 * iy + 1e-3,
            "output wraps seamlessly: seam step %.4f/%.4f vs inner step %.4f/%.4f" % (sx, sy, ix, iy))

    # Centre: where both masks are 1 the output is the input.
    bw = 0.5
    c = np.arange(SIZE[0]) + 0.5
    full_x = np.minimum(c, SIZE[0] - c) / SIZE[0] >= 0.5 * bw
    c = np.arange(SIZE[1]) + 0.5
    full_y = np.minimum(c, SIZE[1] - c) / SIZE[1] >= 0.5 * bw
    reg = full_y[:, None] & full_x[None, :]
    H.check(reg.sum() > 0.2 * reg.size, "centre region exists (%d px)" % reg.sum())
    H.check(np.abs(out[reg] - a[reg]).max() < 1e-6, "centre region is the input unchanged")
    # Borders show the half-offset copy.
    left = np.roll(a, -(SIZE[0] // 2), axis=1)[:, 0]
    H.check(np.abs(out[full_y, 0] - left[full_y]).max() < 5e-3,
            "left border column comes from the half-offset copy (where the vertical mask is 1)")

    # One axis only: the other seam stays.
    outx = H.render_node(NODE, "CPU", SIZE, props={"axes": "X"}, images={"Image": a})
    sx, sy, ix, iy = seam_error(outx)
    H.check(sx < 1.5 * ix + 1e-3 and sy > 20 * iy, "axes=X fixes only the horizontal seam")
    # Alpha / premultiplied data go through the same linear blend.
    aa = smooth_image(*SIZE, alpha=True)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_width": 0.7}, images={"Image": aa})
    H.compare("offset on premultiplied alpha image", out, ref_offset(aa, 0.7), 3e-6)
    # A constant image stays constant.
    k = np.full((16, 24, 4), 0.3, F32)
    out = H.render_node(NODE, "CPU", (24, 16), images={"Image": k})
    H.compare("constant image unchanged", out, k, 1e-6)


@H.guard("mirror")
def test_mirror():
    a = smooth_image(*SIZE, seed=1)
    h, w = SIZE[1], SIZE[0]
    out = H.render_node(NODE, "CPU", SIZE, props={"mode": "MIRROR"}, images={"Image": a})
    H.check(np.array_equal(out[:, ::-1], out) and np.array_equal(out[::-1], out),
            "mirror: symmetric in x and y (opposite edges identical)")
    H.check(np.array_equal(out[:h // 2, :w // 2], a[:h // 2, :w // 2]), "mirror: lower-left quadrant is kept")
    sx, sy, ix, iy = seam_error(out)
    H.check(sx == 0 and sy == 0, "mirror: seam difference exactly 0")
    outx = H.render_node(NODE, "CPU", SIZE, props={"mode": "MIRROR", "axes": "X"}, images={"Image": a})
    H.check(np.array_equal(outx[:, ::-1], outx) and np.array_equal(outx[:, :w // 2], a[:, :w // 2]),
            "mirror X only: symmetric in x, rows untouched")
    outy = H.render_node(NODE, "CPU", SIZE, props={"mode": "MIRROR", "axes": "Y"}, images={"Image": a})
    H.check(np.array_equal(outy[::-1], outy) and np.array_equal(outy[:h // 2], a[:h // 2]),
            "mirror Y only")
    # Odd sizes: x -> min(x, w - 1 - x).
    b = smooth_image(37, 23)
    out = H.render_node(NODE, "CPU", (37, 23), props={"mode": "MIRROR"}, images={"Image": b})
    xs = np.minimum(np.arange(37), 36 - np.arange(37))
    ys = np.minimum(np.arange(23), 22 - np.arange(23))
    H.check(np.array_equal(out, b[ys[:, None], xs[None, :]]), "mirror odd size")


@H.guard("parity")
def test_parity():
    cases = [("smooth", smooth_image(*SIZE)), ("alpha", smooth_image(*SIZE, alpha=True)),
             ("noise", H.test_image(*SIZE, seed=3)), ("hdr", H.test_image(*SIZE, seed=4, hdr=True))]
    for label, a in cases:
        for props in ({"mode": "OFFSET", "blend_width": 0.5}, {"mode": "OFFSET", "blend_width": 0.1},
                      {"mode": "OFFSET", "blend_width": 1.0, "axes": "X"},
                      {"mode": "OFFSET", "axes": "Y"}, {"mode": "MIRROR"},
                      {"mode": "MIRROR", "axes": "X"}):
            cpu = H.render_node(NODE, "CPU", SIZE, props=props, images={"Image": a})
            gpu = H.render_node(NODE, "GPU", SIZE, props=props, images={"Image": a})
            tol = 0.0 if props["mode"] == "MIRROR" else GPU_ATOL
            H.compare("%s %s: GPU vs CPU" % (label, props), gpu, cpu, tol, rtol=2e-6 if tol else 0.0)
    for size in ((4, 4), (7, 5), (33, 17), (4, 40), (51, 4), (97, 61)):
        a = smooth_image(*size, alpha=True)
        for props in ({}, {"mode": "MIRROR"}):
            cpu = H.render_node(NODE, "CPU", size, props=props, images={"Image": a})
            gpu = H.render_node(NODE, "GPU", size, props=props, images={"Image": a})
            H.compare("%dx%d %s: GPU vs CPU" % (size[0], size[1], props or "offset"), gpu, cpu, GPU_ATOL, rtol=2e-6)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for mode in ("OFFSET", "MIRROR"):
            out = H.render_node(NODE, dev, (16, 12), props={"mode": mode})
            H.check(out.shape == (12, 16, 4) and np.isfinite(out).all()
                    and np.allclose(out, [0.5, 0.5, 0.5, 1.0], atol=1e-6),
                    "%s %s: unlinked input -> default grey (%s)" % (dev, mode, out[0, 0]))
            out = H.render_node(NODE, dev, (16, 12), props={"mode": mode},
                                inputs={"Image": (0.2, 0.4, 0.6, 1.0)})
            H.check(np.allclose(out, [0.2, 0.4, 0.6, 1.0], atol=1e-6), "%s %s: colour default" % (dev, mode))
        for size in ((4, 4), (5, 9), (200, 120)):
            a = H.test_image(*size, seed=9)
            for mode in ("OFFSET", "MIRROR"):
                out = H.render_node(NODE, dev, size, props={"mode": mode}, images={"Image": a})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %s %dx%d finite, right shape" % (dev, mode, size[0], size[1]))


for fn in (test_offset, test_mirror, test_parity, test_robustness):
    fn()
H.finish()
