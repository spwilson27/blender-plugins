# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Displace / Glass node: exact integer shifts, float64 references, CPU vs GPU, robustness.
Blender -b --factory-startup --python-exit-code 1 --python test_displace.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H
import imgfx_helpers as X

H.setup()

NODE = "CompositorNodeLabDisplace"
SIZE = (100, 60)
F32 = np.float32
MODES = ("CLAMP", "REPEAT", "MIRROR")


def render(device, img, mp=None, size=SIZE, props=None, inputs=None, extra=None):
    images = {"Image": img}
    if mp is not None:
        images["Map"] = mp
    images.update(extra or {})
    return H.render_node(NODE, device, size, props, inputs, images)


def const_map(r, g, size=SIZE):
    m = np.empty((size[1], size[0], 4), F32)
    m[...] = (r, g, 0.5, 1.0)
    return m


def shifted(img, dx, dy, mode):
    """out[y, x] = img[y + dy, x + dx] with the edge mode (integer shifts)."""
    h, w = img.shape[:2]
    ys = X.ref_index(np.arange(h) + dy, h, mode)
    xs = X.ref_index(np.arange(w) + dx, w, mode)
    return img[ys[:, None], xs[None, :]]


def ref_offset(img, mp, strength, mode, disp=0.0):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    st = np.broadcast_to(np.asarray(strength, np.float64), (h, w))
    ox = (mp[..., 0].astype(np.float64) - 0.5) * 2 * st
    oy = (mp[..., 1].astype(np.float64) - 0.5) * 2 * st
    return ref_displaced(img, x, y, ox, oy, mode, disp)


def ref_glass(img, mp, strength, mode, disp=0.0):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    gx, gy = X.ref_sobel(mp[..., :3], mode)
    gain = max(w, h) / 100.0
    ox = -strength * gain * X.luma(gx)
    oy = -strength * gain * X.luma(gy)
    return ref_displaced(img, x, y, ox, oy, mode, disp)


def ref_displaced(img, x, y, ox, oy, mode, disp):
    out = X.ref_bilinear(img, x + 0.5 + ox, y + 0.5 + oy, mode)
    if disp != 0.0:
        r = X.ref_bilinear(img, x + 0.5 + ox * (1 - disp), y + 0.5 + oy * (1 - disp), mode)
        b = X.ref_bilinear(img, x + 0.5 + ox * (1 + disp), y + 0.5 + oy * (1 + disp), mode)
        out = out.copy()
        out[..., 0] = r[..., 0]
        out[..., 2] = b[..., 2]
    return out


@H.guard("exact shifts")
def test_shifts():
    img = H.test_image(*SIZE, seed=5)
    # Strength 8, map (0.75, 0.375): offset = (0.25 * 16, -0.125 * 16) = (4, -2): exact in float32.
    mp = const_map(0.75, 0.375)
    for dev in ("CPU", "GPU"):
        for mode in MODES:
            out = render(dev, img, mp, props={"edge_mode": mode}, inputs={"Strength": 8.0})
            H.compare("%s %s: constant map = exact translation (4, -2)" % (dev, mode), out,
                      shifted(img, 4, -2, mode), 1e-6)
        # Strength image (per-pixel, constant): same result.
        st = np.full((SIZE[1], SIZE[0]), 8.0, F32)
        out = render(dev, img, mp, extra={"Strength": st})
        H.compare("%s: strength as an image" % dev, out, shifted(img, 4, -2, "CLAMP"), 1e-6)
        # Zero strength is the identity (even with a wild map); also with dispersion.
        wild = H.test_image(*SIZE, seed=8)
        for mode in MODES:
            for interp in ("LINEAR", "CUBIC"):
                out = render(dev, img, wild, props={"edge_mode": mode, "interpolation": interp},
                             inputs={"Strength": 0.0, "Dispersion": 0.7})
                H.compare("%s %s %s: zero strength is identity" % (dev, mode, interp), out, img, 1e-6)
        out = render(dev, img, wild, props={"mode": "GLASS"}, inputs={"Strength": 0.0})
        H.compare("%s: glass, zero strength is identity" % dev, out, img, 1e-6)
        # Dispersion: R uses offset * (1 - d), B * (1 + d): (2, -1), (4, -2), (6, -3).
        out = render(dev, img, mp, inputs={"Strength": 8.0, "Dispersion": 0.5})
        ref = img.copy()
        ref[..., 0] = shifted(img, 2, -1, "CLAMP")[..., 0]
        ref[..., 1] = shifted(img, 4, -2, "CLAMP")[..., 1]
        ref[..., 3] = shifted(img, 4, -2, "CLAMP")[..., 3]
        ref[..., 2] = shifted(img, 6, -3, "CLAMP")[..., 2]
        H.compare("%s: dispersion shifts channels apart" % dev, out, ref, 1e-6)
        # Unlinked map: grey 0.5 = no displacement.
        out = render(dev, img, None, inputs={"Strength": 50.0})
        H.compare("%s: default map (0.5) is identity" % dev, out, img, 1e-6)
        # Cubic interpolation also translates exactly by integers.
        out = render(dev, img, mp, props={"interpolation": "CUBIC", "edge_mode": "MIRROR"},
                     inputs={"Strength": 8.0})
        H.compare("%s: bicubic integer translation" % dev, out, shifted(img, 4, -2, "MIRROR"), 1e-6)


@H.guard("references")
def test_references():
    img = H.test_image(*SIZE, seed=6)
    mp = X.smooth_image(*SIZE, seed=3)
    # Half-pixel translation = average of two neighbours.
    half = const_map(0.75, 0.5)          # Strength 1: offset (0.5, 0)
    out = render("CPU", img, half, inputs={"Strength": 1.0})
    ref = 0.5 * (img + shifted(img, 1, 0, "CLAMP"))
    H.compare("half pixel offset averages neighbours", out, ref, 1e-6)
    for mode in MODES:
        for dev in ("CPU", "GPU"):
            out = render(dev, img, mp, props={"edge_mode": mode}, inputs={"Strength": 17.3})
            H.compare("%s offset %s: float64 reference" % (dev, mode), out,
                      ref_offset(img, mp, 17.3, mode), 2e-5)
            out = render(dev, img, mp, props={"edge_mode": mode, "mode": "GLASS"},
                         inputs={"Strength": 700.0})
            H.compare("%s glass %s: float64 reference" % (dev, mode), out,
                      ref_glass(img, mp, 700.0, mode), 1e-4)
            out = render(dev, img, mp, props={"edge_mode": mode}, inputs={"Strength": 9.0, "Dispersion": 0.4})
            H.compare("%s offset %s dispersion 0.4: float64 reference" % (dev, mode), out,
                      ref_offset(img, mp, 9.0, mode, 0.4), 2e-5)
    # Per-pixel strength image.
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    st = (20.0 * xx / SIZE[0]).astype(F32)
    for dev in ("CPU", "GPU"):
        out = render(dev, img, mp, extra={"Strength": st})
        H.compare("%s: per-pixel strength image" % dev, out, ref_offset(img, mp, st, "CLAMP"), 2e-5)
    # Glass on a linear ramp: slope 0.005 / px, gain 1 (100 px wide), Strength 800: shift -4 px in x
    # (a bump pushes the image away from its slope: out[x] = img[x - 4]); interior only (Sobel edges).
    ramp = np.ones((SIZE[1], SIZE[0], 4), F32)
    ramp[..., :3] = (0.1 + 0.005 * np.arange(SIZE[0]))[None, :, None]
    ramp_y = np.ones((SIZE[1], SIZE[0], 4), F32)
    ramp_y[..., :3] = (0.1 + 0.005 * np.arange(SIZE[1]))[:, None, None]
    sm = X.smooth_image(*SIZE, seed=2)
    for dev in ("CPU", "GPU"):
        out = render(dev, sm, ramp, props={"mode": "GLASS"}, inputs={"Strength": 800.0})
        H.compare("%s: glass ramp in x = shift by -4 px" % dev, out[:, 8:-8], shifted(sm, -4, 0, "CLAMP")[:, 8:-8], 2e-4)
        out = render(dev, sm, ramp_y, props={"mode": "GLASS"}, inputs={"Strength": 800.0})
        H.compare("%s: glass ramp in y = shift by -4 px" % dev, out[8:-8], shifted(sm, 0, -4, "CLAMP")[8:-8], 2e-4)


@H.guard("gpu parity")
def test_gpu():
    img = H.test_image(*SIZE, seed=12, alpha=True)
    mp = H.test_image(*SIZE, seed=13)
    for kind, im, m in (("noisy", img, mp), ("smooth", X.smooth_image(*SIZE, seed=1, alpha=True),
                                              X.smooth_image(*SIZE, seed=2))):
        for mode in ("OFFSET", "GLASS"):
            for edge in MODES:
                for interp in ("LINEAR", "CUBIC"):
                    props = {"mode": mode, "edge_mode": edge, "interpolation": interp}
                    inputs = {"Strength": 30.0 if mode == "OFFSET" else 400.0, "Dispersion": 0.3}
                    c = render("CPU", im, m, props=props, inputs=inputs)
                    g = render("GPU", im, m, props=props, inputs=inputs)
                    # Same float32 formulas; positions differ by ~1 ulp (GPU contracts a*b+c). Observed
                    # max 2.2e-5 (noisy image, glass mode, where the Sobel gradient is amplified by
                    # Strength 400); 5e-7..8e-6 in offset mode.
                    H.compare("%s %s %s %s: GPU vs CPU" % (kind, mode, edge, interp), g, c, 5e-5)
    # Map of a different size than the image: clamp-resampled on both backends.
    small = H.test_image(16, 12, seed=4)
    c = render("CPU", img, small, inputs={"Strength": 12.0})
    g = render("GPU", img, small, inputs={"Strength": 12.0})
    H.compare("map of another size: GPU vs CPU", g, c, 5e-5)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17)):
            img = H.test_image(*size, seed=1)
            mp = H.test_image(*size, seed=2)
            for mode in ("OFFSET", "GLASS"):
                for edge in MODES:
                    out = render(dev, img, mp, size=size, props={"mode": mode, "edge_mode": edge},
                                 inputs={"Strength": 100.0, "Dispersion": 1.0})
                    H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                            "%s %dx%d %s %s: shape / finite" % (dev, size[0], size[1], mode, edge))
        img = H.test_image(24, 16, seed=1)
        mp = H.test_image(24, 16, seed=2)
        for name, inputs in (("huge strength", {"Strength": 1e12}), ("negative", {"Strength": -30.0}),
                             ("huge dispersion", {"Dispersion": 1e6, "Strength": 3.0})):
            for edge in MODES:
                out = render(dev, img, mp, size=(24, 16), props={"edge_mode": edge}, inputs=inputs)
                H.check(np.isfinite(out).all(), "%s %s %s: finite" % (dev, name, edge))
        # Everything unlinked: grey image, grey map -> the grey default.
        out = H.render_node(NODE, dev, (16, 8))
        H.check(np.allclose(out, (0.5, 0.5, 0.5, 1.0), atol=1e-6), "%s: unlinked inputs" % dev)
        # Output range: bilinear never leaves the input range.
        out = render(dev, img, mp, size=(24, 16), inputs={"Strength": 9.0})
        H.check(out.min() >= img.min() - 1e-6 and out.max() <= img.max() + 1e-6,
                "%s: bilinear output stays inside the input range" % dev)


for fn in (test_shifts, test_references, test_gpu, test_robustness):
    fn()
H.finish()
