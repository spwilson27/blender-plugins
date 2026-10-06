# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Kuwahara node: CPU vs an independent float64 per-pixel reference, CPU vs GPU, invariants (flat
image, sharp edges, value range), robustness.
Blender -b --factory-startup --python-exit-code 1 --python test_kuwahara.py"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H
import imgfx_helpers as X

H.setup()

from compositor_lab.lib import np_sampling  # noqa: E402

NODE = "CompositorNodeLabKuwahara"
F32 = np.float32


def ref_kuwahara(img, radius, q, k):
    """Per-pixel float64 reference of the algorithm in nodes/kuwahara.py (angles via atan2)."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    gx, gy = X.ref_sobel(img[..., :3])
    tensor = np.stack([(gx * gx).sum(2), (gx * gy).sum(2), (gy * gy).sum(2)], axis=2)
    tensor = X.ref_gaussian(tensor, 2.0)
    ext = int(math.ceil(radius * (1 + k)))
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            e, f, g = tensor[y, x]
            tr, dd = e + g, e - g
            disc = math.sqrt(max(dd * dd + 4 * f * f, 0.0))
            aniso, t = 0.0, (1.0, 0.0)
            if tr > 1e-20 and disc > 1e-5 * tr:
                aniso = disc / tr
                theta = 0.5 * math.atan2(2 * f, dd)          # gradient direction
                t = (-math.sin(theta), math.cos(theta))     # tangent: along the edge
            n = (-t[1], t[0])
            sc = 1 + k * aniso
            ea, eb = radius * sc, radius / sc
            s0 = np.zeros(8)
            s1 = np.zeros((8, 4))
            s2 = np.zeros(8)
            for dy in range(-ext, ext + 1):
                for dx in range(-ext, ext + 1):
                    vx = (dx * t[0] + dy * t[1]) / ea
                    vy = (dx * n[0] + dy * n[1]) / eb
                    l2 = vx * vx + vy * vy
                    if l2 > 1.0:
                        continue
                    c = img[min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)]
                    for i in range(8):
                        if dx == 0 and dy == 0:
                            wt = 1.0
                        else:
                            ca = math.cos(math.atan2(vy, vx) - i * math.pi / 4)
                            wt = max(ca, 0.0) ** 4 * math.exp(-3.125 * l2) * (1.0 - l2)
                        s0[i] += wt
                        s1[i] += wt * c
                        s2[i] += wt * float(c[:3] @ c[:3])
            acc = np.zeros(4)
            asum = 0.0
            for i in range(8):
                m = s1[i] / s0[i]
                var = max(s2[i] / s0[i] - float(m[:3] @ m[:3]), 0.0)
                base = max(255.0 * math.sqrt(var), 1e-20)
                al = 1.0 / (1.0 + min(base ** q, 1e30))
                acc += m * al
                asum += al
            out[y, x] = acc / asum
    return out


def render(device, img, size=None, inputs=None):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, device, size, inputs=inputs, images={"Image": img})


def edge_width(row, lo=0.1, hi=0.9):
    """Number of samples strictly between lo and hi of the step (normalised to 0..1)."""
    v = (row - row.min()) / (row.max() - row.min())
    return int(((v > lo) & (v < hi)).sum())


@H.guard("cpu reference")
def test_cpu_reference():
    size = (20, 14)
    img = H.test_image(*size, seed=4)
    for name, kw in (("default radius 3", dict(Radius=3.0, Sharpness=8.0, Anisotropy=1.0)),
                     ("q 3 aniso 0.5", dict(Radius=2.5, Sharpness=3.0, Anisotropy=0.5)),
                     ("aniso 0 (circle)", dict(Radius=3.0, Sharpness=8.0, Anisotropy=0.0)),
                     ("q 0", dict(Radius=2.0, Sharpness=0.0, Anisotropy=1.0))):
        out = render("CPU", img, inputs=kw)
        ref = ref_kuwahara(img, kw["Radius"], kw["Sharpness"], kw["Anisotropy"])
        # float32 vs float64: the (255 s)^q weights amplify rounding of the sector variances.
        H.compare("%s: CPU == float64 reference" % name, out, ref, 2e-4)
    # Structured image (diagonal edge + noise) so anisotropy and orientation really matter.
    y, x = np.mgrid[0:14, 0:20]
    st = np.ones((14, 20, 4), F32)
    st[..., :3] = (0.15 + 0.7 * ((x + 1.7 * y) > 18))[..., None]
    st[..., :3] += 0.03 * np.random.default_rng(3).random((14, 20, 3))
    out = render("CPU", st, inputs=dict(Radius=3.0))
    H.compare("diagonal edge: CPU == float64 reference", out, ref_kuwahara(st, 3.0, 8.0, 1.0), 2e-4)


@H.guard("gpu parity")
def test_gpu():
    cases = [
        ("default", {}, 1),
        ("radius 8 q 2", dict(Radius=8.0, Sharpness=2.0), 2),
        ("radius 2.5 aniso 2", dict(Radius=2.5, Anisotropy=2.0), 3),
        ("circle", dict(Anisotropy=0.0), 4),
        ("q 16 radius 6", dict(Sharpness=16.0, Radius=6.0), 5),
    ]
    for name, kw, seed in cases:
        for label, img in (("noisy", H.test_image(80, 56, seed=seed)),
                           ("smooth", X.smooth_image(80, 56, seed=seed))):
            cpu = render("CPU", img, inputs=kw)
            gpu = render("GPU", img, inputs=kw)
            # The tensor orientation and the pow(255 s, q) sector weights amplify last-bit
            # differences (GPU fast-math sqrt/exp/pow): observed max 1e-4 (smooth, q 16), 1e-5
            # typical, hence 5e-4 instead of the 1e-5 of pointwise nodes.
            H.compare("%s [%s]: GPU vs CPU" % (name, label), gpu, cpu, 5e-4)


@H.guard("behaviour")
def test_behaviour():
    size = (64, 48)
    flat = np.empty((size[1], size[0], 4), F32)
    flat[...] = (0.3, 0.6, 0.9, 0.8)
    rng = np.random.default_rng(11)
    noisy = H.test_image(*size, seed=9, hdr=True)
    step = np.ones((size[1], size[0], 4), F32)
    step[..., :3] = np.where(np.arange(size[0]) >= size[0] // 2, 0.9, 0.1)[None, :, None]
    diag = np.ones((size[1], size[0], 4), F32)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    diag[..., :3] = np.where((xx + yy) > (size[0] + size[1]) // 2, 0.9, 0.1)[..., None]
    for dev in ("CPU", "GPU"):
        for name, kw in (("default", {}), ("radius 10", dict(Radius=10.0)), ("q 0", dict(Sharpness=0.0))):
            out = render(dev, flat, inputs=kw)
            H.compare("%s: flat image unchanged (%s)" % (dev, name), out, flat, 1e-5)
        # Output range: a convex combination of inputs, per channel (alpha included).
        out = render(dev, noisy)
        lo = noisy.reshape(-1, 4).min(0)
        hi = noisy.reshape(-1, 4).max(0)
        eps = 1e-5
        H.check(bool(np.all(out.reshape(-1, 4).min(0) >= lo - eps) and
                     np.all(out.reshape(-1, 4).max(0) <= hi + eps)),
                "%s: output within the input range per channel" % dev)
        # Smoothing: flat noisy regions lose variance.
        n = np.ones((size[1], size[0], 4), F32)
        n[..., :3] = 0.5 + 0.1 * (rng.random((size[1], size[0], 1)) - 0.5) * np.ones(3)
        out = render(dev, n, inputs=dict(Radius=5.0))
        core = (slice(8, -8), slice(8, -8), 0)
        H.check(out[core].std() < 0.5 * n[core].std(),
                "%s: noise std %.4f -> %.4f" % (dev, n[core].std(), out[core].std()))
        # Edge preservation: vs a Gaussian blur with the same kernel radius (sigma = radius / 3).
        r = 6.0
        out = render(dev, step, inputs=dict(Radius=r))
        row = out[size[1] // 2, :, 0]
        gblur = np_sampling.blur_gaussian(step[..., :3], r / 3.0)[size[1] // 2, :, 0]
        wk, wg = edge_width(row), edge_width(gblur)
        print("  %s step edge width: Kuwahara %d px, Gaussian(sigma %.1f) %d px" % (dev, wk, r / 3, wg))
        H.check(wk <= 2 and wg >= 2 * max(wk, 1), "%s: step edge stays sharp vs Gaussian blur" % dev)
        H.check(row[:size[0] // 2 - 4].std() < 1e-4 and row[size[0] // 2 + 4:].std() < 1e-4,
                "%s: step plateaus stay flat" % dev)
        # Diagonal edge: still two-level (anisotropic kernel follows the edge): few intermediate
        # values compared with the pixel count of the band a blur would create.
        out = render(dev, diag, inputs=dict(Radius=6.0))
        mid = ((out[..., 0] > 0.2) & (out[..., 0] < 0.8)).mean()
        gd = np_sampling.blur_gaussian(diag[..., :3], 2.0)[..., 0]
        gmid = ((gd > 0.2) & (gd < 0.8)).mean()
        print("  %s diagonal edge: intermediate pixels %.3f vs Gaussian %.3f" % (dev, mid, gmid))
        H.check(mid < 0.5 * gmid, "%s: diagonal edge stays sharp" % dev)
        # Radius 0 is a pass-through.
        H.compare("%s: radius 0 is identity" % dev, render(dev, noisy, inputs=dict(Radius=0.0)),
                  noisy, 1e-6)
        # Higher sharpness keeps more of the lowest-variance sector (crisper corners).
        a = render(dev, H.test_image(*size, seed=2), inputs=dict(Sharpness=0.0))
        b = render(dev, H.test_image(*size, seed=2), inputs=dict(Sharpness=16.0))
        H.check(float(np.abs(a - b).mean()) > 0.003, "%s: Sharpness changes the result" % dev)


@H.guard("performance")
def test_performance():
    big = H.test_image(1920, 1080, seed=7)
    big[..., :3] = 0.5 * big[..., :3] + 0.5 * X.smooth_image(1920, 1080, seed=7)[..., :3]
    t = time.time()
    out = render("CPU", big)
    dt = time.time() - t
    H.note("CPU 1920x1080 radius 4 (noisy image, whole render): %.1f s" % dt)
    H.check(np.isfinite(out).all() and dt < 120.0, "CPU 1080p finishes (%.1f s)" % dt)
    t = time.time()
    g = render("GPU", big)
    H.note("GPU 1920x1080 radius 4: %.2f s" % (time.time() - t))
    H.compare("1080p GPU vs CPU", g, out, 5e-4)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17)):
            img = H.test_image(*size, seed=1)
            out = render(dev, img)
            H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                    "%s %dx%d: shape / finite" % (dev, size[0], size[1]))
        img = H.test_image(24, 16, seed=1)
        for name, kw in (("radius 99", dict(Radius=99.0)), ("negative radius", dict(Radius=-3.0)),
                         ("q -5", dict(Sharpness=-5.0)), ("q 99", dict(Sharpness=99.0)),
                         ("aniso 9", dict(Anisotropy=9.0)), ("aniso -1", dict(Anisotropy=-1.0))):
            out = render(dev, img, inputs=kw)
            H.check(np.isfinite(out).all(), "%s %s: finite" % (dev, name))
        # Unlinked Image: constant grey (the socket default) on the render size.
        out = H.render_node(NODE, dev, (16, 8))
        H.check(out.shape == (8, 16, 4) and np.allclose(out, (0.5, 0.5, 0.5, 1.0), atol=1e-5),
                "%s: unlinked image gives the default grey" % dev)
        # HDR + black + alpha-0 images stay finite.
        hdr = H.test_image(24, 16, seed=3, hdr=True) * 50.0
        H.check(np.isfinite(render(dev, hdr)).all(), "%s: HDR input finite" % dev)
        H.check(np.isfinite(render(dev, np.zeros((16, 24, 4), F32))).all(), "%s: black finite" % dev)


for fn in (test_cpu_reference, test_gpu, test_behaviour, test_performance, test_robustness):
    fn()
H.finish()
