# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Edge Stylise node (XDoG, Sobel, Outline): float64 references, invariants, CPU vs GPU,
robustness.   Blender -b --factory-startup --python-exit-code 1 --python test_edge_stylise.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H
import imgfx_helpers as X

H.setup()

NODE = "CompositorNodeLabEdgeStylise"
SIZE = (96, 64)
F32 = np.float32


def render(device, img, size=None, props=None):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, device, size, props, None, {"Image": img})


def ref_xdog(img, sigma=1.0, k=1.6, tau=0.98, phi=20.0, eps=0.0, ink=(0, 0, 0, 1), paper=(1, 1, 1, 1)):
    lum = X.luma(img[..., :3].astype(np.float64))[..., None]
    d = X.ref_gaussian(lum, sigma)[..., 0] - tau * X.ref_gaussian(lum, k * sigma)[..., 0]
    v = np.where(d >= eps, 1.0, 1.0 + np.tanh(np.clip(phi * (d - eps), -20, 20)))
    v = np.clip(v, 0, 1)[..., None]
    ink, paper = np.asarray(ink, np.float64), np.asarray(paper, np.float64)
    return ink + (paper - ink) * v


def ref_sobel_mag(img, color, gain, invert):
    gx, gy = X.ref_sobel(img[..., :3])
    if color:
        m = np.sqrt(gx * gx + gy * gy) * gain
    else:
        lx, ly = X.luma(gx), X.luma(gy)
        m = (np.sqrt(lx * lx + ly * ly) * gain)[..., None] * np.ones(3)
    if invert:
        m = 1.0 - m
    out = np.ones(img.shape, np.float64)
    out[..., :3] = m
    return out


def ref_morph(a, radius, is_max):
    """Brute-force anti-aliased disc dilation / erosion (float64, clamped edges): offset weight
    clamp(radius + 0.85 - |o|, 0, 1); the centre always counts fully."""
    h, w = a.shape
    r = int(np.ceil(radius + 0.85))
    out = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            wt = min(max(radius + 0.85 - np.hypot(dx, dy), 0.0), 1.0)
            if wt <= 0.0:
                continue
            ys = np.clip(np.arange(h) + dy, 0, h - 1)
            xs = np.clip(np.arange(w) + dx, 0, w - 1)
            sh = a[ys[:, None], xs[None, :]]
            out = np.maximum(out, sh * wt) if is_max else np.minimum(out, 1.0 - (1.0 - sh) * wt)
    return out


def ref_outline(img, width, pos, col, only):
    a = img[..., 3].astype(np.float64)
    radius = width / 2.0 if pos == "CENTER" else width
    s_out = np.clip(ref_morph(a, radius, True) - a, 0, 1) if pos != "INSIDE" else np.zeros_like(a)
    s_in = np.clip(a - ref_morph(a, radius, False), 0, 1) if pos != "OUTSIDE" else np.zeros_like(a)
    col = np.asarray(col, np.float64)
    sin_l, sout_l = col * s_in[..., None], col * s_out[..., None]
    if only:
        return sin_l + sout_l
    res = img + sout_l * (1 - img[..., 3:4])
    return sin_l + res * (1 - sin_l[..., 3:4])


@H.guard("xdog")
def test_xdog():
    img = H.test_image(*SIZE, seed=3)
    smooth = X.smooth_image(*SIZE, seed=1)
    for name, props, kw in (
            ("default", {}, {}),
            ("sigma 2.5 k 2 tau 1 phi 5 eps 0.01", dict(sigma=2.5, k=2.0, tau=1.0, phi=5.0, epsilon=0.01),
             dict(sigma=2.5, k=2.0, tau=1.0, phi=5.0, eps=0.01)),
            ("colours", dict(ink=(0.2, 0.0, 0.1, 1.0), paper=(0.9, 0.8, 0.5, 1.0)),
             dict(ink=(0.2, 0.0, 0.1, 1), paper=(0.9, 0.8, 0.5, 1))),
            ("hard threshold", dict(phi=1000.0, epsilon=-0.005), dict(phi=1000.0, eps=-0.005)),
            ("sigma 0.3", dict(sigma=0.3), dict(sigma=0.3))):
        for kind, im in (("noisy", img), ("smooth", smooth)):
            ref = ref_xdog(im, **kw)
            for dev in ("CPU", "GPU"):
                out = render(dev, im, props=props)
                # float32 blur sums and tanh vs float64; GPU adds exp / tanh approximation error.
                H.compare("%s xdog %s [%s]: float64 reference" % (dev, name, kind), out, ref, 2e-4)
                H.check(out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6 or name == "colours",
                        "%s xdog %s [%s]: output within [0, 1]" % (dev, name, kind))
    for dev in ("CPU", "GPU"):
        # Range [0, 1] with HDR input and extreme settings (default ink / paper).
        hdr = H.test_image(*SIZE, seed=2, hdr=True) * 20.0
        for props in ({}, dict(phi=1000.0, tau=2.0, epsilon=-1.0), dict(phi=0.0), dict(sigma=24.0, k=10.0)):
            out = render(dev, hdr, props=props)
            H.check(np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0 and np.all(out[..., 3] == 1.0),
                    "%s xdog %s: HDR input stays in [0, 1], alpha 1" % (dev, props))
        # Flat image: no edges, plain paper (also dark / bright / coloured).
        for col in ((0.5, 0.5, 0.5, 1.0), (0.0, 0.0, 0.0, 1.0), (0.9, 0.2, 0.1, 1.0), (4.0, 4.0, 4.0, 1.0)):
            flat = np.empty((SIZE[1], SIZE[0], 4), F32)
            flat[...] = col
            out = render(dev, flat)
            H.compare("%s xdog: flat %s gives paper" % (dev, col[:3]), out, np.ones_like(out), 1e-5)
        # A step edge draws a dark line on its dark side; far from the edge stays paper.
        step = np.ones((SIZE[1], SIZE[0], 4), F32)
        step[..., :3] = np.where(np.arange(SIZE[0]) >= SIZE[0] // 2, 0.9, 0.1)[None, :, None]
        out = render(dev, step)
        row = out[SIZE[1] // 2, :, 0]
        edge = SIZE[0] // 2
        H.check(row[edge - 3:edge].min() < 0.5 and row[:edge - 12].min() > 0.99 and row[edge + 8:].min() > 0.99,
                "%s xdog: ink line at the step edge only (min %.2f)" % (dev, row[edge - 3:edge].min()))
        # Hard threshold gives (almost) only black or white.
        def binary_frac(phi):
            o = render(dev, img, props=dict(phi=phi))[..., 0]
            return float(((o < 0.01) | (o > 0.99)).mean())
        hard, soft = binary_frac(1000.0), binary_frac(20.0)
        H.check(hard > 0.9 and hard > soft + 0.1,
                "%s xdog: phi 1000 is a hard threshold (%.3f binary vs %.3f at phi 20)" % (dev, hard, soft))
        # Ink amount: larger epsilon -> more ink.
        a = render(dev, img, props=dict(epsilon=0.0))[..., 0].mean()
        b = render(dev, img, props=dict(epsilon=0.05))[..., 0].mean()
        H.check(b < a, "%s xdog: larger epsilon inks more (%.3f -> %.3f)" % (dev, a, b))


@H.guard("sobel")
def test_sobel():
    img = H.test_image(*SIZE, seed=4)
    for name, props in (("mono", {}), ("colour", dict(colored=True)), ("invert gain 3", dict(gain=3.0, invert=True)),
                        ("colour invert", dict(colored=True, invert=True, gain=0.5))):
        ref = ref_sobel_mag(img, props.get("colored", False), props.get("gain", 2.0), props.get("invert", False))
        for dev in ("CPU", "GPU"):
            out = render(dev, img, props=dict(mode="SOBEL", **props))
            H.compare("%s sobel %s: float64 reference" % (dev, name), out, ref, 1e-5)
    for dev in ("CPU", "GPU"):
        for colored in (False, True):
            flat = np.empty((SIZE[1], SIZE[0], 4), F32)
            flat[...] = (0.3, 0.7, 0.2, 1.0)
            out = render(dev, flat, props=dict(mode="SOBEL", colored=colored))
            H.check(np.abs(out[..., :3]).max() < 1e-6 and np.all(out[..., 3] == 1.0),
                    "%s sobel colour=%s: flat image has no edges" % (dev, colored))
        # Step of height 1 with Gain 1: Sobel / 8 gives 0.5 on the two pixels next to the edge.
        step = np.ones((SIZE[1], SIZE[0], 4), F32)
        step[..., :3] = (np.arange(SIZE[0]) >= SIZE[0] // 2).astype(F32)[None, :, None]
        out = render(dev, step, props=dict(mode="SOBEL", gain=1.0))
        edge = SIZE[0] // 2
        H.check(np.allclose(out[:, edge - 1:edge + 1, 0], 0.5, atol=1e-6) and np.abs(out[:, :edge - 1, 0]).max() < 1e-6
                and np.abs(out[:, edge + 1:, 0]).max() < 1e-6,
                "%s sobel: step edge magnitude 0.5 on two columns" % dev)


@H.guard("outline")
def test_outline():
    n = 96
    disc, mask = X.disc_image(n, n, 24.0, color=(1.0, 0.5, 0.25))
    area0 = float(mask.sum())
    r0 = np.sqrt(area0 / np.pi)
    for dev in ("CPU", "GPU"):
        for width in (1.0, 2.0, 4.0, 7.0, 12.0):
            for pos in ("OUTSIDE", "INSIDE", "CENTER"):
                out = render(dev, disc, props=dict(mode="OUTLINE", width=width, position=pos, stroke_only=True))
                area = float(out[..., 3].sum())
                if pos == "OUTSIDE":
                    eff = np.sqrt(area / np.pi + r0 * r0) - r0
                elif pos == "INSIDE":
                    eff = r0 - np.sqrt(max(r0 * r0 - area / np.pi, 0.0))
                else:
                    # Centred: ring between r0 - w/2 and r0 + w/2.
                    # area = pi ((r0 + e/2)^2 - (r0 - e/2)^2) = 2 pi r0 e  ->  e = area / (2 pi r0)
                    eff = area / (2 * np.pi * r0)
                print("  %s %s width %.0f: measured %.2f px" % (dev, pos, width, eff))
                H.check(abs(eff - width) <= 0.5, "%s outline %s width %.0f measured %.2f (within 0.5 px)" % (
                    dev, pos, width, eff))
    colr = (1.0, 0.0, 0.0, 1.0)
    for dev in ("CPU", "GPU"):
        out = render(dev, disc, props=dict(mode="OUTLINE", width=4.0, position="OUTSIDE", stroke=colr))
        ring = (out[..., 1] == 0.0) & (out[..., 0] == 1.0) & (out[..., 2] == 0.0) & (out[..., 3] == 1.0)
        H.check(bool(np.all(out[mask] == disc[mask])), "%s outside outline: the shape is untouched (drawn under)" % dev)
        far = X.disc_image(n, n, 24.0 + 8.0)[1]
        H.check(bool(np.all(out[~far] == 0.0)), "%s outside outline: nothing beyond the width" % dev)
        H.check(ring.sum() > 300, "%s outside outline draws the stroke colour (%d px)" % (dev, ring.sum()))
        out = render(dev, disc, props=dict(mode="OUTLINE", width=4.0, position="INSIDE", stroke=colr))
        H.check(bool(np.all(out[~mask] == 0.0)), "%s inside outline: nothing outside the shape" % dev)
        core = X.disc_image(n, n, 24.0 - 6.0)[1]
        H.check(bool(np.all(out[core] == disc[core])), "%s inside outline: core untouched" % dev)
        H.check(bool(np.any((out[mask][:, 1] == 0.0))), "%s inside outline: stroke drawn over the shape" % dev)
        # Half-transparent stroke colour composites (premultiplied colour * coverage).
        half = (0.5, 0.0, 0.0, 0.5)
        out = render(dev, disc, props=dict(mode="OUTLINE", width=4.0, position="OUTSIDE", stroke=half, stroke_only=True))
        H.check(abs(float(out[..., 3].max()) - 0.5) < 1e-6 and abs(float(out[..., 0].max()) - 0.5) < 1e-6,
                "%s stroke colour alpha respected (max alpha %.3f)" % (dev, out[..., 3].max()))
    # Float64 brute-force reference of the whole composite (soft alpha, odd width, all positions).
    soft = H.test_image(40, 30, seed=5, alpha=True)
    soft[..., 3] = np.clip(soft[..., 3] - 0.3, 0, 1) * (np.arange(40)[None, :] > 10)
    soft[..., :3] *= soft[..., 3:4]
    for pos in ("OUTSIDE", "INSIDE", "CENTER"):
        for only in (False, True):
            col = (0.8, 0.2, 0.1, 0.9)
            ref = ref_outline(soft, 3.3, pos, col, only)
            for dev in ("CPU", "GPU"):
                out = render(dev, soft, props=dict(mode="OUTLINE", width=3.3, position=pos, stroke=col, stroke_only=only))
                H.compare("%s outline %s only=%s: brute-force reference" % (dev, pos, only), out, ref, 1e-5)
    # Flat opaque image: no alpha edge, no stroke (the image border is not an edge).
    for dev in ("CPU", "GPU"):
        flat = np.empty((SIZE[1], SIZE[0], 4), F32)
        flat[...] = (0.3, 0.7, 0.2, 1.0)
        for pos in ("OUTSIDE", "INSIDE", "CENTER"):
            out = render(dev, flat, props=dict(mode="OUTLINE", width=6.0, position=pos))
            H.compare("%s outline %s: flat image unchanged" % (dev, pos), out, flat, 0.0)
        # Width 0 does nothing.
        out = render(dev, disc, props=dict(mode="OUTLINE", width=0.0))
        H.compare("%s outline: width 0 is identity" % dev, out, disc, 0.0)


@H.guard("gpu parity")
def test_gpu():
    img = H.test_image(*SIZE, seed=6, alpha=True)
    sm = X.smooth_image(*SIZE, seed=7, alpha=True)
    for kind, im in (("noisy", img), ("smooth", sm)):
        for name, props, atol in (
                ("xdog", dict(mode="XDOG"), 2e-4),
                ("xdog sigma 3 k 3", dict(mode="XDOG", sigma=3.0, k=3.0, phi=8.0, epsilon=0.02), 2e-4),
                ("sobel", dict(mode="SOBEL"), 1e-5),
                ("sobel colour", dict(mode="SOBEL", colored=True, invert=True), 1e-5),
                ("outline outside", dict(mode="OUTLINE", width=5.0), 1e-5),
                ("outline centre only", dict(mode="OUTLINE", width=6.5, position="CENTER", stroke_only=True), 1e-5),
                ("outline inside", dict(mode="OUTLINE", width=3.0, position="INSIDE", stroke=(0.1, 0.9, 0.3, 1.0)), 1e-5)):
            c = render("CPU", im, props=props)
            g = render("GPU", im, props=props)
            # XDoG: tanh / exp approximations of the GPU; the rest is exact arithmetic up to rounding.
            H.compare("%s %s: GPU vs CPU" % (kind, name), g, c, atol)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17)):
            img = H.test_image(*size, seed=1, alpha=True)
            for mode in ("XDOG", "SOBEL", "OUTLINE"):
                out = render(dev, img, size=size, props=dict(mode=mode, width=9.0, sigma=6.0))
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %dx%d %s: shape / finite" % (dev, size[0], size[1], mode))
        img = H.test_image(24, 16, seed=1, alpha=True)
        for props in (dict(mode="XDOG", sigma=0.0), dict(mode="XDOG", k=0.0), dict(mode="XDOG", sigma=1e9, k=1e9),
                      dict(mode="OUTLINE", width=1e9), dict(mode="SOBEL", gain=100.0),
                      dict(mode="XDOG", phi=1e9, epsilon=1.0, tau=2.0)):
            out = render(dev, img, props=props)
            H.check(np.isfinite(out).all(), "%s %s: finite" % (dev, props))
        for mode in ("XDOG", "SOBEL", "OUTLINE"):
            out = H.render_node(NODE, dev, (16, 8), props=dict(mode=mode))
            H.check(out.shape == (8, 16, 4) and np.isfinite(out).all(), "%s %s: unlinked image" % (dev, mode))


@H.guard("performance")
def test_performance():
    import time
    big = H.test_image(1920, 1080, seed=3, alpha=True)
    for props in (dict(mode="XDOG", sigma=2.0), dict(mode="SOBEL"), dict(mode="OUTLINE", width=8.0)):
        t = time.time()
        c = render("CPU", big, props=props)
        dt = time.time() - t
        H.note("CPU 1920x1080 %s: %.1f s" % (props, dt))
        H.check(np.isfinite(c).all() and dt < 30.0, "CPU 1080p %s finishes (%.1f s)" % (props["mode"], dt))
        g = render("GPU", big, props=props)
        H.compare("1080p %s GPU vs CPU" % props["mode"], g, c, 2e-4)


for fn in (test_xdog, test_sobel, test_outline, test_gpu, test_performance, test_robustness):
    fn()
H.finish()
