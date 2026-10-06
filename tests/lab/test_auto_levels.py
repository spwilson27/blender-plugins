# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Auto Levels node: numpy reference (own histogram percentile), CPU vs GPU, percentile / monotone
/ gamma invariants, alpha handling, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_auto_levels.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

NODE = "CompositorNodeLabAutoLevels"
F32 = np.float32
LUMA = np.array([0.2126, 0.7152, 0.0722])
NB = 256

# CPU node vs the test's own float64 reference (observed max error 2.3e-7: float32 arithmetic).
# The reference rebuilds the histogram in float64 while the node bins in float32; with the same
# float32 luminance the bin assignments agree on this data. (With a float64 luminance in the
# reference the sparse histogram tails moved the percentiles by up to 1.2e-3 in the output.)
REF_ATOL = 1e-5
# GPU vs CPU: identical host-side parameters from identical histograms, then the same float32
# per-pixel formula (GPU pow() is approximate). Observed max difference 3.6e-7 (hdr images up to
# 4.0, gamma on); 2e-6 leaves a margin.
GPU_ATOL = 2e-6
# Output percentiles land on 0 / 1 within one histogram bin of the stretched range.
PCT_ATOL = 0.02


def low_contrast(w, h, seed=1, alpha=False):
    img = H.test_image(w, h, seed=seed, alpha=alpha)
    a = img[..., 3:4].copy()
    straight = img[..., :3] / np.maximum(a, 1e-6)
    straight = 0.2 + 0.3 * straight
    img[..., :3] = straight * a
    return img


def render(device, img, props=None, inputs=None, size=None):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, device, size, props=props, inputs=inputs, images={"Image": img})


def hist_percentile(v, lo, hi, p):
    """Independent float64 histogram percentile (256 bins, interpolated inside the bin)."""
    if p <= 0:
        return lo
    if p >= 100:
        return hi
    hist, _ = np.histogram(v, bins=NB, range=(lo, hi))
    cum = np.cumsum(hist)
    target = p / 100 * len(v)
    b = min(int(np.searchsorted(cum, target, side="left")), NB - 1)
    prev = cum[b - 1] if b else 0
    frac = (target - prev) / hist[b] if hist[b] else 0.0
    return lo + (b + frac) * (hi - lo) / NB


def reference(img, low=1.0, high=99.0, mode="LUMINANCE", clamp=True, gamma=False, target=0.5, fac=1.0):
    a = img[..., 3].astype(np.float64)
    vis = a > 0
    c = np.zeros(img.shape[:2] + (3,))
    c[vis] = img[..., :3][vis].astype(np.float64) / a[vis][:, None]
    sc = c[vis]
    lo = np.zeros(3)
    scale = np.ones(3)
    gam = np.ones(3)
    if mode == "LUMINANCE":
        # Same float32 luminance as the node (a different rounding could move pixels across a
        # bin edge, and the sparse histogram tails make the percentile sensitive to that).
        s32 = (img[..., :3][vis] / img[..., 3][vis][:, None]).astype(F32)
        lum32 = F32(0.2126) * s32[:, 0] + F32(0.7152) * s32[:, 1] + F32(0.0722) * s32[:, 2]
        chans = [(lum32.astype(np.float64), slice(0, 3))]
    else:
        chans = [(sc[:, i], slice(i, i + 1)) for i in range(3)]
    for v, sel in chans:
        v32 = v.astype(F32).astype(np.float64)
        vmin, vmax = v32.min(), v32.max()
        plo, phi = hist_percentile(v, vmin, vmax, low), hist_percentile(v, vmin, vmax, high)
        if phi > plo:
            lo[sel] = plo
            scale[sel] = 1 / (phi - plo)
            if gamma:
                hist, edges = np.histogram(v, bins=NB, range=(vmin, vmax))
                centres = 0.5 * (edges[:-1] + edges[1:])
                y0 = np.clip((centres - plo) / (phi - plo), 0, 1)
                lo_g, hi_g = 0.05, 20.0

                def mean_at(g, hist=hist, y0=y0):
                    return (hist * y0 ** g).sum() / hist.sum()

                if mean_at(hi_g) < target < mean_at(lo_g):
                    for _ in range(60):
                        mid = 0.5 * (lo_g + hi_g)
                        lo_g, hi_g = (mid, hi_g) if mean_at(mid) > target else (lo_g, mid)
                    gam[sel] = 0.5 * (lo_g + hi_g)
    y = (c - lo) * scale
    if clamp:
        y = np.clip(y, 0, 1)
    if gamma:
        y = np.maximum(y, 0) ** gam
    out = img.astype(np.float64).copy()
    out[..., :3] = img[..., :3] + (y * a[..., None] - img[..., :3]) * fac
    return out


@H.guard("cpu reference")
def test_reference():
    cases = [
        ("default luminance", dict(), dict()),
        ("per channel", dict(props={"mode": "PER_CHANNEL"}), dict(mode="PER_CHANNEL")),
        ("percentiles 5/95", dict(props={"low": 5.0, "high": 95.0}), dict(low=5.0, high=95.0)),
        ("no clamp", dict(props={"clamp": False}), dict(clamp=False)),
        ("auto gamma 0.3", dict(props={"auto_gamma": True, "target": 0.3}), dict(gamma=True, target=0.3)),
        ("per channel + gamma", dict(props={"mode": "PER_CHANNEL", "auto_gamma": True}),
         dict(mode="PER_CHANNEL", gamma=True)),
        ("fac 0.4", dict(inputs={"Fac": 0.4}), dict(fac=0.4)),
        ("0..100", dict(props={"low": 0.0, "high": 100.0}), dict(low=0.0, high=100.0)),
    ]
    for alpha in (False, True):
        img = low_contrast(97, 61, seed=3, alpha=alpha)
        for name, kw, rkw in cases:
            out = render("CPU", img, **kw)
            ref = reference(img, **rkw)
            H.compare("%s%s: CPU node vs numpy reference" % (name, " (alpha)" if alpha else ""), out, ref,
                      REF_ATOL)
            H.check(np.array_equal(out[..., 3], img[..., 3]), "%s: alpha unchanged" % name)


@H.guard("invariants")
def test_invariants():
    img = low_contrast(128, 96, seed=5)
    for dev in ("CPU", "GPU"):
        # Per channel: output percentiles at Low / High land on 0 / 1.
        out = render(dev, img, props={"mode": "PER_CHANNEL", "low": 2.0, "high": 98.0})
        for c in range(3):
            lo = np.percentile(out[..., c], 2.0)
            hi = np.percentile(out[..., c], 98.0)
            H.check(abs(lo) < PCT_ATOL and abs(hi - 1) < PCT_ATOL,
                    "%s per channel %d: P2 -> %.4f (want 0), P98 -> %.4f (want 1)" % (dev, c, lo, hi))
            H.check(out[..., c].min() >= 0.0 and out[..., c].max() <= 1.0, "%s clamp keeps 0..1" % dev)
        # Luminance mode on a grey image behaves the same way.
        g = np.repeat(img[..., :1], 3, axis=2)
        g = np.concatenate([g, np.ones_like(img[..., :1])], axis=2)
        out = render(dev, g)
        H.check(abs(np.percentile(out[..., 0], 1.0)) < PCT_ATOL and abs(np.percentile(out[..., 0], 99.0) - 1) < PCT_ATOL,
                "%s luminance mode: P1 -> 0, P99 -> 1 on grey" % dev)
        # Monotone per channel (both modes, with gamma).
        for props in ({}, {"mode": "PER_CHANNEL"}, {"auto_gamma": True, "target": 0.35},
                      {"clamp": False}):
            out = render(dev, img, props=props)
            for c in range(3):
                order = np.argsort(img[..., c].ravel(), kind="stable")
                d = np.diff(out[..., c].ravel()[order])
                H.check(d.min() >= -2e-6, "%s %s: channel %d monotone (min step %.2g)" % (dev, props, c, d.min()))
        # Luminance mode without clamp is one shared affine map: out = (in - lo) * k for all channels.
        out = render(dev, img, props={"clamp": False})
        ks = []
        for c in range(3):
            k, b = np.polyfit(img[..., c].ravel().astype(np.float64), out[..., c].ravel().astype(np.float64), 1)
            res = np.abs(np.polyval([k, b], img[..., c].ravel()) - out[..., c].ravel()).max()
            ks.append((k, b))
            H.check(res < 1e-5, "%s luminance mode channel %d is affine (residual %.2g)" % (dev, c, res))
        H.check(max(abs(ks[0][0] - ks[i][0]) for i in range(3)) < 1e-4 and
                max(abs(ks[0][1] - ks[i][1]) for i in range(3)) < 1e-4,
                "%s luminance mode: same gain and offset for R, G, B (%s)" % (dev, ks))
        # Auto gamma reaches the target mean luminance (histogram resolution) on a grey image.
        for target in (0.3, 0.5, 0.7):
            out = render(dev, g, props={"auto_gamma": True, "target": target})
            m = float(out[..., 0].mean())
            H.check(abs(m - target) < 0.03, "%s auto gamma target %.1f -> mean %.4f" % (dev, target, m))
        # Fac.
        for fac in (0.0, 0.5, 1.0):
            out = render(dev, img, inputs={"Fac": fac})
            full = render(dev, img)
            H.compare("%s Fac %g = mix(in, full)" % (dev, fac), out, img + (full - img) * fac, GPU_ATOL, quiet=True)
        # Alpha: result premultiplied by the same alpha, alpha unchanged.
        ia = low_contrast(64, 48, seed=8, alpha=True)
        out = render(dev, ia)
        H.check(np.array_equal(out[..., 3], ia[..., 3]), "%s alpha unchanged" % dev)
        H.check((out[..., :3] <= ia[..., 3:4] + 1e-6).all(), "%s premultiplied result stays <= alpha" % dev)


@H.guard("cpu vs gpu")
def test_gpu():
    cases = [dict(), dict(props={"mode": "PER_CHANNEL"}), dict(props={"auto_gamma": True}),
             dict(props={"mode": "PER_CHANNEL", "auto_gamma": True, "target": 0.4, "clamp": False}),
             dict(props={"low": 10.0, "high": 90.0}), dict(inputs={"Fac": 0.3})]
    for size in ((96, 64), (33, 17), (257, 129), (4, 4)):
        for alpha in (False, True):
            img = H.test_image(size[0], size[1], seed=size[0], alpha=alpha, hdr=True)
            for kw in cases:
                cpu = render("CPU", img, **kw)
                gpu = render("GPU", img, **kw)
                H.compare("%dx%d%s %s: GPU vs CPU" % (size[0], size[1], " alpha" if alpha else "", kw),
                          gpu, cpu, GPU_ATOL, rtol=GPU_ATOL)
    # Per-pixel Fac.
    img = low_contrast(48, 32, seed=2)
    fac = np.random.default_rng(3).random((32, 48)).astype(F32)
    cpu = H.render_node(NODE, "CPU", (48, 32), images={"Image": img, "Fac": fac})
    gpu = H.render_node(NODE, "GPU", (48, 32), images={"Image": img, "Fac": fac})
    H.compare("per-pixel Fac: GPU vs CPU", gpu, cpu, GPU_ATOL)
    H.compare("per-pixel Fac: CPU vs reference", cpu,
              img + (render("CPU", img) - img) * fac[..., None], 1e-6)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        # Unlinked input (default grey, degenerate range): identity.
        out = H.render_generator(NODE, dev, (16, 12))
        H.check(np.allclose(out, [0.5, 0.5, 0.5, 1.0], atol=1e-6), "%s: unlinked input: identity (%s)" % (dev, out[0, 0]))
        # Constant, black, transparent images are left untouched.
        for name, val in (("constant", [0.3, 0.4, 0.5, 1.0]), ("black", [0, 0, 0, 1.0]), ("transparent", [0, 0, 0, 0])):
            img = np.zeros((9, 14, 4), F32) + np.array(val, F32)
            out = render(dev, img)
            H.check(np.isfinite(out).all() and np.allclose(out, img, atol=1e-6), "%s %s image unchanged" % (dev, name))
        # Low >= High leaves the image alone (degenerate range), no NaNs.
        img = low_contrast(32, 24, seed=4)
        out = render(dev, img, props={"low": 60.0, "high": 40.0})
        H.check(np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0, "%s: low > high stays finite" % dev)
        # One bright pixel on black: percentiles ignore it, HDR range handled.
        spike = np.zeros((20, 20, 4), F32)
        spike[..., 3] = 1.0
        spike[5, 5, :3] = 1000.0
        out = render(dev, spike)
        H.check(np.isfinite(out).all(), "%s: HDR spike finite" % dev)
        # Sizes.
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (333, 187)):
            img = H.test_image(size[0], size[1], seed=2)
            out = render(dev, img)
            H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                    "%s %dx%d: shape / finite" % (dev, size[0], size[1]))
        # Partially transparent: hidden (alpha 0) pixels do not take part in the statistics.
        img = low_contrast(40, 30, seed=6)
        hidden = img.copy()
        hidden[:, :20] = 0.0                    # left half fully transparent black
        out = render(dev, hidden)
        vis = img[:, 20:].copy()
        out_vis = render(dev, np.ascontiguousarray(vis), size=(20, 30))
        H.compare("%s: transparent pixels ignored by the statistics" % dev, out[:, 20:], out_vis, 1e-5)
    # Large image stays fast and finite.
    big = H.test_image(1920, 1080, seed=4)
    for dev in ("CPU", "GPU"):
        out = render(dev, big)
        H.check(np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0, "%s 1920x1080" % dev)


for fn in (test_reference, test_invariants, test_gpu, test_robustness):
    fn()
H.finish()
