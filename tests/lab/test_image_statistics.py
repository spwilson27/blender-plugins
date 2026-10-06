# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Image Statistics node: single-value outputs vs numpy, CPU vs GPU (including multi-level
reductions), percentile accuracy, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_image_statistics.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H
import ua_helpers as U

H.setup()

NODE = "CompositorNodeLabImageStatistics"
F32 = np.float32
LUMA = np.array([0.2126, 0.7152, 0.0722])

# CPU node vs numpy float64 on the same float32 pixels: the node accumulates in float64, so only
# the final float32 output rounding (<= 6e-8 relative) remains.
CPU_RTOL = 2e-7
# GPU: min / max are exact; the mean and variance are float32 tile sums merged pairwise-ish
# (Chan): observed errors up to 8e-7 absolute (5e-7 relative) on 16x16 .. 1920x1080 images;
# 3e-6 relative leaves a small margin.
GPU_RTOL = 3e-6


def stat(device, out, img, props=None, size=None):
    size = size or (img.shape[1], img.shape[0])
    return U.single(NODE, device, size, out, props=props, images={"Image": img})


def reference(img):
    px = img.reshape(-1, 4).astype(np.float64)
    lum = px[:, :3] @ LUMA
    return dict(min=px.min(axis=0)[:3], max=px.max(axis=0)[:3], mean=px.mean(axis=0)[:3],
                lmean=lum.mean(), lstd=lum.std(), lum=lum)


def check_stats(name, device, img, rtol, atol=1e-7):
    ref = reference(img)
    got = {k: stat(device, k, img) for k in ("Min", "Max", "Mean", "Luminance Mean", "Std Dev")}
    H.compare("%s [%s] Min" % (name, device), got["Min"][None, None, :3], ref["min"][None, None, :], 0.0)
    H.compare("%s [%s] Max" % (name, device), got["Max"][None, None, :3], ref["max"][None, None, :], 0.0)
    H.compare("%s [%s] Mean" % (name, device), got["Mean"][None, None, :3], ref["mean"][None, None, :], atol, rtol)
    H.compare("%s [%s] Luminance Mean" % (name, device), got["Luminance Mean"][:1], np.array([ref["lmean"]]), atol, rtol)
    H.compare("%s [%s] Std Dev" % (name, device), got["Std Dev"][:1], np.array([ref["lstd"]]), atol * 10, rtol * 10)
    H.check(got["Min"][3] == 1.0 and got["Max"][3] == 1.0 and got["Mean"][3] == 1.0,
            "%s [%s]: colour outputs have alpha 1" % (name, device))


@H.guard("cpu vs numpy")
def test_cpu():
    imgs = {
        "opaque 96x64": H.test_image(96, 64, seed=1),
        "alpha 97x33": H.test_image(97, 33, seed=2, alpha=True),
        "hdr 80x50": H.test_image(80, 50, seed=3, hdr=True),
    }
    neg = H.test_image(50, 40, seed=4, hdr=True)
    neg[..., :3] = neg[..., :3] * 2.0 - 1.0
    imgs["negative values"] = neg
    for name, img in imgs.items():
        check_stats(name, "CPU", img, CPU_RTOL)
    # Percentile against numpy: histogram resolution is (lmax - lmin) / 256.
    img = H.test_image(128, 96, seed=7, hdr=True)
    ref = reference(img)
    lum = ref["lum"]
    binw = (lum.max() - lum.min()) / 256.0
    for p in (0, 1, 10, 25, 50, 75, 90, 99, 100):
        got = stat("CPU", "Percentile", img, props={"percentile": float(p)})[0]
        want = np.percentile(lum, p)
        H.check(abs(got - want) <= 1.05 * binw + 1e-6,
                "CPU percentile %d: %.5f vs numpy %.5f (bin width %.5f)" % (p, got, want, binw))
    H.check(abs(stat("CPU", "Percentile", img, props={"percentile": 0.0})[0] - lum.min()) < 1e-6,
            "percentile 0 is the luminance minimum")
    H.check(abs(stat("CPU", "Percentile", img, props={"percentile": 100.0})[0] - lum.max()) < 1e-6,
            "percentile 100 is the luminance maximum")
    # Known values: half the pixels 0.2, half 0.8 (grey).
    h, w = 32, 64
    two = np.ones((h, w, 4), F32)
    two[..., :3] = 0.2
    two[:, w // 2:, :3] = 0.8
    H.check(abs(stat("CPU", "Luminance Mean", two)[0] - 0.5) < 1e-6, "two-tone luminance mean 0.5")
    H.check(abs(stat("CPU", "Std Dev", two)[0] - 0.3) < 1e-6, "two-tone std dev 0.3")
    H.check(abs(stat("CPU", "Percentile", two, props={"percentile": 25.0})[0] - 0.2) < 0.6 / 256 + 1e-6,
            "two-tone P25 ~ 0.2")
    H.check(abs(stat("CPU", "Percentile", two, props={"percentile": 75.0})[0] - 0.8) < 0.6 / 256 + 1e-6,
            "two-tone P75 ~ 0.8")


@H.guard("gpu vs cpu")
def test_gpu():
    # Sizes around the 16 px tile boundary and big enough for three reduction levels (> 256 px).
    for size in ((16, 16), (17, 16), (16, 33), (31, 5), (4, 4), (100, 70), (257, 259), (600, 300)):
        img = H.test_image(size[0], size[1], seed=size[0], alpha=(size[0] % 2 == 1), hdr=True)
        check_stats("%dx%d" % size, "GPU", img, GPU_RTOL)
        c = {k: stat("CPU", k, img) for k in ("Mean", "Std Dev")}
        g = {k: stat("GPU", k, img) for k in ("Mean", "Std Dev")}
        for k in c:
            H.compare("%dx%d %s GPU vs CPU" % (size[0], size[1], k), g[k][None, None, :], c[k][None, None, :],
                      1e-7, GPU_RTOL, quiet=True)
        # Same histogram -> same percentile.
        for p in (5.0, 50.0, 95.0):
            cp = stat("CPU", "Percentile", img, props={"percentile": p})[0]
            gp = stat("GPU", "Percentile", img, props={"percentile": p})[0]
            lum = reference(img)["lum"]
            binw = (lum.max() - lum.min()) / 256.0
            # A pixel exactly on a bin edge may land in the neighbouring bin when the GPU rounds
            # the luminance differently: allow one bin.
            H.check(abs(cp - gp) <= binw + 1e-6, "%dx%d P%g GPU %.5f vs CPU %.5f" % (size[0], size[1], p, gp, cp))


@H.guard("big image")
def test_big():
    img = H.test_image(1920, 1080, seed=9, hdr=True)
    ref = reference(img)
    for dev in ("CPU", "GPU"):
        t0 = time.time()
        got = stat(dev, "Mean", img)
        t1 = time.time()
        H.note("1920x1080 Mean on %s: %.2f s (incl. render + EXR round trip)" % (dev, t1 - t0))
        H.compare("1920x1080 [%s] Mean" % dev, got[None, None, :3], ref["mean"][None, None, :], 1e-7,
                  CPU_RTOL if dev == "CPU" else GPU_RTOL)
        std = stat(dev, "Std Dev", img)[0]
        H.check(abs(std - ref["lstd"]) < 1e-6 + (CPU_RTOL if dev == "CPU" else GPU_RTOL) * 10 * ref["lstd"],
                "1920x1080 [%s] Std Dev %.6f vs %.6f" % (dev, std, ref["lstd"]))


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        # Unlinked input (default black) and constant images.
        v = U.single(NODE, dev, (16, 12), "Mean")
        H.check(np.allclose(v[:3], 0.0), "%s: unlinked input: mean is its value (black)" % dev)
        v = U.single(NODE, dev, (16, 12), "Max")
        H.check(np.allclose(v[:3], 0.0), "%s: unlinked input: max" % dev)
        const = np.full((10, 13, 4), 0.37, F32)
        for out in ("Min", "Max", "Mean"):
            v = stat(dev, out, const)
            H.check(np.allclose(v[:3], 0.37, atol=1e-6), "%s constant image %s = 0.37" % (dev, out))
        H.check(stat(dev, "Std Dev", const)[0] < 1e-6, "%s constant image: std dev 0" % dev)
        H.check(abs(stat(dev, "Percentile", const, props={"percentile": 50.0})[0] - 0.37) < 1e-5,
                "%s constant image: percentile is the constant" % dev)
        # Fully transparent / zero image.
        z = np.zeros((9, 9, 4), F32)
        H.check(np.allclose(stat(dev, "Mean", z), [0, 0, 0, 1]), "%s: all-zero image" % dev)
        H.check(abs(stat(dev, "Percentile", z)[0]) < 1e-9, "%s: all-zero image percentile" % dev)
        # Odd and extreme sizes.
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17)):
            img = H.test_image(size[0], size[1], seed=1)
            ref = reference(img)
            m = stat(dev, "Mean", img)
            H.compare("%s %dx%d mean" % (dev, size[0], size[1]), m[None, None, :3], ref["mean"][None, None, :],
                      1e-7, 1e-5, quiet=True)
        # Large dynamic range stays finite.
        big = H.test_image(20, 20, seed=2)
        big[0, 0, :3] = 1e6
        v = stat(dev, "Std Dev", big)
        H.check(np.isfinite(v).all() and v[0] > 0, "%s: HDR outlier: finite std dev" % dev)


for fn in (test_cpu, test_gpu, test_big, test_robustness):
    fn()
H.finish()
