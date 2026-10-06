# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Posterize+ node: formula vs independent reference, distinct-value / identity / dither-mean
invariants, lightness mode, alpha, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_posterize.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_color, np_dither  # noqa: E402

NODE = "CompositorNodeLabPosterize"
SIZE = (96, 64)
F32 = np.float32
DITHERS = ["NONE", "BAYER2", "BAYER4", "BAYER8", "BLUE", "RANDOM"]

# GPU vs CPU: the quantiser is a floor of a float value, so a pixel whose value sits within a few
# ulp of a step boundary (pow() differs by ~1e-7 between numpy and the GPU) can land one step away.
# Allow a tiny fraction of such pixels (observed: none, max diff 6e-6 on GPU vs CPU, lightness mode;
# 3e-8 otherwise).
TIE_FRAC = 0.0005


def render(dev="CPU", size=SIZE, **kw):
    return H.render_node(NODE, dev, size, **kw)


def ref_quant(img, n, gamma=1.0, thresholds=None):
    """Independent float64 reference: straight colour, per-channel levels n, optional threshold."""
    rgb = img[..., :3].astype(np.float64)
    x = np.clip(rgb, 0, 1) ** (1.0 / gamma)
    n1 = np.asarray(n, np.float64) - 1.0
    t = 0.5 if thresholds is None else thresholds[..., None]
    q = np.floor(x * n1 + t) / n1
    return q ** gamma


def distinct(a):
    return len(np.unique(a))


@H.guard("formula")
def test_formula():
    img = H.test_image(*SIZE, seed=3)
    for n in (2, 3, 4, 7, 16):
        for gamma in (1.0, 2.2):
            out = render(props={"gamma": gamma}, inputs={"Levels": n}, images={"Image": img})
            ref = ref_quant(img, n, gamma)
            H.compare("levels %d gamma %g: CPU == reference" % (n, gamma), out[..., :3], ref,
                      2e-5, max_frac=TIE_FRAC)
            H.check(np.array_equal(out[..., 3], img[..., 3]), "alpha kept")
    # Dither thresholds against the independent Bayer definition (recursive construction).
    def bayer(k):
        m = np.array([[0, 2], [3, 1]])
        b = m
        for _ in range(k - 1):
            b = np.block([[4 * b + 0, 4 * b + 2], [4 * b + 3, 4 * b + 1]])
        return b
    for k, mode in ((1, "BAYER2"), (2, "BAYER4"), (3, "BAYER8")):
        n = 1 << k
        h, w = SIZE[1], SIZE[0]
        # Both are valid Bayer orderings; the node's matrix is the transpose of the classic one.
        got = np_dither.threshold(k, (h, w))
        cands = [(np.tile(bayer(k), (h // n + 1, w // n + 1))[:h, :w] + 0.5) / (n * n),
                 (np.tile(bayer(k).T, (h // n + 1, w // n + 1))[:h, :w] + 0.5) / (n * n)]
        H.check(any(np.allclose(got, cnd) for cnd in cands), "%s matrix is a Bayer matrix" % mode)
        out = render(props={"dither": mode, "gamma": 1.0}, inputs={"Levels": 3},
                     images={"Image": img})
        ref = ref_quant(img, 3, 1.0, got.astype(np.float64))
        H.compare("%s: CPU == reference" % mode, out[..., :3], ref, 2e-5, max_frac=TIE_FRAC)
    # Per-channel levels.
    out = render(props={"per_channel": True, "channel_levels": (2, 4, 8), "gamma": 1.0},
                 images={"Image": img})
    ref = ref_quant(img, [2, 4, 8], 1.0)
    H.compare("per-channel levels 2/4/8", out[..., :3], ref, 2e-5, max_frac=TIE_FRAC)


@H.guard("invariants")
def test_invariants():
    img = H.test_image(128, 96, seed=4)
    for dev in ("CPU", "GPU"):
        for n in (2, 3, 5, 8):
            for dither in DITHERS:
                for gamma in (1.0, 2.2):
                    out = render(dev, (128, 96), props={"dither": dither, "gamma": gamma},
                                 inputs={"Levels": n}, images={"Image": img})
                    counts = [distinct(np.round(out[..., c], 5)) for c in range(3)]
                    H.check(max(counts) <= n, "%s N=%d %s gamma %g: distinct values per channel "
                            "%s <= N" % (dev, n, dither, gamma, counts))
        out = render(dev, (128, 96), props={"per_channel": True, "channel_levels": (2, 3, 5),
                                            "dither": "BAYER4"}, images={"Image": img})
        counts = [distinct(np.round(out[..., c], 5)) for c in range(3)]
        H.check(all(c <= n for c, n in zip(counts, (2, 3, 5))),
                "%s per-channel distinct values %s <= (2, 3, 5)" % (dev, counts))
        # N huge: identity.
        for dither in ("NONE", "BAYER8", "RANDOM"):
            out = render(dev, (128, 96), props={"gamma": 1.0, "dither": dither},
                         inputs={"Levels": 65536}, images={"Image": img})
            H.compare("%s N=65536 %s ~ identity" % (dev, dither), out, img, 3e-5)
        out = render(dev, (128, 96), props={"gamma": 2.2}, inputs={"Levels": 65536},
                     images={"Image": img})
        H.compare("%s N=65536 gamma 2.2 ~ identity" % dev, out, img, 3e-4)
        # Fac 0 is the input, exactly; Fac 1 with dither amount 0 equals no dither.
        out = render(dev, (128, 96), inputs={"Fac": 0.0, "Levels": 2}, images={"Image": img})
        H.check(np.array_equal(out, img), "%s Fac 0 returns the input bit-exactly" % dev)
        a = render(dev, (128, 96), props={"dither": "BAYER8", "dither_amount": 0.0},
                   inputs={"Levels": 3}, images={"Image": img})
        b = render(dev, (128, 96), props={"dither": "NONE"}, inputs={"Levels": 3},
                   images={"Image": img})
        H.check(np.array_equal(a, b), "%s dither amount 0 == no dither" % dev)


@H.guard("dither mean")
def test_dither_mean():
    # A flat field: a dithered 2-level result must average to the input within one step; for the
    # ordered patterns every period (matrix size) block does so to within 1/(cells).
    for dev in ("CPU", "GPU"):
        for v in (0.1, 0.3, 0.5, 0.77):
            flat = np.full((64, 64, 4), v, F32)
            flat[..., 3] = 1.0
            for dither, n in (("BAYER2", 2), ("BAYER4", 4), ("BAYER8", 8)):
                out = render(dev, (64, 64), props={"dither": dither, "gamma": 1.0},
                             inputs={"Levels": 2}, images={"Image": flat})[..., 0]
                blocks = out.reshape(64 // n, n, 64 // n, n).mean(axis=(1, 3))
                err = float(np.abs(blocks - v).max())
                H.check(err <= 0.5 / (n * n) + 1e-6, "%s %s v=%g: every %dx%d block mean within "
                        "%.4f of v (max err %.4f)" % (dev, dither, v, n, n, 0.5 / (n * n), err))
            for dither in ("BLUE", "RANDOM"):
                out = render(dev, (64, 64), props={"dither": dither, "gamma": 1.0, "seed": 5},
                             inputs={"Levels": 2}, images={"Image": flat})[..., 0]
                H.check(abs(float(out.mean()) - v) < 0.03,
                        "%s %s v=%g: mean %.4f ~ v" % (dev, dither, v, float(out.mean())))
        # Without dither the same field has a single value (no spatial structure).
        out = render(dev, (64, 64), props={"gamma": 1.0}, inputs={"Levels": 2},
                     images={"Image": np.full((64, 64, 4), 0.3, F32)})
        H.check(distinct(out[..., 0]) == 1, "%s no dither: flat stays flat" % dev)
    # Seed changes the random pattern; blue-ish pattern is not white (neighbours anticorrelate).
    half = np.full((64, 64, 4), 0.5, F32)
    half[..., 3] = 1.0
    a = render(props={"dither": "RANDOM", "seed": 1}, inputs={"Levels": 2}, images={"Image": half})
    b = render(props={"dither": "RANDOM", "seed": 2}, inputs={"Levels": 2}, images={"Image": half})
    H.check(not np.array_equal(a, b), "random dither depends on the seed")
    t = np_dither.threshold(4, (128, 128), 0)
    c = np.corrcoef(t[:, :-1].ravel(), t[:, 1:].ravel())[0, 1]
    H.check(c < 0.0, "blue-ish threshold: horizontal neighbour correlation %.3f is negative" % c)
    H.check(abs(float(t.mean()) - 0.5) < 0.01, "blue-ish threshold mean %.4f" % float(t.mean()))


@H.guard("lightness")
def test_lightness():
    img = H.test_image(*SIZE, seed=6)
    for dev in ("CPU", "GPU"):
        for dither in ("NONE", "BAYER4"):
            out = render(dev, props={"lightness_only": True, "dither": dither},
                         inputs={"Levels": 5}, images={"Image": img})
            lab_in = np_color.linear_to_oklab(img[..., :3])
            lab_out = np_color.linear_to_oklab(out[..., :3])
            # Pixels whose quantised colour fell out of gamut are clamped (L changes): skip them.
            ok = (out[..., :3] > 1e-4).all(axis=-1)
            lq = np.round(lab_out[..., 0][ok], 3)
            H.check(distinct(lq) <= 5 and ok.mean() > 0.5,
                    "%s %s: <= 5 distinct OKLab lightness values (%d, %.0f%% in gamut)"
                    % (dev, dither, distinct(lq), 100 * ok.mean()))
            if dither == "NONE":
                ref_l = np.floor(np.clip(lab_in[..., 0], 0, 1) * 4 + 0.5) / 4
                H.compare("%s: lightness == round(L * 4) / 4" % dev, lab_out[..., 0][ok],
                          ref_l[ok], 3e-4, max_frac=TIE_FRAC)
            # Hue/chroma kept: a, b unchanged where the result stays in gamut.
            H.check(np.abs(lab_out[..., 1:] - lab_in[..., 1:])[ok].max() < 5e-4,
                    "%s: OKLab a/b preserved (%.2g)" % (
                        dev, float(np.abs(lab_out[..., 1:] - lab_in[..., 1:])[ok].max())))
        # Greys keep zero chroma.
        grey = np.full((8, 8, 4), 0.2, F32)
        grey[..., 3] = 1
        out = render(dev, (8, 8), props={"lightness_only": True}, inputs={"Levels": 3},
                     images={"Image": grey})
        H.check(np.allclose(out[..., 0], out[..., 1], atol=1e-4) and
                np.allclose(out[..., 1], out[..., 2], atol=1e-4), "%s: grey stays grey" % dev)


@H.guard("alpha")
def test_alpha():
    img = H.test_image(*SIZE, seed=7, alpha=True)
    a = img[..., 3:]
    straight = img[..., :3] / a
    for dev in ("CPU", "GPU"):
        out = render(dev, props={"gamma": 1.0}, inputs={"Levels": 4}, images={"Image": img})
        H.check(np.allclose(out[..., 3], img[..., 3], atol=1e-6), "%s: alpha unchanged" % dev)
        ref = np.floor(np.clip(straight, 0, 1) * 3 + 0.5) / 3 * a
        H.compare("%s: premultiplied output = quantised straight * alpha" % dev, out[..., :3],
                  ref, 3e-5, rtol=3e-5, max_frac=0.005)
        # Fully transparent pixels stay transparent black.
        clear = np.zeros((8, 8, 4), F32)
        out = render(dev, (8, 8), inputs={"Levels": 3}, images={"Image": clear})
        H.check(np.array_equal(out, clear), "%s: transparent stays transparent" % dev)


@H.guard("parity")
def test_parity():
    cases = [
        ("default", {}, {}),
        ("levels 3 gamma 1", {"gamma": 1.0}, {"Levels": 3}),
        ("per-channel", {"per_channel": True, "channel_levels": (2, 5, 9)}, {}),
        ("lightness", {"lightness_only": True}, {"Levels": 6}),
        ("fac 0.4", {}, {"Fac": 0.4, "Levels": 3}),
    ] + [("dither " + d, {"dither": d, "seed": 11, "dither_amount": 0.8}, {"Levels": 3})
         for d in DITHERS[1:]]
    for label, props, inputs in cases:
        for img_label, img in (("opaque", H.test_image(*SIZE, seed=8)),
                               ("alpha", H.test_image(*SIZE, seed=9, alpha=True)),
                               ("hdr", H.test_image(*SIZE, seed=10, hdr=True))):
            cpu = render("CPU", props=props, inputs=inputs, images={"Image": img})
            gpu = render("GPU", props=props, inputs=inputs, images={"Image": img})
            tol = 5e-5 if "lightness" in label else 1e-5
            H.compare("%s %s: GPU vs CPU" % (label, img_label), gpu, cpu, tol, max_frac=TIE_FRAC)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        out = render(dev, (16, 8))      # unlinked Image: default grey 0.8, 4 levels, gamma 2.2
        H.check(out.shape == (8, 16, 4) and np.isfinite(out).all() and np.allclose(out[..., 3], 1),
                "%s: unlinked input renders" % dev)
        step = np.floor(0.8 ** (1 / 2.2) * 3 + 0.5) / 3
        H.check(np.allclose(out[..., 0], step ** 2.2, atol=1e-4),
                "%s: unlinked default colour posterised (%g)" % (dev, float(out[0, 0, 0])))
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4), (17, 33)):
            img = H.test_image(*size, seed=5, alpha=True)
            for props in ({}, {"dither": "BAYER8"}, {"lightness_only": True, "dither": "BLUE"}):
                out = render(dev, size, props=props, images={"Image": img})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %dx%d %s: finite, right shape" % (dev, size[0], size[1], props))
        # Extreme values: HDR, negative, huge level counts, levels below 2.
        ext = np.array([[[1e4, -3.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]] * 4, F32)
        for props, inputs in (({}, {"Levels": 1}), ({}, {"Levels": 0}), ({}, {"Levels": 10 ** 7}),
                              ({"lightness_only": True}, {"Levels": 4})):
            out = render(dev, (4, 4), props=props, inputs=inputs,
                         images={"Image": np.tile(ext, (1, 2, 1))})
            H.check(np.isfinite(out).all(), "%s extreme %s %s: finite" % (dev, props, inputs))


for fn in (test_formula, test_invariants, test_dither_mean, test_lightness, test_alpha,
           test_parity, test_robustness):
    fn()
H.finish()
