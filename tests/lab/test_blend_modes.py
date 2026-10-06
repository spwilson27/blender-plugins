# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Blend Modes+ node: formulas (CPU) against independent references, CPU vs GPU, alpha handling,
robustness.   Blender -b --factory-startup --python-exit-code 1 --python test_blend_modes.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.lib.glsl.blend import MODES  # noqa: E402
from refs import ref_nonsep, ref_sep  # noqa: E402

NODE = "CompositorNodeLabBlendModes"
SIZE = (96, 64)
F32 = np.float32
SEP = MODES[:21]

# GPU vs CPU tolerance: both use float32 formulas; the GPU may fuse multiply-adds and divide
# approximately (a few ulp), and divides amplify that by 1/denominator, hence the relative term.
GPU_ATOL, GPU_RTOL = 2e-5, 2e-5
# OKLab modes go through cbrt via pow() on the GPU (exp2/log2) and divide by chroma.
GPU_ATOL_OK = 2e-4
# Modes with a discontinuity (a comparison on a computed value): a pixel sitting within an ulp of
# the threshold may land on the other side on the GPU. Allow a tiny fraction of such pixels.
DISCONT = {"HARD_MIX", "PIN_LIGHT", "OVERLAY", "HARD_LIGHT", "VIVID_LIGHT", "SOFT_LIGHT",
           "DARKER_COLOR", "LIGHTER_COLOR", "DIVIDE", "COLOR_DODGE", "COLOR_BURN",
           "HUE", "SATURATION", "COLOR", "LUMINOSITY"}


def ref_image(mode, a, b, fac=1.0):
    """Reference for opaque inputs: blend(A, B) mixed with A by fac (float64 scalar loops)."""
    rgb_a = a[..., :3].astype(np.float64).reshape(-1, 3)
    rgb_b = b[..., :3].astype(np.float64).reshape(-1, 3)
    if mode in MODES[21:]:
        r = ref_nonsep(mode, rgb_a, rgb_b)
    else:
        r = np.array([[ref_sep(mode, x, y) for x, y in zip(ra, rb)] for ra, rb in zip(rgb_a, rgb_b)])
    r = rgb_a + (r - rgb_a) * fac
    out = np.ones(a.shape, np.float64)
    out[..., :3] = r.reshape(a.shape[:2] + (3,))
    return out


@H.guard("roundtrip")
def test_roundtrip():
    """Alpha < 1 passes through the harness unchanged (no hidden (un)premultiply)."""
    a = H.test_image(*SIZE, seed=3, alpha=True)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": "NORMAL"},
                        images={"A": a, "B": np.zeros_like(a)}, inputs={"Fac": 0.0})
    H.compare("harness roundtrip with alpha (Normal, Fac 0 returns A)", out, a, 1e-6)


@H.guard("formulas")
def test_formulas():
    a = H.test_image(*SIZE, seed=11)
    b = H.test_image(*SIZE, seed=12)
    for mode in MODES:
        out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": mode}, images={"A": a, "B": b})
        tol = 3e-5 if mode in MODES[21:] else 2e-6
        H.compare("%s: CPU node == independent formula" % mode, out, ref_image(mode, a, b), tol,
                  rtol=2e-6, max_frac=0.0005 if mode in DISCONT else 0.0)
    # Fac as a constant and as a per-pixel image.
    fac_img = np.clip(np.linspace(0, 1, SIZE[0], dtype=F32)[None, :].repeat(SIZE[1], 0), 0, 1)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": "SCREEN"}, images={"A": a, "B": b},
                        inputs={"Fac": 0.3})
    H.compare("SCREEN Fac 0.3", out, ref_image("SCREEN", a, b, 0.3), 2e-6)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": "SCREEN"},
                        images={"A": a, "B": b, "Fac": fac_img})
    ref = ref_image("SCREEN", a, b, 1.0)
    ref = a + (ref - a) * fac_img[..., None]
    H.compare("SCREEN Fac image", out, ref, 2e-6)


@H.guard("alpha")
def test_alpha():
    a = H.test_image(*SIZE, seed=21, alpha=True)
    b = H.test_image(*SIZE, seed=22, alpha=True)
    for mode in ("NORMAL", "MULTIPLY", "OVERLAY", "LINEAR_DODGE", "HUE"):
        out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": mode}, images={"A": a, "B": b})
        aa, ab = a[..., 3:], b[..., 3:]
        ca = np.where(aa > 0, a[..., :3] / np.where(aa > 0, aa, 1), 0).astype(np.float64).reshape(-1, 3)
        cb = np.where(ab > 0, b[..., :3] / np.where(ab > 0, ab, 1), 0).astype(np.float64).reshape(-1, 3)
        if mode in MODES[21:]:
            bl = ref_nonsep(mode, ca, cb)
        else:
            bl = np.array([[ref_sep(mode, x, y) for x, y in zip(r1, r2)] for r1, r2 in zip(ca, cb)])
        bl = bl.reshape(SIZE[1], SIZE[0], 3)
        rgb = a[..., :3] * (1 - ab) + b[..., :3] * (1 - aa) + bl * (aa * ab)
        ref = np.concatenate([rgb, aa + ab - aa * ab], -1)
        H.compare("%s with alpha: W3C source-over reference" % mode, out, ref, 5e-5, rtol=5e-5,
                  max_frac=0.0005)
    # Alpha identities.
    clear = np.zeros_like(a)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": "MULTIPLY"}, images={"A": a, "B": clear})
    H.compare("transparent B leaves A", out, a, 1e-6)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": "MULTIPLY"}, images={"A": clear, "B": b})
    H.compare("transparent A gives B", out, b, 1e-6)
    out = H.render_node(NODE, "CPU", SIZE, props={"blend_type": "DIFFERENCE"}, images={"A": a, "B": b},
                        inputs={"Fac": 0.0})
    H.compare("Fac 0 gives A", out, a, 1e-6)


@H.guard("parity")
def test_parity():
    cases = [("opaque", H.test_image(*SIZE, seed=31), H.test_image(*SIZE, seed=32)),
             ("alpha", H.test_image(*SIZE, seed=33, alpha=True), H.test_image(*SIZE, seed=34, alpha=True))]
    fac_img = np.linspace(0, 1, SIZE[1], dtype=F32)[:, None].repeat(SIZE[0], 1)
    for mode in MODES:
        for label, a, b in cases:
            kw = dict(props={"blend_type": mode}, images={"A": a, "B": b}, inputs={"Fac": 0.8})
            cpu = H.render_node(NODE, "CPU", SIZE, **kw)
            gpu = H.render_node(NODE, "GPU", SIZE, **kw)
            tol = GPU_ATOL_OK if mode in MODES[21:] else GPU_ATOL
            H.compare("%s %s: GPU vs CPU" % (mode, label), gpu, cpu, tol, GPU_RTOL,
                      max_frac=0.002 if mode in DISCONT or label == "alpha" else 0.0)
    # Fac image and HDR inputs.
    a, b = H.test_image(*SIZE, seed=35, hdr=True), H.test_image(*SIZE, seed=36, hdr=True)
    for mode in ("MULTIPLY", "LINEAR_DODGE", "SUBTRACT", "DIFFERENCE", "SCREEN", "HUE"):
        kw = dict(props={"blend_type": mode}, images={"A": a, "B": b, "Fac": fac_img})
        cpu = H.render_node(NODE, "CPU", SIZE, **kw)
        gpu = H.render_node(NODE, "GPU", SIZE, **kw)
        H.compare("%s HDR + Fac image: GPU vs CPU" % mode, gpu, cpu,
                  GPU_ATOL_OK if mode == "HUE" else GPU_ATOL, GPU_RTOL * 5, max_frac=0.002)


@H.guard("robustness")
def test_robustness():
    half = np.full((16, 32, 4), 0.5, F32)
    half[..., 3] = 1.0
    for dev in ("CPU", "GPU"):
        # Unlinked inputs: defaults A = B = 0.5 grey, Multiply -> 0.25.
        out = H.render_node(NODE, dev, (32, 16), props={"blend_type": "MULTIPLY"},
                            images={"A": half})
        H.check(np.allclose(out[..., :3], 0.25, atol=1e-5), "%s: unlinked B uses its default (%s)" % (dev, out[0, 0]))
        out = H.render_node(NODE, dev, (32, 16), props={"blend_type": "MULTIPLY"},
                            images={"B": half})
        H.check(np.allclose(out[..., :3], 0.25, atol=1e-5), "%s: unlinked A uses its default" % dev)
        out = H.render_node(NODE, dev, (32, 16), props={"blend_type": "LINEAR_DODGE"},
                            inputs={"A": (0.1, 0.2, 0.3, 1.0), "B": (0.2, 0.2, 0.2, 1.0)},
                            images={"Fac": np.ones((16, 32), F32)})
        H.check(np.allclose(out[..., :3], [0.3, 0.4, 0.5], atol=1e-5),
                "%s: colour defaults A, B with a Fac image" % dev)
        # Odd sizes (Blender renders at least 4 pixels per side).
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4), (17, 33)):
            a, b = H.test_image(*size, seed=5), H.test_image(*size, seed=6, alpha=True)
            for mode in ("OVERLAY", "HUE", "DIVIDE"):
                out = H.render_node(NODE, dev, size, props={"blend_type": mode}, images={"A": a, "B": b})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %s %dx%d: finite, right shape (got %s)" % (dev, mode, size[0], size[1], out.shape))
        # HDR / extreme values stay finite.
        a = np.full((8, 8, 4), 1e4, F32)
        b = np.zeros((8, 8, 4), F32)
        b[..., 3] = 1.0
        for mode in MODES:
            out = H.render_node(NODE, dev, (8, 8), props={"blend_type": mode}, images={"A": a, "B": b})
            H.check(np.isfinite(out).all(), "%s %s: extreme inputs finite" % (dev, mode))


for fn in (test_roundtrip, test_formulas, test_alpha, test_parity, test_robustness):
    fn()
H.finish()
