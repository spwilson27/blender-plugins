# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Gradient Map node: stop interpolation vs independent references, endpoint / luminance
invariants, sources, alpha, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_gradient_map.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_color  # noqa: E402
from compositor_lab.nodes.gradient_map import PRESETS  # noqa: E402

NODE = "CompositorNodeLabGradientMap"
SIZE = (96, 64)
F32 = np.float32

# GPU vs CPU: same float32 formulas; OKLab goes through pow() for cbrt on the GPU. Observed max
# diffs: <= 1.5e-6 in every case (OKLab included).
GPU_ATOL = 1e-5


def render(dev="CPU", size=SIZE, **kw):
    return H.render_node(NODE, dev, size, **kw)


def grey_ramp(w=128, h=8, alpha=1.0):
    t = np.linspace(0.0, 1.0, w, dtype=F32)
    img = np.empty((h, w, 4), F32)
    img[..., :3] = t[None, :, None] * alpha
    img[..., 3] = alpha
    return img, t


def custom(stops, **props):
    d = {"preset": "CUSTOM", "stop_count": len(stops)}
    for i, (pos, col) in enumerate(stops):
        d["stop%d_pos" % i] = pos
        d["stop%d_color" % i] = col
    d.update(props)
    return d


def ref_gradient(t, stops, interp):
    """Independent reference: stops [(pos, rgb)] in the interpolation space; t array."""
    stops = sorted(stops, key=lambda s: s[0])
    pos = np.array([s[0] for s in stops], np.float64)
    col = np.array([s[1] for s in stops], np.float64)
    t = np.clip(t.astype(np.float64), 0, 1)
    if interp == "CONSTANT":
        idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 1)
        return col[idx]
    if interp == "SMOOTH":
        idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 2)
        f = np.clip((t - pos[idx]) / (pos[idx + 1] - pos[idx]), 0, 1)
        f = f * f * (3 - 2 * f)
        return col[idx] * (1 - f[..., None]) + col[idx + 1] * f[..., None]
    return np.stack([np.interp(t, pos, col[:, c]) for c in range(3)], axis=-1)


STOPS = [(0.7, (0.1, 0.9, 0.2)), (0.0, (0.05, 0.0, 0.2)), (0.4, (0.9, 0.1, 0.1)),
         (1.0, (1.0, 0.95, 0.6))]


@H.guard("interpolation")
def test_interpolation():
    img, t = grey_ramp(257)
    for interp in ("LINEAR", "SMOOTH", "CONSTANT"):
        out = render(props=custom(STOPS, interpolation=interp, color_space='LINEAR_RGB'),
                     images={"Image": img}, size=(257, 8))
        ref = ref_gradient(t, STOPS, interp)
        # CONSTANT jumps at the stop positions: pixels right at a position may go either way.
        H.compare("%s linear RGB: CPU == reference" % interp, out[0, :, :3], ref, 2e-5,
                  max_frac=0.01 if interp == "CONSTANT" else 0.0)
        H.check(np.allclose(out[..., 3], 1.0), "alpha 1")
        # OKLab: interpolate OKLab stops with the reference, convert back.
        lab_stops = [(p, np_color.linear_to_oklab(np.array(c, F32))) for p, c in STOPS]
        out = render(props=custom(STOPS, interpolation=interp, color_space='OKLAB'),
                     images={"Image": img}, size=(257, 8))
        ref = np.maximum(np_color.oklab_to_linear(ref_gradient(t, lab_stops, interp).astype(F32)), 0)
        H.compare("%s OKLab: CPU == reference" % interp, out[0, :, :3], ref, 1e-4,
                  max_frac=0.01 if interp == "CONSTANT" else 0.0)


@H.guard("endpoints")
def test_endpoints():
    black = np.zeros((8, 8, 4), F32)
    black[..., 3] = 1
    white = np.ones((8, 8, 4), F32)
    names = list(PRESETS) + ["CUSTOM"]
    for dev in ("CPU", "GPU"):
        for name in names:
            for space in ("LINEAR_RGB", "OKLAB"):
                for interp in ("LINEAR", "SMOOTH", "CONSTANT"):
                    props = {"preset": name, "color_space": space, "interpolation": interp}
                    if name == "CUSTOM":
                        props = custom(STOPS, color_space=space, interpolation=interp)
                        first, last = STOPS[1][1], STOPS[3][1]
                    else:
                        st = PRESETS[name][1]
                        first = np_color.srgb_to_linear(np.array(st[0][1], F32))
                        last = np_color.srgb_to_linear(np.array(st[-1][1], F32))
                    lo = render(dev, (8, 8), props=props, images={"Image": black})
                    hi = render(dev, (8, 8), props=props, images={"Image": white})
                    ok = np.allclose(lo[0, 0, :3], first, atol=1e-4) and \
                        np.allclose(hi[0, 0, :3], last, atol=1e-4)
                    H.check(ok, "%s %s %s %s: luminance 0 -> first stop, 1 -> last (%s, %s)" % (
                        dev, name, space, interp, lo[0, 0, :3], hi[0, 0, :3]))
        # Reverse swaps the ends.
        props = custom(STOPS, reverse=True, color_space='LINEAR_RGB')
        lo = render(dev, (8, 8), props=props, images={"Image": black})
        H.check(np.allclose(lo[0, 0, :3], STOPS[3][1], atol=1e-5), "%s reverse: 0 -> last stop" % dev)


@H.guard("black to white")
def test_identity_gradient():
    img, t = grey_ramp(200, 8)
    bw = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]
    for dev in ("CPU", "GPU"):
        out = render(dev, (200, 8), props=custom(bw, color_space='LINEAR_RGB'),
                     images={"Image": img})
        H.compare("%s: 2-stop black->white map == luminance" % dev, out[0, :, :3],
                  np.repeat(t[:, None], 3, 1), 2e-6)
        # On a colour image the output is the (grey) luminance, monotone in it.
        col = H.test_image(*SIZE, seed=2)
        out = render(dev, props=custom(bw, color_space='LINEAR_RGB'), images={"Image": col})
        luma = (col[..., :3] * np.array([0.2126, 0.7152, 0.0722], F32)).sum(-1)
        H.compare("%s: colour image -> its Rec.709 luminance" % dev, out[..., 0], luma, 3e-6)
        H.check(np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2]),
                "%s: output is grey" % dev)
        # In OKLab the same stops are monotone in luminance (not equal to it).
        out = render(dev, (200, 8), props=custom(bw, color_space='OKLAB'), images={"Image": img})
        g = out[0, :, 0]
        H.check((np.diff(g) >= -1e-6).all() and abs(g[0]) < 1e-5 and abs(g[-1] - 1) < 1e-4,
                "%s: OKLab black->white is monotone from 0 to 1" % dev)


@H.guard("sources")
def test_sources():
    img = H.test_image(*SIZE, seed=3)
    bw = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]
    rgb = img[..., :3]
    lab_l = np_color.linear_to_oklab(rgb)[..., 0]
    refs = {"RED": rgb[..., 0], "GREEN": rgb[..., 1], "BLUE": rgb[..., 2], "VALUE": rgb.max(-1),
            "LIGHTNESS": lab_l}
    for src, ref in refs.items():
        out = render(props=custom(bw, source=src, color_space='LINEAR_RGB'),
                     images={"Image": img})
        H.compare("source %s" % src, out[..., 0], ref, 3e-6)
    # Alpha source: premultiplied input with varying alpha.
    ia = H.test_image(*SIZE, seed=4, alpha=True)
    out = render(props=custom(bw, source="ALPHA", color_space='LINEAR_RGB'), images={"Image": ia})
    H.compare("source ALPHA (preserve alpha: output premultiplied)", out[..., 0],
              ia[..., 3] * ia[..., 3], 3e-6)
    out = render(props=custom(bw, source="ALPHA", color_space='LINEAR_RGB', preserve_alpha=False),
                 images={"Image": ia})
    H.compare("source ALPHA, opaque result", out[..., 0], ia[..., 3], 3e-6)
    H.check(np.allclose(out[..., 3], 1.0), "preserve_alpha off gives alpha 1")


@H.guard("alpha and fac")
def test_alpha_fac():
    img = H.test_image(*SIZE, seed=5, alpha=True)
    a = img[..., 3:]
    straight = img[..., :3] / a
    luma = (straight * np.array([0.2126, 0.7152, 0.0722], F32)).sum(-1)
    props = custom(STOPS, color_space='LINEAR_RGB')
    out = render(props=props, images={"Image": img})
    ref = ref_gradient(luma, STOPS, "LINEAR") * a
    H.compare("alpha preserved: rgb = gradient(luma(straight)) * alpha", out[..., :3], ref, 3e-5,
              rtol=3e-5)
    H.check(np.allclose(out[..., 3], img[..., 3], atol=1e-6), "alpha channel kept")
    out = render(props=props, inputs={"Fac": 0.35}, images={"Image": img})
    ref2 = img * 0.65 + 0.35 * np.concatenate([ref, a], -1)
    H.compare("Fac 0.35 mixes with the input", out, ref2, 3e-5, rtol=3e-5)
    out = render(props=props, inputs={"Fac": 0.0}, images={"Image": img})
    H.check(np.array_equal(out, img), "Fac 0 returns the input")


@H.guard("parity")
def test_parity():
    cases = []
    for name in PRESETS:
        cases.append(("preset " + name, {"preset": name}))
    for interp in ("LINEAR", "SMOOTH", "CONSTANT"):
        for space in ("LINEAR_RGB", "OKLAB"):
            cases.append(("custom %s %s" % (interp, space),
                          custom(STOPS, interpolation=interp, color_space=space)))
    cases.append(("2 stops", custom(STOPS[:2])))
    cases.append(("6 stops", custom([(0.0, (0, 0, 0)), (0.1, (1, 0, 0)), (0.3, (0, 1, 0)),
                                      (0.5, (0, 0, 1)), (0.8, (1, 1, 0)), (1.0, (1, 1, 1))])))
    cases.append(("zero-width segment", custom([(0.0, (0, 0, 0)), (0.5, (1, 0, 0)),
                                                 (0.5, (0, 0, 1)), (1.0, (1, 1, 1))])))
    for src in ("LUMINANCE", "LIGHTNESS", "RED", "GREEN", "BLUE", "VALUE", "ALPHA"):
        cases.append(("source " + src, {"source": src, "reverse": src == "RED"}))
    for label, props in cases:
        for ilabel, img in (("opaque", H.test_image(*SIZE, seed=6)),
                            ("alpha", H.test_image(*SIZE, seed=7, alpha=True)),
                            ("hdr", H.test_image(*SIZE, seed=8, hdr=True))):
            if ilabel != "opaque" and not label.startswith(("custom", "source")):
                continue
            kw = dict(props=props, images={"Image": img}, inputs={"Fac": 0.9})
            cpu = render("CPU", **kw)
            gpu = render("GPU", **kw)
            H.compare("%s %s: GPU vs CPU" % (label, ilabel), gpu, cpu, GPU_ATOL,
                      max_frac=0.002 if "CONSTANT" in label or "zero" in label else 0.0)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        out = render(dev, (16, 8))      # unlinked Image = 0.5 grey, default Inferno preset
        H.check(out.shape == (8, 16, 4) and np.isfinite(out).all() and
                np.allclose(out, out[0, 0]), "%s: unlinked input renders a flat colour" % dev)
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4), (17, 33)):
            img = H.test_image(*size, seed=9, alpha=True)
            for props in ({}, {"preset": "ICE_FIRE", "color_space": "LINEAR_RGB"},
                          custom(STOPS, interpolation="CONSTANT")):
                out = render(dev, size, props=props, images={"Image": img})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %dx%d: finite, right shape" % (dev, size[0], size[1]))
        # HDR / negative / transparent inputs stay finite and within the gradient range.
        ext = np.array([[[1e4, -3.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 1.0],
                         [np.nan, 0.0, 0.0, 1.0]]] * 4, F32)
        ext[..., 3][ext[..., 3] != ext[..., 3]] = 1
        out = render(dev, (4, 4), images={"Image": ext[:, :3].copy()})
        H.check(np.isfinite(out[:, :3]).all(), "%s extreme inputs: finite" % dev)
        # Two stops at the same position / all stops equal.
        out = render(dev, (8, 8), props=custom([(0.5, (1, 0, 0)), (0.5, (0, 0, 1))]),
                     images={"Image": np.full((8, 8, 4), 0.5, F32)})
        H.check(np.isfinite(out).all(), "%s coincident stops: finite" % dev)


for fn in (test_interpolation, test_endpoints, test_identity_gradient, test_sources,
           test_alpha_fac, test_parity, test_robustness):
    fn()
H.finish()
