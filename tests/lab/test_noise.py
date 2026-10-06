# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Noise node: CPU vs numpy, CPU vs GPU, determinism, seed/time behaviour, statistics, robustness.
Uses the real node when the build has F1 (generator domain), else a test subclass with a "Ref"
image input; time tests need F2 (the evaluation context).
   Blender -b --factory-startup --python-exit-code 1 --python test_noise.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_noise  # noqa: E402
from compositor_lab.nodes.noise import _TYPE_INDEX, channel_seed  # noqa: E402

NODE = "CompositorNodeLabNoise"
SIZE = (96, 64)
F32 = np.float32
TYPES = list(_TYPE_INDEX)

# GPU vs CPU: lattice maths and hashes are exact; only float rounding differs (fma contraction,
# approximate division in fast math). Coordinates carry ~1e-6 error at the largest octaves,
# amplified by gradients up to ~3 and normalised by the amplitude sum. Observed errors are far
# smaller; 1e-4 leaves headroom for warp + 8 octaves.
GPU_ATOL = 1e-4
# Simplex is scaled by 32 and its gradient reaches ~10 per unit, so a 1e-6 coordinate rounding
# difference (fma contraction, 4 octaves) shows up as up to ~4e-4 in rare pixels (observed
# 4e-4 in 0.03% of pixels, mean error 1e-7).
SIMPLEX_ATOL = 6e-4


def render(device="CPU", size=SIZE, out="Value", **kw):
    return H.render_generator(NODE, device, size, out_socket=out, **kw)


def reference(size, ntype="PERLIN", scale=5.0, seed=0, octaves=4, lac=2.0, gain=0.5, warp=0.0,
              jitter=1.0, ridged=False, ox=0.0, oy=0.0, z=0.0, channel=0):
    w, h = size
    k = F32(scale / max(w, h))
    xs = (np.arange(w, dtype=F32) + F32(0.5)) * k + F32(ox)
    ys = (np.arange(h, dtype=F32) + F32(0.5)) * k + F32(oy)
    x = np.broadcast_to(xs[None, :], (h, w))
    y = np.broadcast_to(ys[:, None], (h, w))
    return np_noise.noise_field(_TYPE_INDEX[ntype], x, y, F32(z), channel_seed(seed, channel),
                                jitter, octaves, lac, gain, ridged, warp)


def px(img):
    return img[..., 0]


@H.guard("features")
def test_features():
    print("build features:", H.FEATURES)
    if not H.FEATURES.get("F1"):
        H.note("F1 (generator domain) not available in this build: using the Ref-input subclass")
    if not H.FEATURES.get("F2"):
        H.note("F2 (context) not available: time-based checks skipped")
    img = render()
    H.check(img.shape == (SIZE[1], SIZE[0], 4), "output has the render size %s" % (img.shape,))


@H.guard("cpu reference")
def test_cpu_reference():
    cases = [
        ("default perlin", dict(), dict()),
        ("value", dict(props={"noise_type": "VALUE"}), dict(ntype="VALUE")),
        ("simplex", dict(props={"noise_type": "SIMPLEX"}), dict(ntype="SIMPLEX")),
        ("worley f1", dict(props={"noise_type": "WORLEY_F1"}), dict(ntype="WORLEY_F1")),
        ("worley f2-f1", dict(props={"noise_type": "WORLEY_F2_F1"}), dict(ntype="WORLEY_F2_F1")),
        ("ridged simplex", dict(props={"noise_type": "SIMPLEX", "ridged": True}),
         dict(ntype="SIMPLEX", ridged=True)),
        ("params", dict(inputs={"Scale": 9.0, "Octaves": 6, "Lacunarity": 2.4, "Gain": 0.6, "Seed": 77,
                                "Offset X": 1.5, "Offset Y": -2.25, "Phase": 0.8, "Speed": 0.0}),
         dict(scale=9.0, octaves=6, lac=2.4, gain=0.6, seed=77, ox=1.5, oy=-2.25, z=0.8)),
        ("warp", dict(inputs={"Warp": 0.6, "Speed": 0.0}), dict(warp=0.6)),
        ("randomness", dict(props={"noise_type": "WORLEY_F1"}, inputs={"Randomness": 0.4}),
         dict(ntype="WORLEY_F1", jitter=0.4)),
    ]
    for name, kw, ref_kw in cases:
        kw = dict(kw)
        kw.setdefault("inputs", {}).setdefault("Speed", 0.0)
        out = render(**kw)
        ref = reference(SIZE, **ref_kw)
        H.compare("%s: CPU node == numpy field" % name, px(out), ref, 1e-6)
    # Colour output: channels use seeds + c * step; R equals Value.
    col = render(out="Color", inputs={"Seed": 5, "Speed": 0.0})
    for c in range(3):
        H.compare("colour channel %d == field(seed channel)" % c, col[..., c],
                  reference(SIZE, seed=5, channel=c), 1e-6)
    H.check(np.all(col[..., 3] == 1.0), "colour alpha is 1")
    val = render(inputs={"Seed": 5, "Speed": 0.0})
    H.compare("Value output == Color.R", px(val), col[..., 0], 0.0)
    cc = np.corrcoef(col[..., 0].ravel(), col[..., 1].ravel())[0, 1]
    H.check(abs(cc) < 0.3, "colour channels decorrelated (corr %.3f)" % cc)


@H.guard("gpu parity")
def test_gpu():
    cases = [(t, dict(props={"noise_type": t})) for t in TYPES]
    cases += [
        ("ridged perlin", dict(props={"ridged": True})),
        ("ridged worley", dict(props={"ridged": True, "noise_type": "WORLEY_F1"})),
        ("octaves 8 lac 2.7 gain 0.7", dict(inputs={"Octaves": 8, "Lacunarity": 2.7, "Gain": 0.7})),
        ("warp 1.2 simplex", dict(props={"noise_type": "SIMPLEX"}, inputs={"Warp": 1.2})),
        ("offset/phase/seed", dict(inputs={"Offset X": 12.5, "Offset Y": -7.0, "Phase": 3.3, "Seed": -9,
                                           "Scale": 20.0})),
        ("worley randomness 0", dict(props={"noise_type": "WORLEY_F2_F1"}, inputs={"Randomness": 0.0})),
        ("large scale", dict(inputs={"Scale": 300.0}, props={"noise_type": "VALUE"})),
    ]
    for name, kw in cases:
        for out in ("Value", "Color"):
            cpu = render("CPU", out=out, **kw)
            gpu = render("GPU", out=out, **kw)
            # Worley feature-point ties and the F1/F2 swap are continuous; no mismatch allowance.
            simplex = kw.get("props", {}).get("noise_type") == "SIMPLEX"
            H.compare("%s [%s]: GPU vs CPU" % (name, out), gpu, cpu,
                      SIMPLEX_ATOL if simplex else GPU_ATOL)


@H.guard("behaviour")
def test_behaviour():
    a = render()
    b = render()
    H.check(np.array_equal(a, b), "deterministic: same settings, same pixels")
    for dev in ("CPU", "GPU"):
        s0 = render(dev, inputs={"Seed": 1, "Speed": 0.0})
        s1 = render(dev, inputs={"Seed": 2, "Speed": 0.0})
        H.check(float(np.abs(s0 - s1).mean()) > 0.03, "%s: different seed changes the image" % dev)
        z0 = render(dev, inputs={"Phase": 0.0, "Speed": 0.0})
        z1 = render(dev, inputs={"Phase": 0.5, "Speed": 0.0})
        H.check(float(np.abs(z0 - z1).mean()) > 0.01, "%s: Phase (3rd dimension) changes the image" % dev)
    # Resolution independence: a 2x render, block-averaged, resembles the 1x render.
    small = px(render(size=(48, 32), inputs={"Speed": 0.0, "Octaves": 2}))
    big = px(render(size=(96, 64), inputs={"Speed": 0.0, "Octaves": 2}))
    down = big.reshape(32, 2, 48, 2).mean(axis=(1, 3))
    cc = np.corrcoef(small.ravel(), down.ravel())[0, 1]
    H.check(cc > 0.97, "pattern is resolution independent (corr %.3f)" % cc)
    # Statistics per type: values in [0, 1], plausible mean / spread.
    big = (256, 192)
    expect = {"VALUE": (0.5, 0.12), "PERLIN": (0.5, 0.12), "SIMPLEX": (0.5, 0.12),
              "WORLEY_F1": (0.55, 0.1), "WORLEY_F2_F1": (0.25, 0.1)}
    for t in TYPES:
        v = px(render(size=big, props={"noise_type": t}, inputs={"Scale": 12.0, "Speed": 0.0, "Octaves": 1}))
        H.check(v.min() >= 0.0 and v.max() <= 1.0 and np.isfinite(v).all(), "%s in [0, 1]" % t)
        m, s = float(v.mean()), float(v.std())
        print("  %s: mean %.3f std %.3f min %.3f max %.3f" % (t, m, s, v.min(), v.max()))
        H.check(abs(m - expect[t][0]) < 0.12, "%s mean %.3f ~ %.2f" % (t, m, expect[t][0]))
        H.check(0.04 < s < 0.3, "%s has contrast (std %.3f)" % (t, s))
    # fBm adds detail: more octaves -> more high-frequency energy.
    def hf(img):
        return float(np.abs(np.diff(img, axis=1)).mean())
    o1 = hf(px(render(size=big, inputs={"Octaves": 1, "Speed": 0.0})))
    o6 = hf(px(render(size=big, inputs={"Octaves": 6, "Speed": 0.0})))
    H.check(o6 > o1 * 1.2, "more octaves add detail (%.4f -> %.4f)" % (o1, o6))
    g0 = hf(px(render(size=big, inputs={"Octaves": 6, "Gain": 0.2, "Speed": 0.0})))
    g1 = hf(px(render(size=big, inputs={"Octaves": 6, "Gain": 0.8, "Speed": 0.0})))
    H.check(g1 > g0 * 1.2, "higher gain adds detail (%.4f -> %.4f)" % (g0, g1))
    rd = px(render(size=big, props={"ridged": True}, inputs={"Speed": 0.0}))
    pl = px(render(size=big, inputs={"Speed": 0.0}))
    H.check(float(np.abs(rd - pl).mean()) > 0.05, "ridged option changes the pattern")


@H.guard("time")
def test_time():
    if not H.FEATURES.get("F2"):
        H.note("F2 not available: skipping time-dependent noise tests (pending)")
        return
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    for dev in ("CPU", "GPU"):
        f1 = render(dev, frame=1, inputs={"Speed": 0.5})
        f25 = render(dev, frame=25, inputs={"Speed": 0.5})
        H.check(float(np.abs(f1 - f25).mean()) > 0.01, "%s: image changes with the frame" % dev)
        s1 = render(dev, frame=1, inputs={"Speed": 0.0})
        s25 = render(dev, frame=25, inputs={"Speed": 0.0})
        H.check(np.array_equal(s1, s25), "%s: Speed 0 is static over time" % dev)
        rerun = render(dev, frame=25, inputs={"Speed": 0.5})
        H.check(np.array_equal(f25, rerun), "%s: same frame renders identically" % dev)
        ref = reference(SIZE, z=np.float32(25 / fps * 0.5))
        H.compare("%s: frame 25 == numpy at z = time * speed (fps %g)" % (dev, fps), px(f25), ref,
                  1e-6 if dev == "CPU" else GPU_ATOL)
        # Phase adds to time * speed.
        ph = render(dev, frame=25, inputs={"Speed": 0.5, "Phase": 1.0})
        ref = reference(SIZE, z=np.float32(1.0 + 25 / fps * 0.5))
        H.compare("%s: Phase + time * speed" % dev, px(ph), ref, 1e-6 if dev == "CPU" else GPU_ATOL)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (333, 187), (17, 17)):  # Blender renders >= 4 px per side
            for out in ("Value", "Color"):
                img = render(dev, size=size, out=out)
                H.check(img.shape == (size[1], size[0], 4) and np.isfinite(img).all() and
                        img.min() >= 0.0 and img.max() <= 1.0,
                        "%s %dx%d %s: shape/finite/range" % (dev, size[0], size[1], out))
        # Extreme / degenerate parameters stay finite and in range.
        for name, inputs in (("octaves 0", {"Octaves": 0}), ("octaves 99", {"Octaves": 99}),
                             ("scale 0", {"Scale": 0.0}), ("negative scale", {"Scale": -4.0}),
                             ("gain 0", {"Gain": 0.0}), ("gain 3", {"Gain": 3.0, "Octaves": 8}),
                             ("lacunarity 0.5", {"Lacunarity": 0.5}), ("huge offset", {"Offset X": 1e5}),
                             ("big warp", {"Warp": 25.0})):
            img = render(dev, inputs=inputs)
            H.check(np.isfinite(img).all() and img.min() >= 0.0 and img.max() <= 1.0,
                    "%s %s: finite, in range" % (dev, name))
    # Linked scalar inputs fall back to the socket default on both backends (documented).
    scale_img = np.full((SIZE[1], SIZE[0]), 0.5, F32)
    kw = dict(images={"Scale": scale_img}) if H.FEATURES.get("F1") else None
    if kw:
        scene = H.configure_scene(SIZE, "CPU")
        for dev in ("CPU", "GPU"):
            H.configure_scene(SIZE, dev)
            H.build_tree(scene, NODE, images={"Scale": scale_img}, inputs={"Speed": 0.0})
            img = H.render(scene)
            H.check(np.isfinite(img).all(), "%s: linked Scale does not break the node" % dev)
            if dev == "CPU":
                cpu_linked = img
            else:
                H.compare("linked Scale: GPU == CPU (both use the default)", img, cpu_linked, GPU_ATOL)
    else:
        H.note("linked-scalar fallback test needs F1 (image input defines the domain otherwise)")


for fn in (test_features, test_cpu_reference, test_gpu, test_behaviour, test_time, test_robustness):
    fn()
H.finish()
