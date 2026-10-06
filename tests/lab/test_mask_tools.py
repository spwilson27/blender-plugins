# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Mask Tools node: threshold / grow / shrink / feather / outline / invert against brute-force
numpy references, disc and opening invariants, CPU vs GPU, float-socket input, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_mask_tools.py"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

NODE = "CompositorNodeLabMaskTools"
SIZE = (72, 56)
F32 = np.float32

# Binary results (hard threshold, grow, shrink, outline) are computed from integer squared
# distances in both backends: they must agree exactly. Smooth results (soft threshold, feather)
# use float32 sqrt / smoothstep: agreement to float rounding.
SMOOTH_ATOL = 2e-5


def mask_pattern(w, h, seed=0, grey=True):
    """Blobs + speckle + a thin line + a one-pixel hole, as (h, w, 4) with value in all channels."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    v = np.sin(x / 6.0 + seed) + np.cos(y / 5.0) + 0.5 * np.sin((x + y) / 3.7)
    m = (v > 0.9).astype(F32)
    m[(rng.random((h, w)) < 0.02)] = 1.0
    m[h // 2, 3:w - 3] = 1.0
    m[h // 3, w // 3] = 0.0
    px = np.empty((h, w, 4), F32)
    px[..., :3] = m[..., None]
    px[..., 3] = 1.0
    return px


def soft_values(w, h, seed=0):
    """Grey ramp-ish image with all values in [0, 1] (for threshold tests)."""
    rng = np.random.default_rng(seed)
    v = np.clip(0.5 + 0.5 * np.sin(np.arange(w)[None, :] / 5.0 + np.arange(h)[:, None] / 7.0)
                + 0.2 * (rng.random((h, w)) - 0.5), 0, 1).astype(F32)
    px = np.empty((h, w, 4), F32)
    px[..., :3] = v[..., None]
    px[..., 3] = 1.0
    return px


def offsets(r):
    n = int(math.ceil(r))
    return [(dy, dx) for dy in range(-n, n + 1) for dx in range(-n, n + 1)
            if dy * dy + dx * dx <= float(F32(r)) ** 2]


def dilate(b, r):
    """Brute-force binary dilation by the exact disc, no wrap, outside the image is empty."""
    h, w = b.shape
    out = np.zeros_like(b)
    for dy, dx in offsets(r):
        ys, ye = max(0, dy), min(h, h + dy)
        xs, xe = max(0, dx), min(w, w + dx)
        out[ys:ye, xs:xe] |= b[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def erode(b, r):
    """Erosion = complement of the dilation of the complement (image border is not an edge)."""
    return ~dilate(~b, r)


def dist_to(feature):
    """Brute-force Euclidean distance (float64) to the nearest True pixel."""
    h, w = feature.shape
    ys, xs = np.nonzero(feature)
    if len(ys) == 0:
        return np.full((h, w), np.inf)
    Y, X = np.mgrid[0:h, 0:w]
    d2 = np.full((h, w), np.inf)
    for fy, fx in zip(ys, xs):
        np.minimum(d2, (Y - fy) ** 2 + (X - fx) ** 2, out=d2)
    return np.sqrt(d2)


def run(dev, img, size=None, **props):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, dev, size, props=props, images={"Mask": img})[..., 0]


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


@H.guard("threshold")
def test_threshold():
    img = soft_values(*SIZE)
    v = img[..., 0].astype(np.float64)
    out = run("CPU", img, low=0.3, high=0.7)
    H.check(np.array_equal(out, ((v >= 0.3) & (v <= 0.7)).astype(F32)) or
            np.array_equal(out, ((F32(0.3) <= img[..., 0]) & (img[..., 0] <= F32(0.7))).astype(F32)),
            "hard threshold low..high")
    out = run("CPU", img, use_threshold=False)
    H.compare("no threshold: clamped value", out, np.clip(v, 0, 1), 1e-7)
    for lo, hi, soft in ((0.3, 0.7, 0.2), (0.4, 1.0, 0.1), (0.0, 0.5, 0.5)):
        out = run("CPU", img, low=lo, high=hi, softness=soft)
        el = smoothstep((v - (lo - soft / 2)) / soft)
        eh = smoothstep((v - (hi - soft / 2)) / soft)
        H.compare("soft threshold %g..%g soft %g" % (lo, hi, soft), out, el * (1 - eh), 2e-6)
    out = run("CPU", img, low=0.3, high=0.7, invert=True)
    H.check(np.array_equal(out, 1 - run("CPU", img, low=0.3, high=0.7)), "invert")
    # Sources.
    c = H.test_image(*SIZE, seed=5, alpha=True)
    out = run("CPU", c, source='ALPHA', low=0.5, high=1.0)
    H.check(np.array_equal(out, (c[..., 3] >= 0.5).astype(F32)), "source alpha")
    luma = (F32(0.2126) * c[..., 0] + F32(0.7152) * c[..., 1]) + F32(0.0722) * c[..., 2]
    out = run("CPU", c, source='LUMINANCE', low=0.4, high=1.0)
    H.check(np.array_equal(out, (luma >= F32(0.4)).astype(F32)), "source luminance")
    out = run("CPU", c, source='VALUE', low=0.4, high=1.0)
    H.check(np.array_equal(out, (c[..., 0] >= F32(0.4)).astype(F32)), "source value (red)")


@H.guard("morphology")
def test_morphology():
    img = mask_pattern(*SIZE, seed=2)
    b = img[..., 0] > 0.5
    for r in (1.0, 1.5, 2.0, 3.7, 6.0):
        out = run("CPU", img, grow=r)
        H.check(np.array_equal(out > 0.5, dilate(b, r)), "grow %g == brute-force dilation by the disc" % r)
        out = run("CPU", img, grow=-r)
        H.check(np.array_equal(out > 0.5, erode(b, r)), "shrink %g == brute-force erosion" % r)
    # Values are exactly 0 / 1.
    out = run("CPU", img, grow=2.5)
    H.check(set(np.unique(out)) <= {0.0, 1.0}, "binary output")
    # Single pixel -> exact disc.
    for r in (3.0, 5.5, 10.0, 17.3):
        one = np.zeros((61, 81, 4), F32)
        one[..., 3] = 1.0
        one[30, 40, :3] = 1.0
        out = run("CPU", one, grow=r) > 0.5
        yy, xx = np.nonzero(out)
        lattice = sum(1 for dy in range(-20, 21) for dx in range(-20, 21)
                      if dy * dy + dx * dx <= float(F32(r)) ** 2)
        H.check(out.sum() == lattice, "grow %g of a pixel: %d px == lattice disc %d" % (r, out.sum(), lattice))
        H.check(abs(out.sum() - math.pi * r * r) <= 0.1 * math.pi * r * r + 3,
                "grow %g: area %d ~ pi r^2 = %.1f" % (r, out.sum(), math.pi * r * r))
        H.check(((yy - 30) ** 2 + (xx - 40) ** 2).max() <= r * r + 1e-3, "disc radius <= %g" % r)
    # Opening: shrink then grow.
    blob = np.zeros((60, 90, 4), F32)
    blob[..., 3] = 1.0
    blob[10:40, 10:50, :3] = 1.0                 # big block
    blob[25:28, 50:85, :3] = 1.0                 # thin line (3 px)
    blob[45, 20, :3] = 1.0                       # speckle
    b = blob[..., 0] > 0.5
    r = 4.0
    er = run("CPU", blob, grow=-r)
    op = run("CPU", np.repeat(er[..., None], 4, axis=2).astype(F32) * np.array([1, 1, 1, 0], F32)
             + np.array([0, 0, 0, 1], F32), grow=r) > 0.5
    H.check(np.array_equal(er > 0.5, erode(b, r)), "opening step 1 is the erosion")
    H.check(np.array_equal(op, dilate(erode(b, r), r)), "opening = dilation of erosion")
    H.check((op & ~b).sum() == 0, "opening is anti-extensive (subset of the input)")
    H.check(not op[26, 60] and not op[45, 20], "opening removes the thin line and the speckle")
    H.check(op[15:35, 15:45].all() and op.sum() >= 0.9 * b[10:40, 10:50].sum(),
            "opening keeps the big block (only its corners are rounded)")
    # Idempotence of opening.
    op_img = np.repeat(op[..., None].astype(F32), 4, axis=2)
    op_img[..., 3] = 1.0
    er2 = run("CPU", op_img, grow=-r)
    op2 = run("CPU", np.repeat(er2[..., None], 4, axis=2).astype(F32) * np.array([1, 1, 1, 0], F32)
              + np.array([0, 0, 0, 1], F32), grow=r) > 0.5
    H.check(np.array_equal(op2, op), "opening is idempotent")
    # Border is not an edge: a mask covering the image does not erode.
    full = np.ones((20, 30, 4), F32)
    H.check(run("CPU", full, grow=-5.0).min() == 1.0, "image border is not an edge (shrink)")
    H.check(run("CPU", np.zeros((20, 30, 4), F32), grow=5.0).max() == 0.0, "empty stays empty (grow)")


def ref_signed(b):
    """float64 signed distance: inside 0.5 - d_in, outside d_out - 0.5."""
    d_o = dist_to(b)
    d_i = dist_to(~b)
    return np.where(b, 0.5 - d_i, d_o - 0.5)


@H.guard("feather_outline")
def test_feather_outline():
    img = mask_pattern(*SIZE, seed=4)
    b = img[..., 0] > 0.5
    s0 = ref_signed(b)
    for f, g in ((8.0, 0.0), (1.0, 0.0), (5.0, 2.0), (12.0, -3.0), (3.5, 0.0)):
        out = run("CPU", img, edge='FEATHER', feather=f, grow=g)
        ref = smoothstep(0.5 - (s0 - g) / f)
        H.compare("feather %g grow %g vs float64 reference" % (f, g), out, ref, 3e-6)
    out = run("CPU", img, edge='FEATHER', feather=10.0)
    H.check(0.0 <= out.min() and out.max() <= 1.0, "feather in [0, 1]")
    H.check((out[b] >= 0.5).all() and (out[~b] <= 0.5 + 1e-6).all(), "feather: inside >= 0.5 >= outside")
    # Monotone falloff away from an edge: distance-based, not Gaussian (compact support).
    step = np.zeros((20, 60, 4), F32)
    step[..., 3] = 1.0
    step[:, :30, :3] = 1.0
    prof = run("CPU", step, edge='FEATHER', feather=10.0)[10]
    H.check(np.all(np.diff(prof) <= 1e-6), "feather profile monotone")
    H.check(prof[:24].min() == 1.0 and prof[36:].max() == 0.0, "feather: exactly 1 / 0 beyond width/2 (compact)")
    H.check(abs(prof[29] + prof[30] - 1.0) < 1e-5, "feather symmetric about the edge")
    for pos in ('CENTER', 'OUTSIDE', 'INSIDE'):
        for width, g in ((4.0, 0.0), (3.0, 0.0), (1.0, 0.0), (6.0, 2.0), (2.0, -2.0)):
            out = run("CPU", img, edge='OUTLINE', outline_width=width, outline_position=pos, grow=g)
            s = s0 - g
            lo, hi = {'CENTER': (-width / 2, width / 2), 'OUTSIDE': (0, width), 'INSIDE': (-width, 0)}[pos]
            ref = ((s > lo) & (s <= hi)).astype(F32)
            H.compare("outline %s width %g grow %g" % (pos, width, g), out, ref, 0.0, max_frac=0.0)
    # 1-px outline of a square.
    sq = np.zeros((30, 30, 4), F32)
    sq[..., 3] = 1.0
    sq[10:20, 10:20, :3] = 1.0
    out = run("CPU", sq, edge='OUTLINE', outline_width=1.0, outline_position='INSIDE') > 0.5
    ring = np.zeros((30, 30), bool)
    ring[10:20, 10:20] = True
    ring[11:19, 11:19] = False
    H.check(np.array_equal(out, ring), "inside outline of width 1 is the 1px ring")
    out = run("CPU", sq, edge='OUTLINE', outline_width=1.0, outline_position='OUTSIDE') > 0.5
    ring = dilate(sq[..., 0] > 0.5, 1.5) & ~(sq[..., 0] > 0.5)
    H.check(np.array_equal(out, ring), "outside outline of width 1 is the 1px ring (d in (0.5, 1.5])")


@H.guard("parity")
def test_parity():
    from compositor_lab.nodes.mask_tools import CompositorNodeLabMaskTools as cls
    calls = {"gpu": 0}
    orig = cls.gpu

    def counting_gpu(self, *a, **k):
        calls["gpu"] += 1
        return orig(self, *a, **k)

    cls.gpu = counting_gpu
    cases = [
        ("hard threshold", dict(low=0.3, high=0.8), soft_values(*SIZE, seed=1), 0.0),
        ("soft threshold", dict(low=0.3, high=0.8, softness=0.2), soft_values(*SIZE, seed=1), SMOOTH_ATOL),
        ("invert", dict(invert=True), soft_values(*SIZE, seed=1), 0.0),
        ("no threshold", dict(use_threshold=False, grow=3.0), soft_values(*SIZE, seed=1), 0.0),
    ]
    mp = mask_pattern(*SIZE, seed=3)
    for g in (1.0, 2.0, 3.7, 9.0, 30.0, -1.0, -2.0, -3.7, -9.0, -30.0):
        cases.append(("grow %g" % g, dict(grow=g), mp, 0.0))
    for f, g in ((8.0, 0.0), (1.0, 0.0), (6.0, 3.0), (20.0, -4.0)):
        cases.append(("feather %g grow %g" % (f, g), dict(edge='FEATHER', feather=f, grow=g), mp, SMOOTH_ATOL))
    for pos in ('CENTER', 'OUTSIDE', 'INSIDE'):
        for width, g in ((4.0, 0.0), (7.5, 2.0), (3.0, -2.0)):
            cases.append(("outline %s %g grow %g" % (pos, width, g),
                          dict(edge='OUTLINE', outline_width=width, outline_position=pos, grow=g), mp, 0.0))
    cases.append(("outline+invert", dict(edge='OUTLINE', invert=True), mp, 0.0))
    c = H.test_image(*SIZE, seed=8, alpha=True)
    cases.append(("alpha source", dict(source='ALPHA', low=0.4, grow=2.0), c, 0.0))
    cases.append(("luminance source", dict(source='LUMINANCE', low=0.45, grow=-2.0), c, 0.0))
    cases.append(("luminance soft", dict(source='LUMINANCE', low=0.45, softness=0.3, edge='FEATHER', feather=5.0), c, SMOOTH_ATOL))
    for label, props, img, atol in cases:
        cpu = run("CPU", img, **props)
        gpu = run("GPU", img, **props)
        H.compare("%s: GPU vs CPU" % label, gpu, cpu, atol)
    H.check(calls["gpu"] > 0, "the GPU path was really exercised (%d calls)" % calls["gpu"])
    cls.gpu = orig
    # Different sizes: the GPU window passes must also work for odd shapes.
    for size in ((4, 4), (5, 9), (31, 17), (4, 40), (50, 4), (97, 61)):
        img = mask_pattern(*size, seed=6)
        for props in (dict(grow=2.5), dict(grow=-2.5), dict(edge='FEATHER', feather=6.0),
                      dict(edge='OUTLINE', outline_width=3.0)):
            atol = SMOOTH_ATOL if props.get("edge") == 'FEATHER' else 0.0
            H.compare("%dx%d %s: GPU vs CPU" % (size[0], size[1], props),
                      run("GPU", img, **props), run("CPU", img, **props), atol)


@H.guard("float_input")
def test_float_input():
    """A float socket (single-channel buffer / R32F texture) linked to Mask."""
    size = (64, 48)
    results = {}
    for dev in ("CPU", "GPU"):
        for src in ('VALUE', 'LUMINANCE', 'ALPHA'):
            scene = H.configure_scene(size, dev)
            tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
            scene.compositing_node_group = tree
            tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type="NodeSocketColor")
            noise = tree.nodes.new("CompositorNodeLabNoise")
            mt = tree.nodes.new(NODE)
            mt.source = src
            mt.use_threshold = False
            out = tree.nodes.new('NodeGroupOutput')
            tree.links.new(noise.outputs["Value"], mt.inputs["Mask"])
            tree.links.new(mt.outputs["Mask"], out.inputs[0])
            results[(dev, src)] = H.render(scene)[..., 0]
            if src == 'VALUE' and dev == "CPU":
                tree.links.new(noise.outputs["Value"], out.inputs[0])
                results["noise"] = H.render(scene)[..., 0]
            bpy.data.node_groups.remove(tree)
    H.compare("float input, VALUE: equals the noise value", results[("CPU", 'VALUE')],
              np.clip(results["noise"], 0, 1), 1e-6)
    H.compare("float input, VALUE: GPU vs CPU", results[("GPU", 'VALUE')], results[("CPU", 'VALUE')], 1e-6)
    H.compare("float input, LUMINANCE: the value itself (grey)", results[("CPU", 'LUMINANCE')],
              np.clip(results["noise"], 0, 1), 2e-6)
    H.compare("float input, LUMINANCE: GPU vs CPU", results[("GPU", 'LUMINANCE')], results[("CPU", 'LUMINANCE')], 1e-6)
    H.check(np.allclose(results[("CPU", 'ALPHA')], 1.0) and np.allclose(results[("GPU", 'ALPHA')], 1.0),
            "float input has alpha 1 on both backends")


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for props in ({}, {"grow": 5.0}, {"grow": -5.0}, {"edge": 'FEATHER'}, {"edge": 'OUTLINE'}):
            out = H.render_node(NODE, dev, (24, 16), props=props)
            H.check(out.shape == (16, 24, 4) and np.isfinite(out).all(),
                    "%s unlinked %s: finite, right shape" % (dev, props))
        out = H.render_node(NODE, dev, (24, 16), props={"grow": -6.0})
        H.check(np.allclose(out[..., 0], 1.0), "%s: unlinked (white) mask shrinks to itself (no border edge)" % dev)
        out = H.render_node(NODE, dev, (24, 16), inputs={"Mask": (0, 0, 0, 1)}, props={"grow": 6.0})
        H.check(np.allclose(out[..., 0], 0.0), "%s: black mask stays empty" % dev)
        z = np.zeros((16, 24, 4), F32)
        z[..., 3] = 1.0
        o = np.ones_like(z)
        for label, img in (("empty", z), ("full", o)):
            for props in ({"grow": 400.0}, {"grow": -400.0}, {"edge": 'FEATHER', "feather": 500.0},
                          {"edge": 'OUTLINE', "outline_width": 300.0}):
                out = H.render_node(NODE, dev, (24, 16), props=props, images={"Mask": img})
                H.check(np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1,
                        "%s %s %s: finite, in [0, 1]" % (dev, label, props))
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4)):
            img = mask_pattern(*size, seed=1)
            for props in ({"grow": 3.0}, {"edge": 'FEATHER'}, {"edge": 'OUTLINE', "grow": -1.0}):
                out = H.render_node(NODE, dev, size, props=props, images={"Mask": img})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %dx%d %s" % (dev, size[0], size[1], props))


@H.guard("perf")
def test_perf():
    from compositor_lab.lib import distance
    big = mask_pattern(1920, 1080, seed=1)
    for dev in ("CPU", "GPU"):
        for props in ({"grow": 20.0}, {"edge": 'FEATHER', "feather": 60.0}, {"grow": -100.0}):
            t = time.time()
            out = H.render_node(NODE, dev, (1920, 1080), props=props, images={"Mask": big})
            dt = time.time() - t
            H.note("1920x1080 %s %s: %.2f s (incl. render and EXR roundtrip)" % (dev, props, dt))
            H.check(dt < 20 and np.isfinite(out).all(), "%s 1080p %s finishes in reasonable time" % (dev, props))
    big_b = big[..., 0] > 0.5
    t = time.time()
    distance.edt_sq(big_b)
    H.note("edt_sq 1920x1080: %.2f s" % (time.time() - t))


for fn in (test_threshold, test_morphology, test_feather_outline, test_parity, test_float_input,
           test_robustness, test_perf):
    fn()
H.finish()
