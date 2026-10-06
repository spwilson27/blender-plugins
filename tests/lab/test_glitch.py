# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Glitch node: each effect against an independent loop reference, exact CPU/GPU agreement on
the random decisions, determinism, seed / time behaviour, statistics, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_glitch.py"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_noise  # noqa: E402
from compositor_lab.nodes.glitch import sub_seed  # noqa: E402

NODE = "CompositorNodeLabGlitch"
SIZE = (128, 80)
F32 = np.float32

OFF = {"use_split": False, "use_blocks": False, "use_jitter": False, "use_crush": False}
STATIC = {"Speed": 0.0}      # step 0 whatever the frame / context


def render(dev="CPU", size=SIZE, **kw):
    return H.render_node(NODE, dev, size, **kw)


def only(**on):
    p = dict(OFF)
    p.update(on)
    return p


def decide(h3, density):
    """(moves, hy) from a scalar hash triple, exactly as the node does it."""
    hx, hy = int(np.ravel(h3[0])[0]), int(np.ravel(h3[1])[0])
    return F32((hx >> 8) / 16777216.0) < F32(density), hy


def ref_glitch(img, step, seed, shift, density, bw, bh, jitter, jdens, dx, dy, blocks=True,
               jit=True, crush_bits=0):
    """Independent reference: Python loops over blocks and rows."""
    h, w = img.shape[:2]
    xs = np.tile(np.arange(w), (h, 1))
    if blocks:
        amp = min(int(round(shift)), w - 1)
        ss = np_noise.scramble_seed(sub_seed(seed, 1))
        for cy in range((h + bh - 1) // bh):
            for cx in range((w + bw - 1) // bw):
                hh = [int(np.ravel(v)[0]) for v in np_noise.hash3(np.array([cx], np.int32),
                                                     np.array([cy], np.int32), np.int32(step), ss)]
                mv, hy = decide(hh, density)
                if mv and amp > 0:
                    sh = ((hy >> 8) % (2 * amp + 1)) - amp
                    y0, y1, x0, x1 = cy * bh, min((cy + 1) * bh, h), cx * bw, min((cx + 1) * bw, w)
                    xs[y0:y1, x0:x1] = (np.arange(x0, x1) - sh) % w
    if jit:
        amp = min(int(round(jitter)), w - 1)
        ss = np_noise.scramble_seed(sub_seed(seed, 2))
        for y in range(h):
            hh = [int(np.ravel(v)[0]) for v in np_noise.hash3(np.array([y], np.int32), np.int32(0),
                                                 np.int32(step), ss)]
            mv, hy = decide(hh, jdens)
            if mv and amp > 0:
                sh = ((hy >> 8) % (2 * amp + 1)) - amp
                xs[y] = (xs[y] - sh) % w
    yy = np.arange(h)[:, None]
    take = lambda ox, oy: img[np.clip(yy + oy, 0, h - 1), np.clip(xs + ox, 0, w - 1)]
    cr, cg, cb = take(dx, dy), take(0, 0), take(-dx, -dy)
    out = np.stack([cr[..., 0], cg[..., 1], cb[..., 2],
                    np.maximum(cr[..., 3], np.maximum(cg[..., 3], cb[..., 3]))], -1)
    if crush_bits:
        n = (1 << crush_bits) - 1
        out[..., :3] = np.floor(np.clip(out[..., :3], 0, 1) * n + 0.5) / n
    return out


@H.guard("blocks")
def test_blocks():
    img = H.test_image(*SIZE, seed=1, alpha=True)
    for seed in (0, 1, 777):
        for (bw, bh, shift, dens) in ((32, 16, 20.0, 0.4), (48, 10, 60.0, 1.0), (7, 3, 9.0, 0.5),
                                      (500, 500, 30.0, 1.0)):
            props = only(use_blocks=True, block_width=bw, block_height=bh)
            out = render(props=props, inputs=dict(STATIC, Seed=seed, Shift=shift, Density=dens),
                         images={"Image": img})
            ref = ref_glitch(img, 0, seed, shift, dens, bw, bh, 0, 0, 0, 0, jit=False)
            H.check(np.array_equal(out, ref), "CPU seed %d blocks %dx%d shift %g density %g: "
                    "exactly the reference blocks (max diff %g)" % (
                        seed, bw, bh, shift, dens, float(np.abs(out - ref).max())))
    # Moved blocks are shifted copies: every moved row of a block is a roll of the source row.
    props = only(use_blocks=True, block_width=32, block_height=16)
    out = render(props=props, inputs=dict(STATIC, Shift=20.0, Density=1.0), images={"Image": img})
    moved = 0
    for cy in range(SIZE[1] // 16):
        for cx in range(SIZE[0] // 32):
            blk = out[cy * 16:(cy + 1) * 16, cx * 32:(cx + 1) * 32]
            ok = any(np.array_equal(blk, np.roll(img[cy * 16:(cy + 1) * 16], s, axis=1)
                                    [:, cx * 32:(cx + 1) * 32]) for s in range(-20, 21))
            moved += ok
    H.check(moved == (SIZE[1] // 16) * (SIZE[0] // 32), "all blocks are integer rolls (|shift| <= 20)")
    # Density statistics: the fraction of blocks that move follows Density.
    big = H.test_image(256, 256, seed=2)
    props = only(use_blocks=True, block_width=16, block_height=16)
    for dens in (0.0, 0.25, 0.6, 1.0):
        out = render(size=(256, 256), props=props, inputs=dict(STATIC, Shift=40.0, Density=dens),
                     images={"Image": big})
        changed = (out != big).any(axis=-1).reshape(16, 16, 16, 16).any(axis=(1, 3)).mean()
        # (a moved block changes unless its shift is 0: probability 1/81)
        H.check(abs(changed - dens) < 0.08, "density %.2f: %.3f of blocks changed" % (dens, changed))


@H.guard("jitter")
def test_jitter():
    img = H.test_image(*SIZE, seed=3)
    for seed in (0, 5):
        for jitter, dens in ((6.0, 0.25), (30.0, 1.0), (1.0, 0.5)):
            out = render(props=only(use_jitter=True, jitter_density=dens),
                         inputs=dict(STATIC, Seed=seed, Jitter=jitter), images={"Image": img})
            ref = ref_glitch(img, 0, seed, 0, 0, 1, 1, jitter, dens, 0, 0, blocks=False)
            H.check(np.array_equal(out, ref), "CPU seed %d jitter %g density %g: exactly the "
                    "reference rows" % (seed, jitter, dens))
    out = render(props=only(use_jitter=True, jitter_density=1.0), inputs=dict(STATIC, Jitter=6.0),
                 images={"Image": img})
    ok = all(any(np.array_equal(out[y], np.roll(img[y], s, axis=0)) for s in range(-6, 7))
             for y in range(SIZE[1]))
    H.check(ok, "every jittered row is a horizontal roll of its source row by <= 6 px")


@H.guard("split")
def test_split():
    img = H.test_image(*SIZE, seed=4, alpha=True)
    w, h = SIZE
    for angle, (dx, dy) in ((0.0, (6, 0)), (math.pi / 2, (0, 6)), (math.pi / 4, (4, 4))):
        out = render(props=only(use_split=True, split_angle=angle, split_flicker=0.0),
                     inputs=dict(STATIC, Split=6.0), images={"Image": img})
        ys, xs = np.mgrid[0:h, 0:w]
        r = img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1), 0]
        b = img[np.clip(ys - dy, 0, h - 1), np.clip(xs - dx, 0, w - 1), 2]
        H.check(np.array_equal(out[..., 0], r) and np.array_equal(out[..., 2], b) and
                np.array_equal(out[..., 1], img[..., 1]),
                "split angle %.0f: R at +(%d, %d), G in place, B at -(%d, %d)" % (
                    math.degrees(angle), dx, dy, dx, dy))
    out = render(props=only(use_split=True), inputs=dict(STATIC, Split=0.0), images={"Image": img})
    H.check(np.array_equal(out, img), "Split 0 is the identity")
    # Flicker varies the split length with the glitch step (needs the evaluation context).
    if H.FEATURES.get("F2"):
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        lengths = set()
        for frame in range(1, 40, 3):
            out = render(props=only(use_split=True, split_flicker=1.0),
                         inputs={"Speed": fps, "Split": 12.0}, images={"Image": img}, frame=frame)
            lengths.add(int(np.argmax([np.array_equal(out[..., 0][:, :w - 13],
                                                      img[..., 0][:, k:w - 13 + k])
                                       for k in range(0, 13)] + [False])))
        H.check(len(lengths) > 3, "flickering split length takes several values: %s" % sorted(lengths))


@H.guard("crush")
def test_crush():
    img = H.test_image(*SIZE, seed=5, alpha=True)
    for bits in (1, 2, 4, 8):
        out = render(props=only(use_crush=True, crush_bits=bits), inputs=STATIC,
                     images={"Image": img})
        n = (1 << bits) - 1
        ref = img.copy()
        ref[..., :3] = np.floor(np.clip(img[..., :3], 0, 1) * n + 0.5) / n
        H.compare("%d bits: CPU == independent rounding" % bits, out, ref, 1e-6)
        for c in range(3):
            H.check(len(np.unique(out[..., c])) <= n + 1, "%d bits: <= %d levels in channel %d"
                    % (bits, n + 1, c))
        H.check(np.array_equal(out[..., 3], img[..., 3]), "alpha untouched")


@H.guard("combined")
def test_combined_reference():
    img = H.test_image(*SIZE, seed=6, alpha=True)
    props = dict(use_split=True, use_blocks=True, use_jitter=True, use_crush=True, crush_bits=5,
                 split_flicker=0.0, split_angle=0.3, block_width=24, block_height=12,
                 jitter_density=0.3)
    inputs = dict(STATIC, Seed=9, Split=5.0, Shift=25.0, Density=0.5, Jitter=7.0)
    out = render(props=props, inputs=inputs, images={"Image": img})
    dx, dy = int(round(5 * math.cos(0.3))), int(round(5 * math.sin(0.3)))
    ref = ref_glitch(img, 0, 9, 25.0, 0.5, 24, 12, 7.0, 0.3, dx, dy, crush_bits=5)
    H.compare("all effects: CPU == independent reference", out, ref, 1e-6)
    fac = render(props=props, inputs=dict(inputs, Fac=0.3), images={"Image": img})
    H.compare("Fac 0.3 mixes with the input", fac, img * 0.7 + ref * 0.3, 1e-6)
    off = render(props=props, inputs=dict(inputs, Fac=0.0), images={"Image": img})
    H.check(np.array_equal(off, img), "Fac 0 returns the input exactly")
    off = render(props=OFF, inputs=inputs, images={"Image": img})
    H.check(np.array_equal(off, img), "all effects off is the identity")


@H.guard("parity")
def test_parity():
    # Block / row decisions and integer shifts are exact on both backends: pixels are copied.
    for label, props, inputs in (
            ("blocks", only(use_blocks=True, block_width=20, block_height=9),
             dict(STATIC, Seed=3, Shift=33.0, Density=0.5)),
            ("blocks big shift", only(use_blocks=True, block_width=64, block_height=64),
             dict(STATIC, Seed=4, Shift=5000.0, Density=0.7)),
            ("jitter", only(use_jitter=True, jitter_density=0.6), dict(STATIC, Seed=8, Jitter=12.0)),
            ("split", only(use_split=True, split_angle=0.7), dict(STATIC, Split=9.0)),
            ("blocks + jitter + split", only(use_blocks=True, use_jitter=True, use_split=True),
             dict(STATIC, Seed=12)),
            ("seed 2**31 + 5", only(use_blocks=True, use_jitter=True),
             dict(STATIC, Seed=2 ** 31 - 1))):
        for ilabel, img in (("opaque", H.test_image(*SIZE, seed=7)),
                            ("alpha", H.test_image(*SIZE, seed=8, alpha=True))):
            kw = dict(props=props, inputs=inputs, images={"Image": img})
            cpu = render("CPU", **kw)
            gpu = render("GPU", **kw)
            H.compare("%s %s: GPU == CPU (exact)" % (label, ilabel), gpu, cpu, 0.0)
    # Crush and Fac: only float rounding.
    img = H.test_image(*SIZE, seed=9, alpha=True)
    for bits in (1, 3, 8, 16):
        kw = dict(props=dict(use_crush=True, crush_bits=bits), inputs=dict(STATIC, Fac=0.8),
                  images={"Image": img})
        cpu, gpu = render("CPU", **kw), render("GPU", **kw)
        # A value exactly on a rounding tie can land on the other level: a rare pixel.
        H.compare("crush %d bits: GPU vs CPU" % bits, gpu, cpu, 1e-5, max_frac=0.0005)
    if H.FEATURES.get("F2"):
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        for frame in (1, 2, 5, 13, 40):
            kw = dict(props={}, inputs={"Speed": fps / 2, "Seed": 3}, images={"Image": img},
                      frame=frame)
            cpu, gpu = render("CPU", **kw), render("GPU", **kw)
            H.compare("default settings frame %d: GPU vs CPU" % frame, gpu, cpu, 1e-6)


@H.guard("time")
def test_time_seed():
    img = H.test_image(*SIZE, seed=10)
    for dev in ("CPU", "GPU"):
        a = render(dev, inputs={"Seed": 1, "Speed": 0.0}, images={"Image": img})
        b = render(dev, inputs={"Seed": 1, "Speed": 0.0}, images={"Image": img})
        c = render(dev, inputs={"Seed": 2, "Speed": 0.0}, images={"Image": img})
        H.check(np.array_equal(a, b), "%s: same seed renders identically" % dev)
        H.check(not np.array_equal(a, c) and float(np.abs(a - c).mean()) > 1e-3,
                "%s: different seed changes the glitch" % dev)
        H.check(not np.array_equal(a, img), "%s: default settings glitch the image" % dev)
        if not H.FEATURES.get("F2"):
            H.note("F2 not available: time checks skipped")
            continue
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        f1 = render(dev, inputs={"Seed": 1, "Speed": fps}, images={"Image": img}, frame=3)
        f1b = render(dev, inputs={"Seed": 1, "Speed": fps}, images={"Image": img}, frame=3)
        f2 = render(dev, inputs={"Seed": 1, "Speed": fps}, images={"Image": img}, frame=4)
        H.check(np.array_equal(f1, f1b), "%s: same frame renders identically" % dev)
        H.check(not np.array_equal(f1, f2) and float(np.abs(f1 - f2).mean()) > 1e-3,
                "%s: a new glitch step each frame at Speed = fps" % dev)
        s1 = render(dev, inputs={"Seed": 1, "Speed": 0.0}, images={"Image": img}, frame=3)
        s2 = render(dev, inputs={"Seed": 1, "Speed": 0.0}, images={"Image": img}, frame=30)
        H.check(np.array_equal(s1, s2), "%s: Speed 0 is static over time" % dev)
        slow1 = render(dev, inputs={"Seed": 1, "Speed": fps / 4}, images={"Image": img}, frame=5)
        slow2 = render(dev, inputs={"Seed": 1, "Speed": fps / 4}, images={"Image": img}, frame=6)
        H.check(np.array_equal(slow1, slow2), "%s: Speed fps/4 holds a glitch for 4 frames" % dev)
        slow3 = render(dev, inputs={"Seed": 1, "Speed": fps / 4}, images={"Image": img}, frame=9)
        H.check(not np.array_equal(slow1, slow3), "%s: ... then changes" % dev)
        # The reference at the step the time maps to.
        step = math.floor(7 / fps * fps + 1e-6)
        out = render(dev, props=only(use_blocks=True, use_jitter=True),
                     inputs={"Seed": 1, "Speed": fps}, images={"Image": img}, frame=7)
        ref = ref_glitch(img, step, 1, 48.0, 0.2, 96, 24, 6.0, 0.25, 0, 0)
        H.check(np.array_equal(out, ref), "%s: frame 7 == reference at step %d" % (dev, step))


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        out = render(dev, (16, 8))
        H.check(out.shape == (8, 16, 4) and np.isfinite(out).all(), "%s: unlinked input renders" % dev)
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4), (17, 33)):
            img = H.test_image(*size, seed=11, alpha=True)
            for props, inputs in (({}, {}),
                                  ({"use_crush": True}, {"Shift": 1e6, "Jitter": 1e6, "Split": 1e6}),
                                  ({"block_width": 1, "block_height": 1}, {"Density": 1.0}),
                                  ({"block_width": 4096}, {"Seed": -5, "Speed": 1e9})):
                out = render(dev, size, props=props, inputs=inputs, images={"Image": img})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %dx%d %s %s: finite, right shape" % (dev, size[0], size[1], props, inputs))
        ext = np.array([[[1e4, -3.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 1.0],
                         [0.5, 0.5, 0.5, 1.0]]] * 4, F32)
        out = render(dev, (4, 4), props={"use_crush": True}, images={"Image": ext})
        H.check(np.isfinite(out).all(), "%s: extreme input finite" % dev)
        # Image smaller than the shifts: wrap stays inside the image (checked by value).
        img = H.test_image(8, 8, seed=2)
        out = render(dev, (8, 8), props=only(use_blocks=True, block_width=8, block_height=8),
                     inputs=dict(STATIC, Shift=100.0, Density=1.0), images={"Image": img})
        H.check(all(any(np.array_equal(out[y], np.roll(img[y], s, axis=0)) for s in range(8))
                    for y in range(8)), "%s: shift larger than the image wraps" % dev)
    import time
    img = H.test_image(1920, 1080, seed=7)
    t0 = time.time()
    render("CPU", (1920, 1080), props={"use_crush": True}, images={"Image": img})
    H.note("Glitch 1920x1080 CPU render, all effects (incl. IO): %.2fs" % (time.time() - t0))


for fn in (test_blocks, test_jitter, test_split, test_crush, test_combined_reference, test_parity,
           test_time_seed, test_robustness):
    fn()
H.finish()
