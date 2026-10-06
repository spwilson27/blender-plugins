# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Pixel Shuffle node: permutation invariants per mode, amount, seeds, animation, exact CPU vs GPU
agreement (integer maths, so bit-identical), robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_pixel_shuffle.py"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

NODE = "CompositorNodeLabPixelShuffle"
SIZE = (70, 46)
F32 = np.float32


def coord_image(w, h):
    """R = x, G = y (exact in float32), B = a hash-ish value, A = 1: every pixel is unique."""
    y, x = np.mgrid[0:h, 0:w]
    px = np.empty((h, w, 4), F32)
    px[..., 0] = x
    px[..., 1] = y
    px[..., 2] = (x * 7 + y * 13) % 5
    px[..., 3] = 1.0
    return px


def src_coords(out):
    return out[..., 0].astype(np.int64), out[..., 1].astype(np.int64)


def run(dev, img, size=None, frame=None, **props):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, dev, size, props=props, images={"Image": img}, frame=frame)


def check_permutation(name, out, w, h):
    sx, sy = src_coords(out)
    ok = (sx >= 0).all() and (sx < w).all() and (sy >= 0).all() and (sy < h).all()
    lin = sy * w + sx
    ok = ok and np.array_equal(np.sort(lin.ravel()), np.arange(w * h))
    H.check(ok, "%s: output is a permutation of the input pixels" % name)
    # Colour travels with the pixel (B channel is a function of the source coordinates).
    H.check(np.array_equal(out[..., 2], ((sx * 7 + sy * 13) % 5).astype(F32)) and (out[..., 3] == 1).all(),
            "%s: whole pixels are moved" % name)
    return sx, sy


@H.guard("block_pixels")
def test_block_pixels():
    w, h = SIZE
    img = coord_image(w, h)
    for n in (1, 2, 5, 8, 16, 64):
        out = run("CPU", img, mode='BLOCK_PIXELS', block_size=n, seed=3)
        sx, sy = check_permutation("block pixels N=%d" % n, out, w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        H.check(((sx // n) == (xx // n)).all() and ((sy // n) == (yy // n)).all(),
                "N=%d: pixels stay inside their block" % n)
        if n >= 2:
            H.check((sx != xx).mean() > 0.5, "N=%d: most pixels moved (%.2f)" % (n, (sx != xx).mean()))
    # Different blocks use different permutations; seeds differ; same seed reproduces.
    out = run("CPU", img, mode='BLOCK_PIXELS', block_size=8, seed=3)
    sx, sy = src_coords(out)
    loc = (sy % 8) * 8 + sx % 8
    b00, b10, b01 = loc[:8, :8], loc[:8, 8:16], loc[8:16, :8]
    H.check(not np.array_equal(b00, b10) and not np.array_equal(b00, b01), "each block has its own permutation")
    out2 = run("CPU", img, mode='BLOCK_PIXELS', block_size=8, seed=3)
    out3 = run("CPU", img, mode='BLOCK_PIXELS', block_size=8, seed=4)
    H.check(np.array_equal(out, out2), "same seed: identical output")
    H.check((out[..., 0] != out3[..., 0]).mean() > 0.5, "other seed: different output")
    # Uniformity: every local source position equally likely for a given destination (full blocks).
    big = coord_image(320, 160)
    o = run("CPU", big, mode='BLOCK_PIXELS', block_size=4, seed=1)
    sx, sy = src_coords(o)
    dst = (np.mgrid[0:160, 0:320][0] % 4) * 4 + np.mgrid[0:160, 0:320][1] % 4
    srcl = (sy % 4) * 4 + sx % 4
    for d in (0, 5, 15):
        hist = np.bincount(srcl[dst == d], minlength=16) / (dst == d).sum()
        others = np.delete(hist, d)
        # The affected pixels move along one random cycle, so no pixel keeps its place (hist[d] == 0).
        H.check(hist[d] == 0 and others.max() < 2.0 / 15 and others.min() > 0.3 / 15,
                "destination %d draws from the 15 other positions ~uniformly (min %.3f max %.3f, 1/15 = %.3f)"
                % (d, others.min(), others.max(), 1 / 15))
    # Amount.
    out0 = run("CPU", img, mode='BLOCK_PIXELS', block_size=8, amount=0.0, seed=2)
    H.check(np.array_equal(out0, img), "amount 0 is the identity")
    for amt in (0.25, 0.5, 0.75):
        o = run("CPU", img, mode='BLOCK_PIXELS', block_size=8, amount=amt, seed=2)
        sx, sy = check_permutation("amount %g" % amt, o, w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        moved = ((sx != xx) | (sy != yy))
        full = (np.arange(h) // 8 * 8 + 8 <= h)[:, None] & (np.arange(w) // 8 * 8 + 8 <= w)[None, :]
        # Full blocks: exactly round(amount * 64) pixels move.
        per_block = moved[:h // 8 * 8, :w // 8 * 8].reshape(h // 8, 8, w // 8, 8).sum(axis=(1, 3))
        H.check((per_block == round(amt * 64)).all(), "amount %g moves exactly %d of 64 pixels per block"
                % (amt, round(amt * 64)))


@H.guard("swap_pairs")
def test_swap_pairs():
    w, h = SIZE
    img = coord_image(w, h)
    yy, xx = np.mgrid[0:h, 0:w]
    for radius in (1, 3, 8, 20):
        out = run("CPU", img, mode='SWAP_PAIRS', radius=radius, seed=5)
        sx, sy = check_permutation("swap radius %d" % radius, out, w, h)
        H.check(np.abs(sx - xx).max() <= radius and np.abs(sy - yy).max() <= radius,
                "radius %d: displacement <= radius (max %d, %d)" % (radius, np.abs(sx - xx).max(), np.abs(sy - yy).max()))
        lin = sy * w + sx
        H.check(np.array_equal(lin.ravel()[lin.ravel()], np.arange(w * h)), "radius %d: one pass is an involution (pairs swap)" % radius)
        H.check((lin.ravel() != np.arange(w * h)).mean() > 0.8, "radius %d: nearly all pixels swapped" % radius)
    for amt in (0.0, 0.3, 0.6):
        o = run("CPU", img, mode='SWAP_PAIRS', radius=6, amount=amt, seed=5)
        sx, sy = check_permutation("swap amount %g" % amt, o, w, h)
        moved = ((sx != xx) | (sy != yy)).mean()
        H.check(abs(moved - amt) < 0.08, "swap amount %g: %.3f of the pixels swapped" % (amt, moved))
        if amt == 0.0:
            H.check(np.array_equal(o, img), "swap amount 0 is the identity")
    for it in (2, 4):
        o = run("CPU", img, mode='SWAP_PAIRS', radius=4, iterations=it, seed=5)
        sx, sy = check_permutation("swap iterations %d" % it, o, w, h)
        H.check(np.abs(sx - xx).max() <= 4 * it, "iterations %d: displacement <= radius * iterations" % it)


@H.guard("blocks")
def test_blocks():
    w, h = 70, 46
    img = coord_image(w, h)
    for n in (2, 5, 8, 16):
        out = run("CPU", img, mode='BLOCKS', block_size=n, seed=7)
        sx, sy = check_permutation("shuffle blocks N=%d" % n, out, w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        nbx, nby = w // n, h // n
        H.check((sx % n == xx % n).all() and (sy % n == yy % n).all() or
                ((xx >= nbx * n) | (yy >= nby * n)).any(),
                "N=%d: local position inside the block preserved" % n)
        inside = (xx < nbx * n) & (yy < nby * n)
        H.check((sx % n == xx % n)[inside].all() and (sy % n == yy % n)[inside].all(), "N=%d: blocks move intact" % n)
        H.check((sx == xx)[~inside].all() and (sy == yy)[~inside].all(), "N=%d: partial blocks at the edge stay" % n)
        bsrc = (sy // n) * nbx + sx // n
        bdst = (yy // n) * nbx + xx // n
        mp = np.full(nbx * nby, -1)
        mp[bdst[inside]] = bsrc[inside]
        H.check(np.array_equal(np.sort(mp), np.arange(nbx * nby)), "N=%d: block mapping is a permutation" % n)
        H.check((mp != np.arange(nbx * nby)).mean() > 0.8, "N=%d: blocks moved (%.2f)" % (n, (mp != np.arange(nbx * nby)).mean()))
    for amt in (0.5,):
        n = 5
        o = run("CPU", img, mode='BLOCKS', block_size=n, amount=amt, seed=1)
        sx, sy = check_permutation("blocks amount %g" % amt, o, w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        moved = ((sx != xx) | (sy != yy))[::n, ::n][:h // n, :w // n]
        H.check(moved.sum() == round(amt * (w // n) * (h // n)),
                "blocks amount %g moves exactly %d blocks" % (amt, round(amt * (w // n) * (h // n))))
    H.check(np.array_equal(run("CPU", img, mode='BLOCKS', block_size=5, amount=0.0), img), "blocks amount 0 identity")
    # Fewer than two blocks: identity.
    H.check(np.array_equal(run("CPU", img, mode='BLOCKS', block_size=40), img), "single block: identity")


@H.guard("animate")
def test_animate():
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    img = coord_image(40, 24)
    base = dict(mode='BLOCK_PIXELS', block_size=6, seed=11)
    rate = 10.0
    for frame in (0, 7, 30):
        expected_steps = int(math.floor(frame / fps * rate))
        frac = frame / fps * rate - expected_steps
        if min(frac, 1 - frac) < 1e-3:
            continue
        for dev in ("CPU", "GPU"):
            a = run(dev, img, frame=frame, animate=True, rate=rate, **base)
            b = run(dev, img, frame=frame, animate=False, **dict(base, seed=base["seed"] + expected_steps))
            H.check(np.array_equal(a, b), "%s frame %d: animate == seed + floor(time * rate) = +%d" % (dev, frame, expected_steps))
    a = run("CPU", img, frame=0, animate=True, rate=rate, **base)
    c = run("CPU", img, frame=30, animate=True, rate=rate, **base)
    d = run("CPU", img, frame=30, animate=False, **base)
    H.check(not np.array_equal(a, c), "animated output changes over time")
    H.check(np.array_equal(run("CPU", img, frame=0, animate=False, **base), d), "not animated: no change over time")


@H.guard("parity")
def test_parity():
    from compositor_lab.nodes.pixel_shuffle import CompositorNodeLabPixelShuffle as cls
    calls = {"gpu": 0}
    orig = cls.gpu

    def counting_gpu(self, *a, **k):
        calls["gpu"] += 1
        return orig(self, *a, **k)

    cls.gpu = counting_gpu
    cases = []
    for n in (1, 2, 3, 8, 13, 64):
        cases.append(dict(mode='BLOCK_PIXELS', block_size=n, seed=n))
        cases.append(dict(mode='BLOCKS', block_size=n, seed=n + 1))
    for r in (1, 2, 5, 17):
        cases.append(dict(mode='SWAP_PAIRS', radius=r, seed=r))
    cases += [
        dict(mode='BLOCK_PIXELS', block_size=8, amount=0.37, seed=-5),
        dict(mode='BLOCKS', block_size=4, amount=0.61, seed=123456789),
        dict(mode='SWAP_PAIRS', radius=7, amount=0.42, iterations=3, seed=99),
        dict(mode='SWAP_PAIRS', radius=3, iterations=8, seed=2 ** 31 - 1),
        dict(mode='BLOCK_PIXELS', block_size=7, amount=0.0),
        dict(mode='BLOCK_PIXELS', block_size=7, amount=1.0, seed=2 ** 31 - 1),
    ]
    for img_label, img in (("coords", coord_image(*SIZE)), ("hdr+alpha", H.test_image(*SIZE, seed=3, hdr=True)),
                           ("alpha", H.test_image(*SIZE, seed=4, alpha=True))):
        for props in cases:
            cpu = run("CPU", img, **props)
            gpu = run("GPU", img, **props)
            H.compare("%s %s: GPU == CPU (exact)" % (img_label, props), gpu, cpu, 0.0, quiet=True)
        print("  %s: %d cases compared" % (img_label, len(cases)))
    H.check(calls["gpu"] > 0, "the GPU path was really exercised (%d calls)" % calls["gpu"])
    cls.gpu = orig
    for size in ((4, 4), (5, 7), (31, 17), (4, 40), (50, 4), (97, 61)):
        img = coord_image(*size)
        for props in (dict(mode='BLOCK_PIXELS', block_size=3), dict(mode='BLOCKS', block_size=2),
                      dict(mode='SWAP_PAIRS', radius=2, iterations=2), dict(mode='BLOCK_PIXELS', block_size=100)):
            cpu = run("CPU", img, seed=8, **props)
            gpu = run("GPU", img, seed=8, **props)
            H.compare("%dx%d %s: GPU == CPU" % (size[0], size[1], props), gpu, cpu, 0.0, quiet=True)
            check_permutation("%dx%d %s" % (size[0], size[1], props), cpu, *size)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for mode in ('BLOCK_PIXELS', 'SWAP_PAIRS', 'BLOCKS'):
            out = H.render_node(NODE, dev, (16, 12), props={"mode": mode})
            H.check(np.allclose(out, [0.5, 0.5, 0.5, 1.0]) and out.shape == (12, 16, 4),
                    "%s %s: unlinked input -> default grey" % (dev, mode))
            out = H.render_node(NODE, dev, (16, 12), props={"mode": mode}, inputs={"Image": (0.1, 0.2, 0.3, 0.4)})
            H.check(np.allclose(out, [0.1, 0.2, 0.3, 0.4], atol=1e-6), "%s %s: colour default" % (dev, mode))
        for size in ((4, 4), (7, 5), (333, 187)):
            img = H.test_image(*size, seed=2)
            for mode in ('BLOCK_PIXELS', 'SWAP_PAIRS', 'BLOCKS'):
                out = H.render_node(NODE, dev, size, props={"mode": mode}, images={"Image": img})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %s %dx%d" % (dev, mode, size[0], size[1]))
                key = lambda a: a.reshape(-1, 4)[np.lexsort(a.reshape(-1, 4).T)]
                H.check(np.array_equal(key(out), key(img)), "%s %s %dx%d: multiset of pixels unchanged" % (dev, mode, size[0], size[1]))


@H.guard("perf")
def test_perf():
    big = H.test_image(1920, 1080, seed=1)
    for props in (dict(mode='BLOCK_PIXELS', block_size=8), dict(mode='BLOCKS', block_size=2),
                  dict(mode='SWAP_PAIRS', radius=8, iterations=2)):
        for dev in ("CPU", "GPU"):
            t = time.time()
            out = H.render_node(NODE, dev, (1920, 1080), props=props, images={"Image": big})
            H.note("1920x1080 %s %s: %.2f s (incl. render and EXR roundtrip)" % (dev, props, time.time() - t))
            H.check(np.isfinite(out).all(), "%s 1080p %s" % (dev, props))


for fn in (test_block_pixels, test_swap_pairs, test_blocks, test_animate, test_parity,
           test_robustness, test_perf):
    fn()
H.finish()
