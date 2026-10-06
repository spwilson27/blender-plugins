# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Line Sort node: lines are permuted, keys come out sorted, bands, stable ties, exact CPU vs GPU
agreement (integer keys and ranks), robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_line_sort.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.nodes import line_sort as LS  # noqa: E402

NODE = "CompositorNodeLabLineSort"
SIZE = (64, 48)
F32 = np.float32
STATS = list(LS._STAT_INDEX)


def run(dev, img, size=None, **props):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, dev, size, props=props, images={"Image": img})


def lines_of(img, vertical):
    return np.swapaxes(img, 0, 1) if vertical else img


def key_of(img, stat, vertical):
    return LS.line_keys(img, LS._STAT_INDEX[stat], vertical)


def line_multiset_equal(a, b):
    """Same set of lines (as sorted byte strings)."""
    sa = sorted(bytes(r) for r in np.ascontiguousarray(a).reshape(a.shape[0], -1))
    sb = sorted(bytes(r) for r in np.ascontiguousarray(b).reshape(b.shape[0], -1))
    return sa == sb


@H.guard("behaviour")
def test_behaviour():
    img = H.test_image(*SIZE, seed=7, alpha=True)
    for vertical in (False, True):
        d = 'COLUMNS' if vertical else 'ROWS'
        for stat in STATS:
            for desc in (False, True):
                out = run("CPU", img, direction=d, statistic=stat, descending=desc)
                L_in, L_out = lines_of(img, vertical), lines_of(out, vertical)
                H.check(line_multiset_equal(L_in, L_out), "%s %s desc=%s: output is a permutation of the lines" % (d, stat, desc))
                k = key_of(out, stat, vertical).astype(np.int64)   # variance keys: float bits, monotone
                dk = np.diff(k)
                H.check((dk <= 0).all() if desc else (dk >= 0).all(),
                        "%s %s desc=%s: keys of the output lines are sorted" % (d, stat, desc))
    # Independent float64 statistics: sorted by the real mean luminance (quantisation = 1/1024
    # per pixel, so only lines whose means differ by < 1/1024 may swap).
    out = run("CPU", img, statistic='LUMINANCE')
    luma = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).astype(np.float64).mean(axis=1)
    H.check((np.diff(luma) > -2.0 / 1024).all(), "rows sorted by their true mean luminance (to 1/1024)")
    for stat, ch in (('RED', 0), ('GREEN', 1), ('BLUE', 2), ('ALPHA', 3)):
        out = run("CPU", img, statistic=stat)
        m = out[..., ch].astype(np.float64).sum(axis=1)
        H.check((np.diff(m) > -64 / 1024.0).all(), "%s: sorted by the channel sum" % stat)
    out = run("CPU", img, statistic='VARIANCE')
    lu = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).astype(np.float64)
    var = lu.var(axis=1)
    H.check(np.mean(np.diff(var) >= -1e-4) > 0.99, "VARIANCE: sorted by the true variance (float64)")
    out = run("CPU", img, statistic='HUE')
    r, g, b = [out[..., i].astype(np.float64) for i in range(3)]
    mx, mn = np.maximum(r, np.maximum(g, b)), np.minimum(r, np.minimum(g, b))
    delta = np.where(mx > mn, mx - mn, 1)
    hue = np.where(mx == r, ((g - b) / delta) % 6, np.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4)) / 6
    hue = np.where(mx > mn, hue, 0) % 1.0
    H.check((np.diff(hue.mean(axis=1)) > -2.0 / 1024).all(), "HUE: sorted by mean hue")

    # Descending is the reverse for distinct keys.
    asc = run("CPU", img, statistic='LUMINANCE')
    desc = run("CPU", img, statistic='LUMINANCE', descending=True)
    H.check(np.array_equal(asc[::-1], desc), "descending == reversed ascending (distinct keys)")

    # Rows on a known image: row i has constant value perm[i].
    perm = np.random.default_rng(1).permutation(20)
    known = np.zeros((20, 12, 4), F32)
    known[..., :3] = (perm / 20.0)[:, None, None]
    known[..., 3] = 1.0
    out = run("CPU", known, statistic='LUMINANCE')
    H.check(np.allclose(out[:, 0, 0], np.arange(20) / 20.0, atol=1e-6), "known rows come out ordered")

    # Stable ties: equal keys keep their original order.
    tie = np.zeros((16, 8, 4), F32)
    tie[..., 3] = 1.0
    tie[..., 0] = np.arange(16)[:, None] / 16.0          # distinguishes rows in red only
    tie[..., 1] = 0.25
    tie[:, :, 2] = 0.0
    # Sort by BLUE (all zero): all keys tie -> identity; sort by GREEN likewise.
    for stat in ('BLUE', 'GREEN', 'ALPHA'):
        for desc in (False, True):
            out = run("CPU", tie, statistic=stat, descending=desc)
            H.check(np.array_equal(out, tie), "all keys tie (%s, desc=%s): stable, nothing moves" % (stat, desc))
    two = tie.copy()
    two[::2, :, 1] = 0.5
    out = run("CPU", two, statistic='GREEN')
    exp = np.concatenate([two[1::2], two[::2]])
    H.check(np.array_equal(out, exp), "two key groups: each keeps its original order")
    out = run("CPU", two, statistic='GREEN', descending=True)
    H.check(np.array_equal(out, np.concatenate([two[::2], two[1::2]])), "descending ties keep original order too")

    # Bands.
    for band in (1, 5, 16, 100):
        for vertical in (False, True):
            d = 'COLUMNS' if vertical else 'ROWS'
            out = run("CPU", img, direction=d, statistic='LUMINANCE', band_size=band)
            L_in, L_out = lines_of(img, vertical), lines_of(out, vertical)
            n = L_in.shape[0]
            ok = True
            for s in range(0, n, band):
                ok = ok and line_multiset_equal(L_in[s:s + band], L_out[s:s + band])
            H.check(ok, "%s band %d: lines stay inside their band" % (d, band))
            k = key_of(out, 'LUMINANCE', vertical).astype(np.int64)
            sorted_in_band = all((np.diff(k[s:s + band]) >= 0).all() for s in range(0, n, band))
            H.check(sorted_in_band, "%s band %d: sorted within each band" % (d, band))
    H.check(np.array_equal(run("CPU", img, band_size=1), img), "band size 1: identity")
    out = run("CPU", img, band_size=SIZE[1], statistic='LUMINANCE')
    H.check(np.array_equal(out, run("CPU", img, band_size=0, statistic='LUMINANCE')), "band = height equals no band")
    # Sorting twice changes nothing; sorting is a pure permutation on HDR data too.
    hdr = H.test_image(*SIZE, seed=2, hdr=True)
    out = run("CPU", hdr, statistic='SATURATION')
    H.check(line_multiset_equal(hdr, out), "HDR input: lines permuted unchanged")
    H.check(np.array_equal(run("CPU", out, statistic='SATURATION'), out), "sorting a sorted image keeps it")


@H.guard("parity")
def test_parity():
    from compositor_lab.nodes.line_sort import CompositorNodeLabLineSort as cls
    calls = {"gpu": 0}
    orig = cls.gpu

    def counting_gpu(self, *a, **k):
        calls["gpu"] += 1
        return orig(self, *a, **k)

    cls.gpu = counting_gpu
    images = [("alpha", H.test_image(*SIZE, seed=11, alpha=True)),
              ("hdr", H.test_image(*SIZE, seed=12, hdr=True))]
    # Many equal keys: quantised-flat image plus duplicated rows / columns.
    flat = H.test_image(*SIZE, seed=13)
    flat = np.round(flat * 4) / 4
    flat[10:20] = flat[10]
    flat[:, 5:9] = flat[:, 5:6]
    images.append(("ties", flat.astype(F32)))
    neg = H.test_image(*SIZE, seed=14, hdr=True) - 0.6
    neg[..., 3] = 1.0
    images.append(("negative", neg.astype(F32)))
    for label, img in images:
        n = 0
        for d in ('ROWS', 'COLUMNS'):
            for stat in STATS:
                for desc in (False, True):
                    for band in (0, 7):
                        props = dict(direction=d, statistic=stat, descending=desc, band_size=band)
                        cpu = run("CPU", img, **props)
                        gpu = run("GPU", img, **props)
                        H.compare("%s %s: GPU == CPU" % (label, props), gpu, cpu, 0.0, quiet=True)
                        n += 1
        print("  %s: %d cases compared (exact)" % (label, n))
    H.check(calls["gpu"] > 0, "the GPU path was really exercised (%d calls)" % calls["gpu"])
    cls.gpu = orig
    for size in ((4, 4), (5, 9), (31, 17), (4, 40), (50, 4), (97, 61), (300, 5)):
        img = H.test_image(*size, seed=5, alpha=True)
        for d in ('ROWS', 'COLUMNS'):
            for stat in ('LUMINANCE', 'VARIANCE', 'HUE'):
                props = dict(direction=d, statistic=stat, band_size=3)
                H.compare("%dx%d %s: GPU == CPU" % (size[0], size[1], props), run("GPU", img, **props),
                          run("CPU", img, **props), 0.0, quiet=True)


@H.guard("keys")
def test_keys():
    """The CPU keys themselves against a plain float64 evaluation (guards the numpy mirror)."""
    img = H.test_image(40, 30, seed=21, hdr=True)
    q = lambda x: np.floor(np.clip(x, 0, 4) * 1024 + 0.5)
    for stat, f in (('RED', lambda c: c[..., 0]), ('GREEN', lambda c: c[..., 1]), ('BLUE', lambda c: c[..., 2]),
                    ('ALPHA', lambda c: c[..., 3])):
        k = key_of(img, stat, False).astype(np.int64)
        ref = q(f(img).astype(np.float64)).sum(axis=1).astype(np.int64)
        H.check(np.array_equal(k, ref), "key %s == exact integer sum" % stat)
    k = key_of(img, 'LUMINANCE', False).astype(np.int64)
    ref = q((0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float64)).sum(axis=1)
    H.check(np.abs(k - ref).max() <= 40 * 1, "key LUMINANCE within 1 quantum per ~pixel of float64 (%d)" % np.abs(k - ref).max())
    k = key_of(img, 'VARIANCE', False).view(np.float32)
    lu = np.clip((0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float64), 0, 4)
    ref = (q(lu) / 1024).var(axis=1)
    H.check(np.allclose(k, ref * 1024 * 1024, rtol=2e-3, atol=1.0),
            "key VARIANCE == variance of the quantised luminance (units of 1/1024^2): max rel err %.2g"
            % np.max(np.abs(k - ref * 1024 * 1024) / (ref * 1024 * 1024 + 1)))


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for d in ('ROWS', 'COLUMNS'):
            out = H.render_node(NODE, dev, (16, 12), props={"direction": d})
            H.check(np.allclose(out, [0.5, 0.5, 0.5, 1.0]) and out.shape == (12, 16, 4),
                    "%s %s: unlinked -> default grey" % (dev, d))
            out = H.render_node(NODE, dev, (16, 12), props={"direction": d}, inputs={"Image": (0.1, 0.2, 0.3, 0.4)})
            H.check(np.allclose(out, [0.1, 0.2, 0.3, 0.4], atol=1e-6), "%s %s: colour default" % (dev, d))
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4)):
            img = H.test_image(*size, seed=2, alpha=True)
            for d in ('ROWS', 'COLUMNS'):
                for band in (0, 3):
                    out = H.render_node(NODE, dev, size, props={"direction": d, "band_size": band}, images={"Image": img})
                    H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all()
                            and line_multiset_equal(lines_of(out, d == 'COLUMNS'), lines_of(img, d == 'COLUMNS')),
                            "%s %s %dx%d band %d: right shape, permuted lines" % (dev, d, size[0], size[1], band))
        # Band larger than the image, and all-black / NaN-free extremes.
        blk = np.zeros((10, 10, 4), F32)
        blk[..., 3] = 1.0
        for stat in STATS:
            out = H.render_node(NODE, dev, (10, 10), props={"statistic": stat, "band_size": 50}, images={"Image": blk})
            H.check(np.array_equal(out, blk), "%s %s: black image unchanged" % (dev, stat))
        ext = np.full((8, 8, 4), 1e4, F32)
        ext[::2] = -5.0
        for stat in STATS:
            out = H.render_node(NODE, dev, (8, 8), props={"statistic": stat}, images={"Image": ext})
            H.check(np.isfinite(out).all() and line_multiset_equal(out, ext), "%s %s: extreme values" % (dev, stat))


@H.guard("perf")
def test_perf():
    big = H.test_image(1920, 1080, seed=1)
    for props in (dict(direction='ROWS'), dict(direction='COLUMNS', statistic='VARIANCE'),
                  dict(direction='ROWS', statistic='HUE', band_size=16)):
        res = {}
        for dev in ("CPU", "GPU"):
            t = time.time()
            res[dev] = H.render_node(NODE, dev, (1920, 1080), props=props, images={"Image": big})
            H.note("1920x1080 %s %s: %.2f s (incl. render and EXR roundtrip)" % (dev, props, time.time() - t))
        H.compare("1080p %s: GPU == CPU" % props, res["GPU"], res["CPU"], 0.0, quiet=True)


for fn in (test_behaviour, test_keys, test_parity, test_robustness, test_perf):
    fn()
H.finish()
