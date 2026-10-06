# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Palette Extract node: exact recovery of flat colours, quantized output only holds palette
colours (and the nearest one), swatch layout, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_palette_extract.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H
import ua_helpers as U

H.setup()

from compositor_lab.lib import np_color  # noqa: E402

NODE = "CompositorNodeLabPaletteExtract"
F32 = np.float32
NAMES = ["Color %d" % i for i in range(1, 9)]

# Flat colours are recovered to float32 rounding of the weighted mean (observed 0 error).
FLAT_ATOL = 1e-6
# The palette is computed by the same host code from bit-identical samples in both modes.
GPU_PAL_ATOL = 1e-6

COLOURS = np.array([
    [0.9, 0.1, 0.1], [0.1, 0.8, 0.2], [0.1, 0.2, 0.9], [0.95, 0.85, 0.1],
    [0.05, 0.05, 0.05], [0.6, 0.6, 0.6], [0.9, 0.4, 0.8], [0.1, 0.8, 0.8]], F32)


def bands(size, colours, weights=None, alpha=1.0):
    """Vertical bands of the given colours with widths proportional to `weights`
    (default: descending, so the dominant colour is well defined). Returns the image."""
    w, h = size
    n = len(colours)
    weights = np.asarray(weights if weights is not None else np.arange(n, 0, -1), np.float64)
    edges = np.round(np.cumsum(weights) / weights.sum() * w).astype(int)
    img = np.zeros((h, w, 4), F32)
    x0 = 0
    for c, x1 in zip(colours, edges):
        img[:, x0:x1, :3] = np.asarray(c, F32) * alpha
        img[:, x0:x1, 3] = alpha
        x0 = x1
    return img


def palette(device, img, count, props=None, size=None):
    p = dict(count=count)
    p.update(props or {})
    size = size or (img.shape[1], img.shape[0])
    return [U.single(NODE, device, size, n, props=p, images={"Image": img}) for n in NAMES]


def render(device, img, out, props=None, size=None):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, device, size, props=props, images={"Image": img}, out_socket=out)


def match_set(got, want, atol):
    """Every wanted colour has a (distinct) match among `got`."""
    got = [np.asarray(g[:3], np.float64) for g in got]
    used = set()
    for wc in want:
        d = [(np.abs(g - wc).max(), i) for i, g in enumerate(got) if i not in used]
        if not d or min(d)[0] > atol:
            return False
        used.add(min(d)[1])
    return True


@H.guard("flat colours")
def test_flat():
    for dev in ("CPU", "GPU"):
        for k in (1, 2, 3, 5, 8):
            for size in ((96, 64), (130, 40), (200, 17)):
                img = bands(size, COLOURS[:k])
                pal = palette(dev, img, k)
                H.check(match_set(pal[:k], COLOURS[:k], FLAT_ATOL),
                        "%s k=%d %dx%d: flat colours recovered exactly" % (dev, k, size[0], size[1]))
                # Largest band first (widths are descending).
                H.check(np.allclose(pal[0][:3], COLOURS[0], atol=FLAT_ATOL), "%s k=%d: Color 1 is the dominant colour" % (dev, k))
                H.check(all(np.allclose(pal[i][:3], COLOURS[i], atol=FLAT_ATOL) for i in range(k)),
                        "%s k=%d: sorted by population" % (dev, k))
                H.check(all(np.allclose(p, [0, 0, 0, 1]) for p in pal[k:]) and all(p[3] == 1.0 for p in pal[:k]),
                        "%s k=%d: unused outputs are black, alpha 1" % (dev, k))
        # Fewer distinct colours than requested: the extra outputs stay black.
        img = bands((96, 64), COLOURS[:3])
        pal = palette(dev, img, 6)
        H.check(match_set(pal[:3], COLOURS[:3], FLAT_ATOL) and all(np.allclose(p[:3], 0) for p in pal[3:]),
                "%s: 3 colours with Colors=6 -> 3 entries + black" % dev)
        # Both colour spaces.
        for space in ("OKLAB", "LINEAR"):
            pal = palette(dev, bands((96, 64), COLOURS[:5]), 5, props={"space": space})
            H.check(match_set(pal[:5], COLOURS[:5], FLAT_ATOL), "%s %s: flat colours recovered" % (dev, space))


@H.guard("quantized and swatch")
def test_outputs():
    for dev in ("CPU", "GPU"):
        for k in (2, 4, 7):
            flat = bands((96, 64), COLOURS[:k])
            q = render(dev, flat, "Quantized", props={"count": k})
            H.compare("%s k=%d: quantizing a flat-colour image changes nothing" % (dev, k), q, flat, FLAT_ATOL)
            pal = np.array(palette(dev, flat, k)[:k])[:, :3]
            sw = render(dev, flat, "Swatch", props={"count": k})
            w = 96
            for i in range(w):
                bar = min(i * k // w, k - 1)
                if not np.allclose(sw[:, i, :3], pal[bar], atol=FLAT_ATOL) or not np.all(sw[:, i, 3] == 1.0):
                    H.report(False, "%s k=%d: swatch column %d != palette %d" % (dev, k, i, bar))
                    break
            else:
                H.report(True, "%s k=%d: swatch is %d equal bars of the palette colours, alpha 1" % (dev, k, k))
        # Noisy natural image: Quantized only holds palette colours, and each is the nearest.
        img = H.test_image(96, 64, seed=21)
        for space in ("OKLAB", "LINEAR"):
            props = {"count": 5, "space": space}
            pal = np.array(palette(dev, img, 5, props={"space": space})[:5])[:, :3].astype(np.float32)
            q = render(dev, img, "Quantized", props=props)
            colours = np.unique(q[..., :3].reshape(-1, 3), axis=0)
            ok = all(any(np.allclose(c, p, atol=1e-6) for p in pal) for c in colours)
            H.check(ok and len(colours) <= 5, "%s %s: Quantized holds only palette colours (%d distinct)" % (dev, space, len(colours)))
            H.check(np.all(q[..., 3] == 1.0), "%s %s: opaque input stays opaque" % (dev, space))
            sp = (lambda c: np_color.linear_to_oklab(c)) if space == "OKLAB" else (lambda c: c)
            d = ((sp(img[..., :3])[:, :, None, :] - sp(pal)[None, None, :, :]) ** 2).sum(-1)
            chosen = np.array([[np.argmin(np.abs(pal - q[y, x, :3]).sum(1)) for x in range(96)] for y in range(64)])
            dchosen = np.take_along_axis(d, chosen[..., None], axis=2)[..., 0]
            viol = float((dchosen - d.min(axis=2)).max())
            H.check(viol <= 1e-5, "%s %s: every pixel maps to its nearest palette colour (worst excess %.3g)" % (dev, space, viol))
        # Alpha: semi transparent input keeps its alpha, colour is the palette colour premultiplied.
        flat = bands((96, 64), COLOURS[:3], alpha=0.5)
        q = render(dev, flat, "Quantized", props={"count": 3})
        H.compare("%s: premultiplied alpha preserved by Quantized" % dev, q, flat, 1e-6)
        pal = palette(dev, flat, 3)
        H.check(match_set(pal, COLOURS[:3], FLAT_ATOL), "%s: palette of a half-transparent image is the straight colour" % dev)


@H.guard("noise and clusters")
def test_clusters():
    rng = np.random.default_rng(5)
    for dev in ("CPU", "GPU"):
        base = bands((120, 80), COLOURS[:4], weights=[4, 3, 2, 1])
        noisy = base.copy()
        noisy[..., :3] = np.clip(noisy[..., :3] + rng.normal(0, 0.01, noisy[..., :3].shape), 0, 1)
        noisy = noisy.astype(F32)
        pal = palette(dev, noisy, 4)
        H.check(match_set(pal[:4], COLOURS[:4], 0.01), "%s: noisy clusters recovered within 0.01" % dev)
        # The result does not depend on the seed for well separated clusters.
        pal2 = palette(dev, noisy, 4, props={"seed": 123})
        H.check(all(np.allclose(a[:3], b[:3], atol=1e-6) for a, b in zip(pal, pal2)), "%s: seed independent for separated clusters" % dev)
        # Transparent pixels do not take part.
        half = bands((100, 40), COLOURS[:1])
        half[:, 50:] = 0.0
        pal = palette(dev, half, 3)
        H.check(np.allclose(pal[0][:3], COLOURS[0], atol=FLAT_ATOL) and all(np.allclose(p[:3], 0) for p in pal[1:3]),
                "%s: transparent pixels are ignored" % dev)
        q = render(dev, half, "Quantized", props={"count": 3})
        H.check(np.all(q[:, 50:] == 0.0), "%s: transparent pixels stay transparent in Quantized" % dev)
        # More colours -> less quantization error.
        img = H.test_image(96, 64, seed=33)
        errs = []
        for k in (1, 2, 4, 8):
            q = render(dev, img, "Quantized", props={"count": k})
            errs.append(float(np.abs(q[..., :3] - img[..., :3]).mean()))
        H.check(all(errs[i] > errs[i + 1] for i in range(3)), "%s: error falls with the number of colours %s" % (dev, np.round(errs, 4)))


@H.guard("cpu vs gpu")
def test_gpu():
    imgs = {
        "natural 96x64": H.test_image(96, 64, seed=1),
        "alpha 130x70": H.test_image(130, 70, seed=2, alpha=True),
        "hdr 33x17": H.test_image(33, 17, seed=3, hdr=True),
        "big 300x200": H.test_image(300, 200, seed=4),
        "tiny 4x4": H.test_image(4, 4, seed=5),
    }
    for name, img in imgs.items():
        for props in ({"count": 5}, {"count": 8, "space": "LINEAR"}, {"count": 3, "seed": 7}):
            cpu = palette("CPU", img, **{"count": props["count"], "props": {k: v for k, v in props.items() if k != "count"}})
            gpu = palette("GPU", img, **{"count": props["count"], "props": {k: v for k, v in props.items() if k != "count"}})
            H.compare("%s %s: palette GPU vs CPU" % (name, props), np.array(gpu)[None], np.array(cpu)[None], GPU_PAL_ATOL, quiet=True)
            for out in ("Swatch", "Quantized"):
                c = render("CPU", img, out, props=props)
                g = render("GPU", img, out, props=props)
                # Observed: bit-identical (0 difference) in every case. A near distance tie
                # could in principle pick another entry on the GPU (cbrt differs by an ulp).
                H.compare("%s %s %s: GPU vs CPU" % (name, props, out), g, c, 1e-6,
                          quiet=True)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        # Unlinked input (default black).
        pal = [U.single(NODE, dev, (16, 12), n) for n in NAMES[:3]]
        H.check(all(np.allclose(p, [0, 0, 0, 1]) for p in pal), "%s: unlinked input: black palette" % dev)
        out = H.render_generator(NODE, dev, (16, 12), out_socket="Quantized")
        H.check(np.isfinite(out).all() and np.allclose(out[..., :3], 0.0), "%s: unlinked input: Quantized" % dev)
        out = H.render_generator(NODE, dev, (16, 12), out_socket="Swatch")
        H.check(np.isfinite(out).all() and out.shape == (12, 16, 4), "%s: unlinked input: Swatch" % dev)
        # Fully transparent image: empty palette.
        z = np.zeros((20, 20, 4), F32)
        pal = palette(dev, z, 4)
        H.check(all(np.allclose(p, [0, 0, 0, 1]) for p in pal), "%s: transparent image: black palette" % dev)
        for out in ("Swatch", "Quantized"):
            r = render(dev, z, out)
            H.check(np.isfinite(r).all() and r.shape == (20, 20, 4), "%s: transparent image: %s finite" % (dev, out))
        # Sizes.
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17), (333, 187)):
            img = H.test_image(size[0], size[1], seed=2)
            for out in ("Swatch", "Quantized"):
                r = render(dev, img, out)
                H.check(r.shape == (size[1], size[0], 4) and np.isfinite(r).all(),
                        "%s %dx%d %s: shape / finite" % (dev, size[0], size[1], out))
        # Colours == 1: the (alpha weighted) mean colour.
        img = H.test_image(40, 30, seed=9)
        p = U.single(NODE, dev, (40, 30), "Color 1", props={"count": 1, "space": "LINEAR"}, images={"Image": img})
        H.check(np.allclose(p[:3], img[..., :3].reshape(-1, 3).astype(np.float64).mean(0), atol=1e-6),
                "%s: one colour = mean colour (%s)" % (dev, p[:3]))
    # Large image: sampling keeps it fast.
    big = H.test_image(1920, 1080, seed=4)
    for dev in ("CPU", "GPU"):
        t0 = time.time()
        r = render(dev, big, "Quantized")
        H.note("1920x1080 Quantized on %s: %.2f s" % (dev, time.time() - t0))
        H.check(np.isfinite(r).all() and r.shape == (1080, 1920, 4), "%s 1920x1080" % dev)


for fn in (test_flat, test_outputs, test_clusters, test_gpu, test_robustness):
    fn()
H.finish()
