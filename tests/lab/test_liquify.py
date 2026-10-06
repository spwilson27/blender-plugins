# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Liquify node: identity / locality invariants, float64 reference, radial statistics of the
twirl, CPU vs GPU, robustness.
Blender -b --factory-startup --python-exit-code 1 --python test_liquify.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H
import imgfx_helpers as X

H.setup()

NODE = "CompositorNodeLabLiquify"
SIZE = (90, 60)
F32 = np.float32


def render(device, img, size=None, props=None, inputs=None):
    size = size or (img.shape[1], img.shape[0])
    return H.render_node(NODE, device, size, props, inputs, {"Image": img})


def ref_liquify(img, cx=0.5, cy=0.5, radius=0.4, strength=0.5, falloff=1.0, mode="TWIRL",
                aspect=True):
    """float64 reference, written from the documented formulas."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    bx, by = x + 0.5, y + 0.5
    c = (cx * w, cy * h)
    sx, sy = (1.0 / h, 1.0 / h) if aspect else (1.0 / w, 1.0 / h)
    dx, dy = (bx - c[0]) * sx, (by - c[1]) * sy
    t = np.hypot(dx, dy) / radius
    e0 = 1.0 - falloff
    u = np.clip((t - e0) / max(1.0 - e0, 1e-6), 0, 1)
    f = 1.0 - u * u * (3 - 2 * u)
    if mode == "TWIRL":
        ang = strength * np.pi * f
        sxd = np.cos(ang) * dx - np.sin(ang) * dy
        syd = np.sin(ang) * dx + np.cos(ang) * dy
    else:
        k = 1.0 - min(max(strength, -4.0), 1.0) * f
        sxd, syd = dx * k, dy * k
    inside = t < 1.0
    px = np.where(inside, c[0] + sxd / sx, bx)
    py = np.where(inside, c[1] + syd / sy, by)
    return X.ref_bilinear(img, px, py, "CLAMP")


def dist_px(size, cx=0.5, cy=0.5):
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    return np.hypot(x + 0.5 - cx * w, y + 0.5 - cy * h)


@H.guard("invariants")
def test_invariants():
    img = H.test_image(*SIZE, seed=5)
    for dev in ("CPU", "GPU"):
        for mode in ("TWIRL", "PINCH_BULGE"):
            for aspect in (True, False):
                props = {"mode": mode, "aspect_correct": aspect}
                out = render(dev, img, props=props, inputs={"Strength": 0.0})
                H.compare("%s %s aspect %s: strength 0 is identity" % (dev, mode, aspect), out, img, 0.0)
                out = render(dev, img, props=props, inputs={"Radius": 0.0, "Strength": 0.9})
                H.compare("%s %s aspect %s: radius 0 is identity" % (dev, mode, aspect), out, img, 0.0)
                out = render(dev, img, props=props, inputs={"Radius": 0.25, "Strength": 0.8,
                                                            "Center X": 0.3, "Center Y": 0.6, "Falloff": 0.5})
                if aspect:
                    outside = dist_px(SIZE, 0.3, 0.6) > 0.25 * SIZE[1] * 1.001 + 0.01
                else:
                    w, h = SIZE
                    y, x = np.mgrid[0:h, 0:w]
                    outside = np.hypot((x + 0.5) / w - 0.3, (y + 0.5) / h - 0.6) > 0.25 * 1.001
                same = np.all(out[outside] == img[outside])
                H.check(bool(same) and outside.mean() > 0.3 and (~outside).sum() > 50,
                        "%s %s aspect %s: pixels outside the radius are bit-identical (%d of them)" % (
                            dev, mode, aspect, outside.sum()))
                inner = ~outside
                H.check(float(np.abs(out[inner] - img[inner]).mean()) > 0.02,
                        "%s %s aspect %s: pixels inside the radius change" % (dev, mode, aspect))


@H.guard("references")
def test_references():
    img = X.smooth_image(*SIZE, seed=6)
    cases = [
        ("twirl default", {}, {}),
        ("twirl negative", {}, dict(strength=-1.3)),
        ("twirl falloff 0.3", {}, dict(falloff=0.3, cx=0.4, cy=0.7)),
        ("twirl no aspect", {"aspect_correct": False}, dict(aspect=False, radius=0.3)),
        ("bulge", {"mode": "PINCH_BULGE"}, dict(mode="PINCH_BULGE", strength=0.7)),
        ("pinch", {"mode": "PINCH_BULGE"}, dict(mode="PINCH_BULGE", strength=-0.9, falloff=0.6)),
        ("bulge no aspect", {"mode": "PINCH_BULGE", "aspect_correct": False},
         dict(mode="PINCH_BULGE", strength=0.5, aspect=False, radius=0.45)),
        ("strength clamp", {"mode": "PINCH_BULGE"}, dict(mode="PINCH_BULGE", strength=7.0)),
    ]
    for name, props, kw in cases:
        inputs = {"Strength": kw.get("strength", 0.5), "Radius": kw.get("radius", 0.4),
                  "Center X": kw.get("cx", 0.5), "Center Y": kw.get("cy", 0.5),
                  "Falloff": kw.get("falloff", 1.0)}
        ref = ref_liquify(img, **kw)
        for dev in ("CPU", "GPU"):
            out = render(dev, img, props=props, inputs=inputs)
            # float32 positions (cos / sin / sqrt): observed max 2e-6 on the smooth image.
            H.compare("%s %s: float64 reference" % (dev, name), out, ref, 2e-5)


@H.guard("statistics")
def test_statistics():
    n = 160
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    field = np.zeros((n, n))
    for px, py, a in zip(rng.uniform(0, n, 500), rng.uniform(0, n, 500), rng.uniform(0.3, 1.0, 500)):
        field += a * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * 3.0 ** 2))
    field = (field / field.max()).astype(F32)
    img = np.ones((n, n, 4), F32)
    img[..., :3] = field[..., None]
    r = np.hypot(x + 0.5 - n / 2, y + 0.5 - n / 2)
    bins = np.arange(0, 80, 8)
    idx = np.digitize(r, bins)
    for dev in ("CPU", "GPU"):
        out = render(dev, img, inputs={"Strength": 1.0, "Radius": 0.45})
        mass_in = np.array([field[idx == i].sum() for i in range(1, len(bins) + 1)])
        mass_out = np.array([out[..., 0][idx == i].sum() for i in range(1, len(bins) + 1)])
        rel = np.abs(mass_out - mass_in) / mass_in
        print("  %s twirl radial mass per 8 px ring, relative change: max %.3f" % (dev, rel.max()))
        H.check(rel.max() < 0.04, "%s: twirl preserves the radial mass distribution (max %.3f)" % (dev, rel.max()))
        H.check(float(np.abs(out[..., 0] - field).mean()) > 0.02, "%s: twirl really moves pixels" % dev)
    # A radially symmetric image is (almost) invariant under a twirl.
    ring = np.ones((n, n, 4), F32)
    ring[..., :3] = (0.5 + 0.4 * np.cos(r / 5.0))[..., None]
    for dev in ("CPU", "GPU"):
        out = render(dev, ring, inputs={"Strength": 1.2, "Radius": 0.45})
        H.check(float(np.abs(out - ring).max()) < 0.03, "%s: rings survive a twirl (max diff %.4f)" % (
            dev, float(np.abs(out - ring).max())))
    # Bulge enlarges, pinch shrinks a centred disc.
    disc, mask = X.disc_image(n, n, 15.0)
    for dev in ("CPU", "GPU"):
        bulge = render(dev, disc, props={"mode": "PINCH_BULGE"}, inputs={"Strength": 0.6, "Radius": 0.45})
        pinch = render(dev, disc, props={"mode": "PINCH_BULGE"}, inputs={"Strength": -0.6, "Radius": 0.45})
        a0, ab, ap = float(mask.sum()), float(bulge[..., 3].sum()), float(pinch[..., 3].sum())
        print("  %s disc area %.0f -> bulge %.0f, pinch %.0f" % (dev, a0, ab, ap))
        H.check(ab > 1.5 * a0 and ap < 0.7 * a0, "%s: bulge enlarges and pinch shrinks" % dev)


@H.guard("gpu parity")
def test_gpu():
    cases = [
        ("twirl", {}, {}),
        ("twirl off-centre falloff 0.5", {}, dict(Strength=-1.1, **{"Center X": 0.2, "Center Y": 0.8, "Falloff": 0.5})),
        ("twirl no aspect", {"aspect_correct": False}, dict(Radius=0.3)),
        ("bulge", {"mode": "PINCH_BULGE"}, dict(Strength=0.8)),
        ("pinch no aspect cubic", {"mode": "PINCH_BULGE", "aspect_correct": False, "interpolation": "CUBIC"},
         dict(Strength=-1.5, Radius=0.5)),
        ("twirl cubic", {"interpolation": "CUBIC"}, dict(Strength=2.0)),
    ]
    for kind, img in (("noisy", H.test_image(*SIZE, seed=2, alpha=True)),
                      ("smooth", X.smooth_image(*SIZE, seed=3, alpha=True))):
        for name, props, inputs in cases:
            c = render("CPU", img, props=props, inputs=inputs)
            g = render("GPU", img, props=props, inputs=inputs)
            # sin / cos / sqrt differ by a few ulp between numpy and the GPU: sample positions move
            # by ~1e-5 px, which a noisy (pixel-to-pixel contrast ~0.4) image turns into ~1.4e-5
            # (observed max); smooth images agree to 1.4e-6.
            H.compare("%s %s: GPU vs CPU" % (kind, name), g, c, 5e-5)
    # Hard edge (Falloff 0): the t < 1 test may flip for the odd pixel.
    img = X.smooth_image(*SIZE, seed=4)
    c = render("CPU", img, inputs={"Falloff": 0.0})
    g = render("GPU", img, inputs={"Falloff": 0.0})
    H.compare("falloff 0 (hard edge): GPU vs CPU", g, c, 5e-5, max_frac=0.005)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17)):
            img = H.test_image(*size, seed=1)
            for mode in ("TWIRL", "PINCH_BULGE"):
                out = render(dev, img, props={"mode": mode}, inputs={"Strength": 1.0, "Radius": 1.0})
                H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                        "%s %dx%d %s: shape / finite" % (dev, size[0], size[1], mode))
        img = H.test_image(24, 16, seed=1)
        for name, inputs in (("huge strength", {"Strength": 1e9}), ("negative radius", {"Radius": -1.0}),
                             ("huge radius", {"Radius": 1e6}), ("centre far away", {"Center X": 1e6, "Center Y": -1e6}),
                             ("falloff out of range", {"Falloff": 7.0}), ("falloff negative", {"Falloff": -3.0}),
                             ("strength -50 bulge", {"Strength": -50.0})):
            for mode in ("TWIRL", "PINCH_BULGE"):
                out = render(dev, img, props={"mode": mode}, inputs=inputs)
                H.check(np.isfinite(out).all(), "%s %s %s: finite" % (dev, name, mode))
        out = H.render_node(NODE, dev, (16, 8))
        H.check(np.allclose(out, (0.5, 0.5, 0.5, 1.0), atol=1e-6), "%s: unlinked image" % dev)


for fn in (test_invariants, test_references, test_statistics, test_gpu, test_robustness):
    fn()
H.finish()
