# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Halftone node: dot geometry vs independent references, area coverage ~ darkness, cell
periodicity, CMYK inks, lines / cross-hatch, alpha, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_halftone.py"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

NODE = "CompositorNodeLabHalftone"
F32 = np.float32
SIZE = (96, 64)


def render(dev="CPU", size=SIZE, **kw):
    return H.render_node(NODE, dev, size, **kw)


def flat(w, h, rgb, alpha=1.0):
    img = np.empty((h, w, 4), F32)
    img[..., :3] = np.asarray(rgb, F32) * alpha
    img[..., 3] = alpha
    return img


def grey(v, w=192, h=192):
    return flat(w, h, (v, v, v))


def mono(**kw):
    p = {"mode": "MONO", "cell_size": 12.0}
    p.update(kw)
    return p


@H.guard("coverage")
def test_coverage():
    # Mono dots on a flat field: mean output = paper * (1 - d) + ink * d with paper 1, ink 0, so
    # the mean equals the input grey (linear) when the dot area matches the darkness.
    worst = 0.0
    for shape in ("ROUND", "SQUARE", "DIAMOND"):
        for angle in (0.0, math.radians(30.0), math.radians(45.0)):
            for v in (0.05, 0.2, 0.5, 0.8, 0.95):
                out = render(size=(192, 192), props=mono(shape=shape, angle=angle),
                             images={"Image": grey(v)})
                m = float(out[..., 0].mean())
                worst = max(worst, abs(m - v))
                H.check(abs(m - v) < 0.03, "%s angle %.0f v=%.2f: coverage-based mean %.4f"
                        % (shape, math.degrees(angle), v, m))
    H.note("mono dot coverage: worst |mean - darkness| = %.4f" % worst)
    # Coloured ink / paper: mean follows paper * (1 - d) + ink * d per channel.
    out = render(size=(192, 192), props=mono(paper_color=(0.9, 0.8, 0.1), ink_color=(0.1, 0.0, 0.5),
                                              angle=0.3), images={"Image": grey(0.4)})
    d = 0.6
    ref = np.array([0.9, 0.8, 0.1]) * (1 - d) + np.array([0.1, 0.0, 0.5]) * d
    H.check(np.allclose(out[..., :3].mean(axis=(0, 1)), ref, atol=0.03),
            "paper / ink colours: mean %s ~ %s" % (out[..., :3].mean(axis=(0, 1)), ref))
    # Lines: coverage = darkness as well.
    for angle in (0.0, math.radians(30.0), math.radians(45.0)):
        for v in (0.2, 0.5, 0.8):
            out = render(size=(192, 192), props={"mode": "LINES", "cell_size": 10.0,
                                                  "angle": angle}, images={"Image": grey(v)})
            m = float(out[..., 0].mean())
            H.check(abs(m - v) < 0.03, "lines angle %.0f v=%.2f: mean %.4f" % (
                math.degrees(angle), v, m))
    # Extremes: white paper only, black solid ink.
    for mode in ("MONO", "LINES", "CROSSHATCH", "CMYK"):
        out = render(size=(64, 64), props={"mode": mode}, images={"Image": grey(1.0, 64, 64)})
        H.check(np.allclose(out[..., :3], 1.0, atol=1e-6), "%s: white input -> paper only" % mode)
    for mode in ("MONO", "LINES"):
        out = render(size=(64, 64), props={"mode": mode}, images={"Image": grey(0.0, 64, 64)})
        H.check(np.allclose(out[..., :3], 0.0, atol=1e-6), "%s: black input -> solid ink" % mode)
    out = render(size=(64, 64), props={"mode": "CROSSHATCH"}, images={"Image": grey(0.0, 64, 64)})
    H.check(float(out[..., 0].mean()) < 0.08, "cross-hatch black input mean %.3f (dense hatching)"
            % float(out[..., 0].mean()))
    # Cross-hatch gets darker with the input.
    means = [float(render(size=(128, 128), props={"mode": "CROSSHATCH", "cell_size": 8.0},
                          images={"Image": grey(v, 128, 128)})[..., 0].mean())
             for v in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)]
    H.check(all(a > b for a, b in zip(means, means[1:])), "cross-hatch mean decreases: %s"
            % np.round(means, 3))


@H.guard("geometry")
def test_geometry():
    # Hard-edged (softness 0), angle 0, cell 16: compare with the exact geometry of the dot.
    cell = 16
    w = h = 128
    yy, xx = np.mgrid[0:h, 0:w]
    fx = (xx + 0.5) / cell % 1.0 - 0.5
    fy = (yy + 0.5) / cell % 1.0 - 0.5
    for d in (0.1, 0.3, 0.45):   # (not 0.5: diamond pixel centres would sit exactly on the edge)
        for shape, radius_of in (("ROUND", lambda d: math.sqrt(d / math.pi)),
                                 ("SQUARE", lambda d: math.sqrt(d) / 2),
                                 ("DIAMOND", lambda d: math.sqrt(d / 2))):
            dist = {"ROUND": np.hypot(fx, fy), "SQUARE": np.maximum(abs(fx), abs(fy)),
                    "DIAMOND": abs(fx) + abs(fy)}[shape]
            ink = dist < radius_of(d)
            out = render(size=(w, h), props=mono(shape=shape, cell_size=cell, angle=0.0,
                                                  softness=0.0),
                         images={"Image": grey(1 - d, w, h)})
            got = out[..., 0] < 0.5
            frac = float((got != ink).mean())
            H.check(frac < 0.01, "%s d=%.1f: hard dot matches the geometric shape (%.3f%% pixels "
                    "differ)" % (shape, d, frac * 100))
            # (pixel centres that sit exactly on the dot boundary get 0.5)
            nb = float(((out[..., 0] > 1e-4) & (out[..., 0] < 1 - 1e-4)).mean())
            H.check(nb < 0.01, "%s d=%.1f softness 0: %.3f%% non-binary pixels" % (shape, d, nb * 100))
    # Anti-aliasing: with softness 1 there are intermediate values at the dot edges.
    out = render(size=(w, h), props=mono(cell_size=cell, angle=0.0, softness=1.0),
                 images={"Image": grey(0.5, w, h)})
    mid = ((out[..., 0] > 0.02) & (out[..., 0] < 0.98)).mean()
    H.check(0.02 < mid < 0.4, "anti-aliased edge pixels: %.1f%%" % (mid * 100))


@H.guard("periodicity")
def test_periodicity():
    cell = 8
    for mode, shape in (("MONO", "ROUND"), ("MONO", "DIAMOND"), ("LINES", "ROUND"),
                        ("CMYK", "SQUARE")):   # (cross-hatch has 45 degree layers: not 8 px periodic)
        for dev in ("CPU", "GPU"):
            props = {"mode": mode, "shape": shape, "cell_size": float(cell), "angle": 0.0,
                     "angle_c": 0.0, "angle_m": 0.0, "angle_y": 0.0, "angle_k": 0.0}
            out = render(dev, (96, 64), props=props, images={"Image": grey(0.45, 96, 64)})
            H.check(np.allclose(out[:, cell:], out[:, :-cell], atol=1e-5) and
                    np.allclose(out[cell:], out[:-cell], atol=1e-5),
                    "%s %s %s: output repeats every %d px at angle 0" % (dev, mode, shape, cell))
    # At 90 degrees the screen is the same lattice rotated: still periodic in x and y.
    out = render(size=(96, 64), props=mono(cell_size=8.0, angle=math.pi / 2),
                 images={"Image": grey(0.45, 96, 64)})
    H.check(np.allclose(out[:, 8:], out[:, :-8], atol=1e-4), "90 degree screen is periodic")


@H.guard("cmyk")
def test_cmyk():
    big = (192, 192)
    # Flat cyan / magenta / yellow / black / white: pure inks reproduce the colour.
    for rgb in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0), (1, 1, 1)):
        out = render(size=(64, 64), props={"mode": "CMYK", "softness": 0.0, "cell_size": 8.0},
                     images={"Image": flat(64, 64, rgb)})
        H.check(np.allclose(out[..., :3], rgb, atol=1e-5), "CMYK flat %s reproduced" % (rgb,))
    # Mixed colours: area-averaged reproduction of (1-c)(1-k) = r within 0.05.
    for rgb in ((0.6, 0.3, 0.2), (0.2, 0.5, 0.8), (0.7, 0.7, 0.1), (0.4, 0.4, 0.4)):
        out = render(size=big, props={"mode": "CMYK", "cell_size": 9.0},
                     images={"Image": flat(*big, rgb)})
        m = out[..., :3].mean(axis=(0, 1))
        H.check(np.allclose(m, rgb, atol=0.05), "CMYK flat %s: mean %s" % (rgb, np.round(m, 3)))
    # Ink colour properties are used (a blue "cyan" ink on a cyan-only input).
    out = render(size=(64, 64), props={"mode": "CMYK", "softness": 0.0, "ink_c": (0.0, 0.0, 1.0)},
                 images={"Image": flat(64, 64, (0, 1, 1))})
    H.check(np.allclose(out[..., :3], (0, 0, 1), atol=1e-5), "cyan ink colour property is used")
    # Angles change the pattern.
    img = flat(*big, (0.3, 0.5, 0.6))
    a = render(size=big, props={"mode": "CMYK", "angle_c": 0.0}, images={"Image": img})
    b = render(size=big, props={"mode": "CMYK", "angle_c": 0.5}, images={"Image": img})
    H.check(float(np.abs(a - b).mean()) > 0.01, "cyan screen angle changes the pattern")


@H.guard("alpha")
def test_alpha():
    img = H.test_image(*SIZE, seed=3, alpha=True)
    for dev in ("CPU", "GPU"):
        for mode in ("CMYK", "MONO", "LINES", "CROSSHATCH"):
            out = render(dev, props={"mode": mode}, images={"Image": img})
            H.check(np.allclose(out[..., 3], img[..., 3], atol=1e-6),
                    "%s %s: alpha kept" % (dev, mode))
            H.check((out[..., :3] <= out[..., 3:] + 1e-5).all(),
                    "%s %s: premultiplied output" % (dev, mode))
        out = render(dev, props={"mode": "MONO"}, inputs={"Fac": 0.0}, images={"Image": img})
        H.check(np.array_equal(out, img), "%s: Fac 0 returns the input" % dev)


@H.guard("parity")
def test_parity():
    cases = []
    for mode in ("CMYK", "MONO"):
        for shape in ("ROUND", "SQUARE", "DIAMOND"):
            cases.append(("%s %s" % (mode, shape), {"mode": mode, "shape": shape}))
    cases += [("LINES", {"mode": "LINES", "cell_size": 7.0}),
              ("CROSSHATCH", {"mode": "CROSSHATCH", "cell_size": 9.0, "angle": 0.3}),
              ("hard edges", {"mode": "MONO", "softness": 0.0}),
              ("big soft", {"mode": "CMYK", "softness": 3.0, "cell_size": 20.0}),
              ("small cells", {"mode": "MONO", "cell_size": 2.5}),
              ("colours", {"mode": "CMYK", "paper_color": (0.9, 0.85, 0.7), "angle_k": 0.9,
                           "ink_c": (0.0, 0.5, 0.9)}),
              ("angle 0", {"mode": "MONO", "angle": 0.0, "cell_size": 8.0}),
              ("angle 90", {"mode": "LINES", "angle": math.pi / 2, "cell_size": 8.0})]
    big = (160, 120)
    imgs = (("opaque", H.test_image(*big, seed=4)),
            ("alpha", H.test_image(*big, seed=5, alpha=True)))
    for label, props in cases:
        for ilabel, img in imgs:
            kw = dict(props=props, images={"Image": img}, inputs={"Fac": 0.9})
            cpu = render("CPU", big, **kw)
            gpu = render("GPU", big, **kw)
            # Rare pixels differ a lot: a pixel centre within float rounding of a cell boundary
            # (fma contraction in the screen coordinate) reads another cell's tone, or sits on
            # the anti-aliasing ramp where (d - t) / width amplifies rounding. Observed: <= 0.07 %
            # of pixels (max abs diff up to 0.88 on those), mean abs diff <= 2.3e-4.
            H.compare("%s %s: GPU vs CPU" % (label, ilabel), gpu, cpu, 2e-3, max_frac=0.002)
            H.compare("%s %s: GPU vs CPU (mean abs)" % (label, ilabel), gpu.mean(axis=(0, 1)),
                      cpu.mean(axis=(0, 1)), 5e-4)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        out = render(dev, (16, 8))      # unlinked Image: mid grey
        H.check(out.shape == (8, 16, 4) and np.isfinite(out).all(),
                "%s: unlinked input renders" % dev)
        for size in ((4, 4), (7, 5), (333, 187), (4, 40), (50, 4), (17, 33)):
            img = H.test_image(*size, seed=6, alpha=True)
            for mode in ("CMYK", "MONO", "LINES", "CROSSHATCH"):
                for cell in (2.0, 13.0, 400.0):
                    out = render(dev, size, props={"mode": mode, "cell_size": cell},
                                 images={"Image": img})
                    H.check(out.shape == (size[1], size[0], 4) and np.isfinite(out).all(),
                            "%s %dx%d %s cell %g: finite" % (dev, size[0], size[1], mode, cell))
        ext = np.array([[[1e4, -3.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 1.0],
                         [0.5, 0.5, 0.5, 1.0]]] * 4, F32)
        for mode in ("CMYK", "MONO", "LINES", "CROSSHATCH"):
            out = render(dev, (4, 4), props={"mode": mode}, images={"Image": ext})
            H.check(np.isfinite(out).all(), "%s %s: extreme input finite" % (dev, mode))
    img = H.test_image(1920, 1080, seed=7)
    t0 = time.time()
    render("CPU", (1920, 1080), props={"mode": "CMYK"}, images={"Image": img})
    H.note("CMYK halftone 1920x1080 CPU render (incl. IO): %.2fs" % (time.time() - t0))


for fn in (test_coverage, test_geometry, test_periodicity, test_cmyk, test_alpha, test_parity,
           test_robustness):
    fn()
H.finish()
