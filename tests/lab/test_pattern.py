# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Pattern node: every tiling against an independent supersampled binary reference (float64),
stripe period (zero crossings and FFT), checker mean, truchet determinism, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_pattern.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_noise  # noqa: E402

NODE = "CompositorNodeLabPattern"
SIZE = (128, 96)
F32 = np.float32
MODES = ["STRIPES", "CHECKER", "DOTS", "HEX", "TRUCHET_ARC", "TRUCHET_DIAG", "MOIRE", "RINGS"]

# CPU vs GPU: same formulas; the edge ramp has slope 1 / pixel, so a 1e-6 coordinate difference is
# a 1e-6 coverage difference (observed max 6e-6, except the Offset X -123.4 case: float32 spacing at
# 123 is 7.6e-6 and the ramp slope is 1/k ~ 13, observed 9.8e-5); a flipped cell parity / truchet hash bit for a pixel exactly at a
# cell border is the only discontinuity, hence the small allowed pixel fraction.
GPU_ATOL = 1e-4
GPU_FRAC = 1e-3


def render(device="CPU", size=SIZE, out="Mask", props=None, inputs=None, frame=None):
    return H.render_generator(NODE, device, size, props, inputs, out, frame)


# ---------------------------------------------------------------------------
# Independent reference: binary pattern at float64 sub-sample positions
# ---------------------------------------------------------------------------

def pattern_coords(size, scale, rot_deg, off, sub):
    """Pattern-space coordinates of sub x sub samples per pixel: arrays (h, w, sub, sub)."""
    w, h = size
    k = scale / max(w, h)
    offs = (np.arange(sub) + 0.5) / sub
    qx = ((np.arange(w)[:, None] + offs[None, :]) - w / 2) * k      # (w, sub)
    qy = ((np.arange(h)[:, None] + offs[None, :]) - h / 2) * k      # (h, sub)
    qx = qx[None, :, None, :]
    qy = qy[:, None, :, None]
    th = np.radians(rot_deg)
    u = qx * np.cos(th) + qy * np.sin(th) + off[0]
    v = qy * np.cos(th) - qx * np.sin(th) + off[1]
    return np.broadcast_arrays(u, v)


def frac(x):
    return x - np.floor(x)


def stripes_bin(u, duty):
    return frac(u) < duty


def ref_binary(mode, u, v, duty, seed, moire_deg):
    if mode == "STRIPES":
        return stripes_bin(u, duty)
    if mode == "CHECKER":
        return ((np.floor(u) + np.floor(v)) % 2) == 1
    if mode == "DOTS":
        return np.hypot(frac(u) - 0.5, frac(v) - 0.5) < duty / 2
    if mode == "HEX":
        best = np.full(u.shape, np.inf)
        ex = np.zeros(u.shape)
        for r in range(int(np.floor(v.min() / 0.8660254)) - 2, int(np.ceil(v.max() / 0.8660254)) + 3):
            for n in range(int(np.floor(u.min())) - 2, int(np.ceil(u.max())) + 3):
                cx, cy = n + (0.5 if r % 2 else 0.0), 0.8660254 * r
                dx, dy = u - cx, v - cy
                d2 = dx * dx + dy * dy
                e = 0.5 - np.maximum.reduce([np.abs(dx), np.abs(0.5 * dx + 0.8660254 * dy),
                                             np.abs(-0.5 * dx + 0.8660254 * dy)])
                closer = d2 < best
                best = np.where(closer, d2, best)
                ex = np.where(closer, e, ex)
        return ex > (1 - duty) / 2
    if mode in ("TRUCHET_ARC", "TRUCHET_DIAG"):
        ss = np_noise.scramble_seed(seed)
        cu, cv = np.floor(u), np.floor(v)

        def flip(a, b):
            return np_noise.hash3_u01((cu + a).astype(np.int32), (cv + b).astype(np.int32), 0, ss)[0] < 0.5
        if mode == "TRUCHET_ARC":
            fx = np.where(flip(0, 0), 1 - (u - cu), u - cu)
            fy = v - cv
            d = np.minimum(np.abs(np.hypot(fx, fy) - 0.5), np.abs(np.hypot(fx - 1, fy - 1) - 0.5))
            return d < duty * 0.25
        # Diagonal segments with round caps, from this and the 8 neighbouring tiles.
        hit = np.zeros(u.shape, bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                px = np.where(flip(dx, dy), 1 - (u - cu - dx), u - cu - dx)
                py = v - cv - dy
                a_, b_ = np.array([0.0, 0.0]), np.array([1.0, 1.0])
                t = np.clip(((px - a_[0]) * (b_[0] - a_[0]) + (py - a_[1]) * (b_[1] - a_[1])) / 2.0, 0, 1)
                hit |= np.hypot(px - t, py - t) < duty * 0.25
        return hit
    if mode == "MOIRE":
        a = np.radians(moire_deg)
        return stripes_bin(u, duty) & stripes_bin(u * np.cos(a) + v * np.sin(a), duty)
    if mode == "RINGS":
        return stripes_bin(np.hypot(u, v), duty)
    raise ValueError(mode)


def ref_mask(mode, size, scale=6.0, rot=0.0, off=(0.0, 0.0), duty=0.5, seed=0, moire=4.0, sub=6):
    u, v = pattern_coords(size, scale, rot, off, sub)
    return ref_binary(mode, u, v, duty, seed, moire).mean(axis=(2, 3))


@H.guard("features")
def test_features():
    print("build features:", H.FEATURES)
    img = render(out="Color")
    H.check(img.shape == (SIZE[1], SIZE[0], 4), "output has the render size %s" % (img.shape,))


@H.guard("reference")
def test_reference():
    size = (96, 64)
    # (mean-abs-diff limit, fraction of pixels allowed to differ by more than 0.3). The node's ramp
    # is linear in the signed distance; the reference is a box-filtered 6x6 supersample. They agree
    # exactly on straight edges and differ near corners, thin features and curved boundaries.
    limits = {"STRIPES": (0.01, 0.0), "CHECKER": (0.02, 0.003), "DOTS": (0.03, 0.01), "HEX": (0.03, 0.01),
              "TRUCHET_ARC": (0.04, 0.02), "TRUCHET_DIAG": (0.03, 0.02), "MOIRE": (0.03, 0.01),
              "RINGS": (0.03, 0.01)}
    for mode in MODES:
        for kw in (dict(), dict(rot=23.0, off=(0.31, -0.17), duty=0.35, seed=5, scale=9.0),
                   dict(rot=-70.0, duty=0.7, scale=5.0, moire=11.0)):
            inputs = {"Scale": kw.get("scale", 6.0), "Rotation": kw.get("rot", 0.0),
                      "Offset X": kw.get("off", (0.0, 0.0))[0], "Offset Y": kw.get("off", (0.0, 0.0))[1],
                      "Duty": kw.get("duty", 0.5), "Seed": kw.get("seed", 0),
                      "Moire Angle": kw.get("moire", 4.0)}
            out = render(size=size, props={"pattern": mode}, inputs=inputs)[..., 0]
            ref = ref_mask(mode, size, kw.get("scale", 6.0), kw.get("rot", 0.0), kw.get("off", (0.0, 0.0)),
                           kw.get("duty", 0.5), kw.get("seed", 0), kw.get("moire", 4.0))
            d = np.abs(out - ref)
            lim_mean, lim_frac = limits[mode]
            frac_bad = float((d > 0.3).mean())
            print("  %s %s: mean abs diff %.4f, >0.3: %.4f%%, max %.3f" % (
                mode, sorted(kw), d.mean(), 100 * frac_bad, d.max()))
            H.check(d.mean() < lim_mean and frac_bad <= lim_frac,
                    "%s %s: mask matches the supersampled reference" % (mode, kw or "default"))
            H.check(out.min() >= 0.0 and out.max() <= 1.0, "%s: mask in [0, 1]" % mode)


@H.guard("stripes")
def test_stripes():
    size = (256, 192)
    for scale in (8.0, 5.0, 13.0):
        m = render(size=size, props={"pattern": "STRIPES"}, inputs={"Scale": scale})[..., 0]
        row = m[size[1] // 2]
        crossings = int(np.sum(np.diff((row > 0.5).astype(int)) != 0))
        H.check(abs(crossings - 2 * scale) <= 1, "scale %g: %d zero crossings ~ %g" % (scale, crossings, 2 * scale))
        spec = np.abs(np.fft.rfft(row - row.mean()))
        H.check(int(np.argmax(spec)) == int(scale), "scale %g: FFT peak at %d" % (scale, int(np.argmax(spec))))
        H.check(np.array_equal(m, np.broadcast_to(m[:1], m.shape)), "scale %g: vertical stripes (angle 0)" % scale)
        # Duty: fraction of B.
    for duty in (0.1, 0.5, 0.8):
        m = render(size=size, props={"pattern": "STRIPES"}, inputs={"Scale": 8.0, "Duty": duty})[..., 0]
        H.check(abs(m.mean() - duty) < 0.02, "duty %g: B coverage %.3f" % (duty, m.mean()))
    H.check(render(size=size, inputs={"Duty": 0.0})[..., 0].max() == 0.0, "duty 0: all A")
    H.check(render(size=size, inputs={"Duty": 1.0})[..., 0].min() == 1.0, "duty 1: all B")
    # Rotation: 90 degrees swaps the axes; 30 degrees shows a 2D FFT peak at the expected spot.
    m90 = render(size=size, inputs={"Scale": 8.0, "Rotation": 90.0})[..., 0]
    H.check(np.abs(m90 - m90[:, :1]).max() < 1e-4, "rotation 90: horizontal stripes")
    sq = (192, 192)
    m = render(size=sq, inputs={"Scale": 8.0, "Rotation": 30.0})[..., 0]
    spec = np.abs(np.fft.fft2(m - m.mean()))
    ky, kx = np.unravel_index(np.argmax(spec), spec.shape)
    ky, kx = (ky if ky <= 96 else ky - 192), (kx if kx <= 96 else kx - 192)
    ex = (8 * np.sin(np.radians(30)), 8 * np.cos(np.radians(30)))
    ok = (abs(abs(ky) - ex[0]) <= 1.01 and abs(abs(kx) - ex[1]) <= 1.01)
    H.check(ok, "rotation 30: 2D FFT peak (%d, %d) ~ (%.1f, %.1f)" % (ky, kx, ex[0], ex[1]))
    # Offset shifts by whole periods without changing the image.
    a = render(size=size, inputs={"Scale": 8.0})[..., 0]
    b = render(size=size, inputs={"Scale": 8.0, "Offset X": 3.0})[..., 0]
    H.compare("offset by whole periods", b, a, 2e-4)
    a = render(size=size, inputs={"Scale": 7.3})[..., 0]
    # Anti-aliasing: edge pixels carry intermediate values; Softness widens the ramp.
    frac0 = float(((a > 0.02) & (a < 0.98)).mean())
    soft = render(size=size, inputs={"Scale": 7.3, "Softness": 0.5})[..., 0]
    frac1 = float(((soft > 0.02) & (soft < 0.98)).mean())
    H.check(0.0 < frac0 < 0.1, "anti-aliased edges (%.3f of pixels intermediate)" % frac0)
    H.check(frac1 > frac0 * 4, "softness widens the edges (%.3f -> %.3f)" % (frac0, frac1))
    # Exact box filtering of an axis-aligned edge: row sum is the exact B area.
    H.check(abs(a.mean() - 0.5) < 5e-3, "stripes cover 50%% (%.4f)" % a.mean())


@H.guard("checker")
def test_checker():
    for size, scale in (((256, 256), 8.0), ((256, 192), 16.0), ((200, 150), 7.0)):
        m = render(size=size, props={"pattern": "CHECKER"}, inputs={"Scale": scale})
        c = render(size=size, out="Color", props={"pattern": "CHECKER"}, inputs={"Scale": scale})
        tol = 1e-6 if scale in (8.0, 16.0) else 0.03
        H.check(abs(float(c[..., 0].mean()) - 0.5) < tol, "checker %s scale %g: mean %.5f ~ 0.5" % (
            size, scale, float(c[..., 0].mean())))
        H.compare("  Color (A black, B white) == Mask", c[..., 0], m[..., 0], 1e-6, quiet=True)
    # Squares have the expected size.
    size = (256, 256)
    c = render(size=size, inputs={"Scale": 8.0}, props={"pattern": "CHECKER"})[..., 0]
    row = c[10]
    cross = np.flatnonzero(np.diff((row > 0.5).astype(int)))
    H.check(len(cross) == 7 and np.all(np.abs(np.diff(cross) - 32) <= 1), "checker squares are 32 px wide")
    # 4-fold structure: shifting by one square inverts it.
    H.compare("shift by one square inverts", c[:, 32:], 1 - c[:, :-32], 1e-4)


@H.guard("truchet and hex")
def test_truchet_hex():
    size = (128, 128)
    for mode in ("TRUCHET_ARC", "TRUCHET_DIAG"):
        a = render(size=size, props={"pattern": mode}, inputs={"Scale": 8.0, "Seed": 3})
        b = render(size=size, props={"pattern": mode}, inputs={"Scale": 8.0, "Seed": 3})
        c = render(size=size, props={"pattern": mode}, inputs={"Scale": 8.0, "Seed": 4})
        H.check(np.array_equal(a, b), "%s: deterministic per seed" % mode)
        H.check(float(np.abs(a - c).mean()) > 0.02, "%s: different seed, different tiling" % mode)
        if mode == "TRUCHET_DIAG":
            continue    # strokes have round caps that spill into neighbouring tiles
        # Each tile is one of exactly two variants: tile images (16 px) fall into two classes.
        tiles = a[..., 0].reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(64, 16, 16)
        n_a = sum(1 for t in tiles if np.abs(t - tiles[0]).max() < 1e-4)
        n_b = sum(1 for t in tiles if np.abs(t - tiles[0][:, ::-1]).max() < 1e-4)
        H.check(n_a + n_b == 64, "%s: every tile is the base tile or its mirror (%d + %d)" % (mode, n_a, n_b))
        H.check(8 < n_a < 56, "%s: both orientations occur (%d / %d)" % (mode, n_a, n_b))
    # Arcs connect across tile borders: the mask on both sides of every vertical tile border agrees.
    m = render(size=size, props={"pattern": "TRUCHET_ARC"}, inputs={"Scale": 8.0, "Seed": 1})[..., 0]
    left, right = m[:, 15:-1:16], m[:, 16::16]
    H.check(float(np.abs(left - right).max()) < 0.35, "arcs continue across tile borders (max jump %.3f)" %
            float(np.abs(left - right).max()))
    # Hex grid: symmetric cell with the right centre spacing: FFT peak at scale along x.
    hx = render(size=(256, 256), props={"pattern": "HEX"}, inputs={"Scale": 8.0})[..., 0]
    spec = np.abs(np.fft.rfft(hx[128] - hx[128].mean()))
    H.check(abs(int(np.argmax(spec)) - 8) <= 1 or abs(int(np.argmax(spec)) - 16) <= 1,
            "hex grid: horizontal period matches the scale (peak %d)" % int(np.argmax(spec)))
    H.check(0.2 < hx.mean() < 0.8, "hex grid coverage %.3f" % hx.mean())
    # Dots: area of a disc.
    d = render(size=(256, 256), props={"pattern": "DOTS"}, inputs={"Scale": 8.0, "Duty": 0.8})[..., 0]
    H.check(abs(d.mean() - np.pi * 0.4 ** 2) < 0.01, "dots coverage %.4f ~ pi r^2 = %.4f" % (d.mean(), np.pi * 0.16))
    # Rings: radius of ring edges.
    r = render(size=(256, 256), props={"pattern": "RINGS"}, inputs={"Scale": 8.0})[..., 0]
    row = r[128, 128:]
    cross = np.flatnonzero(np.diff((row > 0.5).astype(int)))
    H.check(len(cross) >= 6 and abs((cross[2] - cross[0]) - 32) <= 2, "rings: period 32 px")
    # Moire: product of two gratings, darker than either; beat period ~ 1 / (angle diff in radians).
    mo = render(size=(256, 256), props={"pattern": "MOIRE"}, inputs={"Scale": 16.0, "Moire Angle": 6.0})[..., 0]
    st = render(size=(256, 256), props={"pattern": "STRIPES"}, inputs={"Scale": 16.0})[..., 0]
    H.check(np.all(mo <= st + 1e-5), "moire <= first grating")
    H.check(mo.mean() < st.mean() * 0.8, "moire coverage %.3f < %.3f" % (mo.mean(), st.mean()))
    # Low-frequency beat: blurred moire has structure, blurred stripes do not.
    def blur_std(a):
        b = a.reshape(16, 16, 16, 16).mean(axis=(1, 3))
        return float(b.std())
    H.check(blur_std(mo) > 5 * blur_std(st) + 0.01, "moire shows a low-frequency beat (%.3f vs %.3f)" % (
        blur_std(mo), blur_std(st)))


@H.guard("colours")
def test_colours():
    A, B = (0.1, 0.2, 0.3, 1.0), (0.9, 0.8, 0.7, 0.5)
    for mode in ("STRIPES", "HEX"):
        col = render(out="Color", props={"pattern": mode}, inputs={"Color A": A, "Color B": B, "Rotation": 11.0})
        m = render(props={"pattern": mode}, inputs={"Rotation": 11.0})[..., :1]
        exp = np.array(A) + (np.array(B) - np.array(A)) * m
        H.compare("%s: Color == mix(A, B, Mask)" % mode, col, exp, 1e-6)
    # Default colours: A black, B white.
    c = render(out="Color")
    m = render()
    H.compare("default Color == Mask grey", c[..., :3], np.repeat(m[..., :1], 3, axis=2), 1e-6)


@H.guard("gpu parity")
def test_gpu():
    cases = []
    for mode in MODES:
        cases.append((mode, dict(props={"pattern": mode})))
        cases.append((mode + " rotated/offset", dict(props={"pattern": mode}, inputs={
            "Rotation": 37.0, "Offset X": 1.3, "Offset Y": -0.77, "Duty": 0.3, "Softness": 0.2, "Seed": 11,
            "Scale": 7.0, "Moire Angle": 9.0, "Color A": (0.2, 0.1, 0.0, 1.0), "Color B": (0.9, 0.9, 0.5, 0.8)})))
    cases += [
        ("stripes duty 0", dict(inputs={"Duty": 0.0})), ("stripes duty 1", dict(inputs={"Duty": 1.0})),
        ("scale 60", dict(inputs={"Scale": 60.0})), ("negative offset", dict(props={"pattern": "CHECKER"},
                                                                          inputs={"Offset X": -123.4})),
    ]
    for name, kw in cases:
        for out in ("Color", "Mask"):
            c = render("CPU", out=out, **kw)
            g = render("GPU", out=out, **kw)
            H.compare("%s [%s]: GPU vs CPU" % (name, out), g, c, GPU_ATOL, max_frac=GPU_FRAC)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (333, 187), (17, 17)):
            for mode in MODES:
                for out in ("Color", "Mask"):
                    img = render(dev, size=size, out=out, props={"pattern": mode})
                    H.check(img.shape == (size[1], size[0], 4) and np.isfinite(img).all() and
                            img.min() >= -1e-6 and img.max() <= 1.0 + 1e-6,
                            "%s %dx%d %s %s: shape/finite/range" % (dev, size[0], size[1], mode, out))
        for name, inputs in (("scale 0", {"Scale": 0.0}), ("scale -5", {"Scale": -5.0}),
                             ("scale 1e4", {"Scale": 1e4}), ("duty 2", {"Duty": 2.0}), ("duty -1", {"Duty": -1.0}),
                             ("softness 5", {"Softness": 5.0}), ("huge offset", {"Offset X": 1e5}),
                             ("rotation 1e4", {"Rotation": 1e4}), ("seed min", {"Seed": -2 ** 31})):
            for mode in MODES:
                img = render(dev, props={"pattern": mode}, inputs=inputs)
                H.check(np.isfinite(img).all() and img.min() >= -1e-6 and img.max() <= 1.0 + 1e-6,
                        "%s %s %s: finite, in range" % (dev, mode, name))
    # CPU and GPU agree for extreme settings too.
    for name, inputs in (("scale 0", {"Scale": 0.0}), ("duty 2", {"Duty": 2.0})):
        H.compare("%s: GPU vs CPU" % name, render("GPU", inputs=inputs), render("CPU", inputs=inputs),
                  GPU_ATOL, max_frac=GPU_FRAC)


for fn in (test_features, test_reference, test_stripes, test_checker, test_truchet_hex, test_colours,
           test_gpu, test_robustness):
    fn()
H.finish()
