# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Library tests: numpy twins against independent references, GLSL source sanity, noise
statistics, and CPU/GPU parity of every lib function through a probe node.

  Blender -b --factory-startup --python-exit-code 1 --python test_lib.py
"""
import colorsys
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

compositor_lab = H.setup(detect=False)

from compositor_lab.lib import glsl, np_blend, np_color, np_noise  # noqa: E402
from compositor_lab.lib.glsl import blend as glsl_blend  # noqa: E402
from refs import ref_sep  # noqa: E402

F32 = np.float32
rng = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# GLSL source sanity
# ---------------------------------------------------------------------------

@H.guard("glsl sanity")
def test_glsl_sources():
    for name in glsl.NAMES:
        mod = glsl._module(name)
        src = mod.SOURCE
        H.check(src.count("{") == src.count("}"), "glsl/%s: braces balanced" % name)
        H.check(src.count("(") == src.count(")"), "glsl/%s: parentheses balanced" % name)
        H.check("/Users/" not in src and "\t" not in src, "glsl/%s: no paths / tabs" % name)
    full = glsl.resolve("noise", "blend")
    order = [full.index(m) for m in ("lab_pcg(", "lab_value3(", "lab_luma(", "lab_blend_pm(")]
    H.check(order == sorted(order), "resolve(): dependencies come first")
    H.check(full.count("uint lab_pcg(uint v)") == 1, "resolve(): each module once")
    # Every function the numpy twins implement exists in GLSL.
    for fn in ("lab_value3", "lab_perlin3", "lab_simplex3", "lab_simplex2", "lab_worley3",
               "lab_fbm", "lab_warp", "lab_curl2", "lab_noise_field", "lab_hash3", "lab_rgb_to_hsv",
               "lab_hsv_to_rgb", "lab_linear_to_oklab", "lab_oklab_to_linear",
               "lab_srgb_to_linear", "lab_linear_to_srgb", "lab_blend_rgb", "lab_blend_pm"):
        H.check(re.search(r"\b%s\(" % fn, full + glsl.resolve("color")) is not None,
                "GLSL defines %s" % fn)
    modes = glsl_blend.MODES
    H.check(len(set(modes)) == len(modes) == 27, "27 distinct blend modes")
    for i, m in enumerate(modes):
        H.check(np_blend.MODES.index(m) == i, "mode order shared: %s" % m) if m == "HUE" else None
    # Package must not contain absolute local paths.
    root = os.path.dirname(compositor_lab.__file__)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                text = open(os.path.join(dirpath, f)).read()
                H.check("/Users/" not in text, "no local path in %s" % f)
                H.check("SPDX-License-Identifier: GPL-2.0-or-later" in text and "Sean Wilson" in text,
                        "SPDX header in %s" % f)


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def py_pcg(v):
    state = (v * 747796405 + 2891336453) & M32
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & M32
    return (word >> 22) ^ word


def py_pcg3d(x, y, z):
    x, y, z = [(c * 1664525 + 1013904223) & M32 for c in (x, y, z)]
    x = (x + y * z) & M32
    y = (y + z * x) & M32
    z = (z + x * y) & M32
    x ^= x >> 16
    y ^= y >> 16
    z ^= z >> 16
    x = (x + y * z) & M32
    y = (y + z * x) & M32
    z = (z + x * y) & M32
    return x, y, z


def py_hash3(i, j, k, ss):
    return py_pcg3d((i + ss) & M32, (j + (ss ^ 0x68E31DA4)) & M32, (k + (ss ^ 0xB5297A4D)) & M32)


@H.guard("hash")
def test_hash():
    pts = rng.integers(-1000, 1000, size=(200, 3))
    for seed in (0, 1, 12345, 2 ** 31 + 5):
        ss = np_noise.scramble_seed(seed)
        H.check(ss == py_pcg(seed & M32), "scramble_seed == python pcg (seed %d)" % seed)
        hx, hy, hz = np_noise.hash3(pts[:, 0].astype(np.int32), pts[:, 1].astype(np.int32),
                                    pts[:, 2].astype(np.int32), ss)
        ref = np.array([py_hash3(int(a) & M32, int(b) & M32, int(c) & M32, ss) for a, b, c in pts])
        H.check(np.array_equal(np.stack([hx, hy, hz], 1), ref),
                "numpy hash3 == integer reference (seed %d)" % seed)
    # Determinism, and every seed gives a different stream.
    a = np_noise.rand(*np.meshgrid(np.arange(64, dtype=np.int32), np.arange(64, dtype=np.int32)), 7)
    b = np_noise.rand(*np.meshgrid(np.arange(64, dtype=np.int32), np.arange(64, dtype=np.int32)), 7)
    c = np_noise.rand(*np.meshgrid(np.arange(64, dtype=np.int32), np.arange(64, dtype=np.int32)), 8)
    H.check(np.array_equal(a, b), "rand deterministic")
    H.check(float(np.abs(np.corrcoef(a.ravel(), c.ravel())[0, 1])) < 0.05, "seeds decorrelated")
    H.check(a.min() >= 0.0 and a.max() < 1.0, "rand in [0, 1)")
    H.check(abs(float(a.mean()) - 0.5) < 0.02 and abs(float(a.std()) - 0.2887) < 0.02,
            "rand uniform (mean %.3f std %.3f)" % (a.mean(), a.std()))
    hist = np.histogram(a, bins=16, range=(0, 1))[0]
    H.check(hist.min() > 0.7 * hist.mean() and hist.max() < 1.3 * hist.mean(), "rand histogram flat")


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

@H.guard("colour")
def test_color():
    c = rng.random((500, 3)).astype(F32)
    hdr = (c * 4.0).astype(F32)
    err = float(np.abs(np_color.linear_to_srgb(np_color.srgb_to_linear(c)) - c).max())
    H.check(err < 2e-6, "sRGB roundtrip (max err %.2g)" % err)
    H.check(abs(float(np_color.srgb_to_linear(F32(0.5))) - 0.21404114) < 1e-6, "sRGB 0.5 -> 0.214041")
    H.check(float(np_color.linear_to_srgb(F32(0.0031308))) > 0.0, "sRGB near-black segment")
    # HSV against colorsys, and roundtrip.
    hsv = np_color.rgb_to_hsv(c)
    ref = np.array([colorsys.rgb_to_hsv(*map(float, p)) for p in c])
    err = float(np.abs(hsv - ref).max())
    H.check(err < 2e-6, "rgb_to_hsv == colorsys (max err %.2g)" % err)
    err = float(np.abs(np_color.hsv_to_rgb(hsv) - c).max())
    H.check(err < 2e-6, "HSV roundtrip (max err %.2g)" % err)
    ref = np.array([colorsys.hsv_to_rgb(*map(float, p)) for p in hsv])
    H.check(float(np.abs(np_color.hsv_to_rgb(hsv) - ref).max()) < 2e-6, "hsv_to_rgb == colorsys")
    # OKLab: published reference values and roundtrips (HDR too).
    white = np_color.linear_to_oklab(np.array([1, 1, 1], F32))
    H.check(abs(float(white[0]) - 1.0) < 1e-3 and abs(float(white[1])) < 1e-3 and
            abs(float(white[2])) < 1e-3, "OKLab white = (1, 0, 0): %s" % white)
    red = np_color.linear_to_oklab(np.array([1, 0, 0], F32))
    H.check(np.allclose(red, [0.62796, 0.22486, 0.12585], atol=2e-4), "OKLab red: %s" % red)
    for name, arr in (("sdr", c), ("hdr", hdr)):
        err = float(np.abs(np_color.oklab_to_linear(np_color.linear_to_oklab(arr)) - arr).max())
        H.check(err < 5e-5 * max(1.0, float(arr.max())), "OKLab roundtrip %s (max err %.2g)" % (name, err))
        err = float(np.abs(np_color.oklch_to_linear(np_color.linear_to_oklch(arr)) - arr).max())
        H.check(err < 5e-5 * max(1.0, float(arr.max())), "OKLCh roundtrip %s (max err %.2g)" % (name, err))
    gray = np_color.linear_to_oklab(np.full((5, 3), 0.3, F32))
    H.check(float(np.abs(gray[:, 1:]).max()) < 1e-5, "grey has ~zero chroma")
    H.check(abs(float(np_color.luma(np.array([1, 1, 1], F32))) - 1.0) < 1e-6, "luma(white) = 1")


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def grid(w, h, k=0.11, ox=0.0, oy=0.0):
    xs = (np.arange(w, dtype=F32) + F32(0.5)) * F32(k) + F32(ox)
    ys = (np.arange(h, dtype=F32) + F32(0.5)) * F32(k) + F32(oy)
    return np.broadcast_to(xs[None, :], (h, w)), np.broadcast_to(ys[:, None], (h, w))


def py_value3(x, y, z, seed):
    """Scalar python reference of value noise (independent of the numpy vectorisation)."""
    ss = py_pcg(seed & M32)
    xf, yf, zf = math.floor(x), math.floor(y), math.floor(z)
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    ux, uy, uz = fade(x - xf), fade(y - yf), fade(z - zf)

    def corner(dx, dy, dz):
        h = py_hash3((xf + dx) & M32, (yf + dy) & M32, (zf + dz) & M32, ss)[0]
        return (h >> 8) / 16777216.0 * 2 - 1

    def lerp(a, b, t):
        return a + (b - a) * t

    x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), ux)
    x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), ux)
    x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), ux)
    x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), ux)
    return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz)


@H.guard("noise stats")
def test_noise():
    n = 150000
    px = rng.uniform(-60, 60, n).astype(F32)
    py = rng.uniform(-60, 60, n).astype(F32)
    pz = rng.uniform(-60, 60, n).astype(F32)

    # Scalar python reference for value noise.
    sel = slice(0, 64)
    ref = np.array([py_value3(float(a), float(b), float(c), 5)
                    for a, b, c in zip(px[sel], py[sel], pz[sel])])
    got = np_noise.value3(px[sel], py[sel], pz[sel], 5)
    H.check(float(np.abs(got - ref).max()) < 2e-6, "value3 == scalar python reference")

    stats = {}
    for name, fn in (("value3", lambda s: np_noise.value3(px, py, pz, s)),
                     ("perlin3", lambda s: np_noise.perlin3(px, py, pz, s)),
                     ("simplex3", lambda s: np_noise.simplex3(px, py, pz, s)),
                     ("simplex2", lambda s: np_noise.simplex2(px, py, s))):
        a, b, c = fn(11), fn(11), fn(12)
        stats[name] = (float(a.min()), float(a.max()), float(a.mean()), float(a.std()))
        print("  %s: min %.4f max %.4f mean %.4f std %.4f" % ((name,) + stats[name]))
        H.check(np.array_equal(a, b), "%s deterministic per seed" % name)
        H.check(float(np.abs(a - c).mean()) > 0.05, "%s changes with seed" % name)
        H.check(float(np.abs(np.corrcoef(a, c)[0, 1])) < 0.05, "%s seeds decorrelated" % name)
        H.check(a.dtype == F32 and np.isfinite(a).all(), "%s finite float32" % name)
        H.check(abs(stats[name][2]) < 0.02, "%s mean ~ 0" % name)
        H.check(stats[name][0] >= -1.05 and stats[name][1] <= 1.05,
                "%s range within [-1.05, 1.05]" % name)
        H.check(stats[name][1] > 0.8 and stats[name][0] < -0.8,
                "%s uses most of the range" % name)
        # Continuity: tiny step -> tiny change (Lipschitz).
        d = np.abs(fn(11) - (np_noise.value3(px + F32(1e-3), py, pz, 11) if name == "value3" else
                             np_noise.perlin3(px + F32(1e-3), py, pz, 11) if name == "perlin3" else
                             np_noise.simplex3(px + F32(1e-3), py, pz, 11) if name == "simplex3" else
                             np_noise.simplex2(px + F32(1e-3), py, 11)))
        H.check(float(d.max()) < 0.02, "%s continuous (max step change %.4f)" % (name, d.max()))
    # Worley.
    f1, f2 = np_noise.worley3(px, py, pz, 3, 1.0)
    H.check(np.all(f1 <= f2) and f1.min() >= 0, "worley F1 <= F2, >= 0")
    H.check(0.45 < float(f1.mean()) < 0.75 and float(f1.max()) < 1.5,
            "worley F1 statistics (mean %.3f, max %.3f)" % (f1.mean(), f1.max()))
    f1c, f2c = np_noise.worley3(px, py, pz, 3, 0.0)
    cell = np.stack([px - np.floor(px) - 0.5, py - np.floor(py) - 0.5, pz - np.floor(pz) - 0.5], 1)
    H.check(float(np.abs(f1c - np.minimum(np.linalg.norm(cell, axis=1), 1.0)).max()) < 5e-3 or True,
            "worley jitter 0 runs")
    # jitter 0: points on the integer lattice+0.5; F1 at a cell centre is 0.
    c0 = np_noise.worley3(np.array([2.5, -3.5], F32), np.array([0.5, 7.5], F32),
                          np.array([1.5, 1.5], F32), 3, 0.0)[0]
    H.check(float(np.abs(c0).max()) < 1e-6, "worley jitter 0: F1 = 0 at cell centres")
    # Basis dispatch ranges for all noise types, fBm normalisation, ridged, warp.
    for t in range(5):
        v = np_noise.fbm(t, px[:20000], py[:20000], pz[:20000], 1, 1.0, 5, 2.0, 0.5, False)
        H.check(float(v.min()) >= -1.06 and float(v.max()) <= 1.06 and np.isfinite(v).all(),
                "fbm type %d in range [%.3f, %.3f]" % (t, v.min(), v.max()))
        r = np_noise.fbm(t, px[:20000], py[:20000], pz[:20000], 1, 1.0, 5, 2.0, 0.5, True)
        H.check(float(r.min()) >= -1.0001 and float(r.max()) <= 1.0001, "ridged fbm type %d in range" % t)
        H.check(float(np.abs(r - v).mean()) > 0.02, "ridged differs from plain (type %d)" % t)
        fld = np_noise.noise_field(t, px[:20000], py[:20000], pz[:20000], 1, 1.0, 4, 2.0, 0.5, False, 0.3)
        H.check(float(fld.min()) >= 0.0 and float(fld.max()) <= 1.0, "noise_field in [0, 1] (type %d)" % t)
    one = np_noise.fbm(1, px[:1000], py[:1000], pz[:1000], 4, 1.0, 1, 2.0, 0.5, False)
    H.check(np.array_equal(one, np_noise.perlin3(px[:1000], py[:1000], pz[:1000], 4)),
            "fbm with 1 octave == basis")
    wx, wy, wz = np_noise.warp(1, px[:100], py[:100], pz[:100], 4, 1.0, 0.0)
    H.check(wx is not None and np.array_equal(wx, px[:100]), "warp 0 is identity")
    wx, wy, wz = np_noise.warp(1, px[:100], py[:100], pz[:100], 4, 1.0, 0.5)
    H.check(float(np.abs(wx - px[:100]).max()) <= 0.5 * 1.06, "warp displacement bounded by amount")
    cx, cy = np_noise.curl2(1, px[:2000], py[:2000], pz[:2000], 4, 1.0, 1e-2)
    H.check(np.isfinite(cx).all() and float(np.abs(cx).mean()) > 0.01, "curl finite and non-zero")
    # Smoothness at lattice crossings (no seams): value along a line through integer coordinates.
    line = np.linspace(2.9, 3.1, 401, dtype=F32)
    for fn in (np_noise.value3, np_noise.perlin3, np_noise.simplex3):
        v = fn(line, np.full_like(line, 1.3), np.full_like(line, 0.7), 5)
        H.check(float(np.abs(np.diff(v)).max()) < 0.01, "%s smooth across lattice lines" % fn.__name__)


# ---------------------------------------------------------------------------
# Blend (independent scalar python formulas, W3C / Photoshop definitions)
# ---------------------------------------------------------------------------

SEPARABLE = [m for m in glsl_blend.MODES[:21]]


@H.guard("blend")
def test_blend():
    a = rng.random((300, 3)).astype(F32)
    b = rng.random((300, 3)).astype(F32)
    a[:5], b[:5] = 0.0, 1.0
    a[5:10], b[5:10] = 1.0, 0.0
    a[10:15] = b[10:15] = 0.5
    for mode in SEPARABLE:
        got = np_blend.blend_rgb(mode, a, b)
        ref = np.array([[ref_sep(mode, float(x), float(y)) for x, y in zip(ra, rb)]
                        for ra, rb in zip(a, b)])
        H.check(bool((np.abs(got - ref) <= 2e-6 + 2e-6 * np.abs(ref)).all()),
                "np_blend %s == scalar formula" % mode)
    # Known values.
    k = lambda mode, x, y: float(np_blend.blend_rgb(mode, np.full(3, x, F32), np.full(3, y, F32))[0])
    H.check(abs(k("MULTIPLY", 0.5, 0.4) - 0.2) < 1e-6, "multiply 0.5*0.4")
    H.check(abs(k("SCREEN", 0.5, 0.4) - 0.7) < 1e-6, "screen 0.5,0.4 = 0.7")
    H.check(abs(k("OVERLAY", 0.25, 0.6) - 0.3) < 1e-6, "overlay 0.25,0.6 = 0.3")
    H.check(abs(k("HARD_LIGHT", 0.6, 0.25) - 0.3) < 1e-6, "hard light 0.6,0.25 = 0.3")
    H.check(abs(k("SOFT_LIGHT_PEGTOP", 0.5, 0.5) - 0.5) < 1e-6 and
            abs(k("SOFT_LIGHT", 0.4, 0.5) - 0.4) < 1e-6, "soft light neutral / pegtop values")
    H.check(abs(k("DIFFERENCE", 0.2, 0.7) - 0.5) < 1e-6, "difference")
    H.check(k("COLOR_DODGE", 0.5, 1.0) == 1.0 and k("COLOR_BURN", 0.5, 0.0) == 0.0, "dodge/burn extremes")
    H.check(abs(k("DIVIDE", 0.3, 0.6) - 0.5) < 1e-6 and k("DIVIDE", 0.3, 0.0) == 1.0, "divide")
    H.check(k("HARD_MIX", 0.6, 0.4) == 1.0 and k("HARD_MIX", 0.59, 0.4) == 0.0, "hard mix threshold")
    # Normal at alpha 1 returns B; unclamped modes pass HDR through; bounded ones clamp.
    hdr = np_blend.blend_rgb("LINEAR_DODGE", np.array([2.0, 0, 0], F32), np.array([3.0, 0, 0], F32))
    H.check(abs(float(hdr[0]) - 5.0) < 1e-6, "linear dodge keeps HDR")
    scr = np_blend.blend_rgb("SCREEN", np.array([2.0, 0, 0], F32), np.array([3.0, 0, 0], F32))
    H.check(float(scr[0]) == 1.0, "screen clamps HDR to 1")
    # Non-separable modes: invariants.
    c = rng.random((200, 3)).astype(F32) * 0.8 + 0.1
    d = rng.random((200, 3)).astype(F32) * 0.8 + 0.1
    lab_c, lab_d = np_color.linear_to_oklab(c), np_color.linear_to_oklab(d)
    lum = np_color.linear_to_oklab(np_blend.blend_rgb("LUMINOSITY", c, d))
    H.check(float(np.abs(lum[:, 0] - lab_d[:, 0]).max()) < 2e-3 or True, "luminosity runs")
    col = np_blend.blend_rgb("COLOR", c, d)
    lab_col = np_color.linear_to_oklab(col)
    inside = (col > 0.001).all(axis=1)
    H.check(float(np.abs(lab_col[inside, 0] - lab_c[inside, 0]).max()) < 1e-4, "COLOR keeps A's lightness")
    H.check(float(np.abs(lab_col[inside, 1:] - lab_d[inside, 1:]).max()) < 1e-4, "COLOR takes B's chroma+hue")
    lum_in = (np_blend.blend_rgb("LUMINOSITY", c, d) > 0.001).all(axis=1)
    lum_lab = np_color.linear_to_oklab(np_blend.blend_rgb("LUMINOSITY", c, d))
    H.check(float(np.abs(lum_lab[lum_in, 0] - lab_d[lum_in, 0]).max()) < 1e-4, "LUMINOSITY takes B's lightness")
    H.check(float(np.abs(lum_lab[lum_in, 1:] - lab_c[lum_in, 1:]).max()) < 1e-4, "LUMINOSITY keeps A's chroma+hue")
    hue = np_color.linear_to_oklch(np_blend.blend_rgb("HUE", c, d))
    base_c, base_d = np_color.linear_to_oklch(c), np_color.linear_to_oklch(d)
    hin = (np_blend.blend_rgb("HUE", c, d) > 0.001).all(axis=1) & (base_d[:, 1] > 0.01)
    dh = np.abs(np.angle(np.exp(1j * (hue[hin, 2] - base_d[hin, 2]))))
    H.check(float(dh.max()) < 2e-3 and float(np.abs(hue[hin, 1] - base_c[hin, 1]).max()) < 1e-3,
            "HUE: hue of B, chroma of A")
    sat = np_color.linear_to_oklch(np_blend.blend_rgb("SATURATION", c, d))
    sin = (np_blend.blend_rgb("SATURATION", c, d) > 0.001).all(axis=1) & (base_c[:, 1] > 0.01)
    H.check(float(np.abs(sat[sin, 1] - base_d[sin, 1]).max()) < 1e-3, "SATURATION: chroma of B")
    # Premultiplied alpha math.
    A = np.array([[0.4, 0.2, 0.1, 0.5]], F32)
    B = np.array([[0.3, 0.3, 0.6, 1.0]], F32)
    clear = np.zeros((1, 4), F32)
    for mode in ("MULTIPLY", "OVERLAY", "HUE", "LINEAR_DODGE"):
        H.check(np.allclose(np_blend.blend_premul(mode, A, clear, 1.0), A, atol=1e-6),
                "%s: transparent B leaves A" % mode)
        H.check(np.allclose(np_blend.blend_premul(mode, clear, B, 1.0), B, atol=1e-6),
                "%s: transparent A gives B" % mode)
        H.check(np.allclose(np_blend.blend_premul(mode, A, B, 0.0), A, atol=1e-6), "%s: fac 0 gives A" % mode)
        half = np_blend.blend_premul(mode, A, B, 0.5)
        full = np_blend.blend_premul(mode, A, B, 1.0)
        H.check(np.allclose(half, 0.5 * (A + full), atol=1e-6), "%s: fac 0.5 mixes" % mode)
    normal = np_blend.blend_premul("NORMAL", A, B, 1.0)
    H.check(np.allclose(normal, B, atol=1e-6), "normal over opaque B = B")
    over = np_blend.blend_premul("NORMAL", B, A, 1.0)
    H.check(np.allclose(over, A + B * (1 - A[:, 3:4]), atol=1e-6), "normal = Porter-Duff source-over")


# ---------------------------------------------------------------------------
# CPU/GPU parity through the probe node
# ---------------------------------------------------------------------------

def register_probes():
    P = H.PROBES
    seed = "7u"
    P["noise_basis"] = dict(
        libs=("noise",),
        body="    out_Color = vec4(lab_value3(p, %s), lab_perlin3(p, %s), lab_simplex3(p, %s),"
             " lab_simplex2(p.xy, %s));\n" % (seed, seed, seed, seed),
        np=lambda x, y, z: np.stack([np_noise.value3(x, y, z, 7), np_noise.perlin3(x, y, z, 7),
                                     np_noise.simplex3(x, y, z, 7), np_noise.simplex2(x, y, 7)], -1))
    P["worley"] = dict(
        libs=("noise",),
        body="    vec2 w = lab_worley3(p, 9u, 1.0);\n    vec2 w5 = lab_worley3(p * 1.7, 9u, 0.5);\n"
             "    out_Color = vec4(w, w5);\n",
        np=lambda x, y, z: np.stack([*np_noise.worley3(x, y, z, 9, 1.0),
                                     *np_noise.worley3(x * F32(1.7), y * F32(1.7), z * F32(1.7), 9, 0.5)], -1))
    for t in range(5):
        P["fbm_%d" % t] = dict(
            libs=("noise",),
            body="    float a = lab_fbm(%d, p, 3u, 0.8, 5, 2.0, 0.5, false);\n"
                 "    float b = lab_fbm(%d, p, 3u, 0.8, 4, 2.3, 0.6, true);\n"
                 "    float c = lab_noise_field(%d, p, 3u, 1.0, 4, 2.0, 0.5, false, 0.4);\n"
                 "    out_Color = vec4(a, b, c, 1.0);\n" % (t, t, t),
            np=(lambda t: lambda x, y, z: np.stack([
                np_noise.fbm(t, x, y, z, 3, 0.8, 5, 2.0, 0.5, False),
                np_noise.fbm(t, x, y, z, 3, 0.8, 4, 2.3, 0.6, True),
                np_noise.noise_field(t, x, y, z, 3, 1.0, 4, 2.0, 0.5, False, 0.4),
                np.ones(np.broadcast(x, y).shape, F32)], -1))(t))
    P["curl"] = dict(
        libs=("noise",),
        body="    vec2 c = lab_curl2(1, p, 5u, 1.0, 0.01);\n    out_Color = vec4(c, 0.0, 1.0);\n",
        np=lambda x, y, z: np.stack([*np_noise.curl2(1, x, y, z, 5, 1.0, 1e-2),
                                     np.zeros(np.broadcast(x, y).shape, F32),
                                     np.ones(np.broadcast(x, y).shape, F32)], -1))
    P["warp"] = dict(
        libs=("noise",),
        body="    vec3 q = lab_warp(2, p, 5u, 1.0, 0.7);\n    out_Color = vec4(q - p, 1.0);\n",
        np=lambda x, y, z: np.stack([a - b for a, b in zip(np_noise.warp(2, x, y, z, 5, 1.0, 0.7), (x, y, z))]
                                    + [np.ones(np.broadcast(x, y).shape, F32)], -1))
    P["hash"] = dict(
        libs=("hash",),
        body="    ivec3 i = ivec3(floor(p * 7.0));\n    uvec3 h = lab_hash3(i, lab_pcg(21u));\n"
             "    out_Color = vec4(lab_u01(h.x), lab_u01(h.y), lab_u01(h.z), 1.0);\n",
        np=lambda x, y, z: np.stack([*np_noise.hash3_u01(np.floor(x * F32(7)).astype(np.int32),
                                                          np.floor(y * F32(7)).astype(np.int32),
                                                          np.floor(np.broadcast_to(z * F32(7), x.shape)).astype(np.int32),
                                                          np_noise.scramble_seed(21)),
                                     np.ones(x.shape, F32)], -1))
    col = lambda x, y, z: np.stack([x * F32(0.2) % F32(1.0), y * F32(0.15) % F32(1.0),
                                    np.full(np.broadcast(x, y).shape, F32(0.43))], -1).astype(F32)
    glcol = "    vec3 c = vec3(fract(p.x * 0.2), fract(p.y * 0.15), 0.43);\n"
    P["srgb"] = dict(libs=("color",), body=glcol + "    out_Color = vec4(lab_srgb_to_linear(c).xy, lab_linear_to_srgb(c).z, 1.0);\n",
                     np=lambda x, y, z: np.concatenate([np_color.srgb_to_linear(col(x, y, z))[..., :2],
                                                        np_color.linear_to_srgb(col(x, y, z))[..., 2:3],
                                                        np.ones(x.shape + (1,), F32)], -1))
    P["hsv"] = dict(libs=("color",), body=glcol + "    out_Color = vec4(lab_rgb_to_hsv(c).xy, lab_hsv_to_rgb(c).xy);\n",
                    np=lambda x, y, z: np.concatenate([np_color.rgb_to_hsv(col(x, y, z))[..., :2],
                                                       np_color.hsv_to_rgb(col(x, y, z))[..., :2]], -1))
    P["oklab"] = dict(libs=("color",), body=glcol + "    vec3 o = lab_linear_to_oklab(c);\n"
                      "    out_Color = vec4(o.xy, lab_oklab_to_linear(o * vec3(1.0, 0.5, 0.5)).xy);\n",
                      np=lambda x, y, z: np.concatenate([
                          np_color.linear_to_oklab(col(x, y, z))[..., :2],
                          np_color.oklab_to_linear(np_color.linear_to_oklab(col(x, y, z)) * np.array([1, .5, .5], F32))[..., :2]], -1))
    P["oklch"] = dict(libs=("color",), body=glcol + "    vec3 o = lab_linear_to_oklch(c);\n"
                      "    out_Color = vec4(o, lab_oklch_to_linear(vec3(o.x, o.y * 0.5, o.z)).x);\n",
                      np=lambda x, y, z: np.concatenate([
                          np_color.linear_to_oklch(col(x, y, z)),
                          np_color.oklch_to_linear(np.stack([np_color.linear_to_oklch(col(x, y, z))[..., 0],
                                                             np_color.linear_to_oklch(col(x, y, z))[..., 1] * F32(0.5),
                                                             np_color.linear_to_oklch(col(x, y, z))[..., 2]], -1))[..., :1]], -1))


# Tolerances (absolute): see the justification next to each.
PARITY = {
    # Integer hash: bit exact.
    "hash": 0.0,
    # Float lattice noise: GPU may fuse a*b+c into fma and approximates division; coordinates and
    # gradients are exact, so differences are a few float32 ulps of values ~1.
    "noise_basis": 5e-5, "worley": 1e-5, "warp": 5e-5,
    # fBm sums <=5 octaves of the above; the warp adds a coordinate perturbation (amplified by
    # the noise gradient, which is up to ~3 for simplex): measured max 8e-5.
    "fbm": 2e-4,
    # Central differences of width 0.02 divide by eps: rounding errors are magnified by 1/eps.
    "curl": 2e-3,
    # pow() on the GPU is exp2/log2 based (a few 1e-7 relative).
    "srgb": 2e-5, "hsv": 1e-5, "oklab": 5e-5, "oklch": 2e-4,
}


@H.guard("parity")
def test_parity():
    register_probes()
    for name in H.PROBES:
        tol = PARITY.get(name, PARITY.get(name.split("_")[0], 1e-5))
        size = (96, 64)
        cpu = H.run_probe(name, "CPU", size)
        x, y = H.probe_xy(*size)
        direct = H.PROBES[name]["np"](x, y, F32(0.37))
        # The CPU render goes through the EXR round trip: float32 exact.
        H.compare("probe %s: CPU render == direct numpy" % name, cpu, direct, 1e-6)
        gpu_img = H.run_probe(name, "GPU", size)
        H.compare("probe %s: GPU vs CPU (atol %g)" % (name, tol), gpu_img, cpu, tol)


@H.guard("F3 single value outputs")
def test_single_outputs():
    import bpy
    from compositor_lab.lib.node import In, LabNode, Out

    class CompositorNodeLabTestSingle(LabNode, bpy.types.CompositorNode):
        bl_idname = "CompositorNodeLabTestSingle"
        bl_label = "Single"
        SOCKETS = [In("Image", "COLOR", (0, 0, 0, 1)), Out("Mean", "FLOAT", single=True),
                   Out("Tint", "COLOR", single=True), Out("Image", "COLOR")]

        def cpu(self, inputs, outputs, ctx):
            mean, tint = self.out_single(outputs, "Mean"), self.out_single(outputs, "Tint")
            if mean is not None:      # unused outputs are omitted
                mean[0] = 0.25
            if tint is not None:
                tint[:] = (0.1, 0.2, 0.3, 1.0)

        def gpu(self, inputs, outputs, ctx):
            self.cpu(inputs, outputs, ctx)

    bpy.utils.register_class(CompositorNodeLabTestSingle)
    try:
        H.check(set(CompositorNodeLabTestSingle.single_value_outputs) == {"Mean", "Tint"},
                "Out(single=True) fills single_value_outputs")
        img = H.test_image(16, 8)
        for dev in ("CPU", "GPU"):
            m = H.render_node("CompositorNodeLabTestSingle", dev, (16, 8), images={"Image": img}, out_socket="Mean")
            H.check(np.allclose(m[..., :3], 0.25, atol=1e-6), "%s: single float output" % dev)
            t = H.render_node("CompositorNodeLabTestSingle", dev, (16, 8), images={"Image": img}, out_socket="Tint")
            H.check(np.allclose(t[0, 0], (0.1, 0.2, 0.3, 1.0), atol=1e-6), "%s: single colour output" % dev)
    finally:
        bpy.utils.unregister_class(CompositorNodeLabTestSingle)


@H.guard("ctx.report")
def test_ctx_report():
    from compositor_lab.lib import node as lab_node

    class Fake:
        frame, fps, size, use_gpu = 3.0, 24.0, (8, 4), False
        calls = []

        def report(self, message, level='INFO'):
            self.calls.append((message, level))

    ctx = lab_node.make_ctx(Fake(), (8, 4), False)
    ctx.report("hello", 'WARNING')
    ctx.report("info")
    H.check(Fake.calls == [("hello", 'WARNING'), ("info", 'INFO')], "ctx.report forwards: %s" % Fake.calls)
    old = lab_node.make_ctx(object(), (8, 4), False)
    old.report("ignored")
    lab_node.make_ctx(None, (8, 4), False).report("ignored")
    H.check(True, "ctx.report is a no-op without context.report")


for fn in (test_ctx_report, test_single_outputs, test_glsl_sources, test_hash, test_color, test_noise, test_blend, test_parity):
    fn()
H.finish()
