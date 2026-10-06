# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Voronoi / Mosaic node: CPU against an independent brute-force reference (all cells, float64),
CPU vs GPU, invariants (flat cells, borders only near F2 - F1 = 0, determinism), animation, mosaic,
robustness.   Blender -b --factory-startup --python-exit-code 1 --python test_voronoi.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_noise  # noqa: E402

NODE = "CompositorNodeLabVoronoi"
SIZE = (96, 64)
F32 = np.float32

# CPU vs GPU: the hashes are bit exact; float rounding (fma contraction, fast-math division) only
# matters for pixels within ~1e-6 of a cell boundary or a border edge, where a whole pixel may
# change cell. Observed: max abs difference ~1e-6 (border ramp, slope 1 / (2k) ~ 50 per unit), no
# pixel changing cell in any of the cases below, so the tolerance is tight; GPU_FRAC only leaves
# room for a single such pixel.
GPU_ATOL = 1e-5
GPU_FRAC = 2e-4


def render(device="CPU", size=SIZE, out="Color", props=None, inputs=None, images=None, frame=None):
    inputs = dict(inputs or {})
    inputs.setdefault("Speed", 0.0)
    if images:
        return H.render_node(NODE, device, size, props, inputs, images, out, frame)
    return H.render_generator(NODE, device, size, props, inputs, out, frame)


def brute(size, scale=8.0, seed=0, jitter=1.0, metric="EUCLIDEAN", off=(0.0, 0.0), phi=0.0):
    """Float64 brute force over every cell touching the image. Returns dict with f1, f2, cell id
    (index into the cell list), the cell coordinate arrays and site positions, k and the pixel
    coordinates."""
    w, h = size
    k = float(F32(scale / max(w, h)))
    xs = (np.arange(w) + 0.5) * k + off[0]
    ys = (np.arange(h) + 0.5) * k + off[1]
    X, Y = np.meshgrid(xs, ys)
    ss = np_noise.scramble_seed(seed)
    cx0, cx1 = int(np.floor(xs.min())) - 2, int(np.floor(xs.max())) + 2
    cy0, cy1 = int(np.floor(ys.min())) - 2, int(np.floor(ys.max())) + 2
    cells, sites = [], []
    for cy in range(cy0, cy1 + 1):
        for cx in range(cx0, cx1 + 1):
            ra = np.array([float(a) for a in np_noise.hash3_u01(np.int32(cx), np.int32(cy), 0, ss)[:2]])
            rb = np.array([float(a) for a in np_noise.hash3_u01(np.int32(cx), np.int32(cy), 1, ss)[:2]])
            r = (ra - 0.5) * np.cos(phi) + (rb - 0.5) * np.sin(phi)
            cells.append((cx, cy))
            sites.append(np.array([cx + 0.5, cy + 0.5]) + jitter * r)
    sites = np.array(sites)
    dx = X[None] - sites[:, 0, None, None]
    dy = Y[None] - sites[:, 1, None, None]
    if metric == "EUCLIDEAN":
        D = np.sqrt(dx * dx + dy * dy)
    elif metric == "MANHATTAN":
        D = np.abs(dx) + np.abs(dy)
    else:
        D = np.maximum(np.abs(dx), np.abs(dy))
    two = np.partition(D, 1, axis=0)[:2]
    nearest = np.argmin(D, axis=0)
    return dict(f1=two[0], f2=two[1], cell=nearest, cells=np.array(cells), sites=sites, k=k,
                X=X, Y=Y, ss=ss)


def cell_rgb(ref):
    c = ref["cells"][ref["cell"]]
    return np.stack(np_noise.hash3_u01(c[..., 0].astype(np.int32), c[..., 1].astype(np.int32), 2,
                                       ref["ss"]), axis=-1).astype(np.float64)


def _clean(a):
    return np.asarray(a, np.float64)


@H.guard("features")
def test_features():
    print("build features:", H.FEATURES)
    img = render()
    H.check(img.shape == (SIZE[1], SIZE[0], 4), "output has the render size %s" % (img.shape,))
    H.check(np.all(img[..., 3] == 1.0), "alpha 1")
    img = render(size=(40, 24), images={"Image": H.test_image(40, 24)}, props={"mode": "MOSAIC"})
    H.check(img.shape == (24, 40, 4), "linked Image defines the domain")


@H.guard("distance reference")
def test_f1_reference():
    for metric in ("EUCLIDEAN", "MANHATTAN", "CHEBYSHEV"):
        for jitter, scale, seed in ((1.0, 8.0, 0), (0.6, 5.0, 7), (0.0, 6.0, 3), (1.0, 20.0, -4)):
            ref = brute(SIZE, scale, seed, jitter, metric)
            out = render(props={"mode": "F1", "metric": metric},
                         inputs={"Scale": scale, "Jitter": jitter, "Seed": seed})
            v = H.render_generator(NODE, "CPU", SIZE, {"mode": "F1", "metric": metric},
                                   {"Scale": scale, "Jitter": jitter, "Seed": seed, "Speed": 0.0}, "Value")
            exp = np.clip(ref["f1"], 0, 1)
            H.compare("F1 %s jitter %g scale %g seed %d: Color == min distance" % (
                metric, jitter, scale, seed), out[..., 0], exp, 1e-5)
            H.compare("  Value == Color.R", v[..., 0], out[..., 0], 0.0, quiet=True)
            H.check(np.array_equal(out[..., 0], out[..., 1]), "  grey")


@H.guard("cells and borders")
def test_cells_and_borders():
    for metric in ("EUCLIDEAN", "MANHATTAN", "CHEBYSHEV"):
        ref = brute(SIZE, 8.0, 0, 1.0, metric)
        k = ref["k"]
        # Flat cells, no border: colour is a function of the (brute force) nearest cell only.
        col = render(props={"mode": "CELLS", "metric": metric}, inputs={"Border Width": 0.0})
        exp = cell_rgb(ref)
        H.compare("cells %s: colour == hash colour of nearest cell" % metric, col[..., :3], exp, 1e-6)
        ids = ref["cell"]
        n_cells = len(np.unique(ids))
        rgb = col[..., :3].reshape(-1, 3)
        n_cols = len(np.unique(np.round(rgb, 6), axis=0))
        H.check(n_cols == n_cells, "%s: one flat colour per cell (%d cells, %d colours)" % (
            metric, n_cells, n_cols))
        # Value is the per-cell id, Border output is 0 without a border.
        val = render(out="Value", props={"mode": "CELLS", "metric": metric}, inputs={"Border Width": 0.0})
        H.check(len(np.unique(val[..., 0])) == n_cells, "%s: Value is a per-cell id" % metric)
        bor = render(out="Border", props={"mode": "CELLS", "metric": metric}, inputs={"Border Width": 0.0})
        H.check(bor[..., 0].max() == 0.0, "%s: Border output is 0 for width 0" % metric)
        # Borders: coverage only near F2 - F1 = 0, ramp about one pixel wide.
        bw = 0.08
        bor = render(out="Border", props={"mode": "CELLS", "metric": metric}, inputs={"Border Width": bw})
        d = ref["f2"] - ref["f1"]
        m = bor[..., 0].astype(np.float64)
        expm = np.clip(0.5 + (bw - d) / (2 * k), 0, 1)
        # F2 from the 3x3 neighbourhood can differ from the true F2 only far from borders.
        # The node searches 3x3 cells. For Manhattan the true second-nearest site can lie outside
        # that window (observed 0.016% of pixels, max error 0.09); Euclidean / Chebyshev are exact.
        H.compare("%s: Border == ramp of F2 - F1 (brute force)" % metric, m, expm, 1e-4,
                  max_frac=5e-4 if metric == "MANHATTAN" else 0.0)
        H.check(np.all(m[d > bw + 1.01 * k] == 0.0), "%s: no border away from F2 - F1 ~ 0" % metric)
        H.check(np.all(m[d < bw - 1.01 * k] == 1.0), "%s: full border where F2 - F1 < width" % metric)
        H.check(0.05 < m.mean() < 0.6, "%s: border coverage plausible (%.3f)" % (metric, m.mean()))
        # Colour output: cell colour mixed with Border Color by the coverage.
        bc = (0.9, 0.1, 0.2, 1.0)
        col = render(props={"mode": "CELLS", "metric": metric},
                     inputs={"Border Width": bw, "Border Color": bc})
        exp = exp * (1 - m[..., None]) + np.array(bc[:3]) * m[..., None]
        H.compare("%s: colour = mix(cell, border colour, coverage)" % metric, col[..., :3], exp, 1e-5)
    # Edges mode: Fill Color with borders; Value is F2 - F1.
    ref = brute(SIZE, 8.0, 0, 1.0)
    fill = (0.2, 0.6, 0.9, 1.0)
    col = render(props={"mode": "EDGES"}, inputs={"Fill Color": fill, "Border Width": 0.0})
    H.compare("edges, no border: flat Fill Color", col, np.broadcast_to(np.array(fill), col.shape), 1e-6)
    val = render(out="Value", props={"mode": "EDGES"})
    H.compare("edges: Value == clamp(F2 - F1)", val[..., 0], np.clip(ref["f2"] - ref["f1"], 0, 1), 1e-5)
    # Jitter 0 gives a regular grid: cells are the unit squares.
    col = render(props={"mode": "CELLS"}, inputs={"Jitter": 0.0, "Border Width": 0.0})
    k = ref["k"]
    cx = np.floor((np.arange(SIZE[0]) + 0.5) * k)
    cy = np.floor((np.arange(SIZE[1]) + 0.5) * k)
    k_cells = len(np.unique(col[..., :3].reshape(-1, 3), axis=0))
    H.check(k_cells == len(np.unique(cx)) * len(np.unique(cy)), "jitter 0: regular grid of unit cells (%d)" % k_cells)


@H.guard("determinism and seeds")
def test_determinism():
    a = render()
    b = render()
    H.check(np.array_equal(a, b), "deterministic")
    s1 = render(inputs={"Seed": 1})
    s2 = render(inputs={"Seed": 2})
    H.check(float(np.abs(s1 - s2).mean()) > 0.05, "different seed, different cells")
    for dev in ("CPU", "GPU"):
        o1 = render(dev, inputs={"Offset X": 0.0})
        o2 = render(dev, inputs={"Offset X": 0.3})
        H.check(float(np.abs(o1 - o2).mean()) > 0.02, "%s: offset moves the pattern" % dev)
    # Resolution independent: the same cells at 2x resolution (block averaged edges differ a bit).
    small = render(size=(48, 32), props={"mode": "CELLS"}, inputs={"Border Width": 0.0})
    big = render(size=(96, 64), props={"mode": "CELLS"}, inputs={"Border Width": 0.0})
    agree = (small[..., :3] == big[::2, ::2, :3]).all(axis=-1).mean()
    H.check(agree > 0.9, "cell layout is resolution independent (%.3f of pixels agree)" % agree)


@H.guard("animation")
def test_animation():
    # Phase 0.25 = phi of 90 degrees: sites use the second random vector only.
    phi = 2 * np.pi * 0.25
    ref = brute(SIZE, 8.0, 5, 1.0, phi=phi)
    out = render(props={"mode": "F1"}, inputs={"Seed": 5, "Phase": 0.25})
    H.compare("phase 0.25: F1 matches brute force with the rotated site vectors", out[..., 0],
              np.clip(ref["f1"], 0, 1), 2e-5)
    ref = brute(SIZE, 8.0, 5, 0.7, phi=0.4 * 2 * np.pi)
    out = render(props={"mode": "F1"}, inputs={"Seed": 5, "Phase": 0.4, "Jitter": 0.7})
    H.compare("phase 0.4 jitter 0.7: brute force", out[..., 0], np.clip(ref["f1"], 0, 1), 2e-5)
    static = render(inputs={"Seed": 5})
    looped = render(inputs={"Seed": 5, "Phase": 1.0})
    H.compare("a full phase loop returns to the static pattern", looped, static, 1e-6, max_frac=1e-3)
    moved = render(inputs={"Seed": 5, "Phase": 0.3})
    H.check(float(np.abs(moved - static).mean()) > 0.05, "phase moves the cells")
    if not H.FEATURES.get("F2"):
        H.note("F2 not available: time-based animation checks skipped")
        return
    fps = None
    import bpy
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    for dev in ("CPU", "GPU"):
        f1 = render(dev, frame=1, inputs={"Speed": 0.5})
        f25 = render(dev, frame=25, inputs={"Speed": 0.5})
        H.check(float(np.abs(f1 - f25).mean()) > 0.02, "%s: image changes with the frame" % dev)
        z1 = render(dev, frame=1, inputs={"Speed": 0.0})
        z25 = render(dev, frame=25, inputs={"Speed": 0.0})
        H.check(np.array_equal(z1, z25), "%s: Speed 0 is static" % dev)
        ref = brute(SIZE, 8.0, 0, 1.0, phi=2 * np.pi * (25 / fps * 0.5))
        out = render(dev, frame=25, props={"mode": "F1"}, inputs={"Speed": 0.5})
        H.compare("%s: frame 25 F1 == brute force at phi = 2 pi time speed" % dev, out[..., 0],
                  np.clip(ref["f1"], 0, 1), 1e-5)


@H.guard("mosaic")
def test_mosaic():
    size = (80, 56)
    img = H.test_image(*size, seed=4)
    ref = brute(size, 6.0, 2, 1.0)
    k = ref["k"]
    col = render(size=size, props={"mode": "MOSAIC"}, images={"Image": img},
                 inputs={"Scale": 6.0, "Seed": 2, "Border Width": 0.0})
    site = ref["sites"][ref["cell"]]
    px = np.clip(np.floor(site[..., 0] / k), 0, size[0] - 1).astype(int)
    py = np.clip(np.floor(site[..., 1] / k), 0, size[1] - 1).astype(int)
    exp = img[py, px]
    H.compare("mosaic: every cell takes the image colour at its site", col, exp, 1e-6, max_frac=2e-3)
    n_cells = len(np.unique(ref["cell"]))
    n_cols = len(np.unique(np.round(col.reshape(-1, 4), 6), axis=0))
    H.check(n_cols <= n_cells, "mosaic: at most one colour per cell (%d colours, %d cells)" % (n_cols, n_cells))
    # Mean colour is close to the image mean (the sites sample the image uniformly).
    H.check(np.abs(col[..., :3].mean(axis=(0, 1)) - img[..., :3].mean(axis=(0, 1))).max() < 0.08,
            "mosaic mean colour close to the image mean")
    # Borders over the mosaic.
    bcol = render(size=size, props={"mode": "MOSAIC"}, images={"Image": img},
                  inputs={"Scale": 6.0, "Seed": 2, "Border Width": 0.1, "Border Color": (1.0, 0.0, 0.0, 1.0)})
    d = ref["f2"] - ref["f1"]
    m = np.clip(0.5 + (0.1 - d) / (2 * k), 0, 1)[..., None]
    H.compare("mosaic with border colour", bcol, exp * (1 - m) + np.array([1.0, 0, 0, 1.0]) * m, 1e-4,
              max_frac=2e-3)
    # No image linked: random cell colours (same as Cells).
    nom = render(props={"mode": "MOSAIC"}, inputs={"Border Width": 0.0})
    cel = render(props={"mode": "CELLS"}, inputs={"Border Width": 0.0})
    H.check(np.array_equal(nom, cel), "mosaic without an image falls back to random cell colours")


@H.guard("gpu parity")
def test_gpu():
    cases = []
    for mode in ("CELLS", "F1", "EDGES", "MOSAIC"):
        for metric in ("EUCLIDEAN", "MANHATTAN", "CHEBYSHEV"):
            cases.append(("%s %s" % (mode, metric), dict(props={"mode": mode, "metric": metric})))
    cases += [
        ("jitter 0.5 scale 15 seed 9", dict(inputs={"Jitter": 0.5, "Scale": 15.0, "Seed": 9})),
        ("jitter 0", dict(inputs={"Jitter": 0.0}, props={"mode": "CELLS"})),
        ("offset", dict(inputs={"Offset X": 3.7, "Offset Y": -2.2, "Seed": -3})),
        ("animated phase 0.3", dict(inputs={"Phase": 0.3, "Seed": 1})),
        ("animated F1 manhattan", dict(inputs={"Phase": 0.7}, props={"mode": "F1", "metric": "MANHATTAN"})),
        ("border width 0.3, colours", dict(inputs={"Border Width": 0.3, "Border Color": (0.2, 0.4, 0.8, 1.0)})),
        ("edges fill", dict(props={"mode": "EDGES"}, inputs={"Fill Color": (0.1, 0.5, 0.3, 1.0)})),
    ]
    for name, kw in cases:
        for out in ("Color", "Value", "Border"):
            c = render("CPU", out=out, **kw)
            g = render("GPU", out=out, **kw)
            H.compare("%s [%s]: GPU vs CPU" % (name, out), g, c, GPU_ATOL, max_frac=GPU_FRAC)
    img = H.test_image(80, 56, seed=2, alpha=True)
    for metric in ("EUCLIDEAN", "CHEBYSHEV"):
        kw = dict(size=(80, 56), props={"mode": "MOSAIC", "metric": metric}, images={"Image": img},
                  inputs={"Scale": 7.0, "Seed": 4})
        H.compare("mosaic image (alpha) %s: GPU vs CPU" % metric, render("GPU", **kw), render("CPU", **kw),
                  GPU_ATOL, max_frac=GPU_FRAC)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (333, 187), (17, 17)):
            for mode in ("CELLS", "F1", "EDGES", "MOSAIC"):
                img = render(dev, size=size, props={"mode": mode})
                H.check(img.shape == (size[1], size[0], 4) and np.isfinite(img).all(),
                        "%s %dx%d %s: shape/finite" % (dev, size[0], size[1], mode))
        for name, inputs in (("scale 0", {"Scale": 0.0}), ("scale -3", {"Scale": -3.0}),
                             ("scale 500", {"Scale": 500.0}), ("jitter 2", {"Jitter": 2.0}),
                             ("jitter -1", {"Jitter": -1.0}), ("border -1", {"Border Width": -1.0}),
                             ("border 50", {"Border Width": 50.0}), ("huge offset", {"Offset X": 1e5}),
                             ("seed big", {"Seed": 2 ** 31 - 1}), ("phase 1e3", {"Phase": 1e3})):
            for mode in ("CELLS", "EDGES"):
                img = render(dev, props={"mode": mode}, inputs=inputs)
                H.check(np.isfinite(img).all() and img.min() >= 0.0 and img.max() <= 1.0 + 1e-6,
                        "%s %s %s: finite, in range" % (dev, mode, name))
        # Mosaic with an image of a different aspect ratio and a tiny image.
        for isize in ((4, 4), (7, 33)):
            img = render(dev, size=isize, props={"mode": "MOSAIC"}, images={"Image": H.test_image(*isize)})
            H.check(img.shape == (isize[1], isize[0], 4) and np.isfinite(img).all(),
                    "%s mosaic %dx%d image" % (dev, isize[0], isize[1]))
    # Linked scalar inputs fall back to defaults on both backends (documented).
    scene = H.configure_scene(SIZE, "CPU")
    scale_img = np.full((SIZE[1], SIZE[0]), 0.5, F32)
    res = {}
    for dev in ("CPU", "GPU"):
        H.configure_scene(SIZE, dev)
        H.build_tree(scene, NODE, images={"Scale": scale_img}, inputs={"Speed": 0.0})
        res[dev] = H.render(scene)
        H._clear_images()
        H.check(np.isfinite(res[dev]).all(), "%s: linked Scale does not break the node" % dev)
    H.compare("linked Scale: GPU == CPU", res["GPU"], res["CPU"], GPU_ATOL, max_frac=GPU_FRAC)


for fn in (test_features, test_f1_reference, test_cells_and_borders, test_determinism, test_animation,
           test_mosaic, test_gpu, test_robustness):
    fn()
H.finish()
