# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Cellular Automata: known patterns, an independent reference implementation (exact equality on
CPU and GPU), the rule parser, seeding / inject / cell size, outputs and the stateful semantics
(sequential frames, re-render, scrub back, catch-up, reset, pre-roll). Frames are rendered with
scene.frame_set + bpy.ops.render.render.
   Blender -b --factory-startup --python-exit-code 1 --python test_cellular_automata.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_ca, np_noise  # noqa: E402
from compositor_lab.lib import state as lab_state  # noqa: E402

NODE = "CompositorNodeLabCellularAutomata"
F32 = np.float32
DEVICES = ("CPU", "GPU")
AGE_RANGE = 64          # a power of two: age / range is exact in float32

# name: (preset, B, S, C), written out here independently of the node's preset table
RULES = {
    "Life": ('LIFE', {3}, {2, 3}, 2),
    "HighLife": ('HIGHLIFE', {3, 6}, {2, 3}, 2),
    "Seeds": ('SEEDS', {2}, set(), 2),
    "Day&Night": ('DAY_NIGHT', {3, 6, 7, 8}, {3, 4, 6, 7, 8}, 2),
    "Brian's Brain": ('BRIANS_BRAIN', {2}, set(), 3),
    "Star Wars": ('STAR_WARS', {2}, {3, 4, 5}, 4),
    "Maze": ('MAZE', {3}, {1, 2, 3, 4, 5}, 2),
}


# ---------------------------------------------------------------------------------------------
# Independent reference (plain python loops over cells, np.roll for the neighbour counts)
# ---------------------------------------------------------------------------------------------
def ref_counts(S, moore, wrap):
    alive = (S == 1).astype(np.int64)
    h, w = S.shape
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
            and (moore or not (dy and dx))]
    n = np.zeros_like(alive)
    for dy, dx in offs:
        if wrap:
            n += np.roll(np.roll(alive, -dy, axis=0), -dx, axis=1)       # value at (y+dy, x+dx)
        else:
            for y in range(h):
                for x in range(w):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        n[y, x] += alive[yy, xx]
    return n


def ref_gen(S, A, B, Sv, C, moore=True, wrap=True):
    n = ref_counts(S, moore, wrap)
    h, w = S.shape
    S2 = np.zeros_like(S)
    A2 = np.zeros_like(A)
    for y in range(h):
        for x in range(w):
            s = S[y, x]
            if s == 0:
                ns = 1 if n[y, x] in B else 0
            elif s == 1:
                ns = 1 if n[y, x] in Sv else (2 if C > 2 else 0)
            else:
                ns = s + 1 if s + 1 < C else 0
            S2[y, x] = ns
            A2[y, x] = 0 if (ns == 0 or s == 0) else A[y, x] + 1
    return S2, A2


def ref_inject(S, A, mask):
    new = mask & (S != 1)
    return np.where(new, 1, S), np.where(new, 0, A)


def ref_run(S, A, gens, B, Sv, C, moore=True, wrap=True):
    for _ in range(gens):
        S, A = ref_gen(S, A, B, Sv, C, moore, wrap)
    return S, A


def ref_color(S, A, C, colors, age_range=AGE_RANGE):
    ca, cb, cc, bg = (np.array(c, np.float64) for c in colors)
    out = np.zeros(S.shape + (3,))
    for y in range(S.shape[0]):
        for x in range(S.shape[1]):
            s = S[y, x]
            if s == 0:
                out[y, x] = bg
                continue
            t = min(A[y, x] / age_range, 1.0)
            g = ca + (cb - ca) * t * 2 if t <= 0.5 else cb + (cc - cb) * (t * 2 - 1)
            shade = (C - s) / (C - 1)
            if s >= 2:
                g = cc
            out[y, x] = bg + (g - bg) * shade ** 2
    return out


# ---------------------------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------------------------
def grey(a):
    return np.asarray(a, F32)


def setup_tree(device, size=(64, 48), props=None, inputs=None, seed=None, inject=None, start=1):
    lab_state.clear()
    scene = H.configure_scene(size, device)
    scene.frame_start, scene.frame_end = start, 250
    scene.frame_set(start)
    inp = {"Cell Size": 1, "Generations per Frame": 1}
    inp.update(inputs or {})
    p = {"age_range": AGE_RANGE}
    p.update(props or {})
    images = {}
    if seed is not None:
        images["Seed"] = grey(seed)
    if inject is not None:
        images["Inject"] = grey(inject)
    node = H.build_tree(scene, NODE, p, inp, images=images, out_socket="Cells")
    return scene, node


def show(scene, node, name):
    tree = scene.compositing_node_group
    out = next(n for n in tree.nodes if n.bl_idname == 'NodeGroupOutput')
    tree.links.new(node.outputs[name], out.inputs[0])


def rend(scene, f):
    scene.frame_set(int(f))
    return H.render(scene)


def run_to(scene, f, start=1):
    """Render the frames start .. f sequentially (the state steps along); returns the last."""
    for k in range(start, int(f) + 1):
        out = rend(scene, k)
    return out


def cells_state(img, C):
    sh = img[..., 0].astype(np.float64)
    return np.where(sh > 0, C - np.rint(sh * (C - 1)), 0).astype(np.int64)


def age_of(img, rng=AGE_RANGE):
    return np.rint(img[..., 0].astype(np.float64) * rng).astype(np.int64)


def stream_of(node):
    streams = lab_state.streams_for(node.lab_uid)
    return streams[0] if streams else None


def cleanup():
    H._clear_images()
    lab_state.configure(**{"global_bytes": 1024 * lab_state.MB, "cache_frames": 32})
    lab_state.clear()


def pat(rows, grid_w=16, grid_h=16, x0=0, y0=0):
    """Cells from strings: rows[i] is grid row y0 + i (row 0 = bottom of the image)."""
    g = np.zeros((grid_h, grid_w), np.int64)
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            if ch == "O":
                g[y0 + i, x0 + j] = 1
    return g


def rand_grid(w, h, seed, density=0.3):
    return (np.random.default_rng(seed).random((h, w)) < density).astype(np.int64)


# ---------------------------------------------------------------------------------------------
@H.guard("rule parser")
def test_parser():
    ok = {
        "B3/S23": ({3}, {2, 3}, 2), "b3/s23": ({3}, {2, 3}, 2), " B3 / S23 ": ({3}, {2, 3}, 2),
        "B2/S": ({2}, set(), 2), "B/S23": (set(), {2, 3}, 2), "B/S": (set(), set(), 2),
        "B2/S/3": ({2}, set(), 3), "B2/S345/4": ({2}, {3, 4, 5}, 4),
        "B36/S23/C4": ({3, 6}, {2, 3}, 4), "B3/S23/G5": ({3}, {2, 3}, 5),
        "S23/B3": ({3}, {2, 3}, 2), "B3678/S34678": ({3, 6, 7, 8}, {3, 4, 6, 7, 8}, 2),
        "B0/S8/256": ({0}, {8}, 256), "B33/S2": ({3}, {2}, 2),
    }
    for text, want in ok.items():
        try:
            got = np_ca.parse_rule(text)
        except ValueError as ex:
            H.report(False, "parser accepts %r (%s)" % (text, ex))
            continue
        H.check(got == want, "parser: %r -> %s" % (text, got))
    bad = ["", "B3", "S23", "B3/B2", "B3/S23/1", "B3/S23/0", "B3/S23/257", "B9/S1", "B3/S2x",
           "3/23", "B3/S23/4/5", "B3-S23", "hello", "B3/S23/-2", "B3/S23/C", "B3 S23"]
    for text in bad:
        try:
            r = np_ca.parse_rule(text)
            H.report(False, "parser rejects %r (got %s)" % (text, r))
        except ValueError as ex:
            H.check("Invalid rule" in str(ex), "parser rejects %r" % text)
    for pid, label, rule, _desc in np_ca.PRESETS:
        if pid != 'CUSTOM':
            np_ca.parse_rule(rule)
    for name, (pid, B, Sv, C) in RULES.items():
        H.check(np_ca.parse_rule(np_ca.PRESET_RULES[pid]) == (B, Sv, C), "preset %s" % name)
    # a rule typed into a node: an error shows as the node message and the outputs stay default
    scene, node = setup_tree("CPU", size=(8, 8), props={"preset": 'CUSTOM', "rule": "B3/X23"})
    out = H.render(scene, allow_errors=True)
    H.check(any("Invalid rule" in e for e in H.LAST_ERRORS), "invalid rule reports an error")
    H.check(np.all(out[..., :3] == 0), "invalid rule leaves default outputs")
    node.rule = "B3/S23"
    H.check(not np.any(H.render(scene)[..., 0] < 0), "a valid rule works again")
    node.preset = 'HIGHLIFE'
    H.check(node.rule == "B36/S23", "selecting a preset fills the rule string")
    cleanup()


@H.guard("patterns")
def test_patterns():
    for dev in DEVICES:
        # blinker: period 2
        blinker = pat(["OOO"], x0=5, y0=8)
        scene, node = setup_tree(dev, size=(16, 16), seed=blinker)
        f1, f2, f3 = (cells_state(rend(scene, f), 2) for f in (1, 2, 3))
        H.check(np.array_equal(f1, blinker), "%s: frame 1 is the seed" % dev)
        H.check(np.array_equal(f2, pat(["O", "O", "O"], x0=6, y0=7)), "%s: blinker flips" % dev)
        H.check(np.array_equal(f3, blinker), "%s: blinker has period 2" % dev)
        cleanup()
        # block: still life, ages every generation
        block = pat(["OO", "OO"], x0=3, y0=3)
        scene, node = setup_tree(dev, size=(16, 16), seed=block)
        for f in range(1, 7):
            out = rend(scene, f)
            H.check(np.array_equal(cells_state(out, 2), block), "%s: block frame %d" % (dev, f))
        show(scene, node, "Age")
        age = age_of(rend(scene, 6))
        H.check(np.array_equal(age, block * 5), "%s: block cells are 5 generations old" % dev)
        cleanup()
        # glider: array rows ".O.", "..O", "OOO" (y, x indices) move (+1, +1) every 4 generations
        glider = pat([".O.", "..O", "OOO"], 16, 16, x0=2, y0=2)
        scene, node = setup_tree(dev, size=(16, 16), seed=glider)
        for f in range(1, 14):
            out = rend(scene, f)
            if f in (5, 9, 13):
                k = (f - 1) // 4
                want = np.roll(np.roll(glider, k, axis=0), k, axis=1)
                H.check(np.array_equal(cells_state(out, 2), want),
                        "%s: glider moved (+%d, +%d) after %d generations" % (dev, k, k, f - 1))
        H.check(not np.array_equal(cells_state(rend(scene, 14), 2), glider),
                "%s: glider differs mid-cycle" % dev)
        cleanup()
        # Brian's Brain, one known step
        pair = pat(["OO"], x0=5, y0=5)
        scene, node = setup_tree(dev, size=(16, 16), seed=pair, props={"preset": 'BRIANS_BRAIN'})
        want1 = np.zeros((16, 16), np.int64)
        want1[5, 5] = want1[5, 6] = 2
        for y, x in ((4, 5), (4, 6), (6, 5), (6, 6)):
            want1[y, x] = 1
        rend(scene, 1)
        got1 = cells_state(rend(scene, 2), 3)
        H.check(np.array_equal(got1, want1), "%s: Brian's Brain first step" % dev)
        got2 = cells_state(rend(scene, 3), 3)
        H.check(got2[5, 5] == 0 and got2[5, 6] == 0, "%s: dying cells vanish" % dev)
        H.check(all(got2[y, x] == 2 for y, x in ((4, 5), (4, 6), (6, 5), (6, 6))),
                "%s: firing cells start dying" % dev)
        # Cells output: alive 1, dying 1/2 (C = 3), dead 0
        raw = rend(scene, 2)[..., 0]
        H.check(set(np.unique(raw)) == {0.0, 0.5, 1.0}, "%s: cells shades %s" % (dev, np.unique(raw)))
        cleanup()
        # Star Wars: C = 4, dying shades 2/3, 1/3
        scene, node = setup_tree(dev, size=(16, 16), seed=rand_grid(16, 16, 4, 0.4),
                                 props={"preset": 'STAR_WARS'})
        raw = run_to(scene, 4)[..., 0]
        vals = {round(float(v), 4) for v in np.unique(raw)}
        H.check(vals <= {0.0, 0.3333, 0.6667, 1.0} and len(vals) == 4,
                "%s: Star Wars shades %s" % (dev, sorted(vals)))
        cleanup()


@H.guard("reference")
def test_reference():
    W, Hh = 64, 48
    configs = [
        ("Life", {}, 1, 21), ("HighLife", {}, 1, 21), ("Seeds", {}, 1, 21),
        ("Day&Night", {}, 1, 21), ("Brian's Brain", {}, 1, 21), ("Star Wars", {}, 1, 21),
        ("Maze", {}, 1, 21),
        ("Life", {"edges": 'DEAD'}, 1, 21),
        ("custom B1/S12/4", {"preset": 'CUSTOM', "rule": "B1/S12/4",
                             "neighbourhood": 'VON_NEUMANN', "edges": 'DEAD'}, 1, 21),
        ("custom B13/S1234/6", {"preset": 'CUSTOM', "rule": "B13/S1234/6",
                                "neighbourhood": 'VON_NEUMANN'}, 1, 21),
        ("custom B3/S2345/5", {"preset": 'CUSTOM', "rule": "B3/S2345/5"}, 1, 21),
        ("Life G=3", {}, 3, 8),
        ("Star Wars G=4", {}, 4, 6),
    ]
    for dev in DEVICES:
        for name, extra, gens, frames in configs:
            pid, B, Sv, C = RULES.get(name.split(" G=")[0], (None, None, None, None))
            if pid is None:
                B, Sv, C = {"custom B1/S12/4": ({1}, {1, 2}, 4),
                            "custom B13/S1234/6": ({1, 3}, {1, 2, 3, 4}, 6),
                            "custom B3/S2345/5": ({3}, {2, 3, 4, 5}, 5)}[name]
                pid = 'CUSTOM'
            props = {"preset": pid}
            props.update(extra)
            if "rule" in props:
                props["preset"] = 'CUSTOM'
            moore = props.get("neighbourhood", 'MOORE') == 'MOORE'
            wrap = props.get("edges", 'WRAP') == 'WRAP'
            dens = 0.4 if not moore else (0.12 if C > 2 or Sv == set() else 0.35)
            seed = rand_grid(W, Hh, 11 + len(name), dens)
            scene, node = setup_tree(dev, size=(W, Hh), props=props, seed=seed,
                                     inputs={"Generations per Frame": gens})
            S, A = seed.copy(), np.zeros_like(seed)
            ok = True
            for f in range(1, frames + 1):
                if f > 1:
                    S, A = ref_run(S, A, gens, B, Sv, C, moore, wrap)
                got = cells_state(rend(scene, f), C)
                if not np.array_equal(got, S):
                    ok = False
                    H.report(False, "%s %s: frame %d differs from the reference in %d cells" % (
                        dev, name, f, int((got != S).sum())))
                    break
            H.check(ok and (S > 0).sum() > 0, "%s %s %s: %d frames (%d generations) exactly equal the "
                    "reference, %d live cells at the end" % (dev, name, "moore" if moore else "vN",
                                                             frames, (frames - 1) * gens,
                                                             int((S == 1).sum())))
            show(scene, node, "Age")
            age = age_of(rend(scene, frames))
            H.check(np.array_equal(age, np.minimum(A, AGE_RANGE)),
                    "%s %s: age equals the reference" % (dev, name))
            cleanup()


@H.guard("seeding")
def test_seeding():
    W, Hh = 64, 48
    # Seed image: alive where luminance > threshold, sampled at the cell centre
    img = np.zeros((Hh, W, 4), F32)
    img[..., 3] = 1.0
    img[10, 20] = (0.4, 0.4, 0.4, 1)          # luminance 0.4
    img[11, 20] = (0.6, 0.6, 0.6, 1)          # 0.6
    img[12, 20] = (1.0, 0.0, 0.0, 1)          # red: 0.2126
    img[13, 20] = (0.0, 1.0, 0.0, 1)          # green: 0.7152
    img[14, 20] = (0.0, 0.0, 1.0, 1)          # blue: 0.0722
    for dev in DEVICES:
        for thr, want in ((0.5, {11, 13}), (0.3, {10, 11, 13}), (0.1, {10, 11, 12, 13})):
            scene, node = setup_tree(dev, size=(W, Hh), seed=img, inputs={"Threshold": thr})
            got = cells_state(rend(scene, 1), 2)
            ys = {int(y) for y in np.nonzero(got[:, 20])[0]}
            H.check(ys == want and got.sum() == len(want),
                    "%s: threshold %.1f seeds rows %s (got %s)" % (dev, thr, sorted(want), sorted(ys)))
            cleanup()
    # random density with a seed
    res = {}
    for dev in DEVICES:
        for dens, rs in ((0.3, 0), (0.3, 5), (0.7, 0)):
            scene, node = setup_tree(dev, size=(W, Hh), inputs={"Density": dens, "Random Seed": rs})
            res[(dev, dens, rs)] = cells_state(rend(scene, 1), 2)
            cleanup()
    for dens in (0.3, 0.7):
        frac = res[("CPU", dens, 0)].mean()
        H.check(abs(frac - dens) < 0.03, "random seeding: density %.1f gives %.3f" % (dens, frac))
    H.check(np.array_equal(res[("CPU", 0.3, 0)], res[("GPU", 0.3, 0)]), "random seed: CPU == GPU")
    H.check(np.array_equal(res[("CPU", 0.7, 0)], res[("GPU", 0.7, 0)]), "random seed: CPU == GPU (0.7)")
    H.check(not np.array_equal(res[("CPU", 0.3, 0)], res[("CPU", 0.3, 5)]), "other Random Seed differs")
    gy, gx = np.mgrid[0:Hh, 0:W].astype(np.int32)
    want = np_noise.rand(gx, gy, 5) < F32(0.3)
    H.check(np.array_equal(res[("CPU", 0.3, 5)] == 1, want), "random seeding uses the shared hash")
    H.check(np.array_equal(res[("GPU", 0.3, 5)] == 1, want), "random seeding uses the shared hash (GPU)")
    H.check(np.all(res[("CPU", 0.3, 0)] <= res[("CPU", 0.7, 0)]),
            "a higher density only adds cells (same hash)")


@H.guard("inject")
def test_inject():
    W, Hh = 32, 24
    for dev in DEVICES:
        # painted cells: OR into the alive cells; Seeds rule: a lone cell dies next frame
        mask = np.zeros((Hh, W), F32)
        mask[5, 5] = 1.0
        mask[5, 6] = 0.5            # not > 0.5
        mask[8, 8] = 0.9
        paint = mask > 0.5
        seed = rand_grid(W, Hh, 3, 0.2)
        B, Sv, C = RULES["Brian's Brain"][1:]
        scene, node = setup_tree(dev, size=(W, Hh), seed=seed, inject=mask,
                                 props={"preset": 'BRIANS_BRAIN'})
        S, A = ref_inject(seed.copy(), np.zeros_like(seed), paint)
        H.check(np.array_equal(cells_state(rend(scene, 1), C), S),
                "%s: frame 1 is the seed ORed with the inject mask" % dev)
        for f in range(2, 8):
            S, A = ref_inject(S, A, paint)
            S, A = ref_gen(S, A, B, Sv, C)
            H.check(np.array_equal(cells_state(rend(scene, f), C), S),
                    "%s: frame %d: inject then generation" % (dev, f))
        cleanup()
        # nothing alive but the mask: a block painted each frame stays a block (Life)
        mask = np.zeros((Hh, W), F32)
        mask[3:5, 3:5] = 1.0
        scene, node = setup_tree(dev, size=(W, Hh), seed=np.zeros((Hh, W)), inject=mask)
        for f in (1, 2, 5):
            H.check(np.array_equal(cells_state(rend(scene, f), 2), (mask > 0.5).astype(np.int64)),
                    "%s: painted block frame %d" % (dev, f))
        cleanup()
        # a constant Inject above 0.5 fills the grid
        scene, node = setup_tree(dev, size=(W, Hh), seed=np.zeros((Hh, W)),
                                 inputs={"Inject": 1.0}, props={"preset": 'SEEDS'})
        H.check(cells_state(rend(scene, 1), 2).all(), "%s: Inject 1.0 fills the grid" % dev)
        cleanup()


@H.guard("cell size")
def test_cell_size():
    for dev in DEVICES:
        W, Hh, cell = 33, 25, 4                      # 9 x 7 cells, the last ones partial
        gw, gh = 9, 7
        img = np.zeros((Hh, W, 4), F32)
        img[..., 3] = 1.0
        grid = rand_grid(gw, gh, 9, 0.4)
        for gy in range(gh):
            for gx in range(gw):
                if grid[gy, gx]:
                    img[min(gy * cell + cell // 2, Hh - 1), min(gx * cell + cell // 2, W - 1)] = 1.0
        # a bright pixel off the cell centre must not seed
        off = (0, 1) if not grid[0, 0] else (2, 2)
        scene, node = setup_tree(dev, size=(W, Hh), seed=img, inputs={"Cell Size": cell})
        B, Sv, C = RULES["Life"][1:]
        S, A = grid.copy(), np.zeros_like(grid)
        for f in range(1, 6):
            if f > 1:
                S, A = ref_gen(S, A, B, Sv, C)
            out = rend(scene, f)
            full = np.repeat(np.repeat(S, cell, axis=0), cell, axis=1)[:Hh, :W]
            H.check(out.shape[:2] == (Hh, W) and np.array_equal(out[..., 0] > 0, full > 0),
                    "%s: cell size 4 on %dx%d (%dx%d cells), frame %d" % (dev, W, Hh, gw, gh, f))
        # changing the cell size restarts the simulation on the new grid
        node.inputs["Cell Size"].default_value = 3
        out = rend(scene, 6)
        H.check(stream_of(node).last_kind in ("STEP", "RESET") and out.shape[:2] == (Hh, W),
                "%s: cell size change renders (%s)" % (dev, stream_of(node).last_kind))
        blocks = out[..., 0][0:3, 0:3]
        H.check(np.all(blocks == blocks[0, 0]), "%s: cells are uniform 3x3 blocks" % dev)
        cleanup()
        # cell size larger than the image: a single cell
        scene, node = setup_tree(dev, size=(5, 4), inputs={"Cell Size": 16})
        out = rend(scene, 2)
        H.check(out.shape[:2] == (4, 5) and np.all(out[..., 0] == out[0, 0, 0]),
                "%s: one cell covers a 5x4 image" % dev)
        cleanup()


@H.guard("outputs")
def test_outputs():
    ca, cb, cc, bg = (1.0, 0.8, 0.2), (0.9, 0.1, 0.3), (0.1, 0.2, 0.9), (0.05, 0.1, 0.15)
    props = {"color_a": ca, "color_b": cb, "color_c": cc, "background": bg, "age_range": 8}
    block = pat(["OO", "OO"], x0=3, y0=3)
    for dev in DEVICES:
        scene, node = setup_tree(dev, size=(16, 16), seed=block, props=props)
        show(scene, node, "Color")
        for f, t in ((1, 0.0), (3, 0.25), (5, 0.5), (7, 0.75), (9, 1.0), (14, 1.0)):
            out = rend(scene, f)
            A = block * (f - 1)
            want = ref_color(block, A, 2, (ca, cb, cc, bg), age_range=8)
            H.compare("%s: colour at age %d (t = %.2f)" % (dev, f - 1, t), out[..., :3], want,
                      atol=2e-6, quiet=True)
            H.check(np.all(out[..., 3] == 1.0), "%s: colour alpha is 1" % dev)
        show(scene, node, "Age")
        out = rend(scene, 14)
        H.check(np.allclose(out[..., 0], np.minimum(block * 13 / 8.0, 1.0)), "%s: age saturates" % dev)
        cleanup()
        # dying cells fade to the background (Brian's Brain: shade 1/2)
        pair = pat(["OO"], x0=5, y0=5)
        scene, node = setup_tree(dev, size=(16, 16), seed=pair, props=dict(props, preset='BRIANS_BRAIN'))
        show(scene, node, "Color")
        out = run_to(scene, 2)
        S = np.zeros((16, 16), np.int64)
        S[5, 5] = S[5, 6] = 2
        for y, x in ((4, 5), (4, 6), (6, 5), (6, 6)):
            S[y, x] = 1
        A = np.zeros((16, 16), np.int64)
        A[5, 5] = A[5, 6] = 1
        H.compare("%s: dying and newborn cells colour" % dev, out[..., :3],
                  ref_color(S, A, 3, (ca, cb, cc, bg), age_range=8), atol=2e-6, quiet=True)
        cleanup()
    # unused outputs are fine, all three agree on the same frame
    scene, node = setup_tree("CPU", size=(16, 16), seed=block)
    a = rend(scene, 3)[..., 0]
    show(scene, node, "Color")
    b = rend(scene, 3)
    lum = b[..., :3].sum(axis=-1)
    H.check(a.shape == (16, 16) and np.all(lum[a > 0] > lum[a == 0].max()),
            "cells and colour render from the same state (live cells are brighter than the background)")
    cleanup()


@H.guard("semantics")
def test_semantics():
    W, Hh = 40, 30
    seed = rand_grid(W, Hh, 21, 0.3)
    B, Sv, C = RULES["Life"][1:]
    zeros = np.zeros_like(seed)

    def at(n):
        return ref_run(seed.copy(), zeros, n, B, Sv, C)[0]

    for dev in DEVICES:
        scene, node = setup_tree(dev, size=(W, Hh), seed=seed)
        first = {f: cells_state(rend(scene, f), 2) for f in range(1, 9)}
        H.check(all(np.array_equal(first[f], at(f - 1)) for f in first), "%s: sequential frames" % dev)
        H.check(stream_of(node).last_kind == "STEP", "%s: sequential frames step" % dev)
        again = cells_state(rend(scene, 8), 2)
        H.check(stream_of(node).last_kind == "REPEAT" and np.array_equal(again, first[8]),
                "%s: re-render of a frame is idempotent" % dev)
        H.check(np.array_equal(cells_state(rend(scene, 8), 2), first[8]), "%s: ... twice" % dev)
        # a rule change on the same frame shows live without stepping the simulation forward
        node.preset = 'SEEDS'
        edited = cells_state(rend(scene, 8), 2)
        want, _ = ref_run(first[7], zeros, 1, {2}, set(), 2)
        H.check(np.array_equal(edited, want), "%s: rule tweak recomputes from the pre-step state" % dev)
        node.preset = 'LIFE'
        H.check(np.array_equal(cells_state(rend(scene, 8), 2), first[8]), "%s: tweak undone" % dev)
        back = cells_state(rend(scene, 3), 2)
        H.check(stream_of(node).last_kind == "RESTORE" and np.array_equal(back, first[3]),
                "%s: scrub back restores frame 3" % dev)
        H.check(np.array_equal(cells_state(rend(scene, 4), 2), first[4]), "%s: continues after restore" % dev)
        jump = cells_state(rend(scene, 20), 2)
        H.check(stream_of(node).last_kind == "CATCH_UP" and np.array_equal(jump, at(19)),
                "%s: catch-up to frame 20 equals 19 generations" % dev)
        node.max_catch_up = 5
        held = cells_state(rend(scene, 60), 2)
        H.check(stream_of(node).last_kind == "HOLD" and np.array_equal(held, at(19)),
                "%s: a jump beyond max catch-up holds" % dev)
        H.check(np.array_equal(cells_state(rend(scene, 61), 2), at(20)), "%s: steps on after a hold" % dev)
        node.max_catch_up = 64
        reset = cells_state(rend(scene, 1), 2)
        H.check(stream_of(node).last_kind == "RESET" and np.array_equal(reset, seed),
                "%s: start frame reseeds" % dev)
        bpy.ops.compositor_lab.reset_state(uid=node.lab_uid)
        H.check(lab_state.streams_for(node.lab_uid) == [], "%s: Reset clears the state" % dev)
        H.check(np.array_equal(cells_state(rend(scene, 5), 2), seed), "%s: evaluation after Reset reseeds" % dev)
        cleanup()
        # generations per frame is read every frame
        scene, node = setup_tree(dev, size=(W, Hh), seed=seed, inputs={"Generations per Frame": 2})
        rend(scene, 1)
        H.check(np.array_equal(cells_state(rend(scene, 2), 2), at(2)), "%s: 2 generations per frame" % dev)
        node.inputs["Generations per Frame"].default_value = 0
        H.check(np.array_equal(cells_state(rend(scene, 3), 2), at(2)), "%s: 0 generations holds still" % dev)
        node.inputs["Generations per Frame"].default_value = 5
        H.check(np.array_equal(cells_state(rend(scene, 4), 2), at(7)), "%s: 5 generations" % dev)
        cleanup()
        # pre-roll: the stateless fallback
        scene, node = setup_tree(dev, size=(W, Hh), seed=seed, props={"preroll": 7})
        H.check(np.array_equal(cells_state(rend(scene, 1), 2), at(7)),
                "%s: pre-roll: a still render shows 7 generations" % dev)
        H.check(np.array_equal(cells_state(rend(scene, 2), 2), at(8)), "%s: pre-roll then steps" % dev)
        cleanup()
        # scene start frame other than 1
        scene, node = setup_tree(dev, size=(W, Hh), seed=seed, start=10)
        rend(scene, 10)
        rend(scene, 11)
        H.check(np.array_equal(cells_state(rend(scene, 12), 2), at(2)), "%s: scene start 10: frame 12" % dev)
        cleanup()
        # a duplicated node has its own state
        scene, node = setup_tree(dev, size=(W, Hh), seed=seed)
        tree = scene.compositing_node_group
        t2 = tree.copy()
        for f in range(1, 6):
            rend(scene, f)
        scene.compositing_node_group = t2
        n2 = next(n for n in t2.nodes if n.bl_idname == NODE)
        H.check(np.array_equal(cells_state(rend(scene, 6), 2), seed), "%s: a duplicate restarts" % dev)
        H.check(n2.lab_uid != node.lab_uid, "%s: duplicate has its own uid" % dev)
        scene.compositing_node_group = tree
        H.check(np.array_equal(cells_state(rend(scene, 6), 2), at(5)), "%s: the original continues" % dev)
        bpy.data.node_groups.remove(t2)
        cleanup()


@H.guard("cpu vs gpu")
def test_cpu_gpu():
    W, Hh = 50, 37
    seed = rand_grid(W, Hh, 8, 0.3)
    mask = np.zeros((Hh, W), F32)
    mask[10:14, 20:24] = 1.0
    cases = [
        ("Life cell 3", {}, {"Cell Size": 3}),
        ("Star Wars cell 2 G2 inject", {"preset": 'STAR_WARS'}, {"Cell Size": 2, "Generations per Frame": 2}),
        ("Brian's von Neumann dead", {"preset": 'BRIANS_BRAIN', "neighbourhood": 'VON_NEUMANN',
                                      "edges": 'DEAD'}, {"Cell Size": 1}),
    ]
    for name, props, inputs in cases:
        outs = {}
        for dev in DEVICES:
            scene, node = setup_tree(dev, size=(W, Hh), props=props, inputs=inputs, seed=seed,
                                     inject=mask if "inject" in name else None)
            seq = {}
            for out_name in ("Cells", "Age", "Color"):
                show(scene, node, out_name)
                seq[out_name] = [rend(scene, f) for f in (1,)]
            show(scene, node, "Cells")
            frames = {k: [] for k in seq}
            for f in range(2, 8):
                for out_name in ("Cells", "Age", "Color"):
                    show(scene, node, out_name)
                    frames[out_name].append(rend(scene, f))
            outs[dev] = {k: seq[k] + frames[k] for k in seq}
            cleanup()
        for k, tol in (("Cells", 0.0), ("Age", 0.0), ("Color", 1e-5)):
            for i, (c, g) in enumerate(zip(outs["CPU"][k], outs["GPU"][k])):
                H.compare("cpu vs gpu: %s %s frame %d" % (name, k, i + 1), g, c, atol=tol, quiet=True)
        H.check(not np.array_equal(outs["CPU"]["Cells"][0], outs["CPU"]["Cells"][6]),
                "%s: the sequence changes" % name)


@H.guard("robustness")
def test_robustness():
    for dev in DEVICES:
        # everything unlinked: random seeding, three outputs
        for name in ("Cells", "Age", "Color"):
            scene, node = setup_tree(dev, size=(64, 48), inputs={"Cell Size": 2})
            show(scene, node, name)
            out = rend(scene, 1)
            out2 = rend(scene, 2)
            H.check(np.isfinite(out).all() and out.shape[:2] == (48, 64), "%s: unlinked %s" % (dev, name))
            H.check(name == "Age" or not np.array_equal(out, out2), "%s: %s evolves" % (dev, name))
            cleanup()
        for size, cell in (((4, 4), 1), ((5, 7), 2), ((4, 4), 8), ((7, 5), 3), ((16, 4), 1), ((4, 17), 1)):
            for edges in ('WRAP', 'DEAD'):
                scene, node = setup_tree(dev, size=size, inputs={"Cell Size": cell, "Density": 0.5},
                                         props={"edges": edges, "preset": 'BRIANS_BRAIN'})
                gw, gh = np_ca.grid_size(size, cell)
                S = np_noise.rand(*[a.astype(np.int32) for a in np.mgrid[0:gh, 0:gw][::-1]], 0) < F32(0.5)
                S = S.astype(np.int64)
                A = np.zeros_like(S)
                ok = True
                for f in range(1, 7):
                    if f > 1:
                        S, A = ref_gen(S, A, {2}, set(), 3, True, edges == 'WRAP')
                    out = rend(scene, f)
                    full = np.repeat(np.repeat(S, cell, axis=0), cell, axis=1)[:size[1], :size[0]]
                    ok &= out.shape[:2] == (size[1], size[0]) and np.allclose(
                        out[..., 0], np.where(full > 0, (3 - full) / 2.0, 0.0))
                H.check(ok, "%s: %dx%d cell %d %s agrees with the reference" % (dev, size[0], size[1], cell, edges))
                cleanup()
        # an unused / unlinked-everything node does not fail with an image Inject of another size
        scene, node = setup_tree(dev, size=(32, 24), inject=np.ones((8, 8), F32) * 0.0,
                                 seed=rand_grid(16, 12, 1))
        out = rend(scene, 2)
        H.check(np.isfinite(out).all(), "%s: inputs of other sizes" % dev)
        cleanup()


@H.guard("memory")
def test_memory():
    for dev in DEVICES:
        W, Hh = 64, 48
        scene, node = setup_tree(dev, size=(W, Hh))
        frame_bytes = W * Hh * (8 if dev == "GPU" else 3)
        lab_state.configure(global_bytes=6 * frame_bytes)
        for f in range(1, 21):
            rend(scene, f)
        st = stream_of(node)
        H.check(lab_state.total_bytes() <= 6 * frame_bytes,
                "%s: global budget holds (%d <= %d)" % (dev, lab_state.total_bytes(), 6 * frame_bytes))
        cached = st.cached_frames()
        H.check(20 in cached and 2 not in cached, "%s: old frames evicted, newest kept (%s)" % (dev, cached))
        out = rend(scene, 2)
        H.check(stream_of(node).last_kind == "RESIM" and np.isfinite(out).all(),
                "%s: an evicted frame is re-simulated" % dev)
        cleanup()
        # the state is a grid of cells, not pixels
        scene, node = setup_tree(dev, size=(256, 128), inputs={"Cell Size": 4})
        rend(scene, 1)
        H.check(stream_of(node).nbytes <= 2 * 64 * 32 * (8 if dev == "GPU" else 3) * 3,
                "%s: state is stored per cell (%d bytes)" % (dev, stream_of(node).nbytes))
        cleanup()


test_parser()
test_patterns()
test_reference()
test_seeding()
test_inject()
test_cell_size()
test_outputs()
test_semantics()
test_cpu_gpu()
test_robustness()
test_memory()
H.finish()
