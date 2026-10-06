# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Reaction-Diffusion (Gray-Scott): reference check, simulation semantics, CPU vs GPU, presets,
stability. Frame sequences are rendered with scene.frame_set + bpy.ops.render.render.
   Blender -b --factory-startup --python-exit-code 1 --python test_reaction_diffusion.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import state as lab_state  # noqa: E402

NODE = "CompositorNodeLabReactionDiffusion"
DEVICES = ("CPU", "GPU")
SIZE = (40, 30)
ITERS = 20
F, K, DU, DV, DT = 0.0545, 0.062, 1.0, 0.5, 1.0
# float32 node vs float64 reference: rounding accumulates about 1e-8 per iteration (<= 220 here)
TOL = 5e-6


# ---------------------------------------------------------------------------------------------
# An independent float64 Gray-Scott (does not import lib/np_rd): index-gather neighbours
# ---------------------------------------------------------------------------------------------
def ref_neighbours(a, dy, dx, edge):
    h, w = a.shape
    ys = np.arange(h) + dy
    xs = np.arange(w) + dx
    if edge == 'WRAP':
        ys, xs = ys % h, xs % w
    else:
        ys, xs = np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)
    return a[ys[:, None], xs[None, :]]


def ref_lap(a, edge):
    lap = -a.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            lap += (0.2 if (dx == 0 or dy == 0) else 0.05) * ref_neighbours(a, dy, dx, edge)
    return lap


def ref_run(u, v, n, f=F, k=K, du=DU, dv=DV, dt=DT, edge='WRAP'):
    u, v = u.astype(np.float64), v.astype(np.float64)
    for _ in range(n):
        r = u * v * v
        nu = u + dt * (du * ref_lap(u, edge) - r + f * (1.0 - u))
        nv = v + dt * (dv * ref_lap(v, edge) + r - (f + k) * v)
        u, v = np.clip(nu, 0.0, 1.0), np.clip(nv, 0.0, 1.0)
    return u, v


def mask_image(w, h, seed=2, blobs=3):
    """Black image with a few white discs (the Seed input); returns (image, boolean mask)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w), bool)
    for _ in range(blobs):
        cx, cy, r = rng.integers(3, w - 3), rng.integers(3, h - 3), rng.integers(2, 4)
        m |= (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = m[..., None].astype(np.float32)
    img[..., 3] = 1.0
    return img, m


def init_ref(m):
    return np.where(m, 0.5, 1.0), np.where(m, 1.0, 0.0)


# ---------------------------------------------------------------------------------------------
def setup_tree(device, size=SIZE, props=None, inputs=None, images=None, start=1, out="V"):
    lab_state.clear()
    scene = H.configure_scene(size, device)
    scene.frame_start, scene.frame_end = start, 250
    scene.frame_set(start)
    p = {"scale": 1, "seed_noise": 0.0}
    p.update(props or {})
    i = {"Iterations per Frame": ITERS}
    i.update(inputs or {})
    node = H.build_tree(scene, NODE, p, i, images=images, out_socket=out)
    return scene, node


def rend(scene, f):
    scene.frame_set(int(f))
    return H.render(scene)


def stream_of(node):
    streams = lab_state.streams_for(node.lab_uid)
    return streams[0] if streams else None


def cleanup():
    H._clear_images()
    lab_state.configure(**{"global_bytes": 1024 * lab_state.MB, "cache_frames": 32})
    lab_state.clear()


def rd_node_in(scene):
    return next(n for n in scene.compositing_node_group.nodes if n.bl_idname == NODE)


# ---------------------------------------------------------------------------------------------
@H.guard("reference")
def test_reference():
    img, m = mask_image(*SIZE)
    u0, v0 = init_ref(m)
    cases = [
        ("coral wrap", {}, {}, 'WRAP'),
        ("clamp edges", {"edge_mode": 'CLAMP'}, {}, 'CLAMP'),
        ("mitosis, Du 0.8 Dv 0.4 dt 1.2", {"preset": 'MITOSIS'},
         {"Du": 0.8, "Dv": 0.4, "dt": 1.2}, 'WRAP'),
    ]
    for name, props, inputs, edge in cases:
        kw = {}
        if "preset" in props:
            kw = {"f": 0.0367, "k": 0.0649}
        kw.update({"du": inputs.get("Du", DU), "dv": inputs.get("Dv", DV),
                   "dt": inputs.get("dt", DT)})
        for dev, tol in (("CPU", TOL), ("GPU", TOL)):
            scene, node = setup_tree(dev, props=props, inputs=inputs, images={"Seed": img})
            for f in range(1, 5):
                out = rend(scene, f)
                ru, rv = ref_run(u0, v0, ITERS * (f - 1), edge=edge, **kw)
                H.compare("%s %s: V, frame %d" % (dev, name, f), out[..., 0], rv, atol=tol,
                          quiet=True)
            cleanup()
    # the U output
    for dev in DEVICES:
        scene, node = setup_tree(dev, images={"Seed": img}, out="U")
        rend(scene, 1)
        out = rend(scene, 2)
        ru, rv = ref_run(u0, v0, ITERS)
        H.compare("%s: U output, frame 2" % dev, out[..., 0], ru, atol=TOL, quiet=True)
        cleanup()


@H.guard("per-pixel maps")
def test_maps():
    img, m = mask_image(*SIZE)
    u0, v0 = init_ref(m)
    rng = np.random.default_rng(5)
    fmap = (0.8 + 0.4 * rng.random(SIZE[::-1])).astype(np.float32)
    kmap = (0.9 + 0.2 * rng.random(SIZE[::-1])).astype(np.float32)
    for dev, tol in (("CPU", TOL), ("GPU", TOL)):
        scene, node = setup_tree(dev, images={"Seed": img, "Feed Map": fmap, "Kill Map": kmap})
        rend(scene, 1)
        out = rend(scene, 2)
        ru, rv = ref_run(u0, v0, ITERS, f=F * fmap.astype(np.float64),
                         k=K * kmap.astype(np.float64))
        H.compare("%s feed/kill maps modulate F and k per pixel" % dev, out[..., 0], rv, atol=tol,
                  quiet=True)
        plain = ref_run(u0, v0, ITERS)[1]
        H.check(np.abs(plain - rv).max() > 1e-3, "%s: the maps change the result" % dev)
        cleanup()
        # unlinked map sockets scale the value
        scene, node = setup_tree(dev, images={"Seed": img}, inputs={"Feed Map": 0.9})
        rend(scene, 1)
        out = rend(scene, 2)
        H.compare("%s unlinked Feed Map value scales F" % dev, out[..., 0],
                  ref_run(u0, v0, ITERS, f=F * 0.9)[1], atol=tol, quiet=True)
        cleanup()


@H.guard("seeding")
def test_seeding():
    img, m = mask_image(*SIZE)
    for dev in DEVICES:
        scene, node = setup_tree(dev, images={"Seed": img}, props={"seed_noise": 0.0})
        v = rend(scene, 1)[..., 0]
        H.check(np.array_equal(v > 0.5, m) and np.allclose(v[m], 1.0) and np.allclose(v[~m], 0.0),
                "%s: image seed: V = 1 where the seed is bright, 0 elsewhere (no noise)" % dev)
        cleanup()
        scene, node = setup_tree(dev, images={"Seed": img}, props={"seed_noise": 0.05, "seed": 7},
                                 out="U")
        u = rend(scene, 1)[..., 0]
        H.check(np.allclose(u[m], 0.5) and np.allclose(u[~m], 1.0), "%s: U = 0.5 inside, 1 outside" % dev)
        cleanup()
    # noise seeding: unlinked Seed; hashed blobs, identical on both devices (integer maths)
    res = {}
    for dev in DEVICES:
        scene, node = setup_tree(dev, size=(64, 48), props={"seed_noise": 0.03, "seed": 11,
                                                             "seed_density": 0.5})
        res[dev] = rend(scene, 1)[..., 0]
        cleanup()
    v = res["CPU"]
    H.compare("noise seed: CPU == GPU", res["GPU"], v, atol=1e-7, quiet=True)
    H.check((v == 1.0).sum() > 20 and ((v > 0.0) & (v < 1.0)).sum() > 1000 and v.max() == 1.0,
            "noise seed: blobs (V = 1) in a V-noise background (%d blob pixels)" % (v == 1.0).sum())
    H.check(v[v < 1.0].max() <= 0.03 + 1e-6, "background noise stays below the amplitude")
    for dev in DEVICES:
        scene, node = setup_tree(dev, size=(64, 48), props={"seed_noise": 0.0, "seed_density": 0.0})
        H.check(not rend(scene, 1)[..., 0].any(), "%s: density 0 and no noise: V is all 0" % dev)
        cleanup()
    # a different seed gives different blobs
    scene, node = setup_tree("CPU", size=(64, 48), props={"seed_noise": 0.0, "seed": 12,
                                                          "seed_density": 0.5})
    H.check(not np.array_equal(rend(scene, 1)[..., 0] == 1.0, v == 1.0), "seed property changes the blobs")
    cleanup()


@H.guard("semantics")
def test_semantics():
    img, m = mask_image(*SIZE)
    u0, v0 = init_ref(m)

    def refv(n, **kw):
        return ref_run(u0, v0, n, **kw)[1]

    for dev in DEVICES:
        tol = TOL
        scene, node = setup_tree(dev, images={"Seed": img})
        first = {f: rend(scene, f) for f in range(1, 7)}
        H.check(stream_of(node).last_kind == "STEP", "%s: sequential frames step" % dev)
        for f in (1, 6):
            H.compare("%s frame %d = %d iterations" % (dev, f, ITERS * (f - 1)), first[f][..., 0],
                      refv(ITERS * (f - 1)), atol=tol, quiet=True)
        again = rend(scene, 6)
        H.check(stream_of(node).last_kind == "REPEAT", "%s: re-render repeats" % dev)
        H.compare("%s re-render idempotent" % dev, again, first[6], atol=1e-7, quiet=True)
        # live edit: iterations per frame, shown without advancing the simulation
        node.inputs["Iterations per Frame"].default_value = 5
        H.compare("%s parameter tweak recomputes from the pre-step state" % dev,
                  rend(scene, 6)[..., 0], refv(ITERS * 4 + 5), atol=tol, quiet=True)
        node.inputs["Iterations per Frame"].default_value = ITERS
        H.compare("%s tweak undone" % dev, rend(scene, 6), first[6], atol=1e-7, quiet=True)
        # scrub back
        back = rend(scene, 3)
        H.check(stream_of(node).last_kind == "RESTORE", "%s: scrub back restores" % dev)
        H.compare("%s scrub back restores frame 3" % dev, back, first[3], atol=1e-7, quiet=True)
        H.compare("%s continues after restore" % dev, rend(scene, 4), first[4], atol=1e-7,
                  quiet=True)
        # jump forward: catches up with the same result as stepping
        jump = rend(scene, 12)
        H.check(stream_of(node).last_kind == "CATCH_UP", "%s: jump catches up" % dev)
        H.compare("%s catch-up to frame 12" % dev, jump[..., 0], refv(ITERS * 11), atol=2 * tol,
                  quiet=True)
        node.max_catch_up = 3
        rend(scene, 40)
        H.check(stream_of(node).last_kind == "HOLD", "%s: jump beyond max catch-up holds" % dev)
        node.max_catch_up = 64
        # start frame resets
        reset = rend(scene, 1)
        H.check(stream_of(node).last_kind == "RESET", "%s: start frame resets" % dev)
        H.compare("%s start frame shows the seed" % dev, reset[..., 0], v0, atol=1e-7, quiet=True)
        H.compare("%s restarts after reset" % dev, rend(scene, 2)[..., 0], refv(ITERS), atol=tol,
                  quiet=True)
        # Reset button
        uid = node.lab_uid
        bpy.ops.compositor_lab.reset_state(uid=uid)
        H.check(lab_state.streams_for(uid) == [], "%s: Reset operator clears the streams" % dev)
        H.compare("%s evaluation after Reset starts over" % dev, rend(scene, 5)[..., 0], v0,
                  atol=1e-7, quiet=True)
        cleanup()
    # scene start frame 10
    scene, node = setup_tree("CPU", images={"Seed": img}, start=10)
    rend(scene, 10)
    H.compare("scene start 10: frame 11 is one step", rend(scene, 11)[..., 0], refv(ITERS),
              atol=TOL, quiet=True)
    out = rend(scene, 10)
    st = stream_of(node)
    if st.key[1] == "UNKNOWN":
        H.note("build has no context.frame_start (F4)")
    else:
        H.check(st.last_kind == "RESET", "scene start frame resets")
        H.compare("scene start frame shows the seed", out[..., 0], v0, atol=1e-7, quiet=True)
    cleanup()
    # Pre-roll: the stateless fallback on a still render
    for dev in DEVICES:
        scene, node = setup_tree(dev, images={"Seed": img}, props={"preroll": 3})
        tol = TOL
        H.compare("%s pre-roll 3 frames: a still render shows 3 x %d iterations" % (dev, ITERS),
                  rend(scene, 1)[..., 0], refv(3 * ITERS), atol=tol, quiet=True)
        H.compare("%s pre-roll then steps" % dev, rend(scene, 2)[..., 0], refv(4 * ITERS),
                  atol=tol, quiet=True)
        cleanup()
    # a duplicate node has its own state
    for dev in DEVICES:
        scene, node = setup_tree(dev, images={"Seed": img})
        t2 = scene.compositing_node_group.copy()
        for f in range(1, 4):
            rend(scene, f)
        scene.compositing_node_group = t2
        n2 = rd_node_in(scene)
        out = rend(scene, 4)
        H.check(n2.lab_uid != node.lab_uid and stream_of(n2).last_kind == "RESET",
                "%s: duplicate starts its own state" % dev)
        H.compare("%s duplicate shows its seed, not the original's state" % dev, out[..., 0], v0,
                  atol=1e-7, quiet=True)
        bpy.data.node_groups.remove(t2)
        cleanup()


@H.guard("scale")
def test_scale():
    w, h = 40, 30
    img, m = mask_image(w, h)
    for dev in DEVICES:
        scene, node = setup_tree(dev, props={"scale": 2}, images={"Seed": img})
        out = rend(scene, 1)
        H.check(out.shape[:2] == (h, w), "%s: output has the render size at Scale 2" % dev)
        st = stream_of(node)
        H.check(st.nbytes <= 2 * 8 * 20 * 15 * 2 + 64, "%s: state is stored at simulation size "
                "(%d bytes)" % (dev, st.nbytes))
        # the seed disc is smoothly upsampled: values strictly between 0 and 1 at its rim
        H.check(((out[..., 0] > 0.05) & (out[..., 0] < 0.95)).sum() > 10 and out[..., 0].max() <= 1.0,
                "%s: scale 2 output is bilinearly upsampled" % dev)
        for f in range(2, 6):
            rend(scene, f)
        H.check(stream_of(node).last_kind == "STEP", "%s: scale 2 steps" % dev)
        node.scale = 4
        out = rend(scene, 6)
        H.check(stream_of(node).last_kind == "RESET" and out.shape[:2] == (h, w),
                "%s: changing Scale restarts the state" % dev)
        cleanup()
    # odd sizes, scale not dividing the size
    for dev in DEVICES:
        for size, sc in (((33, 17), 2), ((7, 5), 1), ((4, 4), 3), ((50, 20), 4)):
            scene, node = setup_tree(dev, size=size, props={"scale": sc, "seed_density": 1.0,
                                                            "seed_noise": 0.02})
            for f in range(1, 4):
                out = rend(scene, f)
            H.check(out.shape[:2] == (size[1], size[0]) and np.isfinite(out).all()
                    and out.min() >= 0.0 and out.max() <= 1.0,
                    "%s: size %s scale %d renders, finite and in [0, 1]" % (dev, size, sc))
            cleanup()


@H.guard("cpu vs gpu")
def test_cpu_gpu():
    size = (48, 36)
    img, m = mask_image(*size, seed=4, blobs=4)
    fm = (0.9 + 0.2 * np.random.default_rng(8).random(size[::-1])).astype(np.float32)
    cases = [
        ("coral, image seed", {}, {"Seed": img}, {}),
        ("clamp, maps", {"edge_mode": 'CLAMP'}, {"Seed": img, "Feed Map": fm}, {}),
        ("mitosis, noise seed", {"preset": 'MITOSIS', "seed_density": 0.15, "seed_noise": 0.02}, {},
         {}),
        ("scale 2 worms, noise seed", {"preset": 'WORMS', "scale": 2, "seed_density": 0.6}, {}, {}),
    ]
    for name, props, images, inputs in cases:
        outs = {}
        for dev in DEVICES:
            scene, node = setup_tree(dev, size=size, props=props, images=images, inputs=inputs)
            outs[dev] = [rend(scene, f) for f in range(1, 6)]
            cleanup()
        for i, (c, g) in enumerate(zip(outs["CPU"], outs["GPU"])):
            H.compare("cpu vs gpu: %s, frame %d" % (name, i + 1), g[..., 0], c[..., 0], atol=5e-6,
                      quiet=True)
        H.check(not np.allclose(outs["CPU"][0], outs["CPU"][4], atol=1e-3),
                "%s: the sequence changes" % name)


def coverage(v, thr=0.15):
    return float((v > thr).mean())


@H.guard("statistics")
def test_statistics():
    """Long runs diverge chaotically between the devices: compare pattern statistics."""
    size = (64, 48)
    for preset in ('CORAL', 'MITOSIS'):
        res = {}
        for dev in DEVICES:
            scene, node = setup_tree(dev, size=size, props={"preset": preset, "seed_density": 0.15,
                                                            "seed_noise": 0.02})
            for f in range(1, 31):
                out = rend(scene, f)
            res[dev] = out[..., 0]
            cleanup()
        c, g = coverage(res["CPU"]), coverage(res["GPU"])
        H.note("%s coverage after 29 frames: cpu %.3f gpu %.3f" % (preset, c, g))
        H.check(abs(c - g) <= 0.10, "%s: CPU / GPU pattern coverage agrees (%.3f vs %.3f)" %
                (preset, c, g))
        H.check(abs(float(res["CPU"].mean()) - float(res["GPU"].mean())) < 0.05,
                "%s: CPU / GPU mean V agrees" % preset)


# preset -> (min, max) coverage (fraction of pixels with V > 0.15) after 40 frames of 40 iterations
PRESET_RANGES = {'CORAL': (0.3, 0.8), 'MITOSIS': (0.05, 0.5), 'WORMS': (0.1, 0.7),
                 'SPOTS': (0.03, 0.4), 'MAZE': (0.25, 0.8), 'BUBBLES': (0.04, 0.5)}


@H.guard("presets")
def test_presets():
    size = (64, 48)
    covs = {}
    for preset in ('CORAL', 'MITOSIS', 'WORMS', 'SPOTS', 'MAZE', 'BUBBLES'):
        scene, node = setup_tree("CPU", size=size, props={"preset": preset, "seed_density": 0.12,
                                                          "seed_noise": 0.02},
                                 inputs={"Iterations per Frame": 40})
        for f in range(1, 41):
            out = rend(scene, f)
        v = out[..., 0]
        covs[preset] = coverage(v)
        lo, hi = PRESET_RANGES.get(preset, (0.0, 1.0))
        H.note("%s: coverage %.3f, V max %.3f, std %.3f" % (preset, covs[preset], v.max(), v.std()))
        H.check(lo <= covs[preset] <= hi and v.std() > 0.02,
                "%s: non-trivial pattern after 40 frames (coverage %.3f in [%.2f, %.2f])"
                % (preset, covs[preset], lo, hi))
        cleanup()
    H.check(len({round(c, 2) for c in covs.values()}) >= 4, "presets give different patterns")
    # the Preset enum writes the sockets
    scene, node = setup_tree("CPU")
    node.preset = 'MAZE'
    H.check(abs(node.inputs["Feed"].default_value - 0.029) < 1e-6
            and abs(node.inputs["Kill"].default_value - 0.057) < 1e-6, "preset sets Feed / Kill")
    node.preset = 'CUSTOM'
    node.inputs["Feed"].default_value = 0.03
    H.check(abs(node.inputs["Feed"].default_value - 0.03) < 1e-6, "Custom keeps the edited values")
    cleanup()


@H.guard("stability")
def test_stability():
    img, m = mask_image(*SIZE)
    extreme = [
        ("feed 1, kill 1", {"Feed": 1.0, "Kill": 1.0}),
        ("feed 0, kill 0", {"Feed": 0.0, "Kill": 0.0}),
        ("large diffusion and dt", {"Du": 5.0, "Dv": 5.0, "dt": 5.0, "Iterations per Frame": 200}),
        ("out of range parameters", {"Feed": -3.0, "Kill": 50.0, "Du": -1.0, "dt": 1e9}),
        ("zero iterations", {"Iterations per Frame": 0}),
        ("many iterations", {"Iterations per Frame": 5000}),
    ]
    for dev in DEVICES:
        for name, inputs in extreme:
            for images in ({}, {"Seed": img}):
                scene, node = setup_tree(dev, props={"seed_noise": 0.1}, inputs=inputs,
                                         images=images)
                for f in range(1, 4):
                    o = rend(scene, f)
                H.check(np.isfinite(o).all() and o.min() >= 0.0 and o.max() <= 1.0,
                        "%s: %s%s: finite, V in [0, 1]" % (dev, name, " (image seed)" if images else ""))
                cleanup()
    # mass: U + V stay in [0, 1] and the total V does not blow up on a long default run
    for dev in DEVICES:
        scene, node = setup_tree(dev, props={"seed_density": 0.5, "seed_noise": 0.02})
        for f in range(1, 21):
            v = rend(scene, f)[..., 0]
        H.check(0.0 <= v.min() and v.max() <= 1.0 and v.mean() < 0.6, "%s: V bounded after 20 frames "
                "(mean %.3f)" % (dev, v.mean()))
        cleanup()


@H.guard("color output")
def test_color():
    img, m = mask_image(*SIZE)
    a, b = (0.1, 0.2, 0.3, 1.0), (0.9, 0.5, 0.1, 1.0)
    for dev in DEVICES:
        # one tree per output, same frame: the same state (re-render at a fixed frame repeats)
        scene, node = setup_tree(dev, images={"Seed": img},
                                 props={"color_a": a, "color_b": b, "map_low": 0.1, "map_high": 0.6})
        rend(scene, 1)
        v = rend(scene, 2)[..., 0]
        tree = scene.compositing_node_group
        out_node = next(n for n in tree.nodes if n.bl_idname == 'NodeGroupOutput')
        tree.links.new(node.outputs["Color"], out_node.inputs[0])
        col = rend(scene, 2)
        t = np.clip((v - 0.1) / 0.5, 0, 1)
        t = t * t * (3 - 2 * t)
        ref = np.array(a) + (np.array(b) - np.array(a)) * t[..., None]
        H.compare("%s: Color = smoothstep gradient of V" % dev, col, ref, atol=TOL, quiet=True)
        H.check(stream_of(node).last_kind == "REPEAT", "%s: switching the output re-renders" % dev)
        cleanup()


test_reference()
test_maps()
test_seeding()
test_semantics()
test_scale()
test_cpu_gpu()
test_statistics()
test_presets()
test_stability()
test_color()
H.finish()
