# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Feedback / Trails and the stateful-node machinery, end to end: frame sequences are rendered
with scene.frame_set + bpy.ops.render.render (the RENDER stream), on CPU and GPU.
   Blender -b --factory-startup --python-exit-code 1 --python test_feedback.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import state as lab_state  # noqa: E402

NODE = "CompositorNodeLabFeedback"
SIZE = (32, 24)
F32 = np.float32
X = H.test_image(SIZE[0], SIZE[1], seed=3)
A, D = 0.5, 0.9
TOL = 5e-6
DEVICES = ("CPU", "GPU")


def setup_tree(device, size=SIZE, props=None, inputs=None, image=X, start=1):
    lab_state.clear()
    scene = H.configure_scene(size, device)
    scene.frame_start, scene.frame_end = start, 250
    scene.frame_set(start)
    inp = {"Amount": A, "Decay": D}
    inp.update(inputs or {})
    node = H.build_tree(scene, NODE, props, inp, images={"Image": image})
    return scene, node


def rend(scene, f):
    scene.frame_set(int(f))
    return H.render(scene)


def closed_form(n, x=X, a=A, d=D):
    """Static input, no transform, Mix: out_n = s + (a d)^n (x - s), s = x (1 - a) / (1 - a d)."""
    s = x * (1.0 - a) / (1.0 - a * d)
    return s + (a * d) ** n * (x - s)


def stream_of(node):
    streams = lab_state.streams_for(node.lab_uid)
    return streams[0] if streams else None


def cleanup():
    H._clear_images()
    lab_state.configure(**{"global_bytes": 1024 * lab_state.MB, "cache_frames": 32})
    lab_state.clear()


# ---------------------------------------------------------------------------------------------
@H.guard("identity")
def test_identity():
    scene, node = setup_tree("CPU")
    other = scene.compositing_node_group.nodes.new(NODE)
    H.check(len(node.lab_uid) == 32 and node.lab_uid != other.lab_uid,
            "every node gets its own lab_uid")
    H.check(node.bl_rna.properties["lab_uid"].is_hidden, "lab_uid is a hidden property")
    uid = node.lab_uid
    rend(scene, 1)
    rend(scene, 2)
    st = stream_of(node)
    H.check(st is not None and st.key[0] == uid,
            "evaluation (a copied tree) uses the original node's lab_uid")
    H.check(st is not None and st.last_kind == "STEP",
            "state survives from frame to frame (copy() is not called for evaluation copies)")
    kind = st.key[1] if st else None
    H.note("evaluation kind seen by the node: %s" % kind)
    # duplicating a tree runs Node.copy(): the duplicates get new uids
    t2 = scene.compositing_node_group.copy()
    uids = {n.lab_uid for n in scene.compositing_node_group.nodes if n.bl_idname == NODE}
    uids2 = {n.lab_uid for n in t2.nodes if n.bl_idname == NODE}
    H.check(uids.isdisjoint(uids2) and len(uids2) == 2, "duplicated nodes get new uids (copy())")
    H.check(node.lab_uid == uid, "the original keeps its uid")
    bpy.data.node_groups.remove(t2)
    cleanup()


@H.guard("closed form")
def test_closed_form():
    for dev in DEVICES:
        scene, node = setup_tree(dev)
        for f in range(1, 8):
            out = rend(scene, f)
            H.compare("%s decay closed form, frame %d (n=%d)" % (dev, f, f - 1), out,
                      closed_form(f - 1), atol=TOL, quiet=True)
        st = stream_of(node)
        H.check(st is not None and st.last_kind == "STEP", "%s: sequential frames step" % dev)
        cleanup()


@H.guard("semantics")
def test_semantics():
    for dev in DEVICES:
        scene, node = setup_tree(dev)
        first = {f: rend(scene, f) for f in range(1, 9)}
        # re-render of the same frame is idempotent (repeat from the pre-step state)
        again = rend(scene, 8)
        H.check(stream_of(node).last_kind == "REPEAT", "%s: re-render repeats" % dev)
        H.compare("%s re-render idempotent" % dev, again, first[8], atol=1e-7, quiet=True)
        again = rend(scene, 8)
        H.compare("%s re-render idempotent (twice)" % dev, again, first[8], atol=1e-7, quiet=True)
        # a live edit on the same frame shows without stepping the simulation forward
        node.inputs["Decay"].default_value = 0.5
        edited = rend(scene, 8)
        ref = X * (1 - A) + A * 0.5 * first[7]
        H.compare("%s parameter tweak recomputes from the pre-step state" % dev, edited, ref,
                  atol=TOL, quiet=True)
        node.inputs["Decay"].default_value = D
        H.compare("%s tweak undone" % dev, rend(scene, 8), first[8], atol=1e-7, quiet=True)
        # scrub back to a cached frame
        back = rend(scene, 3)
        H.check(stream_of(node).last_kind == "RESTORE", "%s: scrub back restores" % dev)
        H.compare("%s scrub back restores frame 3" % dev, back, first[3], atol=1e-7, quiet=True)
        H.compare("%s continues after restore" % dev, rend(scene, 4), first[4], atol=1e-7,
                  quiet=True)
        # jump forward: catch up k steps
        jump = rend(scene, 20)
        H.check(stream_of(node).last_kind == "CATCH_UP", "%s: jump catches up" % dev)
        H.compare("%s catch-up to frame 20" % dev, jump, closed_form(19), atol=TOL, quiet=True)
        # beyond max_catch_up: hold the state
        node.max_catch_up = 5
        held = rend(scene, 40)
        H.check(stream_of(node).last_kind == "HOLD", "%s: jump beyond max catch-up holds" % dev)
        H.compare("%s hold shows the held state" % dev, held, closed_form(19), atol=TOL, quiet=True)
        H.compare("%s steps on after a hold" % dev, rend(scene, 41), closed_form(20), atol=TOL,
                  quiet=True)
        node.max_catch_up = 64
        # start frame resets
        reset = rend(scene, 1)
        H.check(stream_of(node).last_kind == "RESET", "%s: start frame resets" % dev)
        H.compare("%s start frame outputs the input" % dev, reset, X, atol=1e-7, quiet=True)
        H.compare("%s restarts after reset" % dev, rend(scene, 2), closed_form(1), atol=TOL,
                  quiet=True)
        # operator: Reset button
        uid = node.lab_uid
        bpy.ops.compositor_lab.reset_state(uid=uid)
        H.check(lab_state.streams_for(uid) == [], "%s: Reset operator clears the node's streams" % dev)
        H.compare("%s evaluation after Reset starts over" % dev, rend(scene, 3), X, atol=1e-7,
                  quiet=True)
        cleanup()


@H.guard("start frame")
def test_start_frame():
    scene, node = setup_tree("CPU", start=10)
    rend(scene, 10)
    rend(scene, 11)
    out = rend(scene, 12)
    H.compare("scene start 10: frame 12 is n=2", out, closed_form(2), atol=TOL, quiet=True)
    out = rend(scene, 10)
    st = stream_of(node)
    if st is not None and st.key[1] == "UNKNOWN":
        H.note("build has no context.frame_start (F4): start frame falls back to 1")
        H.check(True, "start frame test skipped without F4")
    else:
        H.check(st.last_kind == "RESET", "scene start frame resets")
        H.compare("scene start frame outputs the input", out, X, atol=1e-7, quiet=True)
    cleanup()


@H.guard("duplicate")
def test_duplicate_independent():
    for dev in DEVICES:
        scene, node = setup_tree(dev)
        tree = scene.compositing_node_group
        t2 = tree.copy()
        for f in range(1, 6):
            rend(scene, f)
        scene.compositing_node_group = t2
        n2 = next(n for n in t2.nodes if n.bl_idname == NODE)
        out = rend(scene, 6)
        H.check(n2.lab_uid != node.lab_uid, "%s: duplicate has its own uid" % dev)
        H.check(stream_of(n2).last_kind == "RESET", "%s: duplicate starts its own state" % dev)
        H.compare("%s duplicate does not continue the original's simulation" % dev, out, X,
                  atol=1e-7, quiet=True)
        scene.compositing_node_group = tree
        H.compare("%s original continues undisturbed" % dev, rend(scene, 6), closed_form(5),
                  atol=TOL, quiet=True)
        bpy.data.node_groups.remove(t2)
        cleanup()


@H.guard("preroll")
def test_preroll():
    for dev in DEVICES:
        scene, node = setup_tree(dev, props={"preroll": 6})
        H.compare("%s pre-roll: still render = 6 iterations" % dev, rend(scene, 1),
                  closed_form(6), atol=TOL, quiet=True)
        H.compare("%s pre-roll then steps" % dev, rend(scene, 2), closed_form(7), atol=TOL,
                  quiet=True)
        cleanup()


@H.guard("transform")
def test_shift():
    img = H.test_image(32, 24, seed=5)
    for dev in DEVICES:
        scene, node = setup_tree(dev, image=img, inputs={"Amount": 1.0, "Decay": 0.5,
                                                         "Offset X": 3.0})
        H.compare("%s transform: frame 1 is the input" % dev, rend(scene, 1), img, atol=1e-7,
                  quiet=True)
        out = rend(scene, 2)
        ref = np.empty_like(img)
        ref[:, 3:] = 0.5 * img[:, :-3]
        ref[:, :3] = 0.5 * img[:, :1]
        H.compare("%s offset x=3 shifts the previous frame right and fades it" % dev, out, ref,
                  atol=1e-6, quiet=True)
        cleanup()


@H.guard("cpu vs gpu")
def test_cpu_gpu():
    img = H.test_image(40, 30, seed=7, alpha=True)
    cases = [
        ("mix, zoom, rotate, offset, hue", {}, {"Zoom": 1.05, "Rotation": 7.0, "Offset X": 1.5,
                                                 "Offset Y": -2.25, "Hue Shift": 25.0}),
        ("screen blend, zoom", {"blend_type": "SCREEN", "edge_mode": "MIRROR"},
         {"Zoom": 0.96, "Decay": 0.98}),
        ("overlay blend, repeat edges", {"blend_type": "OVERLAY", "edge_mode": "REPEAT"},
         {"Rotation": -12.0, "Amount": 0.7}),
    ]
    for name, props, inputs in cases:
        outs = {}
        for dev in DEVICES:
            scene, node = setup_tree(dev, size=(40, 30), props=props, inputs=inputs, image=img)
            outs[dev] = [rend(scene, f) for f in range(1, 6)]
            cleanup()
        for i, (c, g) in enumerate(zip(outs["CPU"], outs["GPU"])):
            H.compare("cpu vs gpu: %s, frame %d" % (name, i + 1), g, c, atol=2e-4, quiet=True)
        H.check(not np.allclose(outs["CPU"][0], outs["CPU"][4], atol=1e-3),
                "%s: the sequence changes" % name)


@H.guard("memory")
def test_memory():
    frame_bytes = SIZE[0] * SIZE[1] * 16
    for dev in DEVICES:
        scene, node = setup_tree(dev)
        lab_state.configure(global_bytes=6 * frame_bytes)
        outs = {f: rend(scene, f) for f in range(1, 21)}
        st = stream_of(node)
        H.check(lab_state.total_bytes() <= 6 * frame_bytes,
                "%s: global budget holds (%d <= %d)" % (dev, lab_state.total_bytes(),
                                                        6 * frame_bytes))
        cached = st.cached_frames()
        H.check(20 in cached and 2 not in cached and len(cached) <= 4,
                "%s: old frames evicted, newest kept (%s)" % (dev, cached))
        out = rend(scene, 2)
        H.check(stream_of(node).last_kind == "RESIM", "%s: evicted frame is re-simulated" % dev)
        H.compare("%s re-simulated frame is still right" % dev, out, closed_form(1), atol=TOL,
                  quiet=True)
        lab_state.configure(global_bytes=1024 * lab_state.MB)
        node.cache_frames = 3
        for f in range(1, 12):
            rend(scene, f)
        H.check(len(stream_of(node).cache) <= 3, "%s: cache_frames property caps the cache" % dev)
        cleanup()


test_identity()
test_closed_form()
test_semantics()
test_start_frame()
test_duplicate_independent()
test_preroll()
test_shift()
test_cpu_gpu()
test_memory()
H.finish()
