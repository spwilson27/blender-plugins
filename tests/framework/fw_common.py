# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Helpers shared by the framework tests (run in background mode, once per compositor device)."""
import os
import tempfile
import traceback

import bpy
import numpy as np

TMP = tempfile.mkdtemp(prefix="fw_test_")
FAILURES = []
DEVICES = ('CPU', 'GPU')


def check(ok, msg):
    print(("PASS: " if ok else "FAIL: ") + msg)
    if not ok:
        FAILURES.append(msg)


def finish():
    """Raise if any check failed, so `--python-exit-code 1` makes Blender exit nonzero."""
    if FAILURES:
        raise RuntimeError("%d check(s) failed:\n  " % len(FAILURES) + "\n  ".join(FAILURES))
    print("ALL PASSED")


def run_case(device, name, func, *args):
    """Run one test function, turning an exception into a failure."""
    print("--- [%s] %s" % (device, name))
    try:
        func(*args)
    except Exception:
        traceback.print_exc()
        check(False, "[%s] %s raised" % (device, name))


def setup_scene(scene, device, width, height):
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA'
    for owner in (scene.render, scene):
        if hasattr(owner, "compositor_device"):
            owner.compositor_device = device
            if hasattr(owner, "compositor_precision"):
                owner.compositor_precision = 'FULL'
            break
    else:
        raise RuntimeError("no compositor_device property found")


def new_tree(scene):
    """An empty compositing tree with a color group output. Returns (tree, group output node)."""
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
        scene.compositing_node_group = tree
    tree.nodes.clear()
    if not any(i.item_type == 'SOCKET' and i.in_out == 'OUTPUT'
               for i in tree.interface.items_tree):
        tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type="NodeSocketColor")
    return tree, tree.nodes.new('NodeGroupOutput')


def render_pixels(scene, name):
    """Render and return an (H, W, 4) float32 array, row 0 is the bottom."""
    path = os.path.join(TMP, name + ".exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(path):
        raise RuntimeError("render produced no file: " + path)
    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        buf = np.empty(w * h * img.channels, np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(h, w, img.channels).copy()
    finally:
        bpy.data.images.remove(img)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=2)
    return arr


class CompositorTestNode:
    """Base for the test nodes, usable in compositor trees only."""

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CompositorNodeTree'


def fill_output(outputs, name, value, device):
    """Fill the image output `name` with a constant color (CPU buffer or GPU texture)."""
    if device == 'CPU':
        np.asarray(outputs[name])[...] = value
    else:
        outputs[name].clear(format='FLOAT', value=value)
