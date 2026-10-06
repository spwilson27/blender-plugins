# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
GUI test: changing a Python node's options must re-run the compositor.

Background mode never runs the interactive compositor, so this runs the real UI:
    Blender --factory-startup --python test_property_updates.py

For each change it waits for the interactive compositor job and checks that the node's
evaluate method ran again and that the Viewer image changed. Exits Blender with
code 0 when all checks pass, 1 otherwise. Results are also printed as PASS/FAIL lines.
"""
import os
import sys
import traceback

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "addons"))
import pixel_sort_node as psn  # noqa: E402

psn.register()

W, H = 96, 48
calls = []
cls = psn.CompositorNodePixelSort
for _name in ("evaluate_cpu", "evaluate_gpu"):
    _orig = getattr(cls, _name)

    def _wrap(self, inputs, outputs, _orig=_orig, _name=_name):
        calls.append(_name)
        return _orig(self, inputs, outputs)

    setattr(cls, _name, _wrap)

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100

rng = np.random.default_rng(7)
src = rng.random((H, W, 4)).astype(np.float32)
src[..., 3] = 1.0
image = bpy.data.images.new("src", W, H, float_buffer=True)
image.pixels.foreach_set(src.ravel())

tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
scene.compositing_node_group = tree
n_img = tree.nodes.new('CompositorNodeImage')
n_img.image = image
node = tree.nodes.new(cls.bl_idname)
n_out = tree.nodes.new('NodeGroupOutput')
n_view = tree.nodes.new('CompositorNodeViewer')
tree.links.new(n_img.outputs[0], node.inputs["Image"])
tree.links.new(node.outputs[0], n_out.inputs[0])
tree.links.new(node.outputs[0], n_view.inputs[0])
# Wide thresholds so every option visibly changes the result.
node.inputs["Lower"].default_value = 0.0
node.inputs["Upper"].default_value = 1.0


def show_compositor():
    """Turn the largest area into a compositor node editor (the interactive compositor
    only runs when the result is shown somewhere)."""
    area = max(bpy.context.screen.areas, key=lambda a: a.width * a.height)
    area.ui_type = 'CompositorNodeTree'
    area.spaces.active.node_tree = tree
    # The compositor job only runs for visible results, e.g. the backdrop.
    area.spaces.active.show_backdrop = True


def viewer_pixels():
    img = bpy.data.images.get("Viewer Node")
    if img is None or img.size[0] == 0:
        return None
    px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(px)
    return px


# Each step: (description, function applying the change). The first entry primes the cache.
STEPS = [
    ("socket value: Lower (control, built-in RNA property)",
     lambda: setattr(node.inputs["Lower"], "default_value", 0.1)),
    ("bpy.props BoolProperty: vertical", lambda: setattr(node, "vertical", True)),
    ("bpy.props BoolProperty: reverse", lambda: setattr(node, "reverse", True)),
    ("bpy.props EnumProperty: sort_key", lambda: setattr(node, "sort_key", 'HUE')),
    ("bpy.props EnumProperty: mask_key", lambda: setattr(node, "mask_key", 'RED')),
    ("bpy.props BoolProperty: invert_mask", lambda: setattr(node, "invert_mask", True)),
    ("compositor device CPU -> GPU", lambda: setattr(scene.render, "compositor_device", 'GPU')),
    ("bpy.props BoolProperty on GPU: vertical", lambda: setattr(node, "vertical", False)),
]

state = {"step": -2, "wait": 0, "before_calls": 0, "before_px": None}
results = []
TIMEOUT_TICKS = 60  # 6 seconds at 0.1s per tick


def finish():
    failed = [r for r in results if not r[1]]
    print("=" * 60)
    for desc, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}: {desc} ({detail})")
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    sys.stdout.flush()
    os._exit(1 if failed or not results else 0)


def tick():
    try:
        s = state["step"]
        if s == -2:
            show_compositor()
            # A built-in socket change reliably triggers the first evaluation.
            node.inputs["Upper"].default_value = 0.99
            state["step"] = -1
            state["wait"] = 0
            return 0.1
        if s == -1:
            # Wait for the initial evaluation so later steps measure only their own change.
            state["wait"] += 1
            if (calls and viewer_pixels() is not None) or state["wait"] > TIMEOUT_TICKS:
                if not calls:
                    results.append(("initial evaluation", False, "compositor never ran"))
                    finish()
                state["step"] = 0
                state["wait"] = -1
            return 0.1
        if s >= len(STEPS):
            finish()
            return None

        desc, apply = STEPS[s]
        if state["wait"] == -1:
            state["before_calls"] = len(calls)
            state["before_px"] = viewer_pixels()
            apply()
            state["wait"] = 0
            return 0.1

        state["wait"] += 1
        new_calls = len(calls) - state["before_calls"]
        px = viewer_pixels()
        changed = (px is not None and state["before_px"] is not None and
                   not np.array_equal(px, state["before_px"]))
        # Device switches are expected to re-run but the image may be identical.
        needs_change = not desc.startswith("compositor device")
        if new_calls > 0 and (changed or not needs_change):
            results.append((desc, True, f"{new_calls} evaluation(s), viewer changed={changed}"))
        elif state["wait"] > TIMEOUT_TICKS:
            results.append((desc, False,
                            f"{new_calls} evaluation(s) and viewer changed={changed} after "
                            f"{TIMEOUT_TICKS / 10:.0f}s"))
        else:
            return 0.1

        state["step"] += 1
        state["wait"] = -1
        return 0.1
    except Exception:
        traceback.print_exc()
        results.append(("test harness", False, "exception, see traceback"))
        finish()


bpy.app.timers.register(tick, first_interval=1.0)
