# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
GUI stress test for Python calls from compositor worker threads.

The interactive (backdrop) compositor evaluates Python nodes on a job thread while the main
thread keeps drawing the node's Python UI. Before the fix this could unbalance Blender's
Python call level, after which every Python call printed
"ERROR: Python context internal state bug".

Run, and check that the log has no such line and the exit code is 0:
    Blender --factory-startup --python test_thread_stress.py 2>&1 | tee log.txt
"""
import os
import sys
import traceback

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "addons"))
import pixel_sort_node as psn  # noqa: E402

psn.register()

TOGGLES = 300
W, H = 128, 64

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
image = bpy.data.images.new("src", W, H, float_buffer=True)
image.pixels.foreach_set(np.random.default_rng(3).random(W * H * 4).astype(np.float32))

tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
scene.compositing_node_group = tree
prev = tree.nodes.new('CompositorNodeImage')
prev.image = image
nodes = []
for i in range(8):
    n = tree.nodes.new(psn.CompositorNodePixelSort.bl_idname)
    n.location = (200 * (i + 1), 0)
    tree.links.new(prev.outputs[0], n.inputs["Image"])
    prev = n
    nodes.append(n)
tree.links.new(prev.outputs[0], tree.nodes.new('NodeGroupOutput').inputs[0])
tree.links.new(prev.outputs[0], tree.nodes.new('CompositorNodeViewer').inputs[0])

state = {"i": 0, "ready": False}


def setup_editor():
    area = max(bpy.context.screen.areas, key=lambda a: a.width * a.height)
    area.ui_type = 'CompositorNodeTree'
    space = area.spaces.active
    space.node_tree = tree
    space.show_backdrop = True
    for n in nodes:
        n.select = True


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    return None if state["i"] >= TOGGLES else 0.0


def toggle():
    try:
        if not state["ready"]:
            setup_editor()
            state["ready"] = True
            return 0.5
        i = state["i"]
        if i >= TOGGLES:
            print(f"STRESS: done, {TOGGLES} toggles")
            sys.stdout.flush()
            os._exit(0)
        node = nodes[i % len(nodes)]
        node.vertical = not node.vertical
        node.sort_key = ('LUMA', 'HUE', 'SATURATION')[i % 3]
        state["i"] += 1
        return 0.02
    except Exception:
        traceback.print_exc()
        os._exit(1)


bpy.app.timers.register(redraw, first_interval=0.5)
bpy.app.timers.register(toggle, first_interval=1.0)
