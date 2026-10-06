# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""F4: context.kind, context.is_animation_playing, context.frame_start / frame_end.

Background mode only runs the render pipeline, so this checks 'RENDER'. The interactive kinds are
covered by test_eval_kind_gui.py.
"""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw_common as fw  # noqa: E402
from fw_common import check, fill_output  # noqa: E402

REC = []


def snapshot(context):
    REC.append(dict(kind=context.kind, playing=context.is_animation_playing,
                    start=context.frame_start, end=context.frame_end,
                    types=(type(context.kind).__name__, type(context.is_animation_playing).__name__,
                           type(context.frame_start).__name__, type(context.frame_end).__name__)))


def evaluate_cpu(self, inputs, outputs, context):
    snapshot(context)
    fill_output(outputs, "Image", (1, 0, 0, 1), 'CPU')


def evaluate_gpu(self, inputs, outputs, context):
    snapshot(context)
    fill_output(outputs, "Image", (1, 0, 0, 1), 'GPU')


NODE = type("CompositorNodeFwKind", (fw.CompositorTestNode, bpy.types.CompositorNode), {
    "bl_idname": "CompositorNodeFwKind",
    "bl_label": "FW Kind",
    "init": lambda self, context: self.outputs.new('NodeSocketColor', "Image"),
    "evaluate_cpu": evaluate_cpu,
    "evaluate_gpu": evaluate_gpu,
})


def test_render(device, scene, start, end, frame):
    fw.setup_scene(scene, device, 32, 24)
    scene.frame_start, scene.frame_end = start, end
    scene.frame_set(frame)
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new("CompositorNodeFwKind")
    tree.links.new(node.outputs["Image"], out.inputs[0])
    REC.clear()
    px = fw.render_pixels(scene, "kind_%s_%d_%d" % (device, start, end))
    label = "[%s] range %d-%d frame %d" % (device, start, end, frame)
    check(len(REC) >= 1, "%s: node evaluated" % label)
    if not REC:
        return
    for rec in REC:
        check(rec["kind"] == 'RENDER', "%s: kind %r" % (label, rec["kind"]))
        check(rec["playing"] is False, "%s: is_animation_playing %r" % (label, rec["playing"]))
        check(rec["start"] == start and rec["end"] == end,
              "%s: frame range %r-%r" % (label, rec["start"], rec["end"]))
        check(rec["types"] == ("str", "bool", "int", "int"), "%s: types %s" % (label, rec["types"]))
    check((px == (1, 0, 0, 1)).all(), "%s: render" % label)


def main():
    bpy.utils.register_class(NODE)
    scene = bpy.context.scene
    for device in fw.DEVICES:
        fw.run_case(device, "render kind 1-250", test_render, device, scene, 1, 250, 5)
        fw.run_case(device, "render kind 10-42", test_render, device, scene, 10, 42, 20)
        fw.run_case(device, "render kind range from 0", test_render, device, scene, 0, 7, 3)
    fw.finish()


main()
