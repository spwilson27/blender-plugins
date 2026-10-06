# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""F1: Python nodes without image inputs output an image over the compositing domain."""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw_common as fw  # noqa: E402
from fw_common import check, fill_output  # noqa: E402

REC = {}


def record_shape(key, outputs, device):
    out = outputs["Image"]
    REC[key] = tuple(np.asarray(out).shape[:2]) if device == 'CPU' else (out.height, out.width)


class CompositorNodeFwGenNoInput(fw.CompositorTestNode, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeFwGenNoInput"
    bl_label = "FW Gen No Input"

    def init(self, context):
        self.outputs.new('NodeSocketColor', "Image")

    def evaluate_cpu(self, inputs, outputs):
        record_shape("gen", outputs, 'CPU')
        fill_output(outputs, "Image", (0.25, 0.5, 0.75, 1.0), 'CPU')

    def evaluate_gpu(self, inputs, outputs):
        record_shape("gen", outputs, 'GPU')
        fill_output(outputs, "Image", (0.25, 0.5, 0.75, 1.0), 'GPU')


class CompositorNodeFwGenValue(fw.CompositorTestNode, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeFwGenValue"
    bl_label = "FW Gen Value"

    def init(self, context):
        self.inputs.new('NodeSocketFloat', "Value").default_value = 0.5
        self.inputs.new('NodeSocketColor', "Tint").default_value = (0.1, 0.2, 0.3, 1.0)
        self.outputs.new('NodeSocketColor', "Image")

    def _run(self, inputs, outputs, device):
        record_shape("value", outputs, device)
        REC["value_types"] = (type(inputs["Value"]).__name__, type(inputs["Tint"]).__name__)
        r, g, b, _ = inputs["Tint"]
        v = inputs["Value"]
        fill_output(outputs, "Image", (r + v, g + v, b + v, 1.0), device)

    def evaluate_cpu(self, inputs, outputs):
        self._run(inputs, outputs, 'CPU')

    def evaluate_gpu(self, inputs, outputs):
        self._run(inputs, outputs, 'GPU')


class CompositorNodeFwImageIn(fw.CompositorTestNode, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeFwImageIn"
    bl_label = "FW Image In"

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Image")
        self.outputs.new('NodeSocketColor', "Image")

    def evaluate_cpu(self, inputs, outputs):
        record_shape("image", outputs, 'CPU')
        fill_output(outputs, "Image", (0.9, 0.8, 0.7, 1.0), 'CPU')

    def evaluate_gpu(self, inputs, outputs):
        record_shape("image", outputs, 'GPU')
        fill_output(outputs, "Image", (0.9, 0.8, 0.7, 1.0), 'GPU')


CLASSES = (CompositorNodeFwGenNoInput, CompositorNodeFwGenValue, CompositorNodeFwImageIn)


def make_image(width, height):
    img = bpy.data.images.new("FwImage_%dx%d" % (width, height), width, height, alpha=True,
                              float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    px = np.ones((height, width, 4), np.float32)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def test_no_input(device, scene, size):
    w, h = size
    fw.setup_scene(scene, device, w, h)
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new("CompositorNodeFwGenNoInput")
    tree.links.new(node.outputs["Image"], out.inputs[0])
    REC.clear()
    px = fw.render_pixels(scene, "nodeinput_%s_%dx%d" % (device, w, h))
    check(REC.get("gen") == (h, w), "[%s] no input %dx%d: output buffer is %s" %
          (device, w, h, REC.get("gen")))
    check(px.shape == (h, w, 4), "[%s] no input %dx%d: render shape %s" % (device, w, h, px.shape))
    check(np.allclose(px, (0.25, 0.5, 0.75, 1.0)), "[%s] no input %dx%d: pixels" % (device, w, h))


def test_single_value_inputs(device, scene, size, linked):
    w, h = size
    fw.setup_scene(scene, device, w, h)
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new("CompositorNodeFwGenValue")
    if linked:
        # A linked single value (the result of a Math node on constants), not just a default.
        math = tree.nodes.new("ShaderNodeMath")
        math.operation = 'ADD'
        math.inputs[0].default_value = 0.1
        math.inputs[1].default_value = 0.15
        tree.links.new(math.outputs[0], node.inputs["Value"])
        expected = (0.35, 0.45, 0.55, 1.0)
    else:
        expected = (0.6, 0.7, 0.8, 1.0)
    tree.links.new(node.outputs["Image"], out.inputs[0])
    REC.clear()
    px = fw.render_pixels(scene, "valuein_%s_%dx%d_%d" % (device, w, h, linked))
    label = "[%s] single value inputs (%s) %dx%d" % (
        device, "linked" if linked else "unlinked", w, h)
    check(REC.get("value") == (h, w), "%s: output buffer is %s" % (label, REC.get("value")))
    check(REC.get("value_types") == ("float", "tuple"), "%s: input types %s" %
          (label, REC.get("value_types")))
    check(px.shape == (h, w, 4), "%s: render shape %s" % (label, px.shape))
    check(np.allclose(px, expected, atol=1e-6), "%s: pixels" % label)


def test_image_input(device, scene):
    # The image input decides the domain, not the render resolution.
    fw.setup_scene(scene, device, 64, 48)
    tree, out = fw.new_tree(scene)
    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.image = make_image(20, 12)
    node = tree.nodes.new("CompositorNodeFwImageIn")
    tree.links.new(img_node.outputs["Image"], node.inputs["Image"])
    tree.links.new(node.outputs["Image"], out.inputs[0])
    REC.clear()
    fw.render_pixels(scene, "imagein_" + device)
    check(REC.get("image") == (12, 20), "[%s] image input: output buffer is %s, expected (12, 20)"
          % (device, REC.get("image")))


def main():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    scene = bpy.context.scene
    for device in fw.DEVICES:
        for size in ((64, 48), (37, 23)):
            fw.run_case(device, "no input %s" % (size,), test_no_input, device, scene, size)
            for linked in (False, True):
                fw.run_case(device, "single value inputs %s linked=%s" % (size, linked),
                            test_single_value_inputs, device, scene, size, linked)
        fw.run_case(device, "image input", test_image_input, device, scene)
    fw.finish()


main()
