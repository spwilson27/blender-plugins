# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""F3: outputs listed in `single_value_outputs` are single values written through a (channels,)
buffer, in both CPU and GPU evaluation."""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw_common as fw  # noqa: E402
from fw_common import check, fill_output  # noqa: E402

REC = {}

# Output identifier -> (socket type, value written by Python).
OUTPUTS = {
    "Mean": ('NodeSocketFloat', (0.25,)),
    "Palette": ('NodeSocketColor', (0.1, 0.2, 0.3, 1.0)),
    "Count": ('NodeSocketInt', (7,)),
    "Vec": ('NodeSocketVector', (0.5, 0.25, 0.125)),
    "Flag": ('NodeSocketBool', (True,)),
}


def describe(buf):
    view = memoryview(buf)
    arr = np.asarray(buf)
    return dict(type=type(buf).__name__, shape=tuple(buf.shape), view_shape=view.shape,
                format=view.format, readonly=view.readonly, zero=bool((arr == 0).all()))


def run(self, inputs, outputs, device, fail):
    for name, (_socket_type, value) in OUTPUTS.items():
        if name not in outputs:
            continue
        buf = outputs[name]
        REC[name] = describe(buf)
        REC[name]["device"] = device
        arr = np.asarray(buf)
        arr[...] = value
        REC["written_" + name] = True
    if "Image" in outputs:
        out = outputs["Image"]
        REC["image_shape"] = (tuple(np.asarray(out).shape[:2]) if device == 'CPU'
                              else (out.height, out.width))
        fill_output(outputs, "Image", (0.5, 0.5, 0.5, 1.0), device)
    if fail:
        raise RuntimeError("intentional test error")


def init(self, context):
    for name, (socket_type, _value) in OUTPUTS.items():
        self.outputs.new(socket_type, name)
    self.outputs.new('NodeSocketColor', "Image")


def make_class(name, fail=False, evaluate=True, single=True):
    attrs = {
        "bl_idname": "CompositorNodeFw" + name,
        "bl_label": "FW " + name,
        "init": init,
    }
    if single:
        # A set with an unknown identifier, which is ignored.
        attrs["single_value_outputs"] = {"Mean", "Palette", "Count", "Vec", "Flag", "Unknown"}
    if evaluate:
        attrs["evaluate_cpu"] = lambda self, i, o: run(self, i, o, 'CPU', fail)
        attrs["evaluate_gpu"] = lambda self, i, o: run(self, i, o, 'GPU', fail)
    return type("CompositorNodeFw" + name, (fw.CompositorTestNode, bpy.types.CompositorNode),
                attrs)


CLS_OK = make_class("SvOk")
CLS_FAIL = make_class("SvFail", fail=True)
CLS_NONE = make_class("SvNone", evaluate=False)
CLS_IMAGES = make_class("SvImage", single=False)


def combine(tree, r, g, b):
    node = tree.nodes.new("CompositorNodeCombineColor")
    node.inputs[3].default_value = 1.0
    for i, source in enumerate((r, g, b)):
        tree.links.new(source, node.inputs[i])
    return node.outputs[0]


def math(tree, operation, socket, value):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    tree.links.new(socket, node.inputs[0])
    node.inputs[1].default_value = value
    return node.outputs[0]


def build(scene, idname, which):
    """Build a tree that consumes only the `which` output, with a built-in node."""
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new(idname)
    if which == "Mean":
        v = math(tree, 'MULTIPLY', node.outputs["Mean"], 2.0)
        result = combine(tree, v, v, v)
    elif which == "Palette":
        result = node.outputs["Palette"]
    elif which == "Count":
        v = math(tree, 'MULTIPLY', node.outputs["Count"], 0.5)
        result = combine(tree, v, v, v)
    elif which == "Vec":
        sep = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(node.outputs["Vec"], sep.inputs[0])
        result = combine(tree, sep.outputs[0], sep.outputs[1], sep.outputs[2])
    elif which == "Flag":
        v = math(tree, 'ADD', node.outputs["Flag"], 0.5)
        result = combine(tree, v, v, v)
    tree.links.new(result, out.inputs[0])
    return tree


EXPECTED = {
    "Mean": (0.5, 0.5, 0.5, 1.0),
    "Palette": (0.1, 0.2, 0.3, 1.0),
    "Count": (3.5, 3.5, 3.5, 1.0),
    "Vec": (0.5, 0.25, 0.125, 1.0),
    "Flag": (1.5, 1.5, 1.5, 1.0),
}
CHANNELS = {"Mean": 1, "Palette": 4, "Count": 1, "Vec": 3, "Flag": 1}
FORMAT = {"Mean": 'f', "Palette": 'f', "Count": 'i', "Vec": 'f', "Flag": '?'}


def test_outputs(device, scene, size, which):
    w, h = size
    fw.setup_scene(scene, device, w, h)
    build(scene, "CompositorNodeFwSvOk", which)
    REC.clear()
    px = fw.render_pixels(scene, "sv_%s_%s_%dx%d" % (device, which, w, h))
    label = "[%s] single value %s (%dx%d)" % (device, which, w, h)
    info = REC.get(which)
    if info is None:
        check(False, "%s: method was not called" % label)
        return
    check(info["device"] == device, "%s: %s method used" % (label, device))
    check(info["type"] == "Buffer" and info["shape"] == (CHANNELS[which],) and
          info["view_shape"] == (CHANNELS[which],),
          "%s: buffer type/shape %s" % (label, (info["type"], info["shape"])))
    check(info["format"] == FORMAT[which], "%s: format %r" % (label, info["format"]))
    check(not info["readonly"], "%s: writable" % label)
    check(info["zero"], "%s: zero initialized" % label)
    check(px.shape == (h, w, 4), "%s: render shape %s" % (label, px.shape))
    check(np.allclose(px, EXPECTED[which], atol=1e-6, rtol=0),
          "%s: pixels %s, expected %s" % (label, px[0, 0], EXPECTED[which]))
    # The outputs that are not consumed are omitted.
    others = [n for n in OUTPUTS if n != which and n in REC]
    check(not others, "%s: unconsumed outputs omitted (%s)" % (label, others))


def test_exception(device, scene, which):
    w, h = 24, 16
    fw.setup_scene(scene, device, w, h)
    # The default is whatever a node without any evaluate method outputs.
    build(scene, "CompositorNodeFwSvNone", which)
    reference = fw.render_pixels(scene, "svdef_%s_%s" % (device, which))
    build(scene, "CompositorNodeFwSvFail", which)
    REC.clear()
    px = fw.render_pixels(scene, "svfail_%s_%s" % (device, which))
    label = "[%s] exception %s" % (device, which)
    check(REC.get("written_" + which), "%s: the method ran and wrote its buffer" % label)
    check(px.shape == (h, w, 4) and np.array_equal(px, reference),
          "%s: default output %s, expected %s" % (label, px[0, 0], reference[0, 0]))
    check(not np.allclose(px[0, 0], EXPECTED[which], atol=1e-3),
          "%s: the value written before the exception is discarded" % label)


def test_mixed(device, scene):
    """A single value output next to a regular image output of the same node."""
    w, h = 31, 17
    fw.setup_scene(scene, device, w, h)
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new("CompositorNodeFwSvOk")
    v = math(tree, 'MULTIPLY', node.outputs["Mean"], 2.0)
    combined = combine(tree, v, v, v)
    # Multiply the regular image of the node (gray 0.5) with the single value (0.5).
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs[0].default_value = 1.0
    tree.links.new(node.outputs["Image"], mix.inputs[6])
    tree.links.new(combined, mix.inputs[7])
    tree.links.new(mix.outputs[2], out.inputs[0])
    REC.clear()
    px = fw.render_pixels(scene, "svmixed_" + device)
    check(REC.get("image_shape") == (h, w), "[%s] mixed: regular image output over compositing "
          "domain %s" % (device, REC.get("image_shape")))
    check(np.allclose(px, (0.25, 0.25, 0.25, 1.0), atol=1e-6), "[%s] mixed: pixels" % device)
    check(REC.get("Mean", {}).get("shape") == (1,), "[%s] mixed: Mean buffer" % device)


def main():
    for cls in (CLS_OK, CLS_FAIL, CLS_NONE, CLS_IMAGES):
        bpy.utils.register_class(cls)
    scene = bpy.context.scene
    for device in fw.DEVICES:
        for which in OUTPUTS:
            fw.run_case(device, "outputs %s" % which, test_outputs, device, scene, (48, 32),
                        which)
        # The domain of single values is independent of the render size.
        fw.run_case(device, "outputs Mean odd size", test_outputs, device, scene, (37, 23), "Mean")
        for which in OUTPUTS:
            fw.run_case(device, "exception %s" % which, test_exception, device, scene, which)
        fw.run_case(device, "mixed", test_mixed, device, scene)
    fw.finish()


main()
