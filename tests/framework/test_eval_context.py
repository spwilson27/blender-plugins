# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""F2: evaluate_* methods that accept a fourth parameter receive the evaluation context."""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw_common as fw  # noqa: E402
from fw_common import check, fill_output  # noqa: E402

REC = {}


def make_node_class(name, cpu, gpu):
    return type("CompositorNodeFw" + name, (fw.CompositorTestNode, bpy.types.CompositorNode), {
        "bl_idname": "CompositorNodeFw" + name,
        "bl_label": "FW " + name,
        "init": lambda self, context: self.outputs.new('NodeSocketColor', "Image"),
        "evaluate_cpu": cpu,
        "evaluate_gpu": gpu,
    })


def ctx_fields(context):
    return dict(frame=context.frame, fps=context.fps, time=context.time, size=context.size,
                use_gpu=context.use_gpu, types=(type(context.frame).__name__,
                                                type(context.fps).__name__,
                                                type(context.time).__name__,
                                                type(context.size).__name__,
                                                type(context.use_gpu).__name__))


def four_cpu(self, inputs, outputs, context):
    REC["four"] = ("cpu", ctx_fields(context))
    fill_output(outputs, "Image", (1, 0, 0, 1), 'CPU')


def four_gpu(self, inputs, outputs, context):
    REC["four"] = ("gpu", ctx_fields(context))
    fill_output(outputs, "Image", (1, 0, 0, 1), 'GPU')


def three_cpu(self, inputs, outputs):
    REC["three"] = ("cpu", 3)
    fill_output(outputs, "Image", (0, 1, 0, 1), 'CPU')


def three_gpu(self, inputs, outputs):
    REC["three"] = ("gpu", 3)
    fill_output(outputs, "Image", (0, 1, 0, 1), 'GPU')


def star_cpu(self, inputs, outputs, *args):
    REC["star"] = ("cpu", len(args), type(args[0]).__name__ if args else None)
    fill_output(outputs, "Image", (0, 0, 1, 1), 'CPU')


def star_gpu(self, inputs, outputs, *args):
    REC["star"] = ("gpu", len(args), type(args[0]).__name__ if args else None)
    fill_output(outputs, "Image", (0, 0, 1, 1), 'GPU')


def default_cpu(self, inputs, outputs, context=None):
    REC["default"] = ("cpu", context is not None)
    fill_output(outputs, "Image", (1, 1, 0, 1), 'CPU')


def default_gpu(self, inputs, outputs, context=None):
    REC["default"] = ("gpu", context is not None)
    fill_output(outputs, "Image", (1, 1, 0, 1), 'GPU')


def kwonly_cpu(self, inputs, outputs, *, context=None):
    # A keyword only parameter does not count as a positional parameter.
    REC["kwonly"] = ("cpu", context is not None)
    fill_output(outputs, "Image", (0, 1, 1, 1), 'CPU')


def kwonly_gpu(self, inputs, outputs, *, context=None):
    REC["kwonly"] = ("gpu", context is not None)
    fill_output(outputs, "Image", (0, 1, 1, 1), 'GPU')


def req_cpu(self, inputs, outputs, ctx):
    REC["required"] = ("cpu", type(ctx).__name__)
    fill_output(outputs, "Image", (1, 0, 1, 1), 'CPU')


def req_gpu(self, inputs, outputs, ctx):
    REC["required"] = ("gpu", type(ctx).__name__)
    fill_output(outputs, "Image", (1, 0, 1, 1), 'GPU')


def defnamed_cpu(self, inputs, outputs, _orig=123):
    REC["defnamed"] = ("cpu", _orig)
    fill_output(outputs, "Image", (0.5, 0, 0, 1), 'CPU')


def defnamed_gpu(self, inputs, outputs, _orig=123):
    REC["defnamed"] = ("gpu", _orig)
    fill_output(outputs, "Image", (0.5, 0, 0, 1), 'GPU')


# (name, class, expected color)
NODES = (
    ("Four", make_node_class("Four", four_cpu, four_gpu), (1, 0, 0, 1)),
    ("Three", make_node_class("Three", three_cpu, three_gpu), (0, 1, 0, 1)),
    ("Star", make_node_class("Star", star_cpu, star_gpu), (0, 0, 1, 1)),
    ("Default", make_node_class("Default", default_cpu, default_gpu), (1, 1, 0, 1)),
    ("Required", make_node_class("Required", req_cpu, req_gpu), (1, 0, 1, 1)),
    ("DefNamed", make_node_class("DefNamed", defnamed_cpu, defnamed_gpu), (0.5, 0, 0, 1)),
    ("KwOnly", make_node_class("KwOnly", kwonly_cpu, kwonly_gpu), (0, 1, 1, 1)),
)


def render_node(scene, device, name, tag):
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new("CompositorNodeFw" + name)
    tree.links.new(node.outputs["Image"], out.inputs[0])
    REC.clear()
    return fw.render_pixels(scene, "ctx_%s_%s_%s" % (name, device, tag))


def test_four_args(device, scene, width, height, frame, subframe, fps, fps_base):
    fw.setup_scene(scene, device, width, height)
    scene.render.fps = fps
    scene.render.fps_base = fps_base
    scene.frame_set(frame, subframe=subframe)
    px = render_node(scene, device, "Four", "%d_%g" % (frame, subframe))
    label = "[%s] context (%dx%d, frame %d+%g, fps %d/%g)" % (
        device, width, height, frame, subframe, fps, fps_base)
    kind, fields = REC.get("four", (None, None))
    check(kind == ('cpu' if device == 'CPU' else 'gpu'), "%s: called %s method" % (label, kind))
    if fields is None:
        check(False, "%s: no context received" % label)
        return
    check(fields["types"] == ("float", "float", "float", "tuple", "bool"),
          "%s: attribute types %s" % (label, fields["types"]))
    expected_fps = fps / fps_base
    # The subframe is part of the frame when the render keeps it. Accept the integer frame only
    # if the render does not apply the subframe at all.
    check(abs(fields["frame"] - (frame + subframe)) < 1e-5,
          "%s: frame %r, expected %r" % (label, fields["frame"], frame + subframe))
    check(abs(fields["fps"] - expected_fps) < 1e-4, "%s: fps %r, expected %r" %
          (label, fields["fps"], expected_fps))
    check(abs(fields["time"] - fields["frame"] / expected_fps) < 1e-4,
          "%s: time %r, expected frame / fps = %r" %
          (label, fields["time"], fields["frame"] / expected_fps))
    check(fields["size"] == (width, height), "%s: size %r" % (label, fields["size"]))
    check(fields["use_gpu"] == (device == 'GPU'), "%s: use_gpu %r" % (label, fields["use_gpu"]))
    check(px.shape == (height, width, 4) and (px == (1, 0, 0, 1)).all(), "%s: render" % label)


def test_arity(device, scene):
    fw.setup_scene(scene, device, 40, 30)
    scene.frame_set(1)
    for name, _cls, color in NODES[1:]:
        px = render_node(scene, device, name, "arity")
        key = name.lower()
        got = REC.get(key)
        check(got is not None, "[%s] %s: method was called" % (device, name))
        check((px == color).all(), "[%s] %s: output color" % (device, name))
        if name == "Three":
            check(got == (('cpu' if device == 'CPU' else 'gpu'), 3),
                  "[%s] 3-argument method is called as before (%s)" % (device, got))
        elif name == "Star":
            check(got[1] == 1 and got[2] == "SimpleNamespace",
                  "[%s] *args receives the context (%s)" % (device, got))
        elif name == "Default":
            check(got[1] is True, "[%s] 'context=None' receives the context" % device)
        elif name == "Required":
            check(got[1] == "SimpleNamespace", "[%s] required 4th parameter (any name) receives "
                  "the context (%s)" % (device, got))
        elif name == "DefNamed":
            check(got[1] == 123, "[%s] defaulted non-context 4th parameter keeps its default (%s)"
                  % (device, got))
        elif name == "KwOnly":
            check(got[1] is False, "[%s] keyword only parameter is not filled" % device)


def main():
    for _name, cls, _color in NODES:
        bpy.utils.register_class(cls)
    scene = bpy.context.scene
    for device in fw.DEVICES:
        fw.run_case(device, "context 64x48 f12 fps 24", test_four_args, device, scene,
                    64, 48, 12, 0.0, 24, 1.0)
        # NTSC like rate and a subframe. Odd render size.
        fw.run_case(device, "context 37x23 f7+0.25 fps 30/1.001", test_four_args, device, scene,
                    37, 23, 7, 0.25, 30, 1.001)
        fw.run_case(device, "context negative frame", test_four_args, device, scene,
                    16, 16, -3, 0.5, 25, 1.0)
        fw.run_case(device, "arity", test_arity, device, scene)
    fw.finish()


main()
