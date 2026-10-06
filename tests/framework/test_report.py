# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""F5: `context.report` reports non-fatal messages while the node still produces output.

The messages are forwarded to node warnings and the info message of the compositor, neither of
which is exposed to Python, so this tests the API: the call works on CPU and GPU, the output is
still produced, invalid arguments raise, and the function raises once the evaluation is over.
"""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw_common as fw  # noqa: E402
from fw_common import check, fill_output  # noqa: E402

REC = {}


def run(self, outputs, context, device):
    REC["calls"] = REC.get("calls", 0) + 1
    context.report("plain info")
    context.report("careful", 'WARNING')
    context.report("keyword", level='INFO')
    REC["retained"] = context.report
    errors = {}
    for label, args, kwargs in (("level", ("x", 'ERROR'), {}),
                                ("lower", ("x", 'info'), {}),
                                ("type", (1,), {}),
                                ("none", (), {})):
        try:
            context.report(*args, **kwargs)
        except (ValueError, TypeError) as e:
            errors[label] = type(e).__name__
    REC["errors"] = errors
    fill_output(outputs, "Image", (0.25, 0.5, 0.75, 1), device)


def cpu(self, inputs, outputs, context):
    run(self, outputs, context, 'CPU')


def gpu(self, inputs, outputs, context):
    run(self, outputs, context, 'GPU')


class CompositorNodeFwReport(fw.CompositorTestNode, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeFwReport"
    bl_label = "FW Report"

    def init(self, context):
        self.outputs.new('NodeSocketColor', "Image")

    evaluate_cpu = cpu
    evaluate_gpu = gpu


def test_report(device, scene):
    fw.setup_scene(scene, device, 32, 24)
    scene.frame_set(1)
    tree, out = fw.new_tree(scene)
    node = tree.nodes.new("CompositorNodeFwReport")
    tree.links.new(node.outputs["Image"], out.inputs[0])
    REC.clear()
    px = fw.render_pixels(scene, "report_" + device)
    check(REC.get("calls", 0) >= 1, "[%s] method was called" % device)
    check((px == (0.25, 0.5, 0.75, 1)).all(), "[%s] output is still produced" % device)
    check(REC.get("errors") == {"level": "ValueError", "lower": "ValueError",
                                "type": "TypeError", "none": "TypeError"},
          "[%s] invalid arguments raise (%s)" % (device, REC.get("errors")))
    retained = REC.get("retained")
    try:
        retained("late")
        raised = None
    except RuntimeError as e:
        raised = e
    check(raised is not None, "[%s] calling report after the evaluation raises RuntimeError" % device)
    REC.clear()


def main():
    bpy.utils.register_class(CompositorNodeFwReport)
    scene = bpy.context.scene
    for device in fw.DEVICES:
        fw.run_case(device, "report", test_report, device, scene)
    fw.finish()


main()
