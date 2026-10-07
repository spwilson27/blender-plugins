# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Socket ranges: `min_value` / `max_value` on the float and int input sockets of a node (the
slider soft range), set from Python, saved with the .blend, with min <= max kept, and without
clamping `default_value`. Skipped on builds that do not have them."""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw_common as fw  # noqa: E402
from fw_common import check  # noqa: E402

FLOAT_TYPES = ["NodeSocketFloat", "NodeSocketFloatFactor", "NodeSocketFloatDistance",
               "NodeSocketFloatAngle", "NodeSocketFloatPercentage", "NodeSocketFloatTime",
               "NodeSocketFloatUnsigned"]
INT_TYPES = ["NodeSocketInt", "NodeSocketIntFactor", "NodeSocketIntPercentage",
             "NodeSocketIntUnsigned"]
ALL_TYPES = FLOAT_TYPES + INT_TYPES


def init(self, context):
    for t in ALL_TYPES:
        self.inputs.new(t, t)
    self.outputs.new('NodeSocketColor', "Image")


class CompositorNodeFwRanges(fw.CompositorTestNode, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeFwRanges"
    bl_label = "FW Ranges"
    init = init

    def evaluate_cpu(self, inputs, outputs):
        fw.fill_output(outputs, "Image", (0, 0, 0, 1), 'CPU')

    def evaluate_gpu(self, inputs, outputs):
        fw.fill_output(outputs, "Image", (0, 0, 0, 1), 'GPU')


def close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def new_node():
    tree = bpy.data.node_groups.new("Ranges", 'CompositorNodeTree')
    tree.use_fake_user = True
    return tree, tree.nodes.new("CompositorNodeFwRanges")


def test_set_get():
    tree, node = new_node()
    for t in ALL_TYPES:
        sock = node.inputs[t]
        is_int = t in INT_TYPES
        cast = int if is_int else float
        # New sockets are unbounded, as before this feature.
        if is_int:
            check(sock.min_value < -10 ** 9 and sock.max_value > 10 ** 9,
                  "%s: fresh range is unbounded (%s..%s)" % (t, sock.min_value, sock.max_value))
        else:
            check(sock.min_value < -1e30 and sock.max_value > 1e30,
                  "%s: fresh range is unbounded" % t)
        sock.min_value = cast(-2)
        sock.max_value = cast(5)
        check(sock.min_value == -2 and sock.max_value == 5, "%s: set/get" % t)
        # Setting the default outside the soft range is allowed (soft range only).
        sock.default_value = cast(100)
        check(sock.default_value == 100, "%s: default_value outside the soft range" % t)
        # min > max: the other bound follows.
        sock.min_value = cast(10)
        check(sock.min_value == 10 and sock.max_value >= 10,
              "%s: min above max raises max (%s..%s)" % (t, sock.min_value, sock.max_value))
        sock.max_value = cast(-7)
        check(sock.max_value == -7 and sock.min_value <= -7,
              "%s: max below min lowers min (%s..%s)" % (t, sock.min_value, sock.max_value))
        # Narrow again, then the extremes.
        sock.min_value = cast(0)
        sock.max_value = cast(1)
        check(sock.min_value == 0 and sock.max_value == 1, "%s: 0..1" % t)
    f = node.inputs["NodeSocketFloat"]
    f.min_value, f.max_value = 0.25, 0.75
    check(close(f.min_value, 0.25) and close(f.max_value, 0.75), "fractional float range")


def test_save_load():
    tree, node = new_node()
    node.name = "RangesNode"
    tree_name = tree.name
    for i, t in enumerate(ALL_TYPES):
        sock = node.inputs[t]
        cast = int if t in INT_TYPES else float
        sock.min_value = cast(-i - 1)
        sock.max_value = cast(i + 3)
    path = os.path.join(fw.TMP, "ranges.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=path)
    node = bpy.data.node_groups[tree_name].nodes["RangesNode"]
    for i, t in enumerate(ALL_TYPES):
        sock = node.inputs[t]
        check(sock.min_value == -i - 1 and sock.max_value == i + 3,
              "%s: range survives save and reload (%s..%s)" % (t, sock.min_value, sock.max_value))


def test_builtin_unchanged():
    tree = bpy.data.node_groups.new("Builtin", 'CompositorNodeTree')
    node = tree.nodes.new("CompositorNodeAlphaOver")
    found = False
    for sock in node.inputs:
        if sock.bl_idname == 'NodeSocketFloatFactor':
            found = True
            check(sock.min_value == 0.0 and sock.max_value == 1.0,
                  "built-in factor socket %r keeps 0..1 (%s..%s)"
                  % (sock.name, sock.min_value, sock.max_value))
    check(found, "Alpha Over has a factor input")


def main():
    bpy.utils.register_class(CompositorNodeFwRanges)
    probe = bpy.data.node_groups.new("Probe", 'CompositorNodeTree').nodes.new(
        "CompositorNodeFwRanges")
    if not hasattr(probe.inputs[0], "min_value"):
        print("SKIP: node sockets have no min_value / max_value on this build")
        return
    for name, func in (("set/get", test_set_get), ("save/load", test_save_load),
                       ("built-in nodes", test_builtin_unchanged)):
        fw.run_case("-", name, func)
    fw.finish()


main()
