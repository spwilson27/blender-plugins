# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Socket ranges of all Lab nodes (and Pixel Sort): every float / int input has a finite soft range
containing its default, `clamp=` inputs clamp (helpers, and a render on CPU and GPU), and the
load_post handler restores the ranges of nodes whose sockets were widened or saved with old ones.
   Blender -b --factory-startup --python-exit-code 1 --python test_ranges.py"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

lab = H.setup()

from compositor_lab.lib import node as lab_node  # noqa: E402

import pixel_sort_node  # noqa: E402

pixel_sort_node.register()

SIZE = (32, 24)
HUGE = 1e30


def all_classes():
    classes = [c for section in lab.REGISTERED.values() for c in section]
    return classes + [pixel_sort_node.CompositorNodePixelSort]


def new_tree(name="Ranges"):
    return bpy.data.node_groups.new(name, 'CompositorNodeTree')


def numeric_inputs(node):
    return [s for s in node.inputs if s.type in ('VALUE', 'INT')]


def has_ranges():
    node = new_tree("Probe").nodes.new("CompositorNodeLabNoise")
    return hasattr(node.inputs["Scale"], "min_value")


def spec_range(cls, name):
    for s in getattr(cls, "SOCKETS", ()):
        if isinstance(s, lab_node.In) and s.name == name:
            return lab_node.resolve_range(s)
    return None


@H.guard("soft ranges")
def test_soft_ranges():
    tree = new_tree()
    n_checked = 0
    for cls in all_classes():
        node = tree.nodes.new(cls.bl_idname)
        for sock in numeric_inputs(node):
            label = "%s.%s" % (cls.bl_idname, sock.name)
            lo, hi = sock.min_value, sock.max_value
            H.check(abs(lo) < HUGE and abs(hi) < HUGE and lo < hi,
                    "%s: finite soft range (%g..%g)" % (label, lo, hi))
            v = sock.default_value
            H.check(lo <= v <= hi, "%s: default %g inside %g..%g" % (label, v, lo, hi))
            spec = spec_range(cls, sock.name)
            if spec is not None:
                for got, want, what in ((lo, spec[0], "min"), (hi, spec[1], "max")):
                    if want is not None:
                        H.check(abs(got - want) <= 1e-6 * max(1.0, abs(want)),
                                "%s: %s_value %g follows the spec (%g)" % (label, what, got, want))
            n_checked += 1
    H.check(n_checked > 60, "checked %d numeric inputs" % n_checked)


def clamp_inputs():
    """(cls, input name, is_int, lo, hi) of every input declared with clamp=."""
    found = []
    for cls in all_classes():
        for name, (is_int, lo, hi) in lab_node._clamp_table(cls).items() \
                if issubclass(cls, lab_node.LabNode) else ():
            found.append((cls, name, is_int, lo, hi))
    return found


@H.guard("clamp helpers")
def test_clamp_helpers():
    items = clamp_inputs()
    H.check(len(items) > 20, "%d clamped inputs" % len(items))
    for cls, name, is_int, lo, hi in items:
        label = "%s.%s" % (cls.bl_idname, name)
        read = cls.in_int if is_int else cls.in_float
        if lo is not None:
            H.check(read({name: lo - 5}, name, lo) == lo, "%s: below %g clamps" % (label, lo))
            H.check(read({name: lo}, name, lo) == lo, "%s: lower bound itself is kept" % label)
        if hi is not None:
            H.check(read({name: hi + 5}, name, hi) == hi, "%s: above %g clamps" % (label, hi))
        if lo is not None and hi is not None:
            mid = (lo + hi) / 2
            mid = int(mid) if is_int else mid
            H.check(read({name: mid}, name, lo) == mid, "%s: in-range value is kept" % label)
        # Single values read through the per-pixel helpers are clamped too.
        if lo is not None:
            arr = cls.in_image_array({name: lo - 5}, name, (2, 2), 1)
            H.check(np.all(arr == lo), "%s: in_image_array clamps a single value" % label)
            H.check(cls.in_texture_or_value({name: lo - 5}, name) == lo,
                    "%s: in_texture_or_value clamps a single value" % label)
    # Images are not clamped per pixel.
    img = np.full((2, 2, 1), -7.0, np.float32)
    cls, name = next((c, n) for c, n, _i, lo, _h in items if lo is not None and lo > -7)
    H.check(np.all(cls.in_image_array({name: img}, name, (2, 2), 1) == -7.0),
            "linked images pass through in_image_array unclamped")


def out_of_range(lo, hi, is_int):
    """Values below the lower bound and above the upper one (None when open)."""
    below = above = None
    if lo is not None:
        below = lo - max(1.0, abs(lo))
    if hi is not None:
        above = hi + max(1.0, abs(hi))
    if is_int:
        below = None if below is None else int(math.floor(below))
        above = None if above is None else int(math.ceil(above))
    return below, above


@H.guard("clamp render")
def test_clamp_render():
    for dev in ("CPU", "GPU"):
        for cls, name, is_int, lo, hi in clamp_inputs():
            below, above = out_of_range(lo, hi, is_int)
            for bad, bound, side in ((below, lo, "low"), (above, hi, "high")):
                if bad is None:
                    continue
                bound = int(bound) if is_int else float(bound)
                label = "%s %s.%s %s (%g -> %g)" % (dev, cls.bl_idname, name, side, bad, bound)
                got = H.render_node(cls.bl_idname, dev, SIZE, inputs={name: bad})
                ref = H.render_node(cls.bl_idname, dev, SIZE, inputs={name: bound})
                H.compare(label, got, ref, 0.0)


@H.guard("restore")
def test_restore():
    tree = new_tree("Restore")
    tree.use_fake_user = True
    nodes = []
    for cls in all_classes():
        node = tree.nodes.new(cls.bl_idname)
        nodes.append(node)
    expected = {}
    for node in nodes:
        for sock in numeric_inputs(node):
            expected[(node.name, sock.name)] = (sock.min_value, sock.max_value)

    def widen():
        for node in nodes:
            for sock in numeric_inputs(node):
                if sock.type == 'INT':
                    sock.min_value, sock.max_value = -1000000, 1000000
                else:
                    sock.min_value, sock.max_value = -1e6, 1e6

    def restored(label):
        bad = [k for node in tree.nodes for sock in numeric_inputs(node)
               for k in [(node.name, sock.name)]
               if (sock.min_value, sock.max_value) != expected[k]]
        H.check(not bad, "%s: ranges restored (%d differ: %s)" % (label, len(bad), bad[:3]))

    # The handler is registered, and re-applies from the spec.
    names = [getattr(h, "__name__", "") for h in bpy.app.handlers.load_post]
    H.check(names.count("_load_post_ranges") == 2, "load_post handlers (lab + pixel sort): %s"
            % [n for n in names if "ranges" in n])
    widen()
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_load_post_ranges":
            h(None)
    restored("handler call")
    widen()
    lab.refresh_socket_ranges()
    pixel_sort_node._load_post_ranges()
    restored("refresh_socket_ranges")

    # Through a real save + load: the file stores the widened ranges, the handler fixes them.
    widen()
    path = os.path.join(H.TMP, "ranges_restore.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=path)
    tree = bpy.data.node_groups["Restore"]
    restored("save and reload")

    # Registering twice does not stack handlers.
    lab._register_range_handler()
    lab._register_range_handler()
    names = [getattr(h, "__name__", "") for h in bpy.app.handlers.load_post]
    H.check(names.count("_load_post_ranges") == 2, "handlers are not duplicated")


def main():
    if not has_ranges():
        print("SKIP: node sockets have no min_value / max_value on this build")
        return
    test_soft_ranges()
    test_clamp_helpers()
    test_clamp_render()
    test_restore()
    H.finish()


main()
