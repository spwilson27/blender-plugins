# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Benchmark the Pixel Sort compositor node: GPU parallel vs GPU serial vs CPU.

Run:
  <blender> -b --factory-startup --python bench_gpu.py

Compositor-only scene, Image -> PixelSort -> Group Output. Each timing is the
median of 5 bpy.ops.render.render() calls after one warm-up. The "node cost"
column subtracts the same render with the node bypassed (Image -> Output).
"""
import os
import statistics
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "addons"))
import pixel_sort_node  # noqa: E402

SIZES = [(1920, 1080), (3840, 2160)]
RUNS = 5


def make_image(w, h):
    rng = np.random.default_rng(7)
    px = rng.random((h, w, 4), dtype=np.float32)
    px[..., 3] = 1.0
    img = bpy.data.images.new("bench_%dx%d" % (w, h), w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    return img


def setup(scene, device, w, h):
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.use_file_extension = False
    for owner in (scene.render, scene):
        if hasattr(owner, "compositor_device"):
            owner.compositor_device = device
            if hasattr(owner, "compositor_precision"):
                owner.compositor_precision = 'FULL'
            break


def build(scene, image, use_node, vertical):
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
        scene.compositing_node_group = tree
    tree.nodes.clear()
    if not any(i.item_type == 'SOCKET' and i.in_out == 'OUTPUT' for i in tree.interface.items_tree):
        tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type="NodeSocketColor")
    n_img = tree.nodes.new('CompositorNodeImage')
    n_img.image = image
    n_out = tree.nodes.new('NodeGroupOutput')
    if use_node:
        node = tree.nodes.new("CompositorNodePixelSort")
        node.vertical = vertical
        tree.links.new(n_img.outputs["Image"], node.inputs["Image"])
        tree.links.new(node.outputs["Image"], n_out.inputs[0])
    else:
        tree.links.new(n_img.outputs["Image"], n_out.inputs[0])


def time_render(scene):
    bpy.ops.render.render()  # warm-up (shader compile, scratch alloc, uploads)
    ts = []
    for _ in range(RUNS):
        t = time.perf_counter()
        bpy.ops.render.render()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    pixel_sort_node.register()
    scene = bpy.context.scene
    rows = []
    for (w, h) in SIZES:
        img = make_image(w, h)
        for vertical in (False, True):
            res = {}
            for label, device, impl in (("gpu_par", 'GPU', "parallel"),
                                        ("gpu_ser", 'GPU', "serial"),
                                        ("cpu", 'CPU', None)):
                if impl:
                    pixel_sort_node._GPU_IMPL = impl
                setup(scene, device, w, h)
                build(scene, img, False, vertical)
                base = time_render(scene)
                build(scene, img, True, vertical)
                full = time_render(scene)
                res[label] = (full, base)
            rows.append((w, h, vertical, res))
        bpy.data.images.remove(img)

    print()
    print("%-10s %-5s | %-26s | %-26s | %-26s | %s" % (
        "size", "dir", "GPU parallel (ms)", "GPU serial (ms)", "CPU numpy (ms)", "speedup"))
    print("%-10s %-5s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s | %s" % (
        "", "", "total", "base", "node", "total", "base", "node", "total", "base", "node",
        "par vs ser (node)"))
    for w, h, vertical, res in rows:
        cols = []
        nodes = {}
        for k in ("gpu_par", "gpu_ser", "cpu"):
            full, base = res[k]
            nodes[k] = max(full - base, 1e-6)
            cols.append("%8.0f %8.0f %8.0f" % (full * 1e3, base * 1e3, nodes[k] * 1e3))
        print("%-10s %-5s | %s | %s | %s | %.1fx" % (
            "%dx%d" % (w, h), "vert" if vertical else "horiz", cols[0], cols[1], cols[2],
            nodes["gpu_ser"] / nodes["gpu_par"]))
    sys.stdout.flush()


main()
