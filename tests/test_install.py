# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Install test: the add-on must work when installed on its own, like a user installing the
single file from Preferences. Run with an isolated user config, e.g.:

    BLENDER_USER_RESOURCES=$(mktemp -d) Blender -b --factory-startup \
        --python-exit-code 1 --python test_install.py

Nothing from this directory is put on sys.path, so a hidden dependency on another file
(e.g. pixel_sort.py) makes the enable step fail.
"""
import os
import sys

import addon_utils
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADDONS_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "addons"))
ADDON_FILE = os.path.join(ADDONS_DIR, "pixel_sort_node.py")

assert HERE not in sys.path and ADDONS_DIR not in sys.path, "test must not see the sources"
user_resources = os.environ.get("BLENDER_USER_RESOURCES", "")
assert user_resources and user_resources.startswith(("/tmp", "/var", "/private")), (
    "run with BLENDER_USER_RESOURCES pointing at a temporary directory")

bpy.ops.preferences.addon_install(filepath=ADDON_FILE, overwrite=True)
installed = os.path.join(bpy.utils.user_resource('SCRIPTS', path="addons"), "pixel_sort_node.py")
assert os.path.exists(installed), installed
assert os.listdir(os.path.dirname(installed)) == ["pixel_sort_node.py"], (
    os.listdir(os.path.dirname(installed)))

mod = addon_utils.enable("pixel_sort_node", default_set=True, handle_error=None)
assert mod is not None, "add-on failed to enable"
assert mod.CompositorNodePixelSort.is_registered, "node type not registered"
print("PASS: installed and enabled from a clean config")

W, H = 48, 24
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
scene.render.compositor_device = 'CPU'
src = np.random.default_rng(1).random((H, W, 4)).astype(np.float32)
src[..., 3] = 1.0
image = bpy.data.images.new("src", W, H, float_buffer=True)
image.pixels.foreach_set(src.ravel())
tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
scene.compositing_node_group = tree
n_img = tree.nodes.new('CompositorNodeImage')
n_img.image = image
node = tree.nodes.new('CompositorNodePixelSort')
node.inputs["Lower"].default_value = 0.0
node.inputs["Upper"].default_value = 1.0
tree.links.new(n_img.outputs[0], node.inputs["Image"])
tree.links.new(node.outputs[0], tree.nodes.new('NodeGroupOutput').inputs[0])
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.filepath = os.path.join(bpy.app.tempdir, "install_test.exr")
bpy.ops.render.render(write_still=True)

out = bpy.data.images.load(scene.render.filepath)
px = np.empty(W * H * 4, np.float32)
out.pixels.foreach_get(px)
px = px.reshape(H, W, 4)
luma = px[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
assert not np.array_equal(px, src), "node did not change the image"
assert np.all(np.diff(luma, axis=1) >= -1e-6), "rows are not sorted by luminance"
print("PASS: installed node sorts rows in a render")
