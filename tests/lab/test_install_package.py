# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Package install test: zip addons/compositor_lab, install it into a *temporary* user config
(a Blender subprocess with BLENDER_USER_RESOURCES pointing at a temp dir), enable it, check that
every node class registered, the Lab menu exists, and a node renders; then disable it.

The outer process (run by tests/run_tests.sh without any isolation) never installs anything
itself; only the subprocess does, and it refuses to run without the isolated config.

  Blender -b --factory-startup --python-exit-code 1 --python test_install_package.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
INNER = "LAB_INSTALL_TEST_ZIP"


def outer():
    import importlib.util

    import bpy

    spec = importlib.util.spec_from_file_location("build_zip", os.path.join(ROOT, "tools", "build_zip.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tempfile.mkdtemp(prefix="lab_install_")
    zip_path = mod.build_zip(os.path.join(work, "compositor_lab.zip"))
    import zipfile
    names = zipfile.ZipFile(zip_path).namelist()
    assert "compositor_lab/__init__.py" in names, names
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names), "caches in zip"
    assert all(n.startswith("compositor_lab/") for n in names), "single top-level folder"
    print("PASS: zip built with %d files" % len(names))

    resources = os.path.join(work, "resources")
    os.makedirs(resources)
    env = dict(os.environ, BLENDER_USER_RESOURCES=resources, **{INNER: zip_path})
    cmd = [bpy.app.binary_path, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", os.path.abspath(__file__)]
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         timeout=300)
    print(res.stdout)
    assert res.returncode == 0, "install subprocess failed (%d)" % res.returncode
    assert "INSTALL TEST OK" in res.stdout
    print("PASS: isolated install subprocess")


def inner():
    import addon_utils
    import bpy
    import numpy as np

    resources = os.environ.get("BLENDER_USER_RESOURCES", "")
    tmp_root = os.path.realpath(tempfile.gettempdir())
    assert resources and os.path.realpath(resources).startswith(tmp_root), (
        "refusing to run without an isolated BLENDER_USER_RESOURCES: %r" % resources)
    zip_path = os.environ[INNER]
    for p in (ROOT, os.path.join(ROOT, "addons")):
        assert p not in sys.path, "sources must not be importable"

    bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
    addons_dir = bpy.utils.user_resource('SCRIPTS', path="addons")
    assert os.path.realpath(addons_dir).startswith(os.path.realpath(resources)), addons_dir
    assert os.path.isfile(os.path.join(addons_dir, "compositor_lab", "__init__.py")), os.listdir(addons_dir)
    mod = addon_utils.enable("compositor_lab", default_set=True, handle_error=None)
    assert mod is not None, "add-on failed to enable"
    assert not mod.ERRORS, mod.ERRORS
    classes = [c for section in mod.REGISTERED.values() for c in section]
    names = sorted(c.bl_idname for c in classes)
    print("registered:", names)
    for expected in ("CompositorNodeLabNoise", "CompositorNodeLabBlendModes"):
        assert expected in names, expected
    for c in classes:
        assert c.is_registered, c.bl_idname
        assert bpy.types.Node.bl_rna_get_subclass(c.bl_idname) is not None, c.bl_idname
    assert "Generate" in mod.REGISTERED and mod.REGISTERED["Generate"], "Noise in Generate"
    assert any(c.bl_idname == "CompositorNodeLabBlendModes" for c in mod.REGISTERED["Filter"])
    assert hasattr(bpy.types, "NODE_MT_compositor_lab"), "Lab menu not registered"
    print("PASS: all node classes and the Lab menu registered")

    # A render through the installed package.
    W, H = 24, 16
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.compositor_device = 'CPU'
    a = np.full((H, W, 4), 0.5, np.float32)
    img = bpy.data.images.new("a", W, H, float_buffer=True)
    img.pixels.foreach_set(a.ravel())
    tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
    tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    scene.compositing_node_group = tree
    n_img = tree.nodes.new('CompositorNodeImage')
    n_img.image = img
    node = tree.nodes.new('CompositorNodeLabBlendModes')
    node.blend_type = 'LINEAR_DODGE'
    tree.links.new(n_img.outputs[0], node.inputs["A"])
    tree.links.new(n_img.outputs[0], node.inputs["B"])
    tree.links.new(node.outputs[0], tree.nodes.new('NodeGroupOutput').inputs[0])
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.filepath = os.path.join(bpy.app.tempdir, "lab_install.exr")
    bpy.ops.render.render(write_still=True)
    out = bpy.data.images.load(scene.render.filepath)
    px = np.empty(W * H * 4, np.float32)
    out.pixels.foreach_get(px)
    assert np.allclose(px.reshape(H, W, 4)[..., :3], 1.0, atol=1e-5), "0.5 + 0.5 should be 1.0"
    print("PASS: installed Blend Modes+ renders")

    addon_utils.disable("compositor_lab", default_set=True)
    assert not any(c.is_registered for c in classes), "classes still registered after disable"
    assert not hasattr(bpy.types, "NODE_MT_compositor_lab")
    print("PASS: disable unregisters everything")
    print("INSTALL TEST OK")


if os.environ.get(INNER):
    inner()
else:
    outer()
