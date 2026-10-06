# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Test for the Python-evaluated Pixel Sort compositor node.

Run:
  <blender> -b --factory-startup --python test_pixel_sort_node.py -- [--gpu]

Pixels are obtained by rendering the scene (compositor output goes to the
render result), writing the render to a 32-bit float OpenEXR in a temp dir
(bpy.ops.render.render(write_still=True)), and loading that file back.
The 'Viewer Node' image is also tried as a secondary, informational source.
"""
import os
import sys
import tempfile
import traceback

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "addons"))

import pixel_sort_node  # noqa: E402
from reference_pixel_sort import pixel_sort  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
USE_GPU = "--gpu" in argv
ONLY = argv[argv.index("--only") + 1] if "--only" in argv else None
W, H = 640, 360
TMP = tempfile.mkdtemp(prefix="pixel_sort_node_")
FAILURES = []


def report(ok, msg):
    print(("PASS: " if ok else "FAIL: ") + msg)
    if not ok:
        FAILURES.append(msg)


# ---------------------------------------------------------------------------
# Extra dummy nodes for robustness tests
# ---------------------------------------------------------------------------

class _DummyBase:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CompositorNodeTree'

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Image")
        self.outputs.new('NodeSocketColor', "Image")


class CompositorNodeTestNoEval(_DummyBase, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeTestNoEval"
    bl_label = "Test No Eval"


class CompositorNodeTestRaises(_DummyBase, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeTestRaises"
    bl_label = "Test Raises"

    def evaluate_cpu(self, inputs, outputs):
        raise RuntimeError("intentional test error")

    def evaluate_gpu(self, inputs, outputs):
        raise RuntimeError("intentional test error (gpu)")


class CompositorNodeTestGpuOnly(_DummyBase, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeTestGpuOnly"
    bl_label = "Test GPU Only"

    def evaluate_gpu(self, inputs, outputs):
        outputs["Image"].clear(format='FLOAT', value=(0.5, 0.5, 0.5, 1.0))


class CompositorNodeTestCpuOnly(_DummyBase, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeTestCpuOnly"
    bl_label = "Test CPU Only"

    def evaluate_cpu(self, inputs, outputs):
        out = np.asarray(outputs["Image"])
        out[...] = 1.0 - np.asarray(inputs["Image"])
        out[..., 3] = 1.0


_DUMMIES = (CompositorNodeTestNoEval, CompositorNodeTestRaises,
            CompositorNodeTestGpuOnly, CompositorNodeTestCpuOnly)


# ---------------------------------------------------------------------------
# Scene / graph helpers
# ---------------------------------------------------------------------------

def make_test_image(W=W, H=H, name="PixelSortInput"):
    rng = np.random.default_rng(1234)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    noise = rng.random((H, W, 3), dtype=np.float32)
    grad = np.stack([x / W, y / H, 0.5 + 0.5 * np.sin(x / 37.0 + y / 91.0)], axis=-1)
    px = np.empty((H, W, 4), np.float32)
    px[..., :3] = np.clip(0.6 * noise + 0.4 * grad, 0.0, 1.0)
    px[..., 3] = 1.0
    img = bpy.data.images.new(name, W, H, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    return img, px


def setup_scene(scene):
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA'
    dev = 'GPU' if USE_GPU else 'CPU'
    for owner in (scene.render, scene):
        if hasattr(owner, "compositor_device"):
            owner.compositor_device = dev
            if hasattr(owner, "compositor_precision"):
                owner.compositor_precision = 'FULL'
            break
    else:
        print("WARNING: no compositor_device property found")


def make_mask_image(values, name):
    """Float image with R=G=B=values (H, W) so any Color->Float conversion gives `values`."""
    h, w = values.shape
    px = np.ones((h, w, 4), np.float32)
    px[..., :3] = values[..., None]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    return img


def build_tree(scene, image, node_idname, props=None, inputs=None, mask_image=None,
               remove_mask_socket=False):
    """Image -> <node> -> Group Output (+ Viewer). Returns the test node."""
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
        scene.compositing_node_group = tree
    tree.nodes.clear()
    # Group output interface socket "Image".
    if not any(i.item_type == 'SOCKET' and i.in_out == 'OUTPUT' for i in tree.interface.items_tree):
        tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type="NodeSocketColor")

    n_img = tree.nodes.new('CompositorNodeImage')
    n_img.image = image
    node = tree.nodes.new(node_idname)
    n_out = tree.nodes.new('NodeGroupOutput')
    n_view = tree.nodes.new('CompositorNodeViewer')
    for k, v in (props or {}).items():
        setattr(node, k, v)
    for k, v in (inputs or {}).items():
        node.inputs[k].default_value = v
    tree.links.new(n_img.outputs["Image"], node.inputs["Image"])
    if mask_image is not None:
        n_mask = tree.nodes.new('CompositorNodeImage')
        n_mask.image = mask_image
        tree.links.new(n_mask.outputs["Image"], node.inputs["Mask"])
    if remove_mask_socket:
        node.inputs.remove(node.inputs["Mask"])
    tree.links.new(node.outputs["Image"], n_out.inputs[0])
    tree.links.new(node.outputs["Image"], n_view.inputs["Image"])
    return node


def render_pixels(scene, name):
    """Render and return (H, W, 4) float32 array, row 0 = bottom."""
    path = os.path.join(TMP, name + ".exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(path):
        raise RuntimeError("render produced no file: " + path)
    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        buf = np.empty(w * h * img.channels, np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(h, w, img.channels).copy()
    finally:
        bpy.data.images.remove(img)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=2)
    return arr


def viewer_pixels():
    """Informational: pixels from the Viewer Node image, or None."""
    img = bpy.data.images.get("Viewer Node")
    if img is None or img.size[0] == 0:
        return None
    w, h = img.size
    buf = np.empty(w * h * img.channels, np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, img.channels)


def save_png(arr, name):
    img = bpy.data.images.new(name, arr.shape[1], arr.shape[0], alpha=True)
    # Display-ish encoding (gamma 2.2) for a viewable PNG.
    out = np.clip(arr, 0, 1).astype(np.float32).copy()
    out[..., :3] = out[..., :3] ** (1 / 2.2)
    img.pixels.foreach_set(out.ravel())
    img.filepath_raw = os.path.join(tempfile.gettempdir(), name + ".png")
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def compare(name, got, ref, tol):
    if got.shape != ref.shape:
        report(False, "%s: shape mismatch %s vs %s" % (name, got.shape, ref.shape))
        return
    d = np.abs(got - ref)
    max_diff = float(d.max())
    frac = float((d.max(axis=-1) > tol).mean())
    print("  %s: max abs diff %.3g, mismatching pixels %.4f%% (tol %g)" %
          (name, max_diff, frac * 100.0, tol))
    report(max_diff <= tol and frac == 0.0, "%s matches numpy reference" % name)


CONFIGS = [
    ("luma_h", dict(mask_key='LUMA', sort_key='LUMA'), {}),
    ("luma_h_desc", dict(mask_key='LUMA', sort_key='LUMA', reverse=True), {}),
    ("hue_v", dict(mask_key='LUMA', sort_key='HUE', vertical=True), {}),
    ("sat_inv", dict(mask_key='SATURATION', sort_key='VALUE', invert_mask=True),
     dict(Lower=0.3, Upper=0.7)),
    ("red_v_desc", dict(mask_key='RED', sort_key='BLUE', vertical=True, reverse=True),
     dict(Lower=0.2, Upper=0.9)),
    # Odd sizes: line length not a power of two.
    ("odd_h", dict(), {}, (333, 187)),
    ("odd_v", dict(vertical=True), {}, (333, 187)),
    # Wide: P = 8192 > 4096.
    ("wide_h", dict(), {}, (4100, 8)),
    ("wide_v", dict(vertical=True), {}, (8, 4100)),
    ("tall_v", dict(vertical=True, sort_key='SATURATION'), {}, (97, 1500)),
    # Degenerate mask (empty -> output == input) and full mask (one run per line).
    ("empty_mask", dict(), dict(Lower=0.5, Upper=0.5), None, "same"),
    ("full_mask_h", dict(), dict(Lower=0.0, Upper=1.0)),
    ("full_mask_v_desc", dict(vertical=True, reverse=True), dict(Lower=0.0, Upper=1.0)),
]
for _mk in ('LUMA', 'HUE', 'SATURATION', 'VALUE', 'RED', 'GREEN', 'BLUE'):
    for _rev in (False, True):
        for _inv in (False, True):
            CONFIGS.append((
                "key_%s_%s%s" % (_mk, "desc" if _rev else "asc", "_inv" if _inv else ""),
                dict(mask_key=_mk, sort_key=_mk, reverse=_rev, invert_mask=_inv,
                     vertical=(_mk in ('HUE', 'GREEN', 'VALUE'))),
                dict(Lower=0.2, Upper=0.7), (101, 77)))


def main():
    scene = bpy.context.scene
    setup_scene(scene)
    for c in _DUMMIES:
        bpy.utils.register_class(c)

    image, src = make_test_image()
    images0 = (image, src)
    save_png(src, "pixel_sort_node_before")

    # ---- Pixel Sort vs reference ----
    tol = 0.0 if USE_GPU else 1e-5
    last = None
    images = {(W, H): images0}
    for cfg in CONFIGS:
        name, props, sockets = cfg[:3]
        size = cfg[3] if len(cfg) > 3 and cfg[3] else (W, H)
        if ONLY and ONLY not in name:
            continue
        same = len(cfg) > 4 and cfg[4] == "same"
        try:
            if size not in images:
                images[size] = make_test_image(size[0], size[1], "PixelSortInput_%dx%d" % size)
            image, src = images[size]
            scene.render.resolution_x, scene.render.resolution_y = size
            build_tree(scene, image, "CompositorNodePixelSort", props, sockets)
            got = render_pixels(scene, name)
            lo = sockets.get("Lower", 0.25)
            hi = sockets.get("Upper", 0.8)
            ref = pixel_sort(src, props.get('mask_key', 'LUMA'), lo, hi,
                             props.get('sort_key', 'LUMA'), props.get('vertical', False),
                             props.get('reverse', False), props.get('invert_mask', False))
            changed = float(np.abs(got - src).max())
            if same:
                report(np.array_equal(ref, src) and changed == 0.0, "%s: output == input" % name)
            else:
                report(changed > 1e-3, "%s: output differs from input (max change %.3g)" % (name, changed))
            compare(name, got, ref, tol)
            if size == (W, H):
                last = got
        except Exception:
            traceback.print_exc()
            report(False, "%s raised" % name)
    if last is not None:
        save_png(last, "pixel_sort_node_after")

    # ---- Mask input ----
    def mask_tests():
        ref_kw = lambda p: (p.get('mask_key', 'LUMA'), 0.25, 0.8, p.get('sort_key', 'LUMA'),
                            p.get('vertical', False), p.get('reverse', False),
                            p.get('invert_mask', False))
        rng = np.random.default_rng(99)

        def gradient(w, h):
            x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
            y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
            return np.clip(0.5 + 0.9 * (x - 0.5) + 0.3 * (y - 0.5), 0.0, 1.0).astype(np.float32)

        def random_binary(w, h):
            # Blobby-ish: runs of mask along rows so runs are not all length 1.
            m = rng.random((h, w)) < 0.5
            m = m | np.roll(m, 1, axis=1) | np.roll(m, 2, axis=1)
            return m.astype(np.float32)

        cases = [
            # name, props, size, mask values or scalar
            ("mask_binary_img", {}, (W, H), "binary"),
            ("mask_binary_img_inv_desc", dict(invert_mask=True, reverse=True), (W, H), "binary"),
            ("mask_gradient_img", {}, (W, H), "gradient"),
            ("mask_binary_vertical", dict(vertical=True, sort_key='HUE'), (W, H), "binary"),
            ("mask_gradient_vertical_desc", dict(vertical=True, reverse=True), (W, H), "gradient"),
            ("mask_binary_odd", {}, (333, 187), "binary"),
            ("mask_gradient_odd_vertical", dict(vertical=True), (333, 187), "gradient"),
            ("mask_scalar_0", {}, (W, H), 0.0),
            ("mask_scalar_0_vertical", dict(vertical=True), (W, H), 0.0),
            ("mask_scalar_0.5", {}, (W, H), 0.5),
            ("mask_scalar_1", {}, (W, H), 1.0),
        ]
        for name, props, size, mval in cases:
            if ONLY and ONLY not in name:
                continue
            try:
                if size not in images:
                    images[size] = make_test_image(size[0], size[1], "PixelSortInput_%dx%d" % size)
                image, src = images[size]
                scene.render.resolution_x, scene.render.resolution_y = size
                mimg = None
                sockets = {}
                bmask = None
                if isinstance(mval, str):
                    vals = random_binary(*size) if mval == "binary" else gradient(*size)
                    mimg = make_mask_image(vals, "Mask_" + name)
                    bmask = vals > 0.5
                else:
                    sockets["Mask"] = mval
                    if mval <= 0.5:
                        bmask = np.zeros(size[::-1], bool)  # nothing may be sorted
                build_tree(scene, image, "CompositorNodePixelSort", props, sockets, mask_image=mimg)
                got = render_pixels(scene, name)
                ref = pixel_sort(src, *ref_kw(props), mask=bmask)
                if mimg is None and mval <= 0.5:
                    report(np.array_equal(got, src) or float(np.abs(got - src).max()) <= tol,
                           "%s: output == input" % name)
                elif mimg is None:
                    unmasked = pixel_sort(src, *ref_kw(props))
                    report(np.array_equal(ref, unmasked), "%s: reference == unmasked" % name)
                else:
                    unmasked = pixel_sort(src, *ref_kw(props))
                    report(not np.array_equal(ref, unmasked) and not np.array_equal(ref, src),
                           "%s: mask is effective (differs from unmasked and from input)" % name)
                compare(name, got, ref, tol)
            except Exception:
                traceback.print_exc()
                report(False, "%s raised" % name)

        # Backwards compatibility: node without a Mask socket (saved before it existed).
        try:
            name = "mask_socket_removed"
            if not (ONLY and ONLY not in name):
                image, src = images0
                scene.render.resolution_x, scene.render.resolution_y = W, H
                build_tree(scene, image, "CompositorNodePixelSort", {}, {}, remove_mask_socket=True)
                report("Mask" not in [i.name for i in
                                      scene.compositing_node_group.nodes[1].inputs],
                       "%s: socket removed" % name)
                got = render_pixels(scene, name)
                compare(name, got, pixel_sort(src), tol)
        except Exception:
            traceback.print_exc()
            report(False, "mask_socket_removed raised")

    mask_tests()

    # ---- Robustness ----
    scene.render.resolution_x, scene.render.resolution_y = W, H
    image, src = images0

    def run_dummy(idname):
        build_tree(scene, image, idname)
        return render_pixels(scene, idname)

    try:
        out = run_dummy("CompositorNodeTestNoEval")
        const = float(np.abs(out - out[0, 0]).max()) < 1e-6
        report(const and not np.allclose(out, src), "no-evaluate node: no crash, constant default output %s" % (out[0, 0],))
    except Exception:
        traceback.print_exc()
        report(False, "no-evaluate node")

    try:
        out = run_dummy("CompositorNodeTestRaises")
        const = float(np.abs(out - out[0, 0]).max()) < 1e-6
        report(const and not np.allclose(out, src),
               "raising node: traceback printed above, no crash, default output %s" % (out[0, 0],))
    except Exception:
        traceback.print_exc()
        report(False, "raising node")

    try:
        out = run_dummy("CompositorNodeTestGpuOnly")
        if USE_GPU:
            report(np.allclose(out[..., :3], 0.5, atol=2e-3), "GPU-only node in GPU mode fills 0.5")
        else:
            const = float(np.abs(out - out[0, 0]).max()) < 1e-6
            report(const and not np.allclose(out[..., :3], 0.5),
                   "GPU-only node in CPU mode: no crash, default output %s" % (out[0, 0],))
    except Exception:
        traceback.print_exc()
        report(False, "GPU-only node")

    try:
        out = run_dummy("CompositorNodeTestCpuOnly")
        report(np.allclose(out[..., :3], 1.0 - src[..., :3], atol=2e-3),
               "CPU-only node in %s mode inverts input" % ("GPU (round trip)" if USE_GPU else "CPU"))
    except Exception:
        traceback.print_exc()
        report(False, "CPU-only node")


try:
    pixel_sort_node.register()
    main()
except Exception:
    traceback.print_exc()
    report(False, "test aborted with exception")

print("=" * 60)
if FAILURES:
    print("FAILED (%d): %s mode" % (len(FAILURES), "GPU" if USE_GPU else "CPU"))
    sys.stdout.flush()
    sys.exit(1)
print("ALL PASSED (%s mode)" % ("GPU" if USE_GPU else "CPU"))
