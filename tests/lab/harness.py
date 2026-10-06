# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Shared helpers for the Compositor Lab tests (run inside Blender, background mode).

    import harness as H
    H.setup()                                   # registers the add-on, probes the build
    img = H.test_image(64, 48, alpha=True)      # numpy (H, W, 4) premultiplied, row 0 = bottom
    out = H.render_node("CompositorNodeLabBlendModes", device="CPU", size=(64, 48),
                        props={"blend_type": "SCREEN"}, images={"A": a, "B": b})
    H.compare("screen cpu vs gpu", out_cpu, out_gpu, atol=1e-5)
    H.finish()                                  # raises AssertionError listing failures

Rendering goes through the real compositor: image nodes -> node -> Group Output, written to a
32-bit float EXR and read back (so everything the node does is exercised end to end).
"""

import os
import sys
import tempfile
import traceback

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADDONS = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "addons"))
if ADDONS not in sys.path:
    sys.path.insert(0, ADDONS)

TMP = tempfile.mkdtemp(prefix="compositor_lab_test_")
FAILURES = []
NOTES = []
FEATURES = {}          # filled by setup(): F1 / F2 availability of this build


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(ok, msg):
    print(("PASS: " if ok else "FAIL: ") + msg)
    sys.stdout.flush()
    if not ok:
        FAILURES.append(msg)
    return ok


def check(cond, msg):
    return report(bool(cond), msg)


def note(msg):
    print("NOTE: " + msg)
    NOTES.append(msg)


def guard(name):
    """Decorator: run a test function, reporting an exception as a failure."""
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception:
                traceback.print_exc()
                report(False, "%s raised" % name)
        return run
    return deco


def finish():
    print("=" * 60)
    if FAILURES:
        raise AssertionError("%d failure(s):\n  %s" % (len(FAILURES), "\n  ".join(FAILURES)))
    print("ALL PASSED")


def compare(name, got, ref, atol, rtol=0.0, max_frac=0.0, quiet=False):
    """Report max/mean abs diff; pass if the fraction of pixels (any channel) with
    |got - ref| > atol + rtol * |ref| is <= max_frac, and all values are finite."""
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        return report(False, "%s: shape mismatch %s vs %s" % (name, got.shape, ref.shape))
    d = np.abs(got - ref)
    bad = d > (atol + rtol * np.abs(ref))
    badpix = bad.reshape(bad.shape[0], bad.shape[1], -1).any(axis=2) if bad.ndim > 2 else bad
    frac = float(badpix.mean())
    finite = bool(np.isfinite(got).all())
    msg = "%s: max abs diff %.3g, mean %.3g, %.4f%% pixels over tol (atol %g rtol %g)" % (
        name, float(d.max()), float(d.mean()), frac * 100.0, atol, rtol)
    if not quiet:
        print("  " + msg)
    return report(finite and frac <= max_frac, name)


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------

def test_image(w, h, seed=1, alpha=False, hdr=False):
    """Deterministic RGBA float32 image, (h, w, 4). With alpha=True the colours are
    premultiplied by a smoothly varying alpha in (0.05, 1]; alpha=False gives alpha 1."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = 0.6 * rng.random((h, w, 3), dtype=np.float32) + 0.4 * np.stack(
        [x / max(w, 1), y / max(h, 1), 0.5 + 0.5 * np.sin(x / 7.0 + y / 11.0)], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    if hdr:
        rgb = (rgb * (1.0 + 3.0 * rng.random((h, w, 1), dtype=np.float32))).astype(np.float32)
    px = np.empty((h, w, 4), np.float32)
    if alpha:
        a = (0.05 + 0.95 * rng.random((h, w), dtype=np.float32)).astype(np.float32)
        px[..., :3] = rgb * a[..., None]
        px[..., 3] = a
    else:
        px[..., :3] = rgb
        px[..., 3] = 1.0
    return px


def make_image(name, arr):
    """bpy image holding `arr` ((h, w, 4) or (h, w): grey) verbatim (Non-Color, no alpha handling)."""
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.ones_like(arr)], axis=-1)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    img.alpha_mode = 'CHANNEL_PACKED'
    img.pixels.foreach_set(np.ascontiguousarray(arr).ravel())
    img.update()
    img.pack()
    return img


# ---------------------------------------------------------------------------
# Scene, tree, render
# ---------------------------------------------------------------------------

def configure_scene(size, device):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = int(size[0]), int(size[1])
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA'
    for owner in (scene.render, scene):
        if hasattr(owner, "compositor_device"):
            owner.compositor_device = device
            if hasattr(owner, "compositor_precision"):
                owner.compositor_precision = 'FULL'
            break
    return scene


def _clear_images(prefix="lab_"):
    for img in [i for i in bpy.data.images if i.name.startswith(prefix)]:
        bpy.data.images.remove(img)


_img_counter = [0]


def build_tree(scene, node_idname, props=None, inputs=None, images=None, out_socket=None,
               extra_links=None):
    """Image nodes -> node -> Group Output. `inputs`: {socket: default_value};
    `images`: {socket: (h, w, 4|grey) array} linked through Image nodes. Returns the node."""
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
        scene.compositing_node_group = tree
    tree.nodes.clear()
    if not any(i.item_type == 'SOCKET' and i.in_out == 'OUTPUT' for i in tree.interface.items_tree):
        tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type="NodeSocketColor")
    node = tree.nodes.new(node_idname)
    n_out = tree.nodes.new('NodeGroupOutput')
    for k, v in (props or {}).items():
        setattr(node, k, v)
    for k, v in (inputs or {}).items():
        node.inputs[k].default_value = v
    for sock, arr in (images or {}).items():
        _img_counter[0] += 1
        img = make_image("lab_%d_%s" % (_img_counter[0], sock), arr)
        n_img = tree.nodes.new('CompositorNodeImage')
        n_img.image = img
        tree.links.new(n_img.outputs["Image"], node.inputs[sock])
    if out_socket is None:
        out_socket = node.outputs[0].name
    tree.links.new(node.outputs[out_socket], n_out.inputs[0])
    return node


def read_exr(path):
    img = bpy.data.images.load(path)
    try:
        img.alpha_mode = 'CHANNEL_PACKED'
        w, h = img.size
        buf = np.empty(w * h * img.channels, np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(h, w, img.channels).copy()
    finally:
        bpy.data.images.remove(img)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=2)
    return arr


_render_counter = [0]


def render(scene):
    """Render the scene's compositor output; returns (h, w, 4) float32, row 0 = bottom."""
    _render_counter[0] += 1
    path = os.path.join(TMP, "render_%d.exr" % _render_counter[0])
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(path):
        raise RuntimeError("render produced no file")
    arr = read_exr(path)
    os.remove(path)
    return arr


def render_node(node_idname, device="CPU", size=(64, 48), props=None, inputs=None,
                images=None, out_socket=None, frame=None):
    """Build Image -> node -> output and render it with the CPU or GPU compositor."""
    scene = configure_scene(size, device)
    if frame is not None:
        scene.frame_set(int(frame))
    build_tree(scene, node_idname, props, inputs, images, out_socket)
    try:
        return render(scene)
    finally:
        _clear_images()


# ---------------------------------------------------------------------------
# Framework feature probes (F1 generator domain, F2 evaluation context)
# ---------------------------------------------------------------------------

_probe_seen = {}


def _make_feature_probe():
    from compositor_lab.lib.node import LabNode, Out

    class CompositorNodeLabTestFeatureProbe(LabNode, bpy.types.CompositorNode):
        bl_idname = "CompositorNodeLabTestFeatureProbe"
        bl_label = "Feature Probe"
        SOCKETS = [Out("Color", "COLOR")]

        def cpu(self, inputs, outputs, ctx):
            arr = self.out_array(outputs, "Color")
            _probe_seen["shape"] = arr.shape[:2]
            _probe_seen["ctx"] = ctx
            arr[...] = 0.5

        def gpu(self, inputs, outputs, ctx):
            tex = self.out_texture(outputs, "Color")
            _probe_seen["shape"] = (tex.height, tex.width)
            _probe_seen["ctx"] = ctx
            tex.clear(format='FLOAT', value=(0.5, 0.5, 0.5, 1.0))

    return CompositorNodeLabTestFeatureProbe


def detect_features():
    """F1: a node with no image inputs gets the render-sized domain. F2: a context is passed."""
    cls = _make_feature_probe()
    bpy.utils.register_class(cls)
    try:
        size = (40, 24)
        configure_scene(size, "CPU")
        _probe_seen.clear()
        render_node(cls.bl_idname, "CPU", size)
        FEATURES["F1"] = tuple(_probe_seen.get("shape", ())) == (size[1], size[0])
        ctx = _probe_seen.get("ctx")
        FEATURES["F2"] = bool(ctx is not None and ctx.has_context)
    except Exception:
        traceback.print_exc()
        FEATURES.setdefault("F1", False)
        FEATURES.setdefault("F2", False)
    finally:
        bpy.utils.unregister_class(cls)
    print("FEATURES: F1 (generator domain) %s, F2 (context) %s" % (
        FEATURES["F1"], FEATURES["F2"]))


def setup(detect=True):
    """Register the add-on (from the source tree) and probe the build's features."""
    import compositor_lab
    compositor_lab.register()
    if compositor_lab.ERRORS:
        raise RuntimeError("add-on registration errors: %s" % list(compositor_lab.ERRORS))
    if detect:
        detect_features()
    return compositor_lab


# ---------------------------------------------------------------------------
# Generator nodes before F1: a subclass with an extra "Ref" image input gives them a domain.
# ---------------------------------------------------------------------------

_ref_classes = {}


def generator_idname(idname):
    """idname to use for a generator node: the node itself with F1, else a test subclass that
    has an extra "Ref" input (link a reference image to it with `ref_images`)."""
    if FEATURES.get("F1"):
        return idname
    if idname not in _ref_classes:
        py_cls = next(c for c in _find_subclasses(idname))

        class RefNode(py_cls):
            bl_idname = idname + "Ref"
            bl_label = py_cls.bl_label + " (ref)"

            def init(self, context):
                py_cls.init(self, context)
                self.inputs.new('NodeSocketColor', "Ref")

        RefNode.__name__ = RefNode.__qualname__ = idname + "Ref"
        bpy.utils.register_class(RefNode)
        _ref_classes[idname] = RefNode
    return _ref_classes[idname].bl_idname


def _find_subclasses(idname):
    import compositor_lab
    for classes in compositor_lab.REGISTERED.values():
        for c in classes:
            if c.bl_idname == idname:
                yield c


def render_generator(idname, device="CPU", size=(64, 48), props=None, inputs=None,
                     out_socket=None, frame=None):
    """Render a generator node (no required image inputs) at `size`."""
    real = generator_idname(idname)
    images = None
    if real != idname:
        images = {"Ref": np.zeros((size[1], size[0], 4), np.float32)}
    return render_node(real, device, size, props, inputs, images, out_socket, frame)


# ---------------------------------------------------------------------------
# Probe node: evaluates a lib function on both backends (see test_lib.py)
# ---------------------------------------------------------------------------

PROBES = {}   # name -> dict(body=glsl, libs=(...), np=callable(x, y, z) -> (h, w, 4), uniforms={})
PROBE_K = 0.173


def _make_probe_class():
    from compositor_lab.lib import gpu as lab_gpu
    from compositor_lab.lib.node import In, LabNode, Out

    class CompositorNodeLabTestProbe(LabNode, bpy.types.CompositorNode):
        bl_idname = "CompositorNodeLabTestProbe"
        bl_label = "Probe"
        SOCKETS = [In("Ref", "COLOR", (0, 0, 0, 1)), Out("Color", "COLOR")]
        probe_id: bpy.props.StringProperty()

        def cpu(self, inputs, outputs, ctx):
            out = self.out_array(outputs, "Color")
            w, h = ctx.size
            x, y = probe_xy(w, h)
            out[...] = PROBES[self.probe_id]["np"](x, y, np.float32(0.37))

        def gpu(self, inputs, outputs, ctx):
            spec = PROBES[self.probe_id]
            body = "    vec3 p = vec3((vec2(texel) + vec2(0.5)) * pk + vec2(0.13, -0.21), 0.37);\n" \
                   + spec["body"]
            lab_gpu.pointwise(body, {"Color": self.out_texture(outputs, "Color")},
                              uniforms={"pk": ("float", PROBE_K)}, libs=spec["libs"])

    return CompositorNodeLabTestProbe


def probe_xy(w, h):
    f32 = np.float32
    xs = (np.arange(w, dtype=f32) + f32(0.5)) * f32(PROBE_K) + f32(0.13)
    ys = (np.arange(h, dtype=f32) + f32(0.5)) * f32(PROBE_K) + f32(-0.21)
    return np.broadcast_to(xs[None, :], (h, w)), np.broadcast_to(ys[:, None], (h, w))


_probe_cls = []


def run_probe(name, device, size=(64, 48)):
    if not _probe_cls:
        _probe_cls.append(_make_probe_class())
        bpy.utils.register_class(_probe_cls[0])
    return render_node("CompositorNodeLabTestProbe", device, size, props={"probe_id": name},
                       images={"Ref": np.zeros((size[1], size[0], 4), np.float32)})
