# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Flow Field node: LIC invariants (constant image, uniform field == 1D box blur), field sources
against independent numpy formulas, streaks, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_flow_field.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_noise  # noqa: E402

NODE = "CompositorNodeLabFlowField"
SIZE = (96, 64)
F32 = np.float32

# CPU vs GPU: the field is computed twice (numpy / shader): curl noise is a central difference of
# noise values over 2 * 0.01, so float rounding (fma) of ~1e-7 in the noise becomes ~1e-5 in the
# direction; positions drift by about that much per step, and the image changes by at most
# gradient * drift. LIC colour tolerance is set from the observed errors printed by the tests.
GPU_ATOL = 5e-4
GPU_FRAC = 2e-3


def render(device="CPU", size=SIZE, out="Color", props=None, inputs=None, images=None, frame=None):
    inputs = dict(inputs or {})
    inputs.setdefault("Speed", 0.0)
    if images:
        return H.render_node(NODE, device, size, props, inputs, images, out, frame)
    return H.render_generator(NODE, device, size, props, inputs, out, frame)


def white_noise(size, seed=0):
    w, h = size
    ix = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    iy = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))
    return np_noise.rand(ix, iy, seed).astype(np.float64)


def box_blur_x(a, radius, stride=1, kernel="BOX"):
    """Reference: weighted mean of a (h, w) array at offsets -radius..radius (x step `stride`),
    clamp to edge."""
    w = a.shape[1]
    acc = a.copy()
    wsum = 1.0
    for i in range(1, radius + 1):
        wt = 1.0 if kernel == "BOX" else 1.0 - i / (radius + 1)
        for s in (1, -1):
            idx = np.clip(np.arange(w) + s * i * stride, 0, w - 1)
            acc = acc + wt * a[:, idx]
            wsum += wt
    return acc / wsum


def swirl_image(size, centre=(0.4, 0.55)):
    w, h = size
    x = (np.arange(w) + 0.5) / w - centre[0]
    y = (np.arange(h) + 0.5) / h - centre[1]
    X, Y = np.meshgrid(x, y)
    img = np.zeros((h, w, 4), np.float32)
    img[..., 0] = -Y
    img[..., 1] = X
    img[..., 3] = 1.0
    return img


def ramp_image(size, axis=0):
    w, h = size
    img = np.zeros((h, w, 4), np.float32)
    t = (np.arange(w) / w)[None, :] if axis == 0 else (np.arange(h) / h)[:, None]
    img[..., :3] = (0.2 + 0.6 * t)[..., None]
    img[..., 3] = 1.0
    return img


@H.guard("features")
def test_features():
    print("build features:", H.FEATURES)
    img = render()
    H.check(img.shape == (SIZE[1], SIZE[0], 4), "output has the render size %s" % (img.shape,))
    H.check(np.all(img[..., 3] == 1.0), "alpha 1")
    fld = render(out="Field")
    H.check(fld.shape == (SIZE[1], SIZE[0], 4) and np.all(fld[..., 3] == 1.0), "Field output present")
    st = render(out="Streaks")
    H.check(st.shape == (SIZE[1], SIZE[0], 4), "Streaks output present")


@H.guard("lic invariants")
def test_invariants():
    const = np.zeros((SIZE[1], SIZE[0], 4), np.float32)
    const[...] = (0.3, 0.6, 0.9, 1.0)
    for dev, tol in (("CPU", 1e-6), ("GPU", 2e-6)):
        for source in ("CURL", "GRADIENT", "CONTOUR", "VECTOR"):
            for kernel in ("BOX", "TRIANGLE"):
                out = render(dev, props={"source": source, "kernel": kernel}, images={"Image": const},
                             inputs={"Length": 25, "Rotate": 17.0})
                H.compare("%s %s %s: LIC of a constant image is constant" % (dev, source, kernel), out, const, tol)
    # Zero-field regions: constant image stays constant even when streamlines stop immediately.
    for dev in ("CPU", "GPU"):
        out = render(dev, props={"source": "VECTOR"}, images={"Image": const}, inputs={"Vector": (0.0, 0.0, 0.0)})
        H.compare("%s: zero field leaves the image unchanged" % dev, out, const, 1e-6)
    # Uniform horizontal field == 1D horizontal box blur of the source (white noise).
    noise = white_noise(SIZE, 0)
    for radius, step, kernel in ((8, 1.0, "BOX"), (3, 1.0, "BOX"), (12, 1.0, "TRIANGLE"), (5, 2.0, "BOX"),
                                 (0, 1.0, "BOX"), (40, 1.0, "BOX")):
        ref = box_blur_x(noise, radius, int(step), kernel)
        for dev, tol in (("CPU", 2e-6), ("GPU", 2e-5)):
            out = render(dev, props={"source": "VECTOR", "kernel": kernel},
                         inputs={"Length": radius, "Step": step})
            H.compare("%s radius %d step %g %s: uniform horizontal field == 1D box blur" % (
                dev, radius, step, kernel), out[..., 0], ref, tol)
            H.check(np.array_equal(out[..., 0], out[..., 1]), "  grey noise")
    # Same with a linked image, and a vertical field (Rotate 90) blurs along y.
    img = H.test_image(*SIZE, seed=6)
    for dev, tol in (("CPU", 2e-6), ("GPU", 2e-5)):
        out = render(dev, props={"source": "VECTOR"}, images={"Image": img}, inputs={"Length": 6})
        ref = np.stack([box_blur_x(img[..., c].astype(np.float64), 6) for c in range(4)], axis=-1)
        H.compare("%s: linked image, horizontal box blur" % dev, out, ref, tol)
        out = render(dev, props={"source": "VECTOR"}, images={"Image": img}, inputs={"Length": 6, "Rotate": 90.0})
        ref = np.stack([box_blur_x(img[..., c].astype(np.float64).T, 6).T for c in range(4)], axis=-1)
        H.compare("%s: Rotate 90 blurs along y" % dev, out, ref, 5e-5)
    # Length 0 and Step 0 return the source.
    for dev in ("CPU", "GPU"):
        out = render(dev, props={"source": "VECTOR"}, inputs={"Length": 0})
        H.compare("%s: Length 0 returns the white noise" % dev, out[..., 0], noise, 1e-6)
        out = render(dev, props={"source": "VECTOR"}, inputs={"Step": 0.0})
        H.compare("%s: Step 0 returns the white noise" % dev, out[..., 0], noise, 1e-5)
    # LIC reduces variance of the noise strongly along curl streamlines.
    out = render(inputs={"Length": 20})
    H.check(out[..., 0].std() < noise.std() * 0.5, "LIC smooths the noise (std %.3f -> %.3f)" % (
        noise.std(), out[..., 0].std()))


@H.guard("fields")
def test_fields():
    # Constant vector input: Field output encodes the unit direction and magnitude.
    for dev in ("CPU", "GPU"):
        fld = render(dev, out="Field", props={"source": "VECTOR"}, inputs={"Vector": (0.0, 2.0, 0.0)})
        H.compare("%s: Vector (0, 2) -> Field (0.5, 1.0, 2.0, 1)" % dev, fld,
                  np.broadcast_to(np.array([0.5, 1.0, 2.0, 1.0]), fld.shape), 1e-6)
        fld = render(dev, out="Field", props={"source": "VECTOR"}, inputs={"Vector": (3.0, 4.0, 0.0), "Rotate": 90.0})
        H.compare("%s: Vector (3, 4) rotated by 90: direction (-0.8, 0.6)" % dev, fld[..., :3],
                  np.broadcast_to(np.array([0.5 - 0.4, 0.5 + 0.3, 5.0]), fld[..., :3].shape), 1e-6)
        fld = render(dev, out="Field", props={"source": "VECTOR"}, inputs={"Vector": (0.0, 0.0, 0.0)})
        H.compare("%s: zero vector -> Field (0.5, 0.5, 0)" % dev, fld,
                  np.broadcast_to(np.array([0.5, 0.5, 0.0, 1.0]), fld.shape), 0.0)
        # Linked vector image.
        vimg = swirl_image(SIZE)
        fld = render(dev, out="Field", props={"source": "VECTOR"}, images={"Vector": vimg})
        v = vimg[..., :2].astype(np.float64)
        mag = np.hypot(v[..., 0], v[..., 1])
        exp = np.concatenate([0.5 + 0.5 * v / np.maximum(mag, 1e-30)[..., None], mag[..., None]], axis=-1)
        H.compare("%s: linked vector image -> Field" % dev, fld[..., :3], exp, 1e-5)
        # Image gradient: luminance ramp along x flows right (+x); contour flows up (+y).
        ramp = ramp_image(SIZE, 0)
        slope = 0.6 / SIZE[0]
        fld = render(dev, out="Field", props={"source": "GRADIENT"}, images={"Image": ramp})[:, 2:-2]
        H.compare("%s: gradient of an x ramp: direction (1, 0), magnitude = slope" % dev, fld[2:-2],
                  np.broadcast_to(np.array([1.0, 0.5, slope, 1.0]), fld[2:-2].shape), 1e-5)
        fld = render(dev, out="Field", props={"source": "CONTOUR"}, images={"Image": ramp})[:, 2:-2]
        H.compare("%s: contour of an x ramp: direction (0, 1)" % dev, fld[2:-2],
                  np.broadcast_to(np.array([0.5, 1.0, slope, 1.0]), fld[2:-2].shape), 1e-5)
        ramp_y = ramp_image(SIZE, 1)
        fld = render(dev, out="Field", props={"source": "GRADIENT"}, images={"Image": ramp_y})[2:-2]
        H.compare("%s: gradient of a y ramp: direction (0, 1)" % dev, fld[:, 2:-2],
                  np.broadcast_to(np.array([0.5, 1.0, 0.6 / SIZE[1], 1.0]), fld[:, 2:-2].shape), 1e-5)
    # Curl noise: unit directions match np_noise.curl2.
    for ntype, name in ((1, "PERLIN"), (2, "SIMPLEX"), (0, "VALUE")):
        for seed in (0, 7):
            fld = render(out="Field", props={"source": "CURL", "noise_type": name}, inputs={"Scale": 4.0, "Seed": seed,
                                                                                         "Phase": 0.4})
            w, h = SIZE
            k = F32(4.0 / max(w, h))
            xs = np.broadcast_to((np.arange(w, dtype=F32) + F32(0.5))[None, :] * k, (h, w))
            ys = np.broadcast_to((np.arange(h, dtype=F32) + F32(0.5))[:, None] * k, (h, w))
            vx, vy = np_noise.curl2(ntype, xs, ys, F32(0.4), seed, 1.0, 0.01)
            mag = np.hypot(vx.astype(np.float64), vy.astype(np.float64))
            exp = np.stack([0.5 + 0.5 * vx / mag, 0.5 + 0.5 * vy / mag], axis=-1)
            H.compare("curl %s seed %d: Field direction == np_noise.curl2" % (name, seed), fld[..., :2], exp, 1e-5)
            H.compare("  magnitude", fld[..., 2], mag, 1e-5 + 1e-5 * mag.max(), quiet=True)
            # The curl is divergence free: mean divergence of the raw field is tiny vs its gradient scale.
            ux, uy = vx.astype(np.float64), vy.astype(np.float64)
            div = np.gradient(ux, axis=1) + np.gradient(uy, axis=0)
            full = np.abs(np.gradient(ux, axis=1)).mean() + np.abs(np.gradient(uy, axis=0)).mean()
            H.check(np.abs(div[2:-2, 2:-2]).mean() < 0.1 * full, "curl %s: divergence free (%.2e vs %.2e)" % (
                name, np.abs(div[2:-2, 2:-2]).mean(), full))
    # Rotating a curl field by 90 degrees rotates the Field output.
    a = render(out="Field", inputs={"Seed": 2})
    b = render(out="Field", inputs={"Seed": 2, "Rotate": 90.0})
    da = (a[..., :2] - 0.5) * 2
    db = (b[..., :2] - 0.5) * 2
    H.compare("Rotate 90: (x, y) -> (-y, x)", db, np.stack([-da[..., 1], da[..., 0]], axis=-1), 1e-5)
    H.compare("Rotate keeps the magnitude", b[..., 2], a[..., 2], 1e-5)
    # Phase / seed change the field; time with Speed (F2).
    c = render(out="Field", inputs={"Seed": 3})
    d = render(out="Field", inputs={"Seed": 2, "Phase": 0.5})
    H.check(float(np.abs(a - c).mean()) > 0.05 and float(np.abs(a - d).mean()) > 0.02, "seed and phase change the field")
    if H.FEATURES.get("F2"):
        import bpy
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        for dev in ("CPU", "GPU"):
            f25 = render(dev, out="Field", frame=25, inputs={"Speed": 0.5})
            f1 = render(dev, out="Field", frame=1, inputs={"Speed": 0.5})
            ph = render(dev, out="Field", inputs={"Phase": 25 / fps * 0.5})
            H.check(float(np.abs(f25 - f1).mean()) > 0.01, "%s: field animates with time" % dev)
            H.compare("%s: frame 25 speed 0.5 == Phase time * speed" % dev, f25, ph, 1e-5 if dev == "CPU" else 2e-4,
                      max_frac=1e-3)
    else:
        H.note("F2 not available: time animation check skipped")


@H.guard("streaks")
def test_streaks():
    for dev in ("CPU", "GPU"):
        s = render(dev, out="Streaks", props={"source": "VECTOR"}, inputs={"Length": 20, "Density": 0.05})[..., 0]
        H.check(s.min() >= 0.0 and s.max() <= 1.0 and np.isfinite(s).all(), "%s: Streaks in [0, 1]" % dev)
        # Along a horizontal field, streaks are long in x and uncorrelated in y.
        z = s - s.mean()
        cx = (z[:, :-5] * z[:, 5:]).mean() / z.var()
        cy = (z[:-5] * z[5:]).mean() / z.var()
        H.check(cx > 0.5 and abs(cy) < 0.15, "%s: streaks follow the field (corr x %.2f, y %.2f)" % (dev, cx, cy))
        z0 = render(dev, out="Streaks", props={"source": "VECTOR"}, inputs={"Density": 0.0})[..., 0]
        H.check(z0.max() == 0.0, "%s: density 0 -> no streaks" % dev)
        # Streaks == the impulse noise convolved the same way: reference from a box blur.
        from compositor_lab.lib import np_field
        imp = np_field.impulses(SIZE[0], SIZE[1], 0, 0.05).astype(np.float64)
        ref = np.clip(box_blur_x(imp, 20) * 0.5 / 0.05, 0, 1)
        H.compare("%s: Streaks == box-blurred impulses * 0.5 / density" % dev, s, ref, 1e-5 if dev == "CPU" else 1e-4)
    # More density -> brighter / more filled.
    a = render(out="Streaks", inputs={"Density": 0.02, "Length": 15})[..., 0].mean()
    b = render(out="Streaks", inputs={"Density": 0.3, "Length": 15})[..., 0].mean()
    H.check(a > 0.0 and b > 0.0, "streaks present at both densities (%.3f, %.3f)" % (a, b))


@H.guard("gpu parity")
def test_gpu():
    img = H.test_image(80, 56, seed=3)
    smooth = ramp_image((80, 56), 0)
    cases = [
        ("curl perlin (noise source)", dict(inputs={"Length": 25})),
        ("curl simplex", dict(props={"noise_type": "SIMPLEX"}, inputs={"Length": 15, "Seed": 5, "Scale": 5.0})),
        ("curl value triangle", dict(props={"noise_type": "VALUE", "kernel": "TRIANGLE"}, inputs={"Length": 18})),
        ("curl on image", dict(images={"Image": img}, size=(80, 56), inputs={"Length": 12, "Rotate": 30.0})),
        ("gradient", dict(props={"source": "GRADIENT"}, images={"Image": img}, size=(80, 56), inputs={"Length": 10})),
        ("contour", dict(props={"source": "CONTOUR"}, images={"Image": img}, size=(80, 56),
                         inputs={"Length": 16, "Step": 0.7})),
        ("vector swirl", dict(props={"source": "VECTOR"}, images={"Vector": swirl_image((80, 56)), "Image": img},
                              size=(80, 56), inputs={"Length": 20})),
        ("step 1.7 phase", dict(inputs={"Step": 1.7, "Phase": 0.3, "Length": 14})),
        ("length 0", dict(inputs={"Length": 0})),
        ("contour of a ramp", dict(props={"source": "CONTOUR"}, images={"Image": smooth}, size=(80, 56))),
    ]
    smooth = ("vector swirl", "contour of a ramp", "length 0")
    for name, kw in cases:
        for out in ("Color", "Streaks", "Field"):
            c = render("CPU", out=out, **kw)
            g = render("GPU", out=out, **kw)
            label = "%s [%s]: GPU vs CPU" % (name, out)
            if out == "Field":
                # Curl noise differences amplify the 1e-7 noise rounding by 1 / (2 * 0.01).
                H.compare(label, g, c, 1e-3)
            elif name in smooth:
                H.compare(label, g, c, 1e-5 if out == "Color" else 5e-5)
            else:
                # Chaotic cases: a streamline passing a point where the interpolated field is
                # nearly zero (noisy gradient images, curl extrema) turns by an amount that
                # depends on float rounding. Observed over all cases: mean abs diff <= 1.3e-4,
                # <= 1.8% of pixels differ by more than 5e-4, max 0.02 in colour (0.25 in the
                # high-contrast impulse streaks of the noisy gradient). Require the same.
                d = np.abs(g.astype(np.float64) - c)
                over = float((d.max(axis=-1) > 5e-4).mean())
                print("  %s: mean %.2e, %.3f%% over 5e-4, max %.3g" % (label, d.mean(), 100 * over, d.max()))
                H.check(d.mean() < 3e-4 and over < 0.03 and (d.max() < 0.3), label)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (333, 187), (17, 17)):
            for out in ("Color", "Streaks", "Field"):
                img = render(dev, size=size, out=out)
                H.check(img.shape == (size[1], size[0], 4) and np.isfinite(img).all(),
                        "%s %dx%d %s: shape/finite" % (dev, size[0], size[1], out))
        for name, inputs in (("length -5", {"Length": -5}), ("length 9999", {"Length": 9999}),
                             ("step -1", {"Step": -1.0}), ("step 1e3", {"Step": 1e3}),
                             ("scale 0", {"Scale": 0.0}), ("density 2", {"Density": 2.0}),
                             ("huge phase", {"Phase": 1e5}), ("seed min", {"Seed": -2 ** 31})):
            for source in ("CURL", "VECTOR"):
                for out in ("Color", "Streaks", "Field"):
                    img = render(dev, out=out, props={"source": source}, inputs=inputs)
                    H.check(np.isfinite(img).all() and img.min() >= -1e-6 and img.max() <= 1.0 + 1e-6 or out == "Field",
                            "%s %s %s %s: finite, in range" % (dev, source, name, out))
        for source in ("GRADIENT", "CONTOUR", "VECTOR"):
            for isize in ((4, 4), (7, 33)):
                img = render(dev, size=isize, props={"source": source}, images={"Image": H.test_image(*isize)})
                H.check(img.shape == (isize[1], isize[0], 4) and np.isfinite(img).all(),
                        "%s %s image %dx%d" % (dev, source, isize[0], isize[1]))
        # Gradient / contour without an Image: zero field -> the noise is returned untouched.
        for source in ("GRADIENT", "CONTOUR"):
            out = render(dev, props={"source": source})
            H.compare("%s %s without an image: unchanged noise" % (dev, source), out[..., 0], white_noise(SIZE), 1e-6)
    # Linked scalar inputs fall back to defaults on both backends.
    scene = H.configure_scene(SIZE, "CPU")
    res = {}
    for dev in ("CPU", "GPU"):
        H.configure_scene(SIZE, dev)
        H.build_tree(scene, NODE, images={"Scale": np.full((SIZE[1], SIZE[0]), 0.5, F32)}, inputs={"Speed": 0.0})
        res[dev] = H.render(scene)
        H._clear_images()
        H.check(np.isfinite(res[dev]).all(), "%s: linked Scale does not break the node" % dev)
    H.compare("linked Scale: GPU == CPU", res["GPU"], res["CPU"], GPU_ATOL, max_frac=GPU_FRAC)


for fn in (test_features, test_invariants, test_fields, test_streaks, test_gpu, test_robustness):
    fn()
H.finish()
