# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Expression node: >= 25 expressions evaluated by the compiler's numpy backend, the node on the
CPU compositor, the node on the GPU compositor, and an independent direct numpy formula; rejection
of unsafe input; compile cache; robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_expression.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import expr, np_noise  # noqa: E402

NODE = "CompositorNodeLabExpression"
SIZE = (64, 48)
F32 = np.float32

# Tolerances. Compiler numpy backend / CPU node (float32) vs a float64 reference: rounding only.
REF_ATOL = 2e-5
# GPU vs CPU: same float32 formulas, but the GPU's sin / cos / tan / atan / exp / log / pow are
# approximate (a few ulp, larger far from zero) and it may fuse multiply-adds.
GPU_ATOL = 2e-4
# noise() amplifies a ~1e-6 coordinate difference by the gradient (see test_noise.py).
NOISE_ATOL = 6e-4


def env64(size, a, b, t=0.0, frame=0.0):
    w, h = size
    x = np.broadcast_to((np.arange(w) + 0.5)[None, :], (h, w))
    y = np.broadcast_to((np.arange(h) + 0.5)[:, None], (h, w))
    return dict(a=a.astype(np.float64), b=b.astype(np.float64), c=np.zeros((h, w, 4)) + [0, 0, 0, 1],
                x=x, y=y, u=x / w, v=y / h, w=float(w), h=float(h), t=t, frame=frame)


def grey(v):
    v = np.asarray(v, np.float64)
    out = np.ones(v.shape + (4,))
    out[..., :3] = v[..., None]
    return out


def rgb(v):
    v = np.asarray(v, np.float64)
    return np.concatenate([v, np.ones(v.shape[:-1] + (1,))], axis=-1)


def vlen(v):
    return np.sqrt((v * v).sum(axis=-1))


def fr(x):
    return x - np.floor(x)


def _noise_ref(e):
    f = np.float32
    x = (e["u"] * 8).astype(f)
    y = (e["v"] * 8).astype(f)
    z = np.full_like(x, f(0.5))
    return grey(np_noise.noise_field(1, x, y, z, 0, 1.0, 1, 2.0, 0.5, False, 0.0))


# (expression, reference(e) -> (h, w, 4) float64, gpu atol)
CASES = [
    ("u", lambda e: grey(e["u"]), GPU_ATOL),
    ("v * 2 - 0.5", lambda e: grey(e["v"] * 2 - 0.5), GPU_ATOL),
    ("x / w + y / h", lambda e: grey(e["x"] / e["w"] + e["y"] / e["h"]), GPU_ATOL),
    ("sin(u * 6.2831853) * 0.5 + 0.5", lambda e: grey(np.sin(e["u"] * 6.2831853) * 0.5 + 0.5), GPU_ATOL),
    ("cos(v * 3) * cos(u * 5)", lambda e: grey(np.cos(e["v"] * 3) * np.cos(e["u"] * 5)), GPU_ATOL),
    ("sqrt(u * v)", lambda e: grey(np.sqrt(e["u"] * e["v"])), GPU_ATOL),
    ("abs(u - 0.5) * 2", lambda e: grey(np.abs(e["u"] - 0.5) * 2), GPU_ATOL),
    ("floor(u * 8) / 8", lambda e: grey(np.floor(e["u"] * 8) / 8), GPU_ATOL),
    ("fract(u * 4.0 + v * 3.0 + 0.123)", lambda e: grey(fr(e["u"] * 4 + e["v"] * 3 + 0.123)), GPU_ATOL),
    ("u % 0.3", lambda e: grey(e["u"] - 0.3 * np.floor(e["u"] / 0.3)), GPU_ATOL),
    ("u ** 2 + v ** 3 - u ** -1 * 0.1", lambda e: grey(e["u"] ** 2 + e["v"] ** 3 - e["u"] ** -1 * 0.1), GPU_ATOL),
    ("pow(u, 1.7) + pow(v, 0.5)", lambda e: grey(e["u"] ** 1.7 + e["v"] ** 0.5), GPU_ATOL),
    ("atan2(v - 0.5, u - 0.5)", lambda e: grey(np.arctan2(e["v"] - 0.5, e["u"] - 0.5)), GPU_ATOL),
    ("max(min(u, v), 0.25)", lambda e: grey(np.maximum(np.minimum(e["u"], e["v"]), 0.25)), GPU_ATOL),
    ("clamp(u * 2 - 0.5, 0.2, 0.8)", lambda e: grey(np.clip(e["u"] * 2 - 0.5, 0.2, 0.8)), GPU_ATOL),
    ("mix(u, v, 0.25)", lambda e: grey(e["u"] + (e["v"] - e["u"]) * 0.25), GPU_ATOL),
    ("smoothstep(0.2, 0.8, u)", lambda e: grey(
        (lambda t: t * t * (3 - 2 * t))(np.clip((e["u"] - 0.2) / 0.6, 0, 1))), GPU_ATOL),
    ("step(0.5, u) + step(0.25, v)", lambda e: grey((e["u"] >= 0.5) * 1.0 + (e["v"] >= 0.25) * 1.0), GPU_ATOL),
    ("exp(-u * 3) + log(v + 1.0)", lambda e: grey(np.exp(-e["u"] * 3) + np.log(e["v"] + 1.0)), GPU_ATOL),
    ("length(vec(u - 0.5, v - 0.5))", lambda e: grey(np.sqrt((e["u"] - 0.5) ** 2 + (e["v"] - 0.5) ** 2)), GPU_ATOL),
    ("dot(vec(u, v, 1.0), vec(0.5, 0.25, 2.0))", lambda e: grey(e["u"] * 0.5 + e["v"] * 0.25 + 2.0), GPU_ATOL),
    ("normalize(vec(u, v, 1.0))", lambda e: rgb(
        np.stack([e["u"], e["v"], np.ones_like(e["u"])], -1) /
        vlen(np.stack([e["u"], e["v"], np.ones_like(e["u"])], -1))[..., None]), GPU_ATOL),
    ("a * u + b * (1 - u)", lambda e: e["a"] * e["u"][..., None] + e["b"] * (1 - e["u"])[..., None], GPU_ATOL),
    ("vec(a.r, b.g, a.b + b.b, 1.0)", lambda e: np.stack(
        [e["a"][..., 0], e["b"][..., 1], e["a"][..., 2] + e["b"][..., 2], np.ones_like(e["u"])], -1), GPU_ATOL),
    ("a.rgb * b.rgb", lambda e: rgb(e["a"][..., :3] * e["b"][..., :3]), GPU_ATOL),
    ("u > 0.5 and v < 0.5", lambda e: grey(((e["u"] > 0.5) & (e["v"] < 0.5)) * 1.0), GPU_ATOL),
    ("1.0 if u > v else 0.25", lambda e: grey(np.where(e["u"] > e["v"], 1.0, 0.25)), GPU_ATOL),
    ("0.2 if not (u < 0.5) or v > 0.9 else 0.7", lambda e: grey(
        np.where(~(e["u"] < 0.5) | (e["v"] > 0.9), 0.2, 0.7)), GPU_ATOL),
    ("a.bgr * 0.5 + 0.1", lambda e: rgb(e["a"][..., ::-1][..., 1:] * 0.5 + 0.1), GPU_ATOL),
    ("vec(u, v, 0.0) if u < 0.5 else a.rgb", lambda e: rgb(np.where(
        (e["u"] < 0.5)[..., None], np.stack([e["u"], e["v"], np.zeros_like(e["u"])], -1), e["a"][..., :3])), GPU_ATOL),
    ("luma(a) + luma(b.rgb) * 0.5", lambda e: grey(
        (e["a"][..., :3] * [0.2126, 0.7152, 0.0722]).sum(-1) +
        0.5 * (e["b"][..., :3] * [0.2126, 0.7152, 0.0722]).sum(-1)), GPU_ATOL),
    ("noise(u * 8, v * 8, 0.5)", _noise_ref, NOISE_ATOL),
    ("tan(u * 0.7) + asin(u) + acos(v) + atan(u * 3)", lambda e: grey(
        np.tan(e["u"] * 0.7) + np.arcsin(e["u"]) + np.arccos(e["v"]) + np.arctan(e["u"] * 3)), GPU_ATOL),
    ("t * 0.1 + frame * 0.01 + u", lambda e: grey(e["t"] * 0.1 + e["frame"] * 0.01 + e["u"]), GPU_ATOL),
    ("u < v < 0.9", lambda e: grey(((e["u"] < e["v"]) & (e["v"] < 0.9)) * 1.0), GPU_ATOL),
    ("ceil(u * 4.5) * 0.1 - sign(v - 0.5) * 0.1 + -u + +v", lambda e: grey(
        np.ceil(e["u"] * 4.5) * 0.1 - np.sign(e["v"] - 0.5) * 0.1 - e["u"] + e["v"]), GPU_ATOL),
    ("pi * u + tau * v * 0.1 + w * 0.001 + h * 0.001 + x * 0.0001", lambda e: grey(
        np.pi * e["u"] + 2 * np.pi * e["v"] * 0.1 + e["w"] * 0.001 + e["h"] * 0.001 + e["x"] * 0.0001), GPU_ATOL),
    ("vec(a.xy, 0.5) + vec(b.rrr) + vec(0.1)", lambda e: rgb(np.concatenate(
        [e["a"][..., :2], np.full(e["u"].shape + (1,), 0.5)], -1) + e["b"][..., :1] + 0.1), GPU_ATOL),
]

BAD = [
    "__import__('os').system('true')",
    "__import__('os')",
    "a.__class__",
    "().__class__.__bases__",
    "(lambda: 1)()",
    "[i for i in range(3)]",
    "{i: i for i in range(3)}",
    "(i for i in a)",
    "a[0]",
    "a[0:2]",
    "open('/etc/passwd')",
    "eval('1')",
    "exec('1')",
    "print(1)",
    "a.real",
    "a.rgbx",
    "a.r.g",
    "u.r",
    "a.xyzwx",
    "'string'",
    "None",
    "True",
    "1j",
    "u // 2",
    "u & 1",
    "u << 1",
    "u @ v",
    "~u",
    "u if a else v",
    "a < b",
    "u in v",
    "u is v",
    "sin",
    "sin()",
    "sin(u, v)",
    "sin(x=u)",
    "sin(*a)",
    "foo(u)",
    "q",
    "u; v",
    "u = 3",
    "(u := 3)",
    "f'{u}'",
    "vec(a, b)",
    "vec(u, v, a)",
    "vec()",
    "a + vec(u, v)",
    "vec(u, v)",
    "dot(a, vec(u))",
    "luma(u)",
    "a.rgb + a",
    "mix(a, a.rgb, 0.5)",
    "noise(a, u)",
    "not a",
    "u and a",
    "",
    "   ",
    "1 +",
    "((((",
    "u" + " + u" * 600,
    "1e999",
    "u.__class__",
    "globals",
    "getattr(a, 'r')",
    "min",
    "x.y",
    "-" * 300 + "u",
    "u ** -" * 100 + "1",
]


@H.guard("rejection")
def test_rejection():
    for text in BAD:
        try:
            expr.compile_expr(text)
        except expr.ExprError as ex:
            H.check(len(str(ex)) > 0, "rejected %r: %s" % (text[:40], str(ex)[:70]))
        except Exception as ex:   # any other exception type is a bug: the error must be clean
            H.report(False, "%r raised %s instead of ExprError: %s" % (text[:40], type(ex).__name__, ex))
        else:
            H.report(False, "%r was accepted" % text[:40])
    # A few accepted spellings.
    for text in ("u", " u \n", "a.rgb", "A" .lower(), "vec(1)", "vec(a.rg, b.rg)", "-u ** 2", "2 ** 0.5"):
        try:
            expr.compile_expr(text)
            H.report(True, "accepted %r" % text)
        except expr.ExprError as ex:
            H.report(False, "%r wrongly rejected: %s" % (text, ex))


@H.guard("cache")
def test_cache():
    c1 = expr.compile_expr("u * v + 0.25")
    c2 = expr.compile_expr("u * v + 0.25")
    H.check(c1 is c2, "compiled expressions are cached by string")
    H.check(c1.out_type == "f" and "e_time" not in c1.glsl_body, "scalar result type, no time use")
    H.check(expr.compile_expr("a").inputs_used == ["A"], "inputs_used reports only the colours read")
    # The same expression string compiles to the same GLSL body (so the shader cache hits).
    expr._cache.clear()
    H.check(expr.compile_expr("u * v + 0.25").glsl_body == c1.glsl_body, "GLSL generation is deterministic")


@H.guard("compiler numpy vs reference")
def test_compiler_numpy():
    a = H.test_image(*SIZE, seed=11, alpha=True)
    b = H.test_image(*SIZE, seed=12, alpha=False)
    e = env64(SIZE, a, b, t=0.75, frame=18.0)
    assert len(CASES) >= 25
    for text, ref, _ in CASES:
        comp = expr.compile_expr(text)
        out = comp.eval_numpy(SIZE[::-1], {"a": a, "b": b, "c": e["c"], "d": e["c"]}, 0.75, 18.0)
        H.check(out.dtype == F32 and out.shape == (SIZE[1], SIZE[0], 4), "%s: float32 (h, w, 4)" % text)
        r = ref(e)
        if r.shape[-1] != 4:
            r = np.concatenate([r, np.ones(r.shape[:-1] + (1,))], -1)
        H.compare("numpy backend %s vs float64 reference" % text, out, r, REF_ATOL, rtol=REF_ATOL, quiet=True)


def render(device, text, size=SIZE, images=None, inputs=None, frame=None, allow_errors=False):
    if images is None:
        return H.render_generator(NODE, device, size, props={"expression": text}, inputs=inputs,
                                  frame=frame, allow_errors=allow_errors)
    return H.render_node(NODE, device, size, props={"expression": text}, images=images,
                         inputs=inputs, frame=frame, allow_errors=allow_errors)


@H.guard("node cpu/gpu")
def test_node():
    a = H.test_image(*SIZE, seed=11, alpha=True)
    b = H.test_image(*SIZE, seed=12, alpha=False)
    frame = 18
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    e = env64(SIZE, a, b)
    for text, ref, gpu_atol in CASES:
        comp = expr.compile_expr(text)
        cpu = render("CPU", text, images={"A": a, "B": b}, frame=frame)
        gpu = render("GPU", text, images={"A": a, "B": b}, frame=frame)
        if "t" in text.split() or "frame" in text or " t " in text:
            time = frame / fps if H.FEATURES.get("F2") else 0.0
            fr_ = float(frame) if H.FEATURES.get("F2") else 0.0
            e = env64(SIZE, a, b, t=time, frame=fr_)
        else:
            e = env64(SIZE, a, b)
        direct = comp.eval_numpy(SIZE[::-1], {"a": a, "b": b, "c": e["c"], "d": e["c"]},
                                 e["t"], e["frame"])
        r = ref(e)
        if r.shape[-1] != 4:
            r = np.concatenate([r, np.ones(r.shape[:-1] + (1,))], -1)
        H.compare("CPU node %s vs direct numpy formula" % text, cpu, r, REF_ATOL, rtol=REF_ATOL, quiet=True)
        H.compare("CPU node %s == compiler numpy" % text, cpu, direct, 0.0, quiet=True)
        H.compare("GPU node %s vs CPU node" % text, gpu, cpu, gpu_atol, rtol=gpu_atol, quiet=True)


@H.guard("time")
def test_time():
    if not H.FEATURES.get("F2"):
        H.note("F2 not available: t / frame checks skipped")
        return
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    for dev in ("CPU", "GPU"):
        for frame in (1, 25):
            img = render(dev, "vec(t, frame * 0.01, 0.0)", frame=frame)
            H.check(abs(float(img[0, 0, 0]) - frame / fps) < 1e-5 and
                    abs(float(img[0, 0, 1]) - frame * 0.01) < 1e-6,
                    "%s frame %d: t = %.4f, frame" % (dev, frame, frame / fps))


@H.guard("behaviour")
def test_behaviour():
    a = H.test_image(*SIZE, seed=5, alpha=True)
    for dev in ("CPU", "GPU"):
        img = render(dev, "a", images={"A": a})
        H.compare("%s: 'a' reproduces the input (premultiplied alpha preserved)" % dev, img, a, 1e-6)
        img = render(dev, "0.25")
        H.check(np.allclose(img[..., :3], 0.25) and np.all(img[..., 3] == 1.0),
                "%s: scalar result is grey with alpha 1" % dev)
        img = render(dev, "vec(0.1, 0.2, 0.3)")
        H.check(np.allclose(img[0, 0], [0.1, 0.2, 0.3, 1.0], atol=1e-6), "%s: vec3 result has alpha 1" % dev)
        img = render(dev, "vec(0.1, 0.2, 0.3, 0.4)")
        H.check(np.allclose(img[0, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-6), "%s: vec4 result keeps alpha" % dev)
        # Unlinked colour inputs use their socket values.
        img = render(dev, "a + b * 2.0 + c - d", inputs={"A": (0.1, 0.2, 0.3, 1.0), "B": (0.01, 0.02, 0.03, 0.0),
                                                         "C": (1.0, 1.0, 1.0, 0.5), "D": (0.5, 0.5, 0.5, 0.25)})
        H.check(np.allclose(img[0, 0], [0.62, 0.74, 0.86, 1.25], atol=1e-5),
                "%s: unlinked inputs use their values (%s)" % (dev, img[0, 0]))
        # y = 0 is the bottom row.
        img = render(dev, "y / h")
        H.check(img[0, 0, 0] < img[-1, 0, 0], "%s: row 0 is the bottom (y grows with the row)" % dev)
        # Coordinates are pixel centres.
        img = render(dev, "x", size=(8, 4))
        H.check(np.allclose(img[0, :, 0], np.arange(8) + 0.5), "%s: x is texel + 0.5" % dev)


@H.guard("robustness")
def test_robustness():
    for dev in ("CPU", "GPU"):
        for size in ((4, 4), (5, 7), (4, 64), (80, 4), (33, 17)):
            img = render(dev, "vec(u, v, sin(x * 0.3) * 0.5 + 0.5)", size=size)
            H.check(img.shape == (size[1], size[0], 4) and np.isfinite(img).all(),
                    "%s %dx%d: shape / finite" % (dev, size[0], size[1]))
        # An invalid expression must not crash the render (the node reports it and outputs black).
        for text in ("__import__('os')", "1 +", "", "a.q"):
            img = render(dev, text, allow_errors=True)
            H.check(len(H.LAST_ERRORS) == 1 and "ExprError" in H.LAST_ERRORS[0],
                    "%s: invalid expression %r is reported as an error" % (dev, text))
            H.check(np.isfinite(img).all() and img.shape == (SIZE[1], SIZE[0], 4),
                    "%s: invalid expression %r renders without crashing" % (dev, text))
        # Division by zero / out-of-domain arguments follow the documented safe definitions.
        img = render(dev, "sqrt(u - 2.0) + log(0.0 * u) * 0.0 + asin(u * 5.0) * 0.0")
        H.check(np.isfinite(img).all(), "%s: sqrt / log / asin of out-of-range values stay finite" % dev)
        # Larger image through the whole pipeline.
        img = render(dev, "noise(u * 10, v * 10) * mix(a.r, 1.0, u)", size=(640, 360))
        H.check(np.isfinite(img).all(), "%s 640x360 noise: finite" % dev)


for fn in (test_rejection, test_cache, test_compiler_numpy, test_node, test_time, test_behaviour,
           test_robustness):
    fn()
H.finish()
