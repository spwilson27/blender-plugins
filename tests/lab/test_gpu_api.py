# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""The public GPU kernel API (lib/gpu.py): shader errors surface, ``functions=`` helpers,
accessor order, single-channel accessors and the sampling wrapper. Run in Blender (-b)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import harness as H  # noqa: E402

H.setup(detect=False)

from compositor_lab.lib import errors as lab_errors  # noqa: E402
from compositor_lab.lib import glsl, gpu as lab_gpu  # noqa: E402
from compositor_lab.lib.glsl import sampling as gl_sampling  # noqa: E402
from compositor_lab.lib.node import In, LabNode, Out  # noqa: E402

MODE = {"name": "bad"}


class CompositorNodeLabTestKernelApi(LabNode, bpy.types.CompositorNode):
    bl_idname = "CompositorNodeLabTestKernelApi"
    bl_label = "Kernel API"
    SOCKETS = [In("Image", "COLOR", (0.25, 0.5, 0.75, 1.0)), Out("Color", "COLOR")]

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        a = np.asarray(self.in_image_array(inputs, "Image", ctx.shape))
        mode = MODE["name"]
        if mode == "functions":
            # helper(p) = in_Image(p + (1, 0)) * 2 (clamped at the border)
            w = a.shape[1]
            out[...] = a[:, np.minimum(np.arange(w) + 1, w - 1)] * 2.0
        elif mode in ("single_color", "single_vec3", "single_float"):
            r = a[..., 0]
            out[..., 0] = out[..., 1] = out[..., 2] = r
            out[..., 3] = 0.0 if mode == "single_float" else 1.0
        elif mode == "sampling":
            out[...] = a * 0.5
        else:
            out[...] = 0.0

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        src = self.in_texture_or_value(inputs, "Image", (0.25, 0.5, 0.75, 1.0))
        w, h = int(dst.width), int(dst.height)
        mode = MODE["name"]
        if mode == "bad":
            lab_gpu.pointwise("    out_Color = this_is_not_glsl(texel);\n", {"Color": dst})
        elif mode == "functions":
            lab_gpu.pointwise(
                "    out_Color = helper(texel);\n", {"Color": dst},
                inputs={"Image": ("color", src)},
                functions="vec4 helper(ivec2 p)\n{\n  return in_Image(p + ivec2(1, 0)) * 2.0;\n}\n")
        elif mode.startswith("single_"):
            r32 = lab_gpu.scratch(w, h, "api_r32", "R32F")
            lab_gpu.pointwise("    out_R = vec4(in_Image(texel).r);\n", {"R": r32},
                              inputs={"Image": ("color", src)})
            kind = {"single_color": "color", "single_vec3": "vec3", "single_float": "float"}[mode]
            body = {"color": "    out_Color = in_R(texel);\n",
                    "vec3": "    out_Color = vec4(in_R(texel), 1.0);\n",
                    "float": "    out_Color = vec4(vec3(in_R(texel)), 0.0);\n"}[kind]
            lab_gpu.pointwise(body, {"Color": dst}, inputs={"R": (kind, r32)})
        elif mode == "sampling":
            lab_gpu.kernel("    out_Color = lab_fetch_Image(texel) * 0.5;\n", {"Color": dst},
                           {"Image": ("color", src)}, sampling=True)
            tmp = lab_gpu.scratch(w, h, "api_tmp")
            gl_sampling.kernel("    out_Color = lab_fetch_Image(texel);\n", {"Color": tmp},
                               {"Image": src})
            lab_gpu.pointwise("    out_Color = in_T(texel) * 0.5;\n", {"Color": dst},
                              inputs={"T": ("color", tmp)})


bpy.utils.register_class(CompositorNodeLabTestKernelApi)
IDNAME = "CompositorNodeLabTestKernelApi"
IMG = H.test_image(40, 24, seed=5)


def run(mode, device, **kw):
    MODE["name"] = mode
    return H.render_node(IDNAME, device, (40, 24), images={"Image": IMG}, **kw)


@H.guard("shader errors")
def test_shader_errors():
    out = run("bad", "GPU", allow_errors=True)
    H.check(any("failed to compile" in e for e in H.LAST_ERRORS),
            "a shader compile error is recorded (%d error(s))" % len(H.LAST_ERRORS))
    H.check(np.isfinite(out).all(), "default outputs after a failed node")
    # Directly: compile_shader raises LabShaderError, and records it.
    lab_errors.clear()
    info = lab_gpu.create_info((16, 16))
    try:
        lab_gpu.compile_shader(info, "void main() { nonsense; }", "direct")
        raised = False
    except lab_gpu.LabShaderError:
        raised = True
    H.check(raised, "compile_shader raises LabShaderError")
    H.check(len(lab_errors.take()) == 1, "compile_shader records the error once")
    # The harness fails on an unexpected error.
    n_before = len(H.FAILURES)
    run("bad", "GPU")
    H.check(len(H.FAILURES) == n_before + 1, "render fails the test on a node evaluation error")
    del H.FAILURES[n_before:]


@H.guard("functions hook")
def test_functions():
    H.compare("functions= helper uses in_Image: CPU vs GPU", run("functions", "GPU"),
              run("functions", "CPU"), atol=1e-6)
    # Layout: accessors before libs before functions before main.
    src = lab_gpu.build_source("    out_X = vec4(0.0);\n", ("hash",), [("Image", "color", True, False)],
                               ["X"], functions="vec4 my_helper() { return vec4(0.0); }\n")
    order = [src.index(m) for m in ("vec4 in_Image(", "uint lab_pcg(", "my_helper", "void main")]
    H.check(order == sorted(order), "source order: accessors, libs, functions, main")


@H.guard("single channel")
def test_single_channel():
    for mode in ("single_color", "single_vec3", "single_float"):
        H.compare("%s accessor of an R32F texture: CPU vs GPU" % mode, run(mode, "GPU"),
                  run(mode, "CPU"), atol=1e-7)


@H.guard("sampling")
def test_sampling():
    H.compare("sampling kernel / wrapper: CPU vs GPU", run("sampling", "GPU"),
              run("sampling", "CPU"), atol=1e-7)
    H.check(set(glsl.NAMES) >= {"pattern", "field", "dither", "sampling", "reduce", "distance"},
            "glsl.NAMES lists the new modules")
    for name in glsl.NAMES:
        H.check(isinstance(glsl.resolve(name), str), "glsl.resolve(%r)" % name)


for fn in (test_shader_errors, test_functions, test_single_channel, test_sampling):
    fn()
H.finish()
