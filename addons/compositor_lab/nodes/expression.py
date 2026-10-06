# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Expression: a per-pixel formula typed into the node.

The text is a single expression in a safe subset of Python syntax, compiled by ``lib/expr.py`` to
both numpy (CPU) and GLSL (GPU); see that module for the full list of variables (``a b c d`` the
four colour inputs, ``x y u v w h`` coordinates and size, ``t frame``), operators, functions
(``sin mix smoothstep noise vec ...``) and swizzles (``a.rgb``). A scalar result becomes grey,
a vec3 gets alpha 1, a vec4 is used as is (colours are premultiplied scene-linear).

Anything outside the whitelist is rejected at compile time with a message (the node reports it as
its info message and outputs black). Compiled expressions are cached by their text.
"""

import bpy
from bpy.props import StringProperty

from ..lib import expr, gpu as lab_gpu
from ..lib.node import In, LabNode, Out

MENU = "Utility"

DEFAULT_EXPRESSION = "vec(0.5 + 0.5 * sin(6.2832 * (u + 0.1 * t)), v, 1.0 - u) * 0.8 + a.rgb * 0.2"


class CompositorNodeLabExpression(LabNode, bpy.types.CompositorNode):
    '''Evaluate a per-pixel expression (safe Python-syntax subset) on the CPU or the GPU'''
    bl_idname = "CompositorNodeLabExpression"
    bl_label = "Expression"

    SOCKETS = [
        In("A", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("B", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("C", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("D", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        Out("Color", "COLOR"),
    ]
    PROPS = [("expression", {"text": ""})]

    expression: StringProperty(
        name="Expression", default=DEFAULT_EXPRESSION,
        description="Per-pixel expression. Variables: a b c d (colours), x y u v w h, t, frame. "
                    "Functions: sin cos sqrt mix clamp smoothstep noise vec ... Swizzles: a.rgb")

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        comp = expr.compile_expr(self.expression)
        colors = {n.lower(): self.in_image_array(inputs, n, ctx.shape, 4,
                                                 default=(0.0, 0.0, 0.0, 1.0))
                  for n in comp.inputs_used}
        out[...] = comp.eval_numpy(ctx.shape, colors, ctx.time, ctx.frame)

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        comp = expr.compile_expr(self.expression)
        lab_gpu.pointwise(
            comp.glsl_body, {"Color": dst},
            inputs={n: ("color", self.in_texture_or_value(inputs, n, (0.0, 0.0, 0.0, 1.0)))
                    for n in comp.inputs_used},
            uniforms={"e_time": ("float", ctx.time), "e_frame": ("float", ctx.frame)},
            libs=("noise",) if comp.uses_noise else ())


NODE_CLASSES = [CompositorNodeLabExpression]
