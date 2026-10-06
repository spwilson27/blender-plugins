# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Blend Modes+: Photoshop / W3C style blend modes for two colour inputs.

Convention: A is the backdrop (bottom layer), B the source drawn on top, Fac mixes A with the
result. Compositor colours are *premultiplied*, so the node unpremultiplies A and B, blends the
straight colours, composites with W3C source-over (``a' = aA + aB - aA aB``,
``rgb' = (1 - aB) A + (1 - aA) B + aA aB blend(cA, cB)`` in premultiplied terms), and mixes
``A + (result - A) * Fac``. Opaque inputs give exactly ``blend(A, B)`` mixed by Fac.

Modes defined on [0, 1] only (Screen, Overlay, both Soft Lights, Hard Light, Vivid Light, Linear
Light, Pin Light, Hard Mix, Color Dodge, Color Burn, Exclusion) clamp operands and result to
[0, 1]. The others pass HDR values through. Hue / Saturation / Color / Luminosity swap OKLCh
components of A and B (clamped to >= 0 when out of gamut); chroma below 1e-5 counts as no hue.
Darker / Lighter Color compare Rec. 709 luminance.
"""

import bpy
from bpy.props import EnumProperty

from ..lib import gpu as lab_gpu, np_blend
from ..lib.glsl.blend import MODES
from ..lib.node import In, LabNode, Out

MENU = "Filter"

_LABELS = {
    "SOFT_LIGHT": "Soft Light (W3C)",
    "SOFT_LIGHT_PEGTOP": "Soft Light (Pegtop)",
    "LINEAR_DODGE": "Linear Dodge (Add)",
}
_ITEMS = [(m, _LABELS.get(m, m.replace("_", " ").title()), "") for m in MODES]
_INDEX = {m: i for i, m in enumerate(MODES)}

_BODY = """
    out_Color = lab_blend_pm(bl_mode, in_A(texel), in_B(texel), in_Fac(texel));
"""


class CompositorNodeLabBlendModes(LabNode, bpy.types.CompositorNode):
    '''Blend two colours with Photoshop-style blend modes (premultiplied-alpha aware)'''
    bl_idname = "CompositorNodeLabBlendModes"
    bl_label = "Blend Modes+"

    SOCKETS = [
        In("Fac", "FACTOR", 1.0),
        In("A", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("B", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Color", "COLOR"),
    ]
    PROPS = ["blend_type"]

    blend_type: EnumProperty(name="Mode", items=_ITEMS, default='MULTIPLY')

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        a = self.in_image_array(inputs, "A", ctx.shape, 4, default=(0.5, 0.5, 0.5, 1.0))
        b = self.in_image_array(inputs, "B", ctx.shape, 4, default=(0.5, 0.5, 0.5, 1.0))
        fac = self.in_image_array(inputs, "Fac", ctx.shape, 1, default=1.0)
        out[...] = np_blend.blend_premul(self.blend_type, a, b, fac)

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        lab_gpu.pointwise(
            _BODY, {"Color": dst},
            inputs={
                "A": ("color", self.in_texture_or_value(inputs, "A", (0.5, 0.5, 0.5, 1.0))),
                "B": ("color", self.in_texture_or_value(inputs, "B", (0.5, 0.5, 0.5, 1.0))),
                "Fac": ("float", self.in_texture_or_value(inputs, "Fac", 1.0)),
            },
            uniforms={"bl_mode": ("int", _INDEX[self.blend_type])},
            libs=("blend",))


NODE_CLASSES = [CompositorNodeLabBlendModes]
