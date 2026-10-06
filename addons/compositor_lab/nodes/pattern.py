# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Pattern generator: stripes, checker, dots, hex grid, truchet tiles, moire and rings.

Pixel (x, y) maps to pattern space ``q = ((x + 0.5, y + 0.5) - size / 2) * Scale / max(width,
height)``, which is rotated by ``Rotation`` (degrees, about the image centre) and then shifted by
``Offset``: one unit of pattern space is one period / cell, and there are ``Scale`` of them across
the larger image side. Coverage of colour B comes from a signed distance to the pattern's edges
(``lib/np_pattern`` on the CPU, ``lib/glsl/pattern.py`` on the GPU) turned into an anti-aliased
ramp about one pixel wide (plus ``Softness`` / 2 periods); the colour is ``mix(A, B, coverage)``.

``Duty`` controls the stripe width / dot diameter / hex cell fill / truchet line width (0..1;
for Moire both gratings use it). Truchet tiles flip per cell with a hash of (cell, Seed).
Moire is the product of two line gratings, the second rotated by ``Moire Angle`` degrees.
Rings are centred on the pattern origin (the image centre with Offset 0).
Needs F1 (generator domain). Outputs: Color and Mask (coverage of B).
"""

import bpy
import numpy as np
from bpy.props import EnumProperty

from ..lib import gpu as lab_gpu, np_noise, np_pattern
from ..lib.node import In, LabNode, Out

MENU = "Generate"

_ITEMS = [
    ('STRIPES', "Stripes", "Parallel bands"),
    ('CHECKER', "Checker", "Checkerboard"),
    ('DOTS', "Dots", "Grid of discs"),
    ('HEX', "Hex Grid", "Hexagonal cells"),
    ('TRUCHET_ARC', "Truchet Arcs", "Randomly flipped quarter-circle tiles"),
    ('TRUCHET_DIAG', "Truchet Diagonals", "Randomly flipped diagonal tiles"),
    ('MOIRE', "Moire", "Two overlaid line gratings"),
    ('RINGS', "Rings", "Concentric rings"),
]
_INDEX = {m[0]: i for i, m in enumerate(_ITEMS)}
F32 = np.float32

_BODY = """
    vec2 q = (vec2(texel) + vec2(0.5) - 0.5 * vec2(float(lab_w), float(lab_h))) * p_k;
    vec2 uv = vec2(q.x * p_rc + q.y * p_rs, q.y * p_rc - q.x * p_rs) + p_off;
    float m = lab_pattern(p_mode, uv, p_duty, p_invw, lab_pcg(uint(p_seed)), p_mc, p_ms);
"""
_STORE = {
    "Color": "    out_Color = mix(p_a, p_b, m);\n",
    "Mask": "    out_Mask = vec4(m, m, m, 1.0);\n",
}


def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


class CompositorNodeLabPattern(LabNode, bpy.types.CompositorNode):
    '''Anti-aliased patterns: stripes, checker, dots, hex grid, truchet, moire, rings'''
    bl_idname = "CompositorNodeLabPattern"
    bl_label = "Pattern"

    SOCKETS = [
        In("Scale", "FLOAT", 10.0),
        In("Rotation", "FLOAT", 0.0),
        In("Offset X", "FLOAT", 0.0),
        In("Offset Y", "FLOAT", 0.0),
        In("Duty", "FACTOR", 0.5),
        In("Softness", "FACTOR", 0.0),
        In("Moire Angle", "FLOAT", 4.0),
        In("Seed", "INT", 0),
        In("Color A", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("Color B", "COLOR", (1.0, 1.0, 1.0, 1.0)),
        Out("Color", "COLOR"),
        Out("Mask", "FLOAT"),
    ]
    PROPS = ["pattern"]

    pattern: EnumProperty(name="Pattern", items=_ITEMS, default='STRIPES')

    def _params(self, inputs, ctx):
        w, h = ctx.size
        scale = max(self.in_float(inputs, "Scale", 10.0), 1e-3)
        k = float(F32(scale / max(w, h, 1)))
        rot = np.radians(self.in_float(inputs, "Rotation", 0.0))
        mang = np.radians(self.in_float(inputs, "Moire Angle", 4.0))
        soft = min(max(self.in_float(inputs, "Softness", 0.0), 0.0), 1.0)
        seed = self.in_int(inputs, "Seed", 0)
        return dict(
            mode=self.pattern, mode_i=_INDEX[self.pattern], k=k,
            rc=float(F32(np.cos(rot))), rs=float(F32(np.sin(rot))),
            ox=float(F32(self.in_float(inputs, "Offset X", 0.0))),
            oy=float(F32(self.in_float(inputs, "Offset Y", 0.0))),
            duty=float(F32(min(max(self.in_float(inputs, "Duty", 0.5), 0.0), 1.0))),
            invw=float(F32(1.0 / (k + 0.5 * soft))),
            mc=float(F32(np.cos(mang))), ms=float(F32(np.sin(mang))),
            seed=seed, ss=np_noise.scramble_seed(seed),
            a=self.in_color(inputs, "Color A", (0.0, 0.0, 0.0, 1.0)),
            b=self.in_color(inputs, "Color B", (1.0, 1.0, 1.0, 1.0)),
        )

    def cpu(self, inputs, outputs, ctx):
        color = self.out_array(outputs, "Color")
        mask = self.out_array(outputs, "Mask")
        if color is None and mask is None:
            return
        p = self._params(inputs, ctx)
        w, h = ctx.size
        u, v = np_pattern.to_pattern_space(w, h, p["k"], p["rc"], p["rs"], p["ox"], p["oy"])
        m = np_pattern.pattern_mask(p["mode"], u, v, p["duty"], p["invw"], p["ss"],
                                    p["mc"], p["ms"])
        if mask is not None:
            mask[..., 0] = m
        if color is not None:
            a = np.asarray(p["a"], F32)
            b = np.asarray(p["b"], F32)
            color[...] = a + (b - a) * m[..., None]

    def gpu(self, inputs, outputs, ctx):
        outs = {n: self.out_texture(outputs, n) for n in ("Color", "Mask")}
        outs = {n: t for n, t in outs.items() if t is not None}
        if not outs:
            return
        p = self._params(inputs, ctx)
        uniforms = {
            "p_mode": ("int", p["mode_i"]), "p_k": ("float", p["k"]),
            "p_rc": ("float", p["rc"]), "p_rs": ("float", p["rs"]),
            "p_off": ("vec2", (p["ox"], p["oy"])), "p_duty": ("float", p["duty"]),
            "p_invw": ("float", p["invw"]), "p_mc": ("float", p["mc"]),
            "p_ms": ("float", p["ms"]), "p_seed": ("int", _i32(p["seed"])),
            "p_a": ("vec4", p["a"]), "p_b": ("vec4", p["b"]),
        }
        body = _BODY + "".join(_STORE[n] for n in outs)
        lab_gpu.pointwise(body, outs, uniforms=uniforms, libs=("pattern",))


NODE_CLASSES = [CompositorNodeLabPattern]
