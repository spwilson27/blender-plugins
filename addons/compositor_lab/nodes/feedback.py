# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Feedback / Trails: each frame mixes the input with the previous output, transformed and faded.

    out_n = mix(input_n, T(out_{n-1}) * Decay, Amount)          (Blend: Mix)
    out_n = blend(input_n, T(out_{n-1}) * Decay, Amount)        (any Blend Modes+ mode)

``T`` samples the previous output at the inverse of: zoom about the image centre, rotation (degrees,
counter-clockwise), offset (pixels, y up), then optionally rotates the hue (a rotation of RGB about
the grey axis) by ``Hue Shift`` degrees. Zoom 1.01 with a small rotation gives the video-feedback
tunnel; Add / Screen blend modes give light trails. Mix is a premultiplied lerp of all four
channels, so with a static input and no transform the output converges geometrically:
``out_n = s + (a d)^n (in - s)`` with ``s = in (1 - a) / (1 - a d)`` (Amount a, Decay d).

State: the previous output (RGBA32F). The first frame (and any frame at or before the scene start
frame) outputs the input, optionally after ``Pre-roll`` iterations on that frame's input (the
stateless fallback: a still render shows the converged trails). See ``lib/state.py`` for what
happens on repeats, scrubbing and jumps.
"""

import math

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty

from ..lib import gpu as lab_gpu, np_blend, np_sampling
from ..lib.glsl.blend import MODES
from ..lib.node import In, Out, LabNode, StatefulNode

MENU = "Simulate"

F32 = np.float32

_BLEND_ITEMS = [('MIX', "Mix", "Linear mix of the input and the faded previous output")] + [
    (m, m.replace("_", " ").title(), "") for m in MODES]
_BLEND_INDEX = {m: i for i, m in enumerate(MODES)}
_EDGE_ITEMS = [
    ('CLAMP', "Clamp", "Repeat the border pixels"),
    ('REPEAT', "Repeat", "Tile the image"),
    ('MIRROR', "Mirror", "Tile the image mirrored"),
]

_BODY = """
    vec4 inp = lab_fetch_Image(texel);
    vec4 prev;
    if (d_xform != 0) {
      vec2 d = vec2(texel) + vec2(0.5) - d_pivot;
      vec2 s = vec2(d_ct * d.x + d_st * d.y, d_ct * d.y - d_st * d.x) * d_iz + d_centre;
      prev = lab_bilinear_Prev(s, d_edge);
    }
    else {
      prev = lab_fetch_Prev(texel);
    }
    if (d_hue != 0) {
      prev = vec4(dot(d_r0, prev.rgb), dot(d_r1, prev.rgb), dot(d_r2, prev.rgb), prev.a);
    }
    prev = prev * d_decay;
    vec4 res = (d_mode < 0) ? inp + (prev - inp) * d_amount
                            : lab_blend_pm(d_mode, inp, prev, d_amount);
    out_State = res;
"""


def hue_matrix(degrees):
    """3x3 float32 rotation of RGB about the grey axis."""
    a = math.radians(float(degrees))
    c, s = F32(math.cos(a)), F32(math.sin(a))
    k = F32(1.0 / math.sqrt(3.0))
    third = F32(1.0 / 3.0)
    ident = np.eye(3, dtype=F32)
    cross = k * np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=F32)
    return (c * ident + s * cross + (F32(1.0) - c) * third * np.ones((3, 3), F32)).astype(F32)


class CompositorNodeLabFeedback(StatefulNode, LabNode, bpy.types.CompositorNode):
    '''Feedback / trails: mix the input with the previous output, zoomed, rotated, shifted and faded'''
    bl_idname = "CompositorNodeLabFeedback"
    bl_label = "Feedback / Trails"

    SOCKETS = [
        In("Image", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("Amount", "FACTOR", 0.9, clamp=True),
        In("Decay", "FLOAT", 0.97, min=0.0, max=1.0),
        In("Zoom", "FLOAT", 1.0, min=0.1, max=4.0, clamp=(1e-3, None)),
        In("Rotation", "FLOAT", 0.0, min=-180.0, max=180.0),
        In("Offset X", "FLOAT", 0.0, min=-500.0, max=500.0),
        In("Offset Y", "FLOAT", 0.0, min=-500.0, max=500.0),
        In("Hue Shift", "FLOAT", 0.0, min=-180.0, max=180.0),
        Out("Image", "COLOR"),
    ]
    PROPS = ["blend_type", "edge_mode", "preroll"]

    blend_type: EnumProperty(name="Blend", items=_BLEND_ITEMS, default='MIX')
    edge_mode: EnumProperty(name="Edges", items=_EDGE_ITEMS, default='CLAMP')
    preroll: IntProperty(
        name="Pre-roll", default=0, min=0, soft_max=64, max=256,
        description="Iterations run on the input when the state is (re)started, so a still "
                    "render shows the trails (0: the first frame is the input)")

    def draw_buttons(self, context, layout):
        self.draw_props(layout)
        self.draw_state_buttons(layout)

    # -- parameters ----------------------------------------------------------
    def _params(self, inputs, ctx):
        w, h = ctx.size
        zoom = max(self.in_float(inputs, "Zoom", 1.0), 1e-3)
        rot = self.in_float(inputs, "Rotation", 0.0)
        ox = self.in_float(inputs, "Offset X", 0.0)
        oy = self.in_float(inputs, "Offset Y", 0.0)
        hue = self.in_float(inputs, "Hue Shift", 0.0)
        ang = math.radians(rot)
        p = {
            "amount": F32(min(max(self.in_float(inputs, "Amount", 0.9), 0.0), 1.0)),
            "decay": F32(self.in_float(inputs, "Decay", 0.97)),
            "xform": zoom != 1.0 or rot != 0.0 or ox != 0.0 or oy != 0.0,
            "iz": F32(1.0 / zoom), "ct": F32(math.cos(ang)), "st": F32(math.sin(ang)),
            "centre": (F32(w * 0.5), F32(h * 0.5)),
            "pivot": (F32(w * 0.5 + ox), F32(h * 0.5 + oy)),
            "hue": hue != 0.0,
            "m": hue_matrix(hue),
            "edge": self.edge_mode,
            "mode": self.blend_type,
        }
        return p

    # -- CPU -----------------------------------------------------------------
    def _step_cpu(self, prev, inp, p, ctx):
        h, w = ctx.shape
        if p["xform"]:
            dx = (np.arange(w, dtype=F32) + F32(0.5))[None, :] - p["pivot"][0]
            dy = (np.arange(h, dtype=F32) + F32(0.5))[:, None] - p["pivot"][1]
            sx = (p["ct"] * dx + p["st"] * dy) * p["iz"] + p["centre"][0]
            sy = (p["ct"] * dy - p["st"] * dx) * p["iz"] + p["centre"][1]
            prev = np_sampling.sample_bilinear(prev, sx, sy, p["edge"])
        if p["hue"]:
            m = p["m"]
            rgb = np.stack([prev[..., 0] * m[i, 0] + prev[..., 1] * m[i, 1]
                            + prev[..., 2] * m[i, 2] for i in range(3)], axis=-1)
            prev = np.concatenate([rgb, prev[..., 3:4]], axis=-1)
        prev = (prev * p["decay"]).astype(F32)
        if p["mode"] == 'MIX':
            res = inp + (prev - inp) * p["amount"]
        else:
            res = np_blend.blend_premul(p["mode"], inp, prev, p["amount"])
        return np.ascontiguousarray(res, dtype=F32)

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        p = self._params(inputs, ctx)
        inp = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                             default=(0.0, 0.0, 0.0, 1.0)), F32)

        def init():
            s = np.array(inp, dtype=F32, order='C')
            for _ in range(self.preroll):
                s = self._step_cpu(s, inp, p, ctx)
            return s

        plan = self.advance(ctx)
        out[...] = plan.run(init, lambda s: self._step_cpu(s, inp, p, ctx))

    # -- GPU -----------------------------------------------------------------
    def _new_state(self, ctx):
        import gpu
        return gpu.types.GPUTexture(ctx.size, format='RGBA32F')

    def _step_gpu(self, prev, image, p, ctx):
        dst = self._new_state(ctx)
        m = p["m"]
        lab_gpu.kernel(
            _BODY, {"State": dst},
            {"Image": ("color", image), "Prev": ("color", prev)},
            uniforms={
                "d_xform": ("int", int(p["xform"])),
                "d_hue": ("int", int(p["hue"])),
                "d_edge": ("int", np_sampling.edge_mode_index(p["edge"])),
                "d_mode": ("int", -1 if p["mode"] == 'MIX' else _BLEND_INDEX[p["mode"]]),
                "d_amount": ("float", float(p["amount"])),
                "d_decay": ("float", float(p["decay"])),
                "d_iz": ("float", float(p["iz"])),
                "d_ct": ("float", float(p["ct"])),
                "d_st": ("float", float(p["st"])),
                "d_centre": ("vec2", tuple(float(v) for v in p["centre"])),
                "d_pivot": ("vec2", tuple(float(v) for v in p["pivot"])),
                "d_r0": ("vec3", tuple(float(v) for v in m[0])),
                "d_r1": ("vec3", tuple(float(v) for v in m[1])),
                "d_r2": ("vec3", tuple(float(v) for v in m[2])),
            },
            libs=("blend",), sampling=True)
        return dst

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        p = self._params(inputs, ctx)
        image = self.in_texture_or_value(inputs, "Image", (0.0, 0.0, 0.0, 1.0))

        def init():
            s = self._new_state(ctx)
            lab_gpu.pointwise("    out_State = in_Image(texel);\n", {"State": s},
                              inputs={"Image": ("color", image)})
            for _ in range(self.preroll):
                s = self._step_gpu(s, image, p, ctx)
            return s

        plan = self.advance(ctx)
        final = plan.run(init, lambda s: self._step_gpu(s, image, p, ctx))
        lab_gpu.pointwise("    out_Image = in_Src(texel);\n", {"Image": dst},
                          inputs={"Src": ("color", final)})


NODE_CLASSES = [CompositorNodeLabFeedback]
