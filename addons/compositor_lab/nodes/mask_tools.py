# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Mask Tools: threshold, grow / shrink, feather, outline and invert a mask, by exact distance.

Pipeline (each stage optional):

1. **Value**: the input's red channel / alpha / Rec. 709 luminance (a single-channel float input
   is its own value).
2. **Threshold**: ``low <= v <= high`` -> 1, else 0; ``Softness`` > 0 replaces each hard edge by
   a smoothstep of that width centred on the edge. Without it the value is clamped to [0, 1].
3. **Grow / Shrink** (pixels, signed) and **Edge**; both work on the binary mask ``m >= 0.5``:
   * ``None``: the result is binary. Grow by r keeps every pixel within Euclidean distance r of
     an inside pixel (a single pixel becomes an exact disc); shrink by r keeps pixels whose
     distance to the nearest outside pixel is > r (dual of grow, so shrink-then-grow is a
     morphological opening).
   * ``Feather``: ``s`` is the signed distance to the edge (``0.5 - d_in`` inside, ``d_out - 0.5``
     outside, minus the grow amount); the result is ``smoothstep(0.5 - s / width)``: a smooth
     falloff centred on the (grown) edge, ``width`` pixels wide. Width 1 = hard edge.
   * ``Outline``: a band of ``Width`` pixels around the (grown) edge, centred on it, outside or
     inside (``lo < s <= hi``; binary).
4. **Invert**: ``1 - m``.

The image border is not an edge: nothing is eroded from it and nothing grows into it.

CPU: Felzenszwalb-Huttenlocher exact EDT (``lib/distance.py``). GPU: separable exact passes with
a search window sized to the needed distance (``lib/glsl/distance.py``). Squared distances are
integers, so binary results agree exactly; feather / soft threshold agree to float rounding.
"""

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from ..lib import distance, gpu as lab_gpu
from ..lib.node import In, LabNode, Out

MENU = "Utility"

F32 = np.float32
_SRC = {'VALUE': 0, 'ALPHA': 1, 'LUMINANCE': 2}
_EDGE = {'NONE': 0, 'FEATHER': 1, 'OUTLINE': 2}
_POS = {'CENTER': 0, 'OUTSIDE': 1, 'INSIDE': 2}
MAX_RADIUS = 1024

# Kinds of final pass.
K_PASS, K_GROW, K_SHRINK, K_FEATHER, K_OUTLINE = range(5)

# Per-pixel value + threshold. Body of the first GPU pass (output M).
_BODY_VALUE = """
    vec4 c = in_Mask(texel);
    if (mt_scalar != 0) {
      c = vec4(c.r, c.r, c.r, 1.0);
    }
    float v = c.r;
    if (mt_src == 1) {
      v = c.a;
    }
    else if (mt_src == 2) {
      float la = lab_rnd32(0.2126 * c.r);
      float lb = lab_rnd32(0.7152 * c.g);
      float lc = lab_rnd32(0.0722 * c.b);
      v = lab_rnd32(la + lb) + lc;
    }
    float m;
    if (mt_thr != 0) {
      float el;
      float eh;
      if (mt_lo_inv > 0.0) {
        float t = clamp((v - mt_lo0) * mt_lo_inv, 0.0, 1.0);
        el = t * t * (3.0 - 2.0 * t);
      }
      else {
        el = (v >= mt_lo0) ? 1.0 : 0.0;
      }
      if (mt_hi_inv > 0.0) {
        float t = clamp((v - mt_hi0) * mt_hi_inv, 0.0, 1.0);
        eh = t * t * (3.0 - 2.0 * t);
      }
      else {
        eh = (v > mt_hi0) ? 1.0 : 0.0;
      }
      m = el * (1.0 - eh);
    }
    else {
      m = clamp(v, 0.0, 1.0);
    }
    out_M = vec4(m, 0.0, 0.0, 1.0);
"""

# Final pass: inputs M (value), Do (squared distance to the nearest inside pixel), Di (squared
# distance to the nearest outside pixel).
_BODY_FINAL = """
    float m = in_M(texel);
    float res_v = m;
    bool inside = m >= 0.5;
    float d_o = in_Do(texel);
    float d_i = in_Di(texel);
    if (mt_kind == 1) {
      res_v = lab_d_le(d_o, mt_r) ? 1.0 : 0.0;
    }
    else if (mt_kind == 2) {
      res_v = lab_d_gt(d_i, mt_r) ? 1.0 : 0.0;
    }
    else if (mt_kind == 3) {
      float s = inside ? (0.5 - sqrt(d_i)) : (sqrt(d_o) - 0.5);
      s = s - mt_off;
      float t = clamp(0.5 - s * mt_inv_f, 0.0, 1.0);
      res_v = t * t * (3.0 - 2.0 * t);
    }
    else if (mt_kind == 4) {
      bool band;
      if (inside) {
        band = lab_d_ge(d_i, mt_c) && lab_d_lt(d_i, mt_e);
      }
      else {
        band = lab_d_gt(d_o, mt_a) && lab_d_le(d_o, mt_b);
      }
      res_v = band ? 1.0 : 0.0;
    }
    if (mt_invert != 0) {
      res_v = 1.0 - res_v;
    }
    out_Mask = vec4(res_v, 0.0, 0.0, 1.0);
"""


def _thr_params(lo, hi, soft):
    soft = max(float(soft), 0.0)
    if soft > 1e-6:
        return (float(F32(lo - soft / 2)), float(F32(1.0 / soft)),
                float(F32(hi - soft / 2)), float(F32(1.0 / soft)))
    return float(F32(lo)), 0.0, float(F32(hi)), 0.0


def _smoothstep01(t):
    return t * t * (F32(3) - F32(2) * t)


class CompositorNodeLabMaskTools(LabNode, bpy.types.CompositorNode):
    '''Threshold, grow / shrink, feather, outline and invert a mask using exact distances'''
    bl_idname = "CompositorNodeLabMaskTools"
    bl_label = "Mask Tools"

    SOCKETS = [
        In("Mask", "COLOR", (1.0, 1.0, 1.0, 1.0)),
        Out("Mask", "FLOAT"),
    ]
    PROPS = ["source", "use_threshold", "low", "high", "softness", "grow", "edge", "feather",
             "outline_width", "outline_position", "invert"]

    source: EnumProperty(name="Value", default='VALUE', items=[
        ('VALUE', "Value", "First channel (red) of the input, or the float value"),
        ('ALPHA', "Alpha", "Alpha of the input"),
        ('LUMINANCE', "Luminance", "Rec. 709 luminance of the input"),
    ])
    use_threshold: BoolProperty(name="Threshold", default=True,
                                description="Keep values between Low and High")
    low: FloatProperty(name="Low", default=0.5, description="Lower threshold")
    high: FloatProperty(name="High", default=1.0, description="Upper threshold")
    softness: FloatProperty(name="Softness", default=0.0, min=0.0, soft_max=1.0,
                            description="Width of the smooth transition at each threshold edge")
    grow: FloatProperty(name="Grow", default=0.0, min=-MAX_RADIUS, max=MAX_RADIUS, soft_min=-64,
                        soft_max=64, subtype='PIXEL',
                        description="Grow (positive) or shrink (negative) the mask by this "
                                    "distance in pixels (exact Euclidean disc)")
    edge: EnumProperty(name="Edge", default='NONE', items=[
        ('NONE', "None", "Binary result"),
        ('FEATHER', "Feather", "Smooth distance-based falloff at the edge"),
        ('OUTLINE', "Outline", "Band of pixels around the edge"),
    ])
    feather: FloatProperty(name="Feather", default=8.0, min=0.0, max=2.0 * MAX_RADIUS,
                           soft_max=128.0, subtype='PIXEL',
                           description="Width of the falloff in pixels, centred on the edge")
    outline_width: FloatProperty(name="Width", default=4.0, min=0.0, max=MAX_RADIUS,
                                 soft_max=64.0, subtype='PIXEL', description="Outline width")
    outline_position: EnumProperty(name="Position", default='CENTER', items=[
        ('CENTER', "Centered", "Centred on the edge"),
        ('OUTSIDE', "Outside", "Outside the edge"),
        ('INSIDE', "Inside", "Inside the edge"),
    ])
    invert: BoolProperty(name="Invert", default=False)

    def draw_buttons(self, context, layout):
        layout.prop(self, "source", text="")
        layout.prop(self, "use_threshold")
        if self.use_threshold:
            col = layout.column(align=True)
            col.prop(self, "low")
            col.prop(self, "high")
            col.prop(self, "softness")
        layout.prop(self, "grow")
        layout.prop(self, "edge")
        if self.edge == 'FEATHER':
            layout.prop(self, "feather")
        elif self.edge == 'OUTLINE':
            layout.prop(self, "outline_width")
            layout.prop(self, "outline_position", text="")
        layout.prop(self, "invert")

    # -- parameters ---------------------------------------------------------
    def _params(self):
        grow = float(F32(min(max(self.grow, -MAX_RADIUS), MAX_RADIUS)))
        lo0, lo_inv, hi0, hi_inv = _thr_params(self.low, self.high, self.softness)
        p = dict(src=_SRC[self.source], thr=bool(self.use_threshold), lo0=lo0, lo_inv=lo_inv,
                 hi0=hi0, hi_inv=hi_inv, invert=bool(self.invert), kind=K_PASS, radius=0,
                 r=0.0, off=grow, inv_f=0.0, a=0.0, b=0.0, c=0.0, e=0.0)
        edge = _EDGE[self.edge]
        feather = float(F32(self.feather))
        if edge == 1 and feather > 1e-3:
            p["kind"] = K_FEATHER
            p["inv_f"] = float(F32(1.0 / feather))
            p["radius"] = int(math.ceil(feather / 2 + 0.5 + abs(grow))) + 1
        elif edge == 2:
            width = float(F32(max(self.outline_width, 0.0)))
            lo, hi = {0: (-width / 2, width / 2), 1: (0.0, width), 2: (-width, 0.0)}[
                _POS[self.outline_position]]
            p["kind"] = K_OUTLINE
            p["a"] = float(F32(lo + 0.5 + grow))
            p["b"] = float(F32(hi + 0.5 + grow))
            p["c"] = float(F32(0.5 - grow - hi))
            p["e"] = float(F32(0.5 - grow - lo))
            p["radius"] = int(math.ceil(max(abs(p["a"]), abs(p["b"]), abs(p["c"]),
                                            abs(p["e"])))) + 1
        elif grow > 0:
            p["kind"] = K_GROW
            p["r"] = grow
            p["radius"] = int(math.ceil(grow))
        elif grow < 0:
            p["kind"] = K_SHRINK
            p["r"] = -grow
            p["radius"] = int(math.ceil(-grow))
        p["radius"] = min(max(p["radius"], 1), 2 * MAX_RADIUS + 2)
        return p

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Mask")
        if out is None:
            return
        p = self._params()
        c = self.in_image_array(inputs, "Mask", ctx.shape, 4, default=(1.0, 1.0, 1.0, 1.0))
        c = np.asarray(c, F32)
        if p["src"] == 0:
            v = c[..., 0]
        elif p["src"] == 1:
            v = c[..., 3]
        else:
            v = (F32(0.2126) * c[..., 0] + F32(0.7152) * c[..., 1]) + F32(0.0722) * c[..., 2]
        if p["thr"]:
            if p["lo_inv"] > 0:
                t = np.clip((v - F32(p["lo0"])) * F32(p["lo_inv"]), F32(0), F32(1))
                el = _smoothstep01(t)
            else:
                el = (v >= F32(p["lo0"])).astype(F32)
            if p["hi_inv"] > 0:
                t = np.clip((v - F32(p["hi0"])) * F32(p["hi_inv"]), F32(0), F32(1))
                eh = _smoothstep01(t)
            else:
                eh = (v > F32(p["hi0"])).astype(F32)
            m = el * (F32(1) - eh)
        else:
            m = np.clip(v, F32(0), F32(1))
        kind = p["kind"]
        res = m
        if kind != K_PASS:
            inside = m >= F32(0.5)
            cap = distance.cap_for(p["radius"])
            d_o = d_i = None
            if kind in (K_GROW, K_FEATHER, K_OUTLINE):
                d_o = distance.edt_sq(inside, cap)
            if kind in (K_SHRINK, K_FEATHER, K_OUTLINE):
                d_i = distance.edt_sq(~inside, cap)
            if kind == K_GROW:
                res = distance.d_le(d_o, p["r"]).astype(F32)
            elif kind == K_SHRINK:
                res = distance.d_gt(d_i, p["r"]).astype(F32)
            elif kind == K_FEATHER:
                s = np.where(inside, F32(0.5) - np.sqrt(d_i), np.sqrt(d_o) - F32(0.5))
                s = s - F32(p["off"])
                t = np.clip(F32(0.5) - s * F32(p["inv_f"]), F32(0), F32(1))
                res = _smoothstep01(t)
            else:
                band_in = distance.d_ge(d_i, p["c"]) & distance.d_lt(d_i, p["e"])
                band_out = distance.d_gt(d_o, p["a"]) & distance.d_le(d_o, p["b"])
                res = np.where(inside, band_in, band_out).astype(F32)
        if p["invert"]:
            res = F32(1) - res
        out[..., 0] = res

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Mask")
        if dst is None:
            return
        p = self._params()
        w, h = int(dst.width), int(dst.height)
        mask = self.in_texture_or_value(inputs, "Mask", (1.0, 1.0, 1.0, 1.0))
        scalar = lab_gpu.is_texture(mask) and mask.format in ("R16F", "R32F")
        m_tex = distance.scratch(w, h, "mt_m", "R32F")
        lab_gpu.pointwise(
            _BODY_VALUE, {"M": m_tex}, inputs={"Mask": ("color", mask)},
            uniforms={
                "mt_src": ("int", p["src"]), "mt_scalar": ("int", int(scalar)),
                "mt_thr": ("int", int(p["thr"])), "mt_lo0": ("float", p["lo0"]),
                "mt_lo_inv": ("float", p["lo_inv"]), "mt_hi0": ("float", p["hi0"]),
                "mt_hi_inv": ("float", p["hi_inv"]),
            }, libs=("exact",))
        kind = p["kind"]
        do_tex = di_tex = m_tex
        if kind in (K_GROW, K_FEATHER, K_OUTLINE):
            do_tex = distance.scratch(w, h, "mt_do", "R32F")
            distance.gpu_edt_sq(m_tex, do_tex, p["radius"], polarity=1, threshold=0.5)
        if kind in (K_SHRINK, K_FEATHER, K_OUTLINE):
            di_tex = distance.scratch(w, h, "mt_di", "R32F")
            distance.gpu_edt_sq(m_tex, di_tex, p["radius"], polarity=-1, threshold=0.5)
        lab_gpu.pointwise(
            _BODY_FINAL, {"Mask": dst},
            inputs={"M": ("float", m_tex), "Do": ("float", do_tex), "Di": ("float", di_tex)},
            uniforms={
                "mt_kind": ("int", kind), "mt_invert": ("int", int(p["invert"])),
                "mt_r": ("float", p["r"]), "mt_off": ("float", p["off"]),
                "mt_inv_f": ("float", p["inv_f"]), "mt_a": ("float", p["a"]),
                "mt_b": ("float", p["b"]), "mt_c": ("float", p["c"]), "mt_e": ("float", p["e"]),
            }, libs=("distance",))


NODE_CLASSES = [CompositorNodeLabMaskTools]
