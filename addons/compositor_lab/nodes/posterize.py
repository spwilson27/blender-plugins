# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Posterize: quantise each colour channel (or OKLab lightness) to N levels, with optional
gamma and dithering.

The node works on straight colour (un-premultiplied; alpha is kept). Values are clamped to
[0, 1] before quantising. For a channel with N levels:

    x = clamp(c, 0, 1) ^ (1 / gamma)
    q = floor(x * (N - 1) + t) / (N - 1)        t = dither threshold, 0.5 without dither
    c' = q ^ gamma

so without dither each channel has at most N distinct values. ``t = 0.5 + (threshold - 0.5) *
amount``; thresholds come from ``lib/np_dither.py`` (Bayer 2/4/8, R2 blue-noise-like, seeded white
hash). Dithering is mean preserving: the local average moves by less than one step.

Lightness mode quantises OKLab L (no gamma) with N = Levels and keeps OKLab a, b (clamped to
>= 0 in RGB). The result is mixed with the input by Fac.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, IntVectorProperty

from ..lib import gpu as lab_gpu, np_color, np_dither
from ..lib.node import In, LabNode, Out

MENU = "Filter"

F32 = np.float32
MAX_LEVELS = 65536

_DITHER_ITEMS = [
    ('NONE', "None", "Plain rounding"),
    ('BAYER2', "Bayer 2x2", "Ordered dither, 2x2 matrix"),
    ('BAYER4', "Bayer 4x4", "Ordered dither, 4x4 matrix"),
    ('BAYER8', "Bayer 8x8", "Ordered dither, 8x8 matrix"),
    ('BLUE', "Blue-ish Noise", "Low-discrepancy (R2) pattern with blue-noise-like spectrum"),
    ('RANDOM', "Random", "Per-pixel white noise from the seed"),
]
_DITHER_INDEX = {d[0]: i for i, d in enumerate(_DITHER_ITEMS)}

_BODY = """
    vec4 c = in_Image(texel);
    float a = c.a;
    vec3 s = (a > 0.0) ? c.rgb / a : vec3(0.0);
    float th = 0.5 + (lab_dither_threshold(p_dither, texel, uint(p_seed)) - 0.5) * p_amount;
    vec3 r;
    if (p_light == 0) {
      vec3 x = clamp(s, 0.0, 1.0);
      if (p_gamma != 1.0) {
        x = pow(x, vec3(p_igamma));
      }
      vec3 q = vec3(floor(lab_rnd32(x.r * p_n1.r) + th),
                    floor(lab_rnd32(x.g * p_n1.g) + th),
                    floor(lab_rnd32(x.b * p_n1.b) + th));
      r = q / p_n1;
      if (p_gamma != 1.0) {
        r = pow(r, vec3(p_gamma));
      }
    }
    else {
      vec3 lab = lab_linear_to_oklab(max(s, vec3(0.0)));
      float l = clamp(lab.x, 0.0, 1.0);
      float q = floor(lab_rnd32(l * p_n1.x) + th) / p_n1.x;
      r = max(lab_oklab_to_linear(vec3(q, lab.y, lab.z)), vec3(0.0));
    }
    vec4 res = vec4(r * a, a);
    float fac = in_Fac(texel);
    out_Color = c * (1.0 - fac) + res * fac;
"""


class CompositorNodeLabPosterize(LabNode, bpy.types.CompositorNode):
    '''Reduce colours to a few levels per channel, with gamma, lightness-only mode and dithering'''
    bl_idname = "CompositorNodeLabPosterize"
    bl_label = "Posterize+"

    SOCKETS = [
        In("Fac", "FACTOR", 1.0),
        In("Image", "COLOR", (0.8, 0.8, 0.8, 1.0)),
        In("Levels", "INT", 4),
        Out("Color", "COLOR"),
    ]
    PROPS = ["per_channel", "channel_levels", "lightness_only", "gamma", None, "dither",
             "dither_amount", "seed"]

    per_channel: BoolProperty(name="Per Channel", description="Use separate level counts for "
                              "R, G and B instead of the Levels input", default=False)
    channel_levels: IntVectorProperty(name="Levels", size=3, default=(4, 4, 4), min=2,
                                      max=MAX_LEVELS)
    lightness_only: BoolProperty(name="Lightness Only", description="Quantise OKLab lightness "
                                 "and keep the colour (gamma is ignored)", default=False)
    gamma: FloatProperty(name="Gamma", description="Quantise in a gamma-encoded space "
                         "(c ^ 1/gamma); 1 quantises scene-linear values", default=2.2,
                         min=0.1, max=5.0)
    dither: EnumProperty(name="Dither", items=_DITHER_ITEMS, default='NONE')
    dither_amount: FloatProperty(name="Amount", description="Dither strength", default=1.0,
                                 min=0.0, max=1.0, subtype='FACTOR')
    seed: IntProperty(name="Seed", description="Seed of the Random and Blue-ish dither",
                      default=0, min=0, max=2**31 - 1)

    # -- shared -------------------------------------------------------------
    def _params(self, inputs):
        base = min(max(self.in_int(inputs, "Levels", 4), 2), MAX_LEVELS)
        if self.per_channel and not self.lightness_only:
            lv = [min(max(int(v), 2), MAX_LEVELS) for v in self.channel_levels]
        else:
            lv = [base] * 3
        gamma = 1.0 if self.lightness_only else float(self.gamma)
        return dict(
            n1=[float(v - 1) for v in lv],
            gamma=gamma, igamma=1.0 / gamma,
            light=int(self.lightness_only),
            dither=_DITHER_INDEX[self.dither],
            amount=float(self.dither_amount),
            seed=int(self.seed))

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        c = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                           default=(0.8, 0.8, 0.8, 1.0)))
        fac = self.in_image_array(inputs, "Fac", ctx.shape, 1, default=1.0)
        out[...] = posterize(c, fac, **self._params(inputs))

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        p = self._params(inputs)
        lab_gpu.pointwise(
            _BODY, {"Color": dst},
            inputs={
                "Image": ("color", self.in_texture_or_value(inputs, "Image",
                                                            (0.8, 0.8, 0.8, 1.0))),
                "Fac": ("float", self.in_texture_or_value(inputs, "Fac", 1.0)),
            },
            uniforms={
                "p_n1": ("vec3", p["n1"]), "p_gamma": ("float", p["gamma"]),
                "p_igamma": ("float", p["igamma"]), "p_light": ("int", p["light"]),
                "p_dither": ("int", p["dither"]), "p_amount": ("float", p["amount"]),
                "p_seed": ("int", p["seed"]),
            },
            libs=("color", "exact", "dither"))


def posterize(c, fac, n1, gamma, igamma, light, dither, amount, seed):
    """Reference / CPU implementation on a premultiplied (H, W, 4) float32 image."""
    h, w = c.shape[:2]
    a = c[..., 3:4]
    pos = a > 0
    s = np.where(pos, c[..., :3] / np.where(pos, a, F32(1.0)), F32(0.0)).astype(F32)
    t = np_dither.threshold(dither, (h, w), seed)
    th = (F32(0.5) + (t - F32(0.5)) * F32(amount))[..., None].astype(F32)
    n1 = np.asarray(n1, F32)
    if not light:
        x = np.clip(s, 0, 1)
        if gamma != 1.0:
            x = np.power(x, F32(igamma))
        q = np.floor(x * n1 + th)
        r = q / n1
        if gamma != 1.0:
            r = np.power(r, F32(gamma))
    else:
        lab = np_color.linear_to_oklab(np.maximum(s, F32(0.0)))
        l = np.clip(lab[..., :1], 0, 1)
        q = np.floor(l * n1[0] + th) / n1[0]
        r = np.maximum(np_color.oklab_to_linear(np.concatenate([q, lab[..., 1:]], axis=-1)),
                       F32(0.0))
    res = np.concatenate([r * a, a], axis=-1).astype(F32)
    fac = np.asarray(fac, F32)
    return (c * (F32(1.0) - fac) + res * fac).astype(F32)


NODE_CLASSES = [CompositorNodeLabPosterize]
