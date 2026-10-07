# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Displace / Glass: warp an image by a displacement map, with edge modes, bilinear or bicubic
sampling and optional chromatic dispersion.

Every output pixel ``p`` (pixel centre ``(x + 0.5, y + 0.5)``, y up) samples the image at
``p + offset * s_c``:

* **Offset** mode: ``offset = (Map.rg - 0.5) * 2 * Strength`` (pixels), i.e. the map's red / green
  channels centred on 0.5 move the sample point right / up by up to ``Strength`` pixels.
* **Glass** mode (refraction-like): ``offset = -Strength * gain * grad(luma(Map))`` with the Sobel
  gradient of the map's Rec. 709 luma (per pixel, unit-less) and ``gain = max(width, height) /
  100``, so the effect is resolution independent: a slope of 1 luma per 1 % of the image gives an
  offset of ``Strength`` pixels. Bright bumps push the image away from their slope (a lens).

``Strength`` may be an image (per-pixel strength, red channel). ``Dispersion`` d scales the
offset per channel: ``s_c = 1 + d (c - 1)`` for ``c = 0, 1, 2`` (red ``1 - d``, green 1, blue
``1 + d``); alpha follows green. With Dispersion 0 all channels use one sample.

Zero strength is an exact identity; a constant offset map is a pure translation (integer
offsets reproduce the image shifted exactly). Edge modes: clamp, repeat, mirror (also used for the
map's Sobel gradient). Cubic interpolation is Catmull-Rom and may overshoot at sharp edges.
"""

import bpy
import numpy as np
from bpy.props import EnumProperty

from ..lib import np_sampling
from ..lib.glsl import sampling as gl_sampling
from ..lib.node import In, LabNode, Out

MENU = "Filter"

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

_MODE_ITEMS = [
    ('OFFSET', "Offset", "Move by the map's red / green channels (0.5 = no movement)"),
    ('GLASS', "Glass", "Refract by the gradient of the map's luminance"),
]
_EDGE_ITEMS = [
    ('CLAMP', "Clamp", "Repeat the border pixels"),
    ('REPEAT', "Repeat", "Tile the image"),
    ('MIRROR', "Mirror", "Tile the image mirrored"),
]
_INTERP_ITEMS = [
    ('LINEAR', "Bilinear", ""),
    ('CUBIC', "Bicubic", "Catmull-Rom (sharper, may overshoot)"),
]

_BODY = """
    float st = lab_fetch_Strength(texel).r;
    vec2 off;
    if (d_glass != 0) {
      vec4 gx, gy;
      lab_sobel_Map(texel, d_edge, gx, gy);
      off = (-st * d_gain) * vec2(dot(gx.rgb, vec3(0.2126, 0.7152, 0.0722)),
                                  dot(gy.rgb, vec3(0.2126, 0.7152, 0.0722)));
    }
    else {
      vec4 m = lab_fetch_Map(texel);
      off = (m.rg - vec2(0.5)) * (2.0 * st);
    }
    vec2 base = vec2(texel) + vec2(0.5);
    vec4 sg = (d_cubic != 0) ? lab_bicubic_Image(base + off, d_edge)
                             : lab_bilinear_Image(base + off, d_edge);
    vec4 res = sg;
    if (d_disp != 0.0) {
      vec2 pr = base + off * (1.0 - d_disp);
      vec2 pb = base + off * (1.0 + d_disp);
      vec4 sr = (d_cubic != 0) ? lab_bicubic_Image(pr, d_edge) : lab_bilinear_Image(pr, d_edge);
      vec4 sb = (d_cubic != 0) ? lab_bicubic_Image(pb, d_edge) : lab_bilinear_Image(pb, d_edge);
      res = vec4(sr.r, sg.g, sb.b, sg.a);
    }
    out_Color = res;
"""


class CompositorNodeLabDisplace(LabNode, bpy.types.CompositorNode):
    '''Displace an image by a map (offset or glass refraction) with edge modes and dispersion'''
    bl_idname = "CompositorNodeLabDisplace"
    bl_label = "Displace / Glass"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("Map", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("Strength", "FLOAT", 20.0, min=-100.0, max=100.0),
        In("Dispersion", "FLOAT", 0.0, min=-1.0, max=1.0),
        Out("Image", "COLOR"),
    ]
    PROPS = ["mode", "edge_mode", "interpolation"]

    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default='OFFSET')
    edge_mode: EnumProperty(name="Edges", items=_EDGE_ITEMS, default='CLAMP')
    interpolation: EnumProperty(name="Interpolation", items=_INTERP_ITEMS, default='LINEAR')

    def _disp(self, inputs):
        return float(np.float32(self.in_float(inputs, "Dispersion", 0.0)))

    # -- CPU ---------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        f32 = np.float32
        h, w = ctx.shape
        img = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                             default=(0.5, 0.5, 0.5, 1.0)), f32)
        mp = np.asarray(self.in_image_array(inputs, "Map", ctx.shape, 4,
                                            default=(0.5, 0.5, 0.5, 1.0)), f32)
        st = np.asarray(self.in_image_array(inputs, "Strength", ctx.shape, 1, default=20.0), f32)
        st = st.reshape(h, w)
        mode = self.edge_mode
        cubic = self.interpolation == 'CUBIC'
        if self.mode == 'GLASS':
            gx, gy = np_sampling.sobel(mp[..., :3], mode)
            gain = f32(max(w, h) / 100.0)
            gxl = gx[..., 0] * _LUMA[0] + gx[..., 1] * _LUMA[1] + gx[..., 2] * _LUMA[2]
            gyl = gy[..., 0] * _LUMA[0] + gy[..., 1] * _LUMA[1] + gy[..., 2] * _LUMA[2]
            k = (-st) * gain
            ox, oy = k * gxl, k * gyl
        else:
            k = f32(2.0) * st
            ox = (mp[..., 0] - f32(0.5)) * k
            oy = (mp[..., 1] - f32(0.5)) * k
        bx = (np.arange(w, dtype=f32) + f32(0.5))[None, :]
        by = (np.arange(h, dtype=f32) + f32(0.5))[:, None]
        disp = self._disp(inputs)
        res = np_sampling.sample(img, bx + ox, by + oy, mode, cubic)
        if disp != 0.0:
            sr = np_sampling.sample(img, bx + ox * f32(1.0 - disp), by + oy * f32(1.0 - disp),
                                    mode, cubic)
            sb = np_sampling.sample(img, bx + ox * f32(1.0 + disp), by + oy * f32(1.0 + disp),
                                    mode, cubic)
            res = res.copy()
            res[..., 0] = sr[..., 0]
            res[..., 2] = sb[..., 2]
        out[...] = res

    # -- GPU ---------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        w, h = int(dst.width), int(dst.height)
        gl_sampling.kernel(
            _BODY, {"Color": dst},
            {"Image": self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0)),
             "Map": self.in_texture_or_value(inputs, "Map", (0.5, 0.5, 0.5, 1.0)),
             "Strength": self.in_texture_or_value(inputs, "Strength", 20.0)},
            uniforms={
                "d_glass": ("int", int(self.mode == 'GLASS')),
                "d_edge": ("int", np_sampling.edge_mode_index(self.edge_mode)),
                "d_cubic": ("int", int(self.interpolation == 'CUBIC')),
                "d_gain": ("float", float(np.float32(max(w, h) / 100.0))),
                "d_disp": ("float", self._disp(inputs)),
            })


NODE_CLASSES = [CompositorNodeLabDisplace]
