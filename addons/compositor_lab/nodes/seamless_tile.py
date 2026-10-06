# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Seamless Tile: make an image tile without visible seams (output has the input's size).

Offset Blend: the image is cross-faded with a copy of itself rolled by half its size. The rolled
copy is continuous across the image border (its own seam is in the middle), the original is
continuous in the middle; a smooth mask that is 0 at the border and 1 from ``Blend Width`` (a
fraction of the half size) inwards switches between them. Done per axis:
``C = lerp(roll_x(A), A, mx(x))`` then ``D = lerp(roll_y(C), C, my(y))`` (x then y, fused into one
four-sample formula). The centre region where both masks are 1 is the input unchanged.

Mirror: the image is folded onto itself (``x -> min(x, W - 1 - x)``), so the output is mirror
symmetric and its opposite edges are identical. It keeps the left / bottom half of the input.

``Axes`` limits either mode to one direction.
"""

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty

from ..lib import gpu as lab_gpu
from ..lib.node import In, LabNode, Out

MENU = "Utility"

F32 = np.float32

_BODY = """
    ivec2 sz = res;
    vec2 e = min(vec2(texel) + 0.5, vec2(res) - (vec2(texel) + 0.5)) / vec2(res);
    vec2 t = clamp(e * st_k, vec2(0.0), vec2(1.0));
    vec2 m = t * t * (3.0 - 2.0 * t);
    if (st_ax == 2) {
      m.x = 1.0;
    }
    if (st_ax == 1) {
      m.y = 1.0;
    }
    ivec2 sh = res / 2;
    ivec2 px = (texel + sh) % res;
    vec4 c0 = in_Image(texel);
    vec4 cx = in_Image(ivec2(px.x, texel.y));
    vec4 cy = in_Image(ivec2(texel.x, px.y));
    vec4 cxy = in_Image(px);
    vec4 r0 = cx + (c0 - cx) * m.x;
    vec4 r1 = cxy + (cy - cxy) * m.x;
    out_Image = r1 + (r0 - r1) * m.y;
"""

_BODY_MIRROR = """
    ivec2 q = texel;
    if (st_ax != 2) {
      q.x = min(q.x, res.x - 1 - q.x);
    }
    if (st_ax != 1) {
      q.y = min(q.y, res.y - 1 - q.y);
    }
    out_Image = in_Image(q);
"""

_AXES = [
    ('BOTH', "Both", "Make both directions seamless"),
    ('X', "Horizontal", "Make left / right edges match"),
    ('Y', "Vertical", "Make top / bottom edges match"),
]
_AX_INDEX = {'BOTH': 0, 'X': 1, 'Y': 2}


def _smooth_mask(n, k):
    """1-D mask over n pixels: smoothstep(0..1) of (distance to the nearest border / n) * k."""
    c = np.arange(n, dtype=F32) + F32(0.5)
    e = np.minimum(c, F32(n) - c) / F32(n)
    t = np.clip(e * F32(k), F32(0), F32(1))
    return t * t * (F32(3) - F32(2) * t)


class CompositorNodeLabSeamlessTile(LabNode, bpy.types.CompositorNode):
    '''Make an image tile seamlessly (offset cross-fade or mirror)'''
    bl_idname = "CompositorNodeLabSeamlessTile"
    bl_label = "Seamless Tile"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Image", "COLOR"),
    ]
    PROPS = ["mode", "axes", "blend_width"]

    mode: EnumProperty(name="Mode", default='OFFSET', items=[
        ('OFFSET', "Offset Blend", "Cross-fade with a copy offset by half the image size"),
        ('MIRROR', "Mirror", "Fold the image onto itself (mirror symmetric, loses half the image)"),
    ])
    axes: EnumProperty(name="Axes", items=_AXES, default='BOTH')
    blend_width: FloatProperty(
        name="Blend Width", default=0.5, min=0.01, max=1.0, subtype='FACTOR',
        description="Width of the cross-fade as a fraction of half the image size (Offset Blend)")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        layout.prop(self, "axes", text="")
        if self.mode == 'OFFSET':
            layout.prop(self, "blend_width")

    def _k(self):
        # e = distance to border / size in (0, 0.5]; mask reaches 1 at e = 0.5 * blend_width.
        return float(F32(2.0) / F32(self.blend_width))

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        h, w = ctx.shape
        a = self.in_image_array(inputs, "Image", ctx.shape, 4, default=(0.5, 0.5, 0.5, 1.0))
        ax = _AX_INDEX[self.axes]
        if self.mode == 'MIRROR':
            xs = np.arange(w)
            ys = np.arange(h)
            if ax != 2:
                xs = np.minimum(xs, w - 1 - xs)
            if ax != 1:
                ys = np.minimum(ys, h - 1 - ys)
            out[...] = a[ys[:, None], xs[None, :]]
            return
        k = self._k()
        mx = _smooth_mask(w, k)[None, :, None]
        my = _smooth_mask(h, k)[:, None, None]
        if ax == 2:
            mx = np.ones_like(mx)
        if ax == 1:
            my = np.ones_like(my)
        xs = (np.arange(w) + w // 2) % w
        ys = (np.arange(h) + h // 2) % h
        c0 = np.asarray(a, F32)
        cx = c0[:, xs]
        cy = c0[ys]
        cxy = c0[ys[:, None], xs[None, :]]
        r0 = cx + (c0 - cx) * mx
        r1 = cxy + (cy - cxy) * mx
        out[...] = r1 + (r0 - r1) * my

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        src = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        body = _BODY_MIRROR if self.mode == 'MIRROR' else _BODY
        lab_gpu.pointwise(
            body, {"Image": dst}, inputs={"Image": ("color", src)},
            uniforms={"st_k": ("float", self._k()), "st_ax": ("int", _AX_INDEX[self.axes])})


NODE_CLASSES = [CompositorNodeLabSeamlessTile]
