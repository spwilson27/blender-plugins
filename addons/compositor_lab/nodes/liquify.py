# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Liquify: twirl and pinch / bulge warps around a point.

Geometry: ``Center X`` / ``Center Y`` are normalised image coordinates (0 = left / bottom,
1 = right / top). With *Aspect Correction* (default) distances are measured in pixels and
``Radius`` is a fraction of the image **height**, so the warp is circular; without it distances
are measured in normalised units (an ellipse on non-square images, ``Radius`` is a fraction of
the width / height).

For a pixel at offset ``d`` from the centre and ``t = |d| / Radius`` the weight is
``f = 1 - smoothstep(1 - Falloff, 1, t)``: 1 near the centre, 0 at ``t >= 1`` (so pixels outside the
radius are untouched, bit exact). ``Falloff`` 1 gives a fully smooth bell, 0 a hard edge.

The output samples the input at the *source* position (so the effect looks like the image
being pushed around):

* **Twirl**: ``d`` rotated by ``Strength * pi * f`` (counter-clockwise for positive Strength).
* **Pinch / Bulge**: ``d * (1 - Strength * f)``; positive Strength magnifies the centre (bulge),
  negative pulls the surroundings in (pinch). Strength is clamped to [-4, 1] (1 would magnify
  the centre infinitely).

Zero strength is an exact identity. Sampling is bilinear (or Catmull-Rom bicubic) with clamped
edges.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty

from ..lib import np_sampling
from ..lib.glsl import sampling as gl_sampling
from ..lib.node import In, LabNode, Out

MENU = "Filter"

_MODE_ITEMS = [
    ('TWIRL', "Twirl", "Rotate the image around the centre"),
    ('PINCH_BULGE', "Pinch / Bulge", "Magnify (positive) or shrink (negative) around the centre"),
]
_INTERP_ITEMS = [
    ('LINEAR', "Bilinear", ""),
    ('CUBIC', "Bicubic", "Catmull-Rom (sharper, may overshoot)"),
]

_BODY = """
    vec2 base = vec2(texel) + vec2(0.5);
    vec2 pos = base;
    vec2 d = (base - l_centre) * l_sc;
    float t = length(d) / l_radius;
    if (t < 1.0) {
      float e0 = 1.0 - l_falloff;
      float u = clamp((t - e0) / max(1.0 - e0, 1.0e-6), 0.0, 1.0);
      float f = 1.0 - u * u * (3.0 - 2.0 * u);
      vec2 src;
      if (l_twirl != 0) {
        float ang = l_strength * f;
        float cs = cos(ang);
        float sn = sin(ang);
        src = vec2(cs * d.x - sn * d.y, sn * d.x + cs * d.y);
      }
      else {
        src = d * (1.0 - l_strength * f);
      }
      pos = l_centre + src / l_sc;
    }
    out_Color = (l_cubic != 0) ? lab_bicubic_Image(pos, 0) : lab_bilinear_Image(pos, 0);
"""


class CompositorNodeLabLiquify(LabNode, bpy.types.CompositorNode):
    '''Twirl and pinch / bulge warps around a point'''
    bl_idname = "CompositorNodeLabLiquify"
    bl_label = "Liquify"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("Center X", "FLOAT", 0.5, min=0.0, max=1.0),
        In("Center Y", "FLOAT", 0.5, min=0.0, max=1.0),
        In("Radius", "FLOAT", 0.4, min=0.0, max=2.0, clamp=(0.0, None)),
        In("Strength", "FLOAT", 2.0, min=-4.0, max=4.0),
        In("Falloff", "FACTOR", 1.0, clamp=True),
        Out("Image", "COLOR"),
    ]
    PROPS = ["mode", "aspect_correct", "interpolation"]

    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default='TWIRL')
    aspect_correct: BoolProperty(name="Aspect Correction", default=True,
                                 description="Measure distances in pixels (circular warp)")
    interpolation: EnumProperty(name="Interpolation", items=_INTERP_ITEMS, default='LINEAR')

    def _params(self, inputs, size):
        """Shared float32 parameters: centre (pixels), scale to warp space, radius, strength."""
        f32 = np.float32
        w, h = size
        cx = self.in_float(inputs, "Center X", 0.5)
        cy = self.in_float(inputs, "Center Y", 0.5)
        radius = max(self.in_float(inputs, "Radius", 0.4), 0.0)
        strength = self.in_float(inputs, "Strength", 0.5)
        falloff = min(max(self.in_float(inputs, "Falloff", 1.0), 0.0), 1.0)
        twirl = self.mode == 'TWIRL'
        if twirl:
            strength = strength * float(np.pi)
        else:
            strength = min(max(strength, -4.0), 1.0)
        if self.aspect_correct:
            sc = (1.0 / h, 1.0 / h)
        else:
            sc = (1.0 / w, 1.0 / h)
        return dict(
            centre=(float(f32(cx * w)), float(f32(cy * h))),
            sc=(float(f32(sc[0])), float(f32(sc[1]))),
            radius=float(f32(radius)), strength=float(f32(strength)),
            falloff=float(f32(falloff)), twirl=twirl,
            cubic=self.interpolation == 'CUBIC')

    # -- CPU ---------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        f32 = np.float32
        h, w = ctx.shape
        img = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                             default=(0.5, 0.5, 0.5, 1.0)), f32)
        p = self._params(inputs, (w, h))
        if p["radius"] <= 0.0 or p["strength"] == 0.0:
            out[...] = img
            return
        bx = np.broadcast_to((np.arange(w, dtype=f32) + f32(0.5))[None, :], (h, w))
        by = np.broadcast_to((np.arange(h, dtype=f32) + f32(0.5))[:, None], (h, w))
        cx, cy = f32(p["centre"][0]), f32(p["centre"][1])
        sx, sy = f32(p["sc"][0]), f32(p["sc"][1])
        dx = (bx - cx) * sx
        dy = (by - cy) * sy
        t = np.sqrt(dx * dx + dy * dy) / f32(p["radius"])
        inside = t < f32(1.0)
        e0 = f32(1.0) - f32(p["falloff"])
        u = np.clip((t - e0) / np.maximum(f32(1.0) - e0, f32(1e-6)), f32(0.0), f32(1.0))
        f = f32(1.0) - u * u * (f32(3.0) - f32(2.0) * u)
        if p["twirl"]:
            ang = f32(p["strength"]) * f
            cs, sn = np.cos(ang), np.sin(ang)
            sxd = cs * dx - sn * dy
            syd = sn * dx + cs * dy
        else:
            k = f32(1.0) - f32(p["strength"]) * f
            sxd, syd = dx * k, dy * k
        px = np.where(inside, cx + sxd / sx, bx)
        py = np.where(inside, cy + syd / sy, by)
        out[...] = np_sampling.sample(img, px.astype(f32), py.astype(f32), "CLAMP", p["cubic"])

    # -- GPU ---------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        p = self._params(inputs, (int(dst.width), int(dst.height)))
        # An inactive warp (no radius or strength) gets a vanishing radius: every pixel is "outside".
        active = p["radius"] > 0.0 and p["strength"] != 0.0
        radius = p["radius"] if active else 1e-30
        gl_sampling.kernel(
            _BODY, {"Color": dst},
            {"Image": self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))},
            uniforms={
                "l_centre": ("vec2", p["centre"]), "l_sc": ("vec2", p["sc"]),
                "l_radius": ("float", radius),
                "l_strength": ("float", p["strength"]),
                "l_falloff": ("float", p["falloff"]),
                "l_twirl": ("int", int(p["twirl"])), "l_cubic": ("int", int(p["cubic"])),
            })


NODE_CLASSES = [CompositorNodeLabLiquify]
