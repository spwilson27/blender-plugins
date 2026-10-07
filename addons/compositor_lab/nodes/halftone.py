# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Halftone: print-style screens (CMYK dots, mono dots, lines, cross-hatch) on a paper colour.

Each screen is a rotated square lattice of ``Cell Size`` pixels. For a pixel p the screen
coordinate is ``q = R(-angle) p / cell`` and ``f = q - floor(q) - 0.5`` is the position inside
its cell (-0.5 .. 0.5). A spot function maps ``f`` to a threshold ``t`` that is uniformly
distributed over [0, 1] across the cell (it is the cumulative area of the dot, so a dot of tone
``d`` covers exactly the fraction ``d`` of the cell, even where round dots overlap):

* round: ``t = area of the circle of radius |f|`` clipped to the cell (``pi r^2`` up to 0.5, then
  minus the four circular segments beyond the cell edges)
* square: ``t = (2 max(|fx|, |fy|))^2``; diamond: ``|fx| + |fy|`` likewise
* lines: ``t = 2 |fy|`` (the line is centred in its cell)

A pixel is inked by ``clamp((d - t) / w + 0.5, 0, 1)`` with ``w = softness * dt/dr / cell``, the
tone change over one pixel (anti-aliased edges; Softness 0 is a hard edge). ``d`` is the tone
sampled from the input *at the cell centre* (dots) or at the pixel's projection onto the line
centre (lines), so every dot is clean and lines vary smoothly in thickness.

Tone: ``d = 1 - luminance`` of the straight linear colour (mono, lines). Cross-hatch uses four
line layers at angle, +90, +45, +135 degrees which fade in at darkness 0, 0.25, 0.5, 0.75 and
each cover at most 60 % of their cell (union ~97 % when fully dark). CMYK separates the colour
with ``k = 1 - max(r, g, b)``, ``c = (max - r) / max`` etc. (so ``(1 - c)(1 - k) = r``) and
prints the four screens as subtractive inks on the paper: ``rgb = paper * prod(1 - cov_i (1 -
ink_i))``. Since all maths is in linear light, a flat field of darkness d has mean mono output
``paper * (1 - d) + ink * d``.

Alpha is kept (output premultiplied by the input alpha); Fac mixes with the input.
"""

import math

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty

from ..lib import gpu as lab_gpu, np_color
from ..lib.node import In, LabNode, Out

MENU = "Filter"

F32 = np.float32
PI = F32(math.pi)
HATCH_FILL = 0.6

_MODE_ITEMS = [
    ('CMYK', "CMYK Dots", "Four rotated screens with subtractive inks"),
    ('MONO', "Mono Dots", "One screen, one ink"),
    ('LINES', "Lines", "Parallel lines whose thickness follows the tone"),
    ('CROSSHATCH', "Cross-hatch", "Four line layers that fade in with darkness"),
]
_SHAPE_ITEMS = [
    ('ROUND', "Round", ""),
    ('SQUARE', "Square", ""),
    ('DIAMOND', "Diamond", ""),
]
_MODE = {m[0]: i for i, m in enumerate(_MODE_ITEMS)}
_SHAPE = {m[0]: i for i, m in enumerate(_SHAPE_ITEMS)}

_BODY = """
    vec4 c0 = in_Image(texel);
    vec2 p = vec2(texel) + vec2(0.5);
    vec3 acc = h_paper;
    float covu = 1.0;
    for (int k = 0; k < 4; k++) {
      if (k >= h_nl) {
        break;
      }
      vec2 cs = (k == 0) ? h_cs0 : ((k == 1) ? h_cs1 : ((k == 2) ? h_cs2 : h_cs3));
      vec3 ink = (k == 0) ? h_ink0 : ((k == 1) ? h_ink1 : ((k == 2) ? h_ink2 : h_ink3));
      vec2 q = vec2(p.x * cs.x + p.y * cs.y, p.y * cs.x - p.x * cs.y) * h_inv;
      vec2 cl = floor(q);
      vec2 f = (q - cl) - vec2(0.5);
      float t;
      float dt;
      vec2 sp;
      if (h_line != 0) {
        sp = vec2(p.x + (f.y * h_cell) * cs.y, p.y - (f.y * h_cell) * cs.x);
        t = 2.0 * abs(f.y);
        dt = 2.0;
      }
      else {
        vec2 cc = cl + vec2(0.5);
        sp = vec2(cc.x * cs.x - cc.y * cs.y, cc.x * cs.y + cc.y * cs.x) * h_cell;
        if (h_shape == 0) {
          float r2 = dot(f, f);
          float r = sqrt(r2);
          float ac = acos(min(0.5 / max(r, 1e-6), 1.0));
          if (r <= 0.5) {
            t = 3.14159265 * r2;
            dt = 2.0 * 3.14159265 * r;
          }
          else {
            t = min(3.14159265 * r2 - 4.0 * (r2 * ac - 0.5 * sqrt(max(r2 - 0.25, 0.0))), 1.0);
            dt = max(2.0 * 3.14159265 * r - 8.0 * r * ac, 0.0);
          }
        }
        else if (h_shape == 1) {
          float r = max(abs(f.x), abs(f.y));
          t = 4.0 * r * r;
          dt = 8.0 * r;
        }
        else {
          float r = abs(f.x) + abs(f.y);
          if (r <= 0.5) {
            t = 2.0 * r * r;
            dt = 4.0 * r;
          }
          else {
            t = 1.0 - 2.0 * (1.0 - r) * (1.0 - r);
            dt = 4.0 * (1.0 - r);
          }
        }
      }
      vec4 sc = in_Image(ivec2(floor(sp)));
      vec3 s = clamp((sc.a > 0.0) ? sc.rgb / sc.a : vec3(0.0), 0.0, 1.0);
      float d;
      if (h_mode == 0) {
        float mx = max(s.r, max(s.g, s.b));
        float sk = (k == 0) ? s.r : ((k == 1) ? s.g : s.b);
        d = (k == 3) ? 1.0 - mx : ((mx > 0.0) ? (mx - sk) / mx : 0.0);
      }
      else if (h_mode == 3) {
        float dd = 1.0 - clamp(lab_luma(s), 0.0, 1.0);
        d = 0.6 * clamp(dd * 4.0 - float(k), 0.0, 1.0);
      }
      else {
        d = 1.0 - clamp(lab_luma(s), 0.0, 1.0);
      }
      float cov = 0.0;
      if (d >= 1.0) {
        cov = 1.0;
      }
      else if (d > 0.0) {
        cov = clamp((d - t) / max(h_soft * dt * h_inv, 1e-6) + 0.5, 0.0, 1.0);
      }
      if (h_mode == 0) {
        acc = acc * (vec3(1.0) - cov * (vec3(1.0) - ink));
      }
      else {
        covu = covu * (1.0 - cov);
      }
    }
    vec3 rgb = (h_mode == 0) ? acc : (h_paper * covu + h_ink0 * (1.0 - covu));
    vec4 res = vec4(rgb * c0.a, c0.a);
    float fac = in_Fac(texel);
    out_Color = c0 * (1.0 - fac) + res * fac;
"""


def _lin(c):
    return tuple(float(v) for v in c[:3])


class CompositorNodeLabHalftone(LabNode, bpy.types.CompositorNode):
    '''Print halftone: CMYK or mono dots, lines or cross-hatch on a paper colour'''
    bl_idname = "CompositorNodeLabHalftone"
    bl_label = "Halftone"

    SOCKETS = [
        In("Fac", "FACTOR", 1.0),
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Color", "COLOR"),
    ]

    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default='CMYK')
    shape: EnumProperty(name="Shape", items=_SHAPE_ITEMS, default='ROUND')
    cell_size: FloatProperty(name="Cell Size", description="Screen period in pixels",
                             default=10.0, min=2.0, max=400.0, soft_max=100.0, subtype='PIXEL')
    softness: FloatProperty(name="Softness", description="Edge anti-aliasing width in pixels "
                            "(0 = hard edges)", default=1.0, min=0.0, max=8.0)
    angle: FloatProperty(name="Angle", description="Screen angle (mono, lines, cross-hatch)",
                         default=math.radians(45.0), soft_min=-math.pi, soft_max=math.pi,
                         subtype='ANGLE')
    angle_c: FloatProperty(name="Cyan", default=math.radians(15.0),
                           soft_min=-math.pi, soft_max=math.pi, subtype='ANGLE')
    angle_m: FloatProperty(name="Magenta", default=math.radians(75.0),
                           soft_min=-math.pi, soft_max=math.pi, subtype='ANGLE')
    angle_y: FloatProperty(name="Yellow", default=0.0,
                           soft_min=-math.pi, soft_max=math.pi, subtype='ANGLE')
    angle_k: FloatProperty(name="Black", default=math.radians(45.0),
                           soft_min=-math.pi, soft_max=math.pi, subtype='ANGLE')
    paper_color: FloatVectorProperty(name="Paper", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=(1.0, 1.0, 1.0))
    ink_color: FloatVectorProperty(name="Ink", description="Ink of the mono / lines / "
                                   "cross-hatch modes", subtype='COLOR', size=3, min=0.0,
                                   max=1.0, default=(0.0, 0.0, 0.0))
    ink_c: FloatVectorProperty(name="Cyan Ink", subtype='COLOR', size=3, min=0.0, max=1.0,
                               default=(0.0, 1.0, 1.0))
    ink_m: FloatVectorProperty(name="Magenta Ink", subtype='COLOR', size=3, min=0.0, max=1.0,
                               default=(1.0, 0.0, 1.0))
    ink_y: FloatVectorProperty(name="Yellow Ink", subtype='COLOR', size=3, min=0.0, max=1.0,
                               default=(1.0, 1.0, 0.0))
    ink_k: FloatVectorProperty(name="Black Ink", subtype='COLOR', size=3, min=0.0, max=1.0,
                               default=(0.0, 0.0, 0.0))

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        if self.mode in ('CMYK', 'MONO'):
            layout.prop(self, "shape", text="")
        layout.prop(self, "cell_size")
        layout.prop(self, "softness")
        layout.prop(self, "paper_color")
        if self.mode == 'CMYK':
            col = layout.column(align=True)
            for ch in "cmyk":
                row = col.row(align=True)
                row.prop(self, "angle_" + ch)
                row.prop(self, "ink_" + ch, text="")
        else:
            layout.prop(self, "angle")
            layout.prop(self, "ink_color")

    # -- shared -------------------------------------------------------------
    def _params(self):
        mode = _MODE[self.mode]
        if mode == 0:
            angles = [self.angle_c, self.angle_m, self.angle_y, self.angle_k]
            inks = [_lin(self.ink_c), _lin(self.ink_m), _lin(self.ink_y), _lin(self.ink_k)]
        else:
            th = float(self.angle)
            angles = [th, th + math.pi / 2, th + math.pi / 4, th + 3 * math.pi / 4]
            inks = [_lin(self.ink_color)] * 4
        nl = 4 if mode in (0, 3) else 1
        cs = [(F32(math.cos(a)), F32(math.sin(a))) for a in angles]
        cell = float(self.cell_size)
        return dict(mode=mode, shape=_SHAPE[self.shape], line=int(mode in (2, 3)), nl=nl,
                    cs=cs, inks=[np.array(i, F32) for i in inks], cell=F32(cell),
                    inv=F32(1.0 / cell), soft=F32(self.softness),
                    paper=np.array(_lin(self.paper_color), F32))

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        c = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                           default=(0.5, 0.5, 0.5, 1.0)))
        fac = self.in_image_array(inputs, "Fac", ctx.shape, 1, default=1.0)
        out[...] = halftone(c, fac, **self._params())

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        p = self._params()
        uniforms = {
            "h_mode": ("int", p["mode"]), "h_shape": ("int", p["shape"]),
            "h_line": ("int", p["line"]), "h_nl": ("int", p["nl"]),
            "h_cell": ("float", float(p["cell"])), "h_inv": ("float", float(p["inv"])),
            "h_soft": ("float", float(p["soft"])),
            "h_paper": ("vec3", [float(v) for v in p["paper"]]),
        }
        for k in range(4):
            uniforms["h_cs%d" % k] = ("vec2", [float(v) for v in p["cs"][k]])
            uniforms["h_ink%d" % k] = ("vec3", [float(v) for v in p["inks"][k]])
        lab_gpu.pointwise(
            _BODY, {"Color": dst},
            inputs={
                "Image": ("color", self.in_texture_or_value(inputs, "Image",
                                                            (0.5, 0.5, 0.5, 1.0))),
                "Fac": ("float", self.in_texture_or_value(inputs, "Fac", 1.0)),
            },
            uniforms=uniforms, libs=("color",))


def spot(shape, fx, fy):
    """(t, dt): threshold (cumulative dot area) and its derivative w.r.t. the radius."""
    if shape == 0:
        r2 = fx * fx + fy * fy
        r = np.sqrt(r2)
        ac = np.arccos(np.minimum(F32(0.5) / np.maximum(r, F32(1e-6)), F32(1.0)))
        inner = r <= F32(0.5)
        t = np.where(inner, PI * r2,
                     np.minimum(PI * r2 - F32(4.0) * (r2 * ac - F32(0.5) * np.sqrt(
                         np.maximum(r2 - F32(0.25), F32(0.0)))), F32(1.0)))
        dt = np.where(inner, F32(2.0) * PI * r,
                      np.maximum(F32(2.0) * PI * r - F32(8.0) * r * ac, F32(0.0)))
    elif shape == 1:
        r = np.maximum(np.abs(fx), np.abs(fy))
        t = F32(4.0) * r * r
        dt = F32(8.0) * r
    else:
        r = np.abs(fx) + np.abs(fy)
        inner = r <= F32(0.5)
        t = np.where(inner, F32(2.0) * r * r, F32(1.0) - F32(2.0) * (F32(1.0) - r) * (F32(1.0) - r))
        dt = np.where(inner, F32(4.0) * r, F32(4.0) * (F32(1.0) - r))
    return t.astype(F32), dt.astype(F32)


def halftone(c, fac, mode, shape, line, nl, cs, inks, cell, inv, soft, paper):
    """Reference / CPU implementation on a premultiplied (H, W, 4) float32 image."""
    h, w = c.shape[:2]
    px = (np.arange(w, dtype=F32) + F32(0.5))[None, :]
    py = (np.arange(h, dtype=F32) + F32(0.5))[:, None]
    px, py = np.broadcast_arrays(px, py)
    a0 = c[..., 3]
    acc = np.broadcast_to(paper, (h, w, 3)).astype(F32)
    covu = np.ones((h, w), F32)
    for k in range(nl):
        cx, sn = cs[k]
        qx = (px * cx + py * sn) * inv
        qy = (py * cx - px * sn) * inv
        clx, cly = np.floor(qx), np.floor(qy)
        fx = (qx - clx) - F32(0.5)
        fy = (qy - cly) - F32(0.5)
        if line:
            spx = px + (fy * cell) * sn
            spy = py - (fy * cell) * cx
            t = F32(2.0) * np.abs(fy)
            dt = np.full_like(t, 2.0)
        else:
            ccx, ccy = clx + F32(0.5), cly + F32(0.5)
            spx = (ccx * cx - ccy * sn) * cell
            spy = (ccx * sn + ccy * cx) * cell
            t, dt = spot(shape, fx, fy)
        ix = np.clip(np.floor(spx), 0, w - 1).astype(np.intp)
        iy = np.clip(np.floor(spy), 0, h - 1).astype(np.intp)
        sc = c[iy, ix]
        sa = sc[..., 3:4]
        s = np.clip(np.where(sa > 0, sc[..., :3] / np.where(sa > 0, sa, F32(1.0)), F32(0.0)),
                    0, 1).astype(F32)
        if mode == 0:
            mx = s.max(axis=-1)
            if k == 3:
                d = F32(1.0) - mx
            else:
                d = np.where(mx > 0, (mx - s[..., k]) / np.where(mx > 0, mx, F32(1.0)), F32(0.0))
        elif mode == 3:
            dd = F32(1.0) - np.clip(np_color.luma(s), 0, 1)
            d = F32(HATCH_FILL) * np.clip(dd * F32(4.0) - F32(k), 0, 1)
        else:
            d = F32(1.0) - np.clip(np_color.luma(s), 0, 1)
        d = d.astype(F32)
        wdt = np.maximum(soft * dt * inv, F32(1e-6))
        cov = np.where(d >= 1, F32(1.0),
                       np.where(d > 0, np.clip((d - t) / wdt + F32(0.5), 0, 1), F32(0.0)))
        cov = cov.astype(F32)
        if mode == 0:
            acc = acc * (F32(1.0) - cov[..., None] * (F32(1.0) - inks[k]))
        else:
            covu = covu * (F32(1.0) - cov)
    if mode == 0:
        rgb = acc
    else:
        rgb = paper * covu[..., None] + inks[0] * (F32(1.0) - covu)[..., None]
    res = np.concatenate([rgb * a0[..., None], a0[..., None]], axis=-1)
    fac = np.asarray(fac, F32)
    return (c * (F32(1.0) - fac) + res * fac).astype(F32)


NODE_CLASSES = [CompositorNodeLabHalftone]
