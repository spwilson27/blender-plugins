# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Voronoi / Mosaic generator: cells, F1 distance, F2-F1 borders, and stained-glass mosaic.

Pixel (x, y) maps to cell space ``p = (x + 0.5, y + 0.5) * Scale / max(width, height) + Offset``
(so the cell count does not depend on the resolution). One jittered feature point per unit cell
(3x3 neighbourhood; ``lib/np_pattern.voronoi`` on the CPU, ``lab_voronoi`` in
``lib/glsl/pattern.py`` on the GPU, both using the shared PCG hash so the sites agree exactly).
Animation moves every site on a closed path: ``site = centre + Jitter * (ra cos(phi) +
rb sin(phi))`` with ra, rb two random vectors per cell and ``phi = 2 pi (Phase + time * Speed)``
(phi = 0 is the static pattern). Needs F1 (generator domain) when no Image is linked.

Modes: Cells (flat random colour per cell), F1 Distance (grey distance to the nearest site),
Edges (Fill Color with borders from F2 - F1), Mosaic (each cell takes the Image colour at its
site pixel; random cell colours if no Image is linked). In Cells / Edges / Mosaic a border of
``Border Width`` (in cell units, measured as F2 - F1) is drawn in ``Border Color`` with about a
pixel of anti-aliasing. Cell fills are not anti-aliased.

Outputs: Color; Value (Cells / Mosaic: random id of the cell, F1: distance (clamped to 0..1),
Edges: F2 - F1 (clamped)); Border (border coverage, also in F1 mode).
"""

import bpy
import numpy as np
from bpy.props import EnumProperty

from ..lib import gpu as lab_gpu, np_noise, np_pattern
from ..lib.node import In, LabNode, Out, is_single

MENU = "Generate"

_MODE_ITEMS = [
    ('CELLS', "Cells", "Flat random colour per cell"),
    ('F1', "F1 Distance", "Distance to the nearest feature point"),
    ('EDGES', "Edges", "Fill colour with borders where F2 - F1 is small"),
    ('MOSAIC', "Mosaic", "Each cell takes the input image colour at its feature point"),
]
_MODE_INDEX = {m[0]: i for i, m in enumerate(_MODE_ITEMS)}
_METRIC_ITEMS = [
    ('EUCLIDEAN', "Euclidean", ""),
    ('MANHATTAN', "Manhattan", ""),
    ('CHEBYSHEV', "Chebyshev", ""),
]
_METRIC_INDEX = {m[0]: i for i, m in enumerate(_METRIC_ITEMS)}

_CHUNK_PIXELS = 1 << 18
F32 = np.float32


def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


_BODY = """
    uint ss = lab_pcg(uint(v_seed));
    vec2 p = (vec2(texel) + vec2(0.5)) * v_k + v_off;
    float f1;
    float f2;
    ivec2 cell;
    vec2 site;
    lab_voronoi(p, v_metric, v_jit, v_cphi, v_sphi, v_anim != 0, ss, f1, f2, cell, site);
    float mb = lab_vor_border(f1, f2, v_bw, v_k);
    vec4 col;
    float val;
    if (v_mode == 1) {
      float g = clamp(f1, 0.0, 1.0);
      col = vec4(g, g, g, 1.0);
      val = g;
    }
    else {
      val = lab_vor_id(cell, ss);
      if (v_mode == 2) {
        col = v_fill;
        val = clamp(f2 - f1, 0.0, 1.0);
      }
      else if (v_mode == 3 && v_has_img != 0) {
        ivec2 sp = ivec2(floor((site - v_off) * v_invk));
        col = IMAGE_AT_SP;
      }
      else {
        col = vec4(lab_vor_rgb(cell, ss), 1.0);
      }
      col = mix(col, v_border, mb);
    }
"""

_STORE = {
    "Color": "    out_Color = col;\n",
    "Value": "    out_Value = vec4(val, val, val, 1.0);\n",
    "Border": "    out_Border = vec4(mb, mb, mb, 1.0);\n",
}


class CompositorNodeLabVoronoi(LabNode, bpy.types.CompositorNode):
    '''Voronoi cells, distance fields, edges and stained-glass mosaic'''
    bl_idname = "CompositorNodeLabVoronoi"
    bl_label = "Voronoi / Mosaic"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0), hide_value=True),
        In("Scale", "FLOAT", 8.0, min=0.0, max=100.0),
        In("Jitter", "FACTOR", 1.0),
        In("Border Width", "FLOAT", 0.06, min=0.0, max=1.0),
        In("Border Color", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("Fill Color", "COLOR", (1.0, 1.0, 1.0, 1.0)),
        In("Offset X", "FLOAT", 0.0, min=-10.0, max=10.0),
        In("Offset Y", "FLOAT", 0.0, min=-10.0, max=10.0),
        In("Phase", "FLOAT", 0.0, min=-10.0, max=10.0),
        In("Speed", "FLOAT", 0.0, min=-10.0, max=10.0),
        In("Seed", "INT", 0, min=0, max=1000),
        Out("Color", "COLOR"),
        Out("Value", "FLOAT"),
        Out("Border", "FLOAT"),
    ]
    PROPS = ["mode", "metric"]

    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default='CELLS')
    metric: EnumProperty(name="Metric", items=_METRIC_ITEMS, default='EUCLIDEAN')

    # -- shared parameters ---------------------------------------------------
    def _params(self, inputs, ctx):
        w, h = ctx.size
        scale = max(self.in_float(inputs, "Scale", 8.0), 1e-3)
        k = float(F32(scale / max(w, h, 1)))
        phi = 2.0 * np.pi * (self.in_float(inputs, "Phase", 0.0) +
                             ctx.time * self.in_float(inputs, "Speed", 0.0))
        cphi, sphi = float(F32(np.cos(phi))), float(F32(np.sin(phi)))
        seed = self.in_int(inputs, "Seed", 0)
        return dict(
            mode=_MODE_INDEX[self.mode], metric=_METRIC_INDEX[self.metric],
            k=k, invk=float(F32(1.0) / F32(k)),
            ox=float(F32(self.in_float(inputs, "Offset X", 0.0))),
            oy=float(F32(self.in_float(inputs, "Offset Y", 0.0))),
            jit=float(F32(min(max(self.in_float(inputs, "Jitter", 1.0), 0.0), 1.0))),
            cphi=cphi, sphi=sphi, anim=int(sphi != 0.0),
            bw=float(F32(max(self.in_float(inputs, "Border Width", 0.06), 0.0))),
            border=self.in_color(inputs, "Border Color", (0.0, 0.0, 0.0, 1.0)),
            fill=self.in_color(inputs, "Fill Color", (1.0, 1.0, 1.0, 1.0)),
            seed=seed, ss=np_noise.scramble_seed(seed),
            has_img=not is_single(inputs.get("Image")),
        )

    # -- CPU -----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        color = self.out_array(outputs, "Color")
        value = self.out_array(outputs, "Value")
        border = self.out_array(outputs, "Border")
        if color is None and value is None and border is None:
            return
        p = self._params(inputs, ctx)
        w, h = ctx.size
        img = self.in_image_array(inputs, "Image", ctx.shape, 4) if p["has_img"] else None
        k = F32(p["k"])
        xs = (np.arange(w, dtype=F32) + F32(0.5)) * k + F32(p["ox"])
        rows_per = max(1, _CHUNK_PIXELS // max(w, 1))
        border_c = np.asarray(p["border"], F32)
        fill_c = np.asarray(p["fill"], F32)
        for y0 in range(0, h, rows_per):
            y1 = min(h, y0 + rows_per)
            ys = (np.arange(y0, y1, dtype=F32) + F32(0.5)) * k + F32(p["oy"])
            x = np.broadcast_to(xs[None, :], (y1 - y0, w))
            y = np.broadcast_to(ys[:, None], (y1 - y0, w))
            f1, f2, cx, cy, sx, sy = np_pattern.voronoi(
                x, y, p["metric"], p["jit"], p["cphi"], p["sphi"], bool(p["anim"]), p["ss"])
            mb = np_pattern.border_mask(f1, f2, p["bw"], k)
            if p["mode"] == 1:
                g = np.clip(f1, F32(0), F32(1))
                col = np.stack([g, g, g, np.ones_like(g)], axis=-1)
                val = g
            else:
                val = np_pattern.cell_id(cx, cy, p["ss"])
                if p["mode"] == 2:
                    col = np.broadcast_to(fill_c, f1.shape + (4,)).astype(F32)
                    val = np.clip(f2 - f1, F32(0), F32(1))
                elif p["mode"] == 3 and img is not None:
                    sxp = np.floor((sx - F32(p["ox"])) * F32(p["invk"])).astype(np.int64)
                    syp = np.floor((sy - F32(p["oy"])) * F32(p["invk"])).astype(np.int64)
                    col = img[np.clip(syp, 0, h - 1), np.clip(sxp, 0, w - 1)]
                else:
                    r, g, b = np_pattern.cell_rgb(cx, cy, p["ss"])
                    col = np.stack([r, g, b, np.ones_like(r)], axis=-1)
                m = mb[..., None]
                col = (col + (border_c - col) * m).astype(F32)
            if color is not None:
                color[y0:y1] = col
            if value is not None:
                value[y0:y1, :, 0] = val
            if border is not None:
                border[y0:y1, :, 0] = mb

    # -- GPU -----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        outs = {n: self.out_texture(outputs, n) for n in ("Color", "Value", "Border")}
        outs = {n: t for n, t in outs.items() if t is not None}
        if not outs:
            return
        p = self._params(inputs, ctx)
        uniforms = {
            "v_mode": ("int", p["mode"]), "v_metric": ("int", p["metric"]),
            "v_k": ("float", p["k"]), "v_invk": ("float", p["invk"]),
            "v_off": ("vec2", (p["ox"], p["oy"])), "v_jit": ("float", p["jit"]),
            "v_cphi": ("float", p["cphi"]), "v_sphi": ("float", p["sphi"]),
            "v_anim": ("int", p["anim"]), "v_bw": ("float", p["bw"]),
            "v_border": ("vec4", p["border"]), "v_fill": ("vec4", p["fill"]),
            "v_seed": ("int", _i32(p["seed"])), "v_has_img": ("int", int(p["has_img"])),
        }
        body = _BODY + "".join(_STORE[n] for n in outs)
        gpu_inputs = {}
        if p["has_img"]:
            body = body.replace("IMAGE_AT_SP", "in_Image(sp)")
            gpu_inputs["Image"] = ("color", self.in_texture_or_value(inputs, "Image"))
        else:
            body = body.replace("IMAGE_AT_SP", "vec4(0.0)")
        lab_gpu.pointwise(body, outs, inputs=gpu_inputs, uniforms=uniforms, libs=("pattern",))


NODE_CLASSES = [CompositorNodeLabVoronoi]
