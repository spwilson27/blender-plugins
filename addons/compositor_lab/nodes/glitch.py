# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Glitch: RGB channel split, block displacement, scanline jitter and bit-crush, seeded and
animated by the evaluation time.

All random decisions come from the shared PCG hash (``lib/glsl/hash.py`` / ``lib/np_noise.py``)
with integer maths only, so the CPU and GPU pick exactly the same blocks and rows, and use
integer pixel shifts, so those pixels are copied exactly. The glitch pattern changes
``Speed`` times per second: ``step = floor(time * Speed)`` (0 without a context or with Speed 0)
and is hashed together with the Seed.

Order per output pixel (x, y):
1. Block displacement: the image is cut into ``Block Width x Block Height`` cells. A cell moves
   if ``u01(hash(cell, step, seed_b).x) < Density``; it then reads the image shifted by an
   integer in [-Shift, Shift] pixels (``hash.y`` modulo 2 Shift + 1), wrapping around horizontally.
2. Scanline jitter: each row moves with probability ``Jitter Density`` by an integer in
   [-Jitter, Jitter], also wrapping.
3. RGB split: red is read at +offset, blue at -offset (integer pixels, direction ``Angle``,
   length ``Split`` scaled per step by ``1 + Flicker * (2 u - 1)``), green in place; coordinates
   clamp at the border. Alpha is the maximum of the three samples.
4. Bit-crush: premultiplied RGB is clamped to [0, 1] and rounded to ``2^bits - 1`` levels.
The result is mixed with the input by Fac.
"""

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, IntProperty

from ..lib import gpu as lab_gpu, np_noise
from ..lib.node import In, LabNode, Out

MENU = "Filter"

F32 = np.float32
U32 = np.uint32
_SEED_STEP = 0x9E3779B1


def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def sub_seed(seed, k):
    """Seed of the k-th random stream (python int in [0, 2**32))."""
    return (int(seed) + k * _SEED_STEP) & 0xFFFFFFFF


_BODY = """
    vec4 c = in_Image(texel);
    int gx = texel.x;
    int gy = texel.y;
    int xs = gx;
    if (g_blocks != 0) {
      uvec3 hh = lab_hash3(ivec3(gx / g_bw, gy / g_bh, g_step), lab_pcg(uint(g_seed_b)));
      if (lab_u01(hh.x) < g_density) {
        int sh = int((hh.y >> 8u) % uint(2 * g_amp + 1)) - g_amp;
        xs -= sh;
        if (xs < 0) { xs += res.x; }
        if (xs >= res.x) { xs -= res.x; }
      }
    }
    if (g_jit != 0) {
      uvec3 hh = lab_hash3(ivec3(gy, 0, g_step), lab_pcg(uint(g_seed_j)));
      if (lab_u01(hh.x) < g_jdens) {
        int sh = int((hh.y >> 8u) % uint(2 * g_jamp + 1)) - g_jamp;
        xs -= sh;
        if (xs < 0) { xs += res.x; }
        if (xs >= res.x) { xs -= res.x; }
      }
    }
    vec4 cr = in_Image(ivec2(xs + g_dx, gy + g_dy));
    vec4 cg = in_Image(ivec2(xs, gy));
    vec4 cb = in_Image(ivec2(xs - g_dx, gy - g_dy));
    vec4 gr = vec4(cr.r, cg.g, cb.b, max(cr.a, max(cg.a, cb.a)));
    if (g_crush != 0) {
      vec3 cq = clamp(gr.rgb, 0.0, 1.0);
      gr.r = floor(lab_rnd32(cq.r * g_levels) + 0.5) / g_levels;
      gr.g = floor(lab_rnd32(cq.g * g_levels) + 0.5) / g_levels;
      gr.b = floor(lab_rnd32(cq.b * g_levels) + 0.5) / g_levels;
    }
    float fac = in_Fac(texel);
    out_Color = c * (1.0 - fac) + gr * fac;
"""


class CompositorNodeLabGlitch(LabNode, bpy.types.CompositorNode):
    '''Digital glitch: RGB split, block displacement, scanline jitter and bit-crush, animated'''
    bl_idname = "CompositorNodeLabGlitch"
    bl_label = "Glitch"

    SOCKETS = [
        In("Fac", "FACTOR", 1.0),
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("Seed", "INT", 0),
        In("Speed", "FLOAT", 8.0),
        In("Split", "FLOAT", 6.0),
        In("Shift", "FLOAT", 48.0),
        In("Density", "FACTOR", 0.2),
        In("Jitter", "FLOAT", 6.0),
        Out("Color", "COLOR"),
    ]
    PROPS = ["use_split", "split_angle", "split_flicker", None, "use_blocks", "block_width",
             "block_height", None, "use_jitter", "jitter_density", None, "use_crush",
             "crush_bits"]

    use_split: BoolProperty(name="RGB Split", default=True)
    split_angle: FloatProperty(name="Angle", default=0.0, subtype='ANGLE')
    split_flicker: FloatProperty(name="Flicker", description="Random variation of the split "
                                 "length from one glitch step to the next", default=0.5,
                                 min=0.0, max=1.0, subtype='FACTOR')
    use_blocks: BoolProperty(name="Block Displacement", default=True)
    block_width: IntProperty(name="Block Width", default=96, min=1, max=4096, subtype='PIXEL')
    block_height: IntProperty(name="Block Height", default=24, min=1, max=4096, subtype='PIXEL')
    use_jitter: BoolProperty(name="Scanline Jitter", default=True)
    jitter_density: FloatProperty(name="Row Density", description="Fraction of rows that move",
                                  default=0.25, min=0.0, max=1.0, subtype='FACTOR')
    use_crush: BoolProperty(name="Bit Crush", default=False)
    crush_bits: IntProperty(name="Bits", description="Bits per channel", default=4, min=1,
                            max=16)

    # -- shared -------------------------------------------------------------
    def _params(self, inputs, ctx):
        w = ctx.size[0]
        seed = self.in_int(inputs, "Seed", 0)
        speed = self.in_float(inputs, "Speed", 8.0)
        step = 0
        if ctx.has_context and speed != 0.0:
            step = int(math.floor(ctx.time * speed + 1e-6))
            step = max(-(1 << 30), min(1 << 30, step))
        split = self.in_float(inputs, "Split", 6.0) if self.use_split else 0.0
        dx = dy = 0
        if split != 0.0:
            u = float(np_noise.rand(np.array([step], np.int32), np.array([0], np.int32),
                                    sub_seed(seed, 3))[0])
            length = split * (1.0 + float(self.split_flicker) * (2.0 * u - 1.0))
            dx = int(round(length * math.cos(self.split_angle)))
            dy = int(round(length * math.sin(self.split_angle)))
        amp = min(max(int(round(self.in_float(inputs, "Shift", 48.0))), 0), w - 1)
        jamp = min(max(int(round(self.in_float(inputs, "Jitter", 6.0))), 0), w - 1)
        levels = float((1 << int(self.crush_bits)) - 1)
        return dict(
            step=step, seed_b=sub_seed(seed, 1), seed_j=sub_seed(seed, 2),
            blocks=bool(self.use_blocks) and amp > 0, amp=amp,
            bw=int(self.block_width), bh=int(self.block_height),
            density=F32(self.in_float(inputs, "Density", 0.2)),
            jit=bool(self.use_jitter) and jamp > 0, jamp=jamp,
            jdens=F32(self.jitter_density), dx=dx, dy=dy,
            crush=bool(self.use_crush), levels=F32(levels))

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        c = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                           default=(0.5, 0.5, 0.5, 1.0)))
        fac = self.in_image_array(inputs, "Fac", ctx.shape, 1, default=1.0)
        out[...] = glitch(c, fac, **self._params(inputs, ctx))

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        p = self._params(inputs, ctx)
        uniforms = {
            "g_blocks": ("int", int(p["blocks"])), "g_amp": ("int", p["amp"]),
            "g_bw": ("int", p["bw"]), "g_bh": ("int", p["bh"]),
            "g_density": ("float", float(p["density"])), "g_step": ("int", p["step"]),
            "g_seed_b": ("int", _i32(p["seed_b"])), "g_seed_j": ("int", _i32(p["seed_j"])),
            "g_jit": ("int", int(p["jit"])), "g_jamp": ("int", p["jamp"]),
            "g_jdens": ("float", float(p["jdens"])),
            "g_dx": ("int", p["dx"]), "g_dy": ("int", p["dy"]),
            "g_crush": ("int", int(p["crush"])), "g_levels": ("float", float(p["levels"])),
        }
        lab_gpu.pointwise(
            _BODY, {"Color": dst},
            inputs={
                "Image": ("color", self.in_texture_or_value(inputs, "Image",
                                                            (0.5, 0.5, 0.5, 1.0))),
                "Fac": ("float", self.in_texture_or_value(inputs, "Fac", 1.0)),
            },
            uniforms=uniforms, libs=("hash", "exact"))


def _shift_map(hx, hy, density, amp):
    """Integer shift (0 where the cell/row does not move) from hash words."""
    move = np_noise.u01(hx) < density
    sh = ((hy >> U32(8)) % U32(2 * amp + 1)).astype(np.int32) - np.int32(amp)
    return np.where(move, sh, 0).astype(np.int32)


def block_shifts(shape, step, seed_b, density, amp, bw, bh):
    """(H, W) int32 horizontal shift of every pixel's block (the decisions the GPU repeats)."""
    h, w = shape
    cx = np.arange(w, dtype=np.int32) // bw
    cy = np.arange(h, dtype=np.int32) // bh
    ncx, ncy = int(cx[-1]) + 1, int(cy[-1]) + 1
    hx, hy, _ = np_noise.hash3(np.arange(ncx, dtype=np.int32)[None, :],
                               np.arange(ncy, dtype=np.int32)[:, None], np.int32(step),
                               np_noise.scramble_seed(seed_b))
    grid = _shift_map(hx, hy, density, amp)
    return grid[cy][:, cx]


def row_shifts(h, step, seed_j, density, amp):
    hx, hy, _ = np_noise.hash3(np.arange(h, dtype=np.int32), np.int32(0), np.int32(step),
                               np_noise.scramble_seed(seed_j))
    return _shift_map(hx, hy, density, amp)


def glitch(c, fac, step, seed_b, seed_j, blocks, amp, bw, bh, density, jit, jamp, jdens, dx, dy,
           crush, levels):
    """Reference / CPU implementation on a premultiplied (H, W, 4) float32 image."""
    h, w = c.shape[:2]
    xs = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    ys = np.arange(h, dtype=np.int32)[:, None]
    if blocks:
        xs = xs - block_shifts((h, w), step, seed_b, density, amp, bw, bh)
        xs = np.where(xs < 0, xs + w, xs)
        xs = np.where(xs >= w, xs - w, xs)
    if jit:
        xs = xs - row_shifts(h, step, seed_j, jdens, jamp)[:, None]
        xs = np.where(xs < 0, xs + w, xs)
        xs = np.where(xs >= w, xs - w, xs)

    def fetch(ox, oy):
        return c[np.clip(ys + oy, 0, h - 1), np.clip(xs + ox, 0, w - 1)]

    cr, cg, cb = fetch(dx, dy), fetch(0, 0), fetch(-dx, -dy)
    r = np.stack([cr[..., 0], cg[..., 1], cb[..., 2],
                  np.maximum(cr[..., 3], np.maximum(cg[..., 3], cb[..., 3]))], axis=-1)
    if crush:
        r[..., :3] = np.floor(np.clip(r[..., :3], 0, 1) * levels + F32(0.5)) / levels
    fac = np.asarray(fac, F32)
    return (c * (F32(1.0) - fac) + r * fac).astype(F32)


NODE_CLASSES = [CompositorNodeLabGlitch]
