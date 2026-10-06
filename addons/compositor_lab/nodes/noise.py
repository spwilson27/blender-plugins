# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Noise generator: value / Perlin / simplex / Worley with fBm, ridged, domain warp, animation.

The field is ``lib/np_noise.noise_field`` (CPU) and ``lab_noise_field`` in ``lib/glsl/noise.py``
(GPU). Pixel (x, y) samples the field at
``p = ((x + 0.5, y + 0.5) * Scale / max(width, height) + Offset, z)`` with
``z = Phase + time * Speed``, so the pattern does not change with resolution. Needs F1 (generator
domain): with no image input the output is render-sized.

Colour output: channel c uses seed ``Seed + c * 0x9E3779B1`` (R equals the Value output),
alpha 1.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty

from ..lib import glsl, gpu as lab_gpu, np_noise
from ..lib.node import In, LabNode, Out

MENU = "Generate"

_TYPE_ITEMS = [
    ('VALUE', "Value", "Smooth value noise"),
    ('PERLIN', "Perlin", "Gradient noise"),
    ('SIMPLEX', "Simplex", "Simplex gradient noise"),
    ('WORLEY_F1', "Worley F1", "Distance to the nearest feature point"),
    ('WORLEY_F2_F1', "Worley F2-F1", "Cell borders: second minus nearest distance"),
]
_TYPE_INDEX = {t[0]: i for i, t in enumerate(_TYPE_ITEMS)}
_CHANNEL_STEP = 0x9E3779B1


def channel_seed(seed, channel):
    """Seed of colour channel `channel` as a python int in [0, 2**32)."""
    return (int(seed) + channel * _CHANNEL_STEP) & 0xFFFFFFFF


def _i32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


class CompositorNodeLabNoise(LabNode, bpy.types.CompositorNode):
    '''Procedural noise: value, Perlin, simplex or Worley, with fBm, ridged and domain warp'''
    bl_idname = "CompositorNodeLabNoise"
    bl_label = "Noise"

    SOCKETS = [
        In("Scale", "FLOAT", 5.0),
        In("Octaves", "INT", 4),
        In("Lacunarity", "FLOAT", 2.0),
        In("Gain", "FLOAT", 0.5),
        In("Warp", "FLOAT", 0.0),
        In("Randomness", "FACTOR", 1.0),
        In("Offset X", "FLOAT", 0.0),
        In("Offset Y", "FLOAT", 0.0),
        In("Phase", "FLOAT", 0.0),
        In("Speed", "FLOAT", 0.5),
        In("Seed", "INT", 0),
        Out("Value", "FLOAT"),
        Out("Color", "COLOR"),
    ]
    PROPS = ["noise_type", "ridged"]

    noise_type: EnumProperty(name="Type", items=_TYPE_ITEMS, default='PERLIN')
    ridged: BoolProperty(name="Ridged", default=False,
                         description="Fold each octave into sharp ridges")

    # -- shared parameter parsing -------------------------------------------
    def _params(self, inputs, ctx):
        w, h = ctx.size
        scale = self.in_float(inputs, "Scale", 5.0)
        phase = self.in_float(inputs, "Phase", 0.0)
        speed = self.in_float(inputs, "Speed", 0.5)
        return dict(
            ntype=_TYPE_INDEX[self.noise_type],
            ridged=bool(self.ridged),
            k=float(np.float32(scale / max(w, h, 1))),
            ox=float(np.float32(self.in_float(inputs, "Offset X", 0.0))),
            oy=float(np.float32(self.in_float(inputs, "Offset Y", 0.0))),
            z=float(np.float32(phase + ctx.time * speed)),
            octaves=min(max(self.in_int(inputs, "Octaves", 4), 1), 16),
            lacunarity=float(np.float32(self.in_float(inputs, "Lacunarity", 2.0))),
            gain=float(np.float32(self.in_float(inputs, "Gain", 0.5))),
            warp=float(np.float32(self.in_float(inputs, "Warp", 0.0))),
            jitter=float(np.float32(min(max(self.in_float(inputs, "Randomness", 1.0), 0.0), 1.0))),
            seed=self.in_int(inputs, "Seed", 0),
        )

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        value = self.out_array(outputs, "Value")
        color = self.out_array(outputs, "Color")
        if value is None and color is None:
            return
        p = self._params(inputs, ctx)
        w, h = ctx.size
        f32 = np.float32
        xs = (np.arange(w, dtype=f32) + f32(0.5)) * f32(p["k"]) + f32(p["ox"])
        ys = (np.arange(h, dtype=f32) + f32(0.5)) * f32(p["k"]) + f32(p["oy"])
        x = np.broadcast_to(xs[None, :], (h, w))
        y = np.broadcast_to(ys[:, None], (h, w))
        z = f32(p["z"])

        def field(channel):
            return np_noise.noise_field(
                p["ntype"], x, y, z, channel_seed(p["seed"], channel), p["jitter"], p["octaves"],
                p["lacunarity"], p["gain"], p["ridged"], p["warp"])

        v0 = field(0)
        if value is not None:
            value[..., 0] = v0
        if color is not None:
            color[..., 0] = v0
            color[..., 1] = field(1)
            color[..., 2] = field(2)
            color[..., 3] = 1.0

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        value = self.out_texture(outputs, "Value")
        color = self.out_texture(outputs, "Color")
        if value is None and color is None:
            return
        p = self._params(inputs, ctx)
        uniforms = {
            "n_type": ("int", p["ntype"]),
            "n_ridged": ("int", int(p["ridged"])),
            "n_k": ("float", p["k"]),
            "n_off": ("vec2", (p["ox"], p["oy"])),
            "n_z": ("float", p["z"]),
            "n_octaves": ("int", p["octaves"]),
            "n_lac": ("float", p["lacunarity"]),
            "n_gain": ("float", p["gain"]),
            "n_warp": ("float", p["warp"]),
            "n_jitter": ("float", p["jitter"]),
            "n_seed0": ("int", _i32(channel_seed(p["seed"], 0))),
        }
        body = _BODY_HEAD
        if color is not None:
            uniforms["n_seed1"] = ("int", _i32(channel_seed(p["seed"], 1)))
            uniforms["n_seed2"] = ("int", _i32(channel_seed(p["seed"], 2)))
            body += _BODY_COLOR
        if value is not None:
            body += "    out_Value = vec4(v0, v0, v0, 1.0);\n"
        outs = {}
        if value is not None:
            outs["Value"] = value
        if color is not None:
            outs["Color"] = color
        lab_gpu.pointwise(body, outs, uniforms=uniforms, libs=("noise",))


_FIELD = ("lab_noise_field(n_type, p, uint(%s), n_jitter, n_octaves, n_lac, n_gain, "
          "n_ridged != 0, n_warp)")

_BODY_HEAD = """
    vec3 p = vec3((vec2(texel) + vec2(0.5)) * n_k + n_off, n_z);
    float v0 = %s;
""" % (_FIELD % "n_seed0")

_BODY_COLOR = """
    float v1 = %s;
    float v2 = %s;
    out_Color = vec4(v0, v1, v2, 1.0);
""" % (_FIELD % "n_seed1", _FIELD % "n_seed2")

NODE_CLASSES = [CompositorNodeLabNoise]
