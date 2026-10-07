# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Flow Field: line integral convolution (LIC) and streamlines along a vector field.

The vector field comes from (Source): curl noise (divergence-free swirls; Scale, Phase, Speed,
Seed, noise type), the gradient of the Image luminance (Sobel; flows uphill), its rotation by 90
degrees (Contour: flows along edges), or the RG channels of the Vector input (an unlinked Vector
is the constant (1, 0): a uniform rightward flow). ``Rotate`` turns every vector by that many
degrees. Only the direction matters: streamlines are traced with unit speed.

LIC: every pixel integrates its streamline ``Length`` steps in both directions (midpoint / RK2,
``Step`` pixels per step, bilinear field and image sampling, clamp to edge) and averages the
image samples along it (Kernel Box: equal weights, Triangle: linearly fading). The smeared image
is the Image input, or grey white noise (per pixel, from Seed) when no Image is linked. A
streamline stops where the field vanishes (squared length < 1e-12); the weights are renormalised
over the samples taken, so a constant image stays constant and a uniform horizontal field gives
a horizontal box blur (length 2 * Length + 1) of the source.

Outputs: Color (the LIC image); Streaks (grey, the same integration applied to sparse impulse
noise of ``Density``, scaled by 0.5 / Density and clamped: bright hair-like streamlines);
Field (RG = unit direction * 0.5 + 0.5, B = raw field magnitude, A = 1). Needs F1 (generator
domain) when no Image / Vector is linked. Two GPU passes: the field goes to a scratch
RGBA32F texture, then the LIC pass samples it; the CPU path precomputes the same field and
integrates in pixel chunks on a thread pool.
"""

import re

import bpy
import numpy as np
from bpy.props import EnumProperty

from ..lib import gpu as lab_gpu, np_field, np_noise
from ..lib.node import In, LabNode, Out, is_single

MENU = "Generate"

_SOURCE_ITEMS = [
    ('CURL', "Curl Noise", "Divergence-free swirling flow from noise"),
    ('GRADIENT', "Image Gradient", "Flow along the gradient of the image luminance"),
    ('CONTOUR', "Image Contour", "Flow along edges: the gradient rotated by 90 degrees"),
    ('VECTOR', "Vector Image", "Use the RG channels of the Vector input"),
]
_NOISE_ITEMS = [
    ('PERLIN', "Perlin", ""),
    ('SIMPLEX', "Simplex", ""),
    ('VALUE', "Value", ""),
]
_NOISE_INDEX = {"VALUE": 0, "PERLIN": 1, "SIMPLEX": 2}
_KERNEL_ITEMS = [
    ('BOX', "Box", "Equal weights along the streamline"),
    ('TRIANGLE', "Triangle", "Weights fade linearly towards the ends"),
]
_KERNEL_INDEX = {"BOX": 0, "TRIANGLE": 1}
F32 = np.float32
MAX_LENGTH = 256
_CURL_EPS = 0.01

_scratch = {}


def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _get_scratch(w, h):
    import gpu

    key = (int(w), int(h))
    tex = _scratch.get(key)
    if tex is None:
        if len(_scratch) >= 3:
            _scratch.clear()
        tex = gpu.types.GPUTexture(key, format='RGBA32F')
        _scratch[key] = tex
    return tex


# Pass 1: the vector field. Source-specific parts are substituted in.
_FIELD_HEAD = """
    vec2 v = vec2(0.0);
"""
_FIELD_BODY = {
    'CURL': """
    vec3 p = vec3((vec2(texel) + vec2(0.5)) * f_k, f_z);
    v = lab_curl2(f_type, p, uint(fl_seed), 1.0, 0.01);
""",
    'GRADIENT': """
    float lm = in_Lum(texel + ivec2(-1, -1)); float l0 = in_Lum(texel + ivec2(0, -1));
    float l1 = in_Lum(texel + ivec2(1, -1)); float l2 = in_Lum(texel + ivec2(-1, 0));
    float l3 = in_Lum(texel + ivec2(1, 0)); float l4 = in_Lum(texel + ivec2(-1, 1));
    float l5 = in_Lum(texel + ivec2(0, 1)); float l6 = in_Lum(texel + ivec2(1, 1));
    float gx = ((l6 + 2.0 * l3 + l1) - (l4 + 2.0 * l2 + lm)) * 0.125;
    float gy = ((l4 + 2.0 * l5 + l6) - (lm + 2.0 * l0 + l1)) * 0.125;
    v = vec2(gx, gy);
""",
    'VECTOR': """
    v = in_Vector(texel).xy;
""",
}
_FIELD_TAIL = """
    v = vec2(v.x * f_rc - v.y * f_rs, v.x * f_rs + v.y * f_rc);
    out_Fld = vec4(v, 0.0, 1.0);
"""
def _lum(expr):
    return "dot(in_Image(%s).rgb, vec3(0.2126, 0.7152, 0.0722))" % expr


_LIC_BODY = """
    vec4 col = vec4(0.0);
    float streak = 0.0;
    lab_fl_lic(texel, fl_len, fl_step, fl_kernel, fl_winv, col, streak);
"""
_FINAL_BODY = """
    vec2 fv = in_Fld(texel).xy;
    float fl2 = fv.x * fv.x + fv.y * fv.y;
    vec2 fd = (fl2 < 1e-12) ? vec2(0.0) : fv * (1.0 / sqrt(fl2));
"""
_STORE = {
    "Color": "    out_Color = col;\n",
    "Streaks": "    float sg = clamp(streak * fl_idens, 0.0, 1.0);\n"
               "    out_Streaks = vec4(sg, sg, sg, 1.0);\n",
    "Field": "    out_Field = vec4(0.5 + 0.5 * fd, sqrt(fl2), 1.0);\n",
}


class CompositorNodeLabFlowField(LabNode, bpy.types.CompositorNode):
    '''Line integral convolution and streamlines along a vector field'''
    bl_idname = "CompositorNodeLabFlowField"
    bl_label = "Flow Field"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0), hide_value=True),
        In("Vector", "VECTOR", (1.0, 0.0, 0.0), hide_value=True),
        In("Length", "INT", 20, min=0, max=MAX_LENGTH),
        In("Step", "FLOAT", 1.0, min=-10.0, max=10.0),
        In("Scale", "FLOAT", 3.0, min=0.0, max=50.0),
        In("Rotate", "FLOAT", 0.0, min=-180.0, max=180.0),
        In("Phase", "FLOAT", 0.0, min=-10.0, max=10.0),
        In("Speed", "FLOAT", 0.0, min=-10.0, max=10.0),
        In("Density", "FACTOR", 0.05),
        In("Seed", "INT", 0, min=0, max=1000),
        Out("Color", "COLOR"),
        Out("Streaks", "FLOAT"),
        Out("Field", "COLOR"),
    ]
    PROPS = ["source", "noise_type", "kernel"]

    source: EnumProperty(name="Source", items=_SOURCE_ITEMS, default='CURL')
    noise_type: EnumProperty(name="Noise", items=_NOISE_ITEMS, default='PERLIN')
    kernel: EnumProperty(name="Kernel", items=_KERNEL_ITEMS, default='BOX')

    def _params(self, inputs, ctx):
        w, h = ctx.size
        scale = self.in_float(inputs, "Scale", 3.0)
        rot = np.radians(self.in_float(inputs, "Rotate", 0.0))
        length = min(max(self.in_int(inputs, "Length", 20), 0), MAX_LENGTH)
        density = min(max(self.in_float(inputs, "Density", 0.05), 0.0), 1.0)
        seed = self.in_int(inputs, "Seed", 0)
        z = self.in_float(inputs, "Phase", 0.0) + ctx.time * self.in_float(inputs, "Speed", 0.0)
        return dict(
            source=self.source, type=_NOISE_INDEX[self.noise_type],
            kernel=self.kernel, kernel_i=_KERNEL_INDEX[self.kernel],
            k=float(F32(scale / max(w, h, 1))), z=float(F32(z)),
            rc=float(F32(np.cos(rot))), rs=float(F32(np.sin(rot))),
            length=length, step=float(F32(self.in_float(inputs, "Step", 1.0))),
            winv=float(F32(1.0) / F32(length + 1)),
            density=float(F32(density)),
            idens=float(F32(0.5 / density)) if density > 0.0 else 0.0,
            seed=seed, has_img=not is_single(inputs.get("Image")),
        )

    # -- CPU -----------------------------------------------------------------
    def _field_cpu(self, inputs, ctx, p):
        w, h = ctx.size
        if p["source"] == 'CURL':
            xs = (np.arange(w, dtype=F32) + F32(0.5)) * F32(p["k"])
            ys = (np.arange(h, dtype=F32) + F32(0.5)) * F32(p["k"])
            vx, vy = np_noise.curl2(p["type"], np.broadcast_to(xs[None, :], (h, w)),
                                    np.broadcast_to(ys[:, None], (h, w)), F32(p["z"]),
                                    p["seed"] & 0xFFFFFFFF, 1.0, _CURL_EPS)
        elif p["source"] == 'VECTOR':
            vec = self.in_image_array(inputs, "Vector", ctx.shape, 3, default=(1.0, 0.0, 0.0))
            vx, vy = vec[..., 0], vec[..., 1]
        else:
            img = self.in_image_array(inputs, "Image", ctx.shape, 4)
            gx, gy = np_field.sobel(np_field.luminance(img))
            vx, vy = (gx, gy) if p["source"] == 'GRADIENT' else (-gy, gx)
        vx, vy = np_field.rotate(vx, vy, p["rc"], p["rs"])
        return np.stack([vx, vy], axis=-1).astype(F32)

    def cpu(self, inputs, outputs, ctx):
        color = self.out_array(outputs, "Color")
        streaks = self.out_array(outputs, "Streaks")
        field_out = self.out_array(outputs, "Field")
        if color is None and streaks is None and field_out is None:
            return
        p = self._params(inputs, ctx)
        w, h = ctx.size
        fld = self._field_cpu(inputs, ctx, p)
        if field_out is not None:
            unit, mag = np_field.unit_field(fld)
            field_out[..., 0:2] = unit * F32(0.5) + F32(0.5)
            field_out[..., 2] = mag
            field_out[..., 3] = 1.0
        if color is None and streaks is None:
            return
        imp = np_field.impulses(w, h, p["seed"], p["density"])
        if p["has_img"]:
            src = np.ascontiguousarray(self.in_image_array(inputs, "Image", ctx.shape, 4))
        else:
            src, _ = np_field.noise_source(w, h, p["seed"], p["density"])
        lic, streak = np_field.lic(fld, src, imp, p["length"], p["step"], p["kernel"])
        if color is not None:
            color[...] = lic
        if streaks is not None:
            streaks[..., 0] = np.clip(streak * F32(p["idens"]), F32(0.0), F32(1.0))

    # -- GPU -----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        outs = {n: self.out_texture(outputs, n) for n in ("Color", "Streaks", "Field")}
        outs = {n: t for n, t in outs.items() if t is not None}
        if not outs:
            return
        p = self._params(inputs, ctx)
        w, h = ctx.size
        scratch = _get_scratch(w, h)
        image = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        shared = {"fl_seed": ("int", _i32(p["seed"]))}

        # Pass 1: the vector field.
        body = _FIELD_HEAD + _FIELD_BODY['GRADIENT' if p["source"] == 'CONTOUR' else p["source"]] + _FIELD_TAIL
        in1 = {}
        libs = ("noise",)
        if p["source"] in ('GRADIENT', 'CONTOUR'):
            in1["Image"] = ("color", image)
            body = re.sub(r"in_Lum\((.*?)\);", lambda m: _lum(m.group(1)) + ";", body)
            if p["source"] == 'CONTOUR':
                body = body.replace("v = vec2(gx, gy);", "v = vec2(-gy, gx);")
        elif p["source"] == 'VECTOR':
            in1["Vector"] = ("vec3", self.in_texture_or_value(inputs, "Vector", (1.0, 0.0, 0.0)))
        uni1 = dict(shared)
        uni1.update({"f_type": ("int", p["type"]), "f_k": ("float", p["k"]),
                     "f_z": ("float", p["z"]), "f_rc": ("float", p["rc"]),
                     "f_rs": ("float", p["rs"])})
        lab_gpu.pointwise(body, {"Fld": scratch}, inputs=in1, uniforms=uni1, libs=libs)

        # Pass 2: LIC + outputs.
        body2 = _FINAL_BODY
        need_lic = "Color" in outs or "Streaks" in outs
        if need_lic:
            body2 += _LIC_BODY
        body2 += "".join(_STORE[n] for n in outs if n != "Streaks" or need_lic)
        uni2 = dict(shared)
        uni2.update({
            "fl_has_img": ("int", int(p["has_img"])), "fl_density": ("float", p["density"]),
            "fl_len": ("int", p["length"]), "fl_step": ("float", p["step"]),
            "fl_kernel": ("int", p["kernel_i"]), "fl_winv": ("float", p["winv"]),
            "fl_idens": ("float", p["idens"]),
        })
        lab_gpu.pointwise(body2, outs, inputs={"Fld": ("color", scratch),
                                         "Image": ("color", image if p["has_img"] else scratch)},
                          uniforms=uni2, libs=("field",))


NODE_CLASSES = [CompositorNodeLabFlowField]
