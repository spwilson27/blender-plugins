# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Edge Stylise: XDoG ink lines, Sobel edges and outlines from alpha.

**XDoG** (Winnemoeller 2012, extended difference of Gaussians) on the Rec. 709 luma ``L`` of the
image: ``D = G(sigma) * L - Tau * G(K sigma) * L`` (Gaussians with radius ``ceil(3 sigma)``,
clamped edges) and ``v = 1`` where ``D >= Epsilon`` else ``v = 1 + tanh(Phi (D - Epsilon))``,
in [0, 1]. The output is ``Ink + (Paper - Ink) * v`` (default black ink on white paper): a large
Phi gives a hard threshold, a small one soft pencil shading. A flat image gives plain paper
(for Tau < 1 and Epsilon <= 0).

**Sobel**: gradient magnitude of the image (Sobel / 8, so a step of height h gives about h / 2) times
Gain. *Mono* uses the luma gradient; *Colour* the magnitude per channel. Optionally inverted
(``1 - m``: dark lines on white). Alpha is 1.

**Outline** from the image's alpha: a stroke of ``Width`` pixels on the inside, outside or centred on
the alpha edge, in the stroke colour (``Stroke Color`` is premultiplied like all compositor
colours). The stroke coverage is built from morphological dilation / erosion of alpha with a
disc of radius ``r`` whose rim is anti-aliased: the offset ``o`` (in whole pixels, clamped edges, so
the image border is not an edge) has weight ``w(o) = clamp(r + 0.85 - |o|, 0, 1)`` and
``dil(a) = max(a, max_o a(p + o) w(o))``, ``ero(a) = min(a, min_o 1 - (1 - a(p + o)) w(o))``.
Outside coverage is ``dil(a) - a`` (r = Width), inside ``a - ero(a)`` (r = Width), centre both with
r = Width / 2. Outside parts are drawn under the image, inside parts over it; *Stroke Only* outputs
just the stroke. The measured width of a stroke on a disc is within 1 px of ``Width`` . Width is limited to 32 px.
"""

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty

from ..lib import np_sampling
from ..lib.glsl import sampling as gl_sampling
from ..lib.node import In, LabNode, Out

MENU = "Filter"

MAX_WIDTH = 32.0
RIM = 0.85    # an offset at distance d has weight clamp(r + RIM - d, 0, 1); calibrates the stroke width
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

_MODE_ITEMS = [
    ('XDOG', "XDoG", "Difference-of-Gaussians ink lines"),
    ('SOBEL', "Sobel", "Gradient magnitude edges"),
    ('OUTLINE', "Outline", "Stroke along the edge of the alpha channel"),
]
_POS_ITEMS = [
    ('OUTSIDE', "Outside", "Stroke outside the alpha edge (drawn under the image)"),
    ('INSIDE', "Inside", "Stroke inside the alpha edge (drawn over the image)"),
    ('CENTER', "Center", "Stroke centred on the alpha edge"),
]

_LUMA_BODY = """
    out_L = vec4(vec3(dot(lab_fetch_Image(texel).rgb, vec3(0.2126, 0.7152, 0.0722))), 1.0);
"""

_XDOG_BODY = """
    float g1 = lab_fetch_B1(texel).r;
    float g2 = lab_fetch_B2(texel).r;
    float D = g1 - x_tau * g2;
    float v = (D >= x_eps) ? 1.0 : 1.0 + tanh(clamp(x_phi * (D - x_eps), -20.0, 20.0));
    v = clamp(v, 0.0, 1.0);
    out_Color = x_ink + (x_paper - x_ink) * v;
"""

_SOBEL_BODY = """
    vec4 gx, gy;
    lab_sobel_Image(texel, 0, gx, gy);
    vec3 m;
    if (s_color != 0) {
      m = sqrt(gx.rgb * gx.rgb + gy.rgb * gy.rgb) * s_gain;
    }
    else {
      float lx = dot(gx.rgb, vec3(0.2126, 0.7152, 0.0722));
      float ly = dot(gy.rgb, vec3(0.2126, 0.7152, 0.0722));
      m = vec3(sqrt(lx * lx + ly * ly) * s_gain);
    }
    if (s_invert != 0) {
      m = vec3(1.0) - m;
    }
    out_Color = vec4(m, 1.0);
"""

_OUTLINE_BODY = """
    vec4 img = lab_fetch_Image(texel);
    float a0 = img.a;
    float mx = a0;
    float mn = a0;
    for (int dy = -o_r; dy <= o_r; ++dy) {
      for (int dx = -o_r; dx <= o_r; ++dx) {
        float wgt = clamp(o_rad + 0.85 - sqrt(float(dx * dx + dy * dy)), 0.0, 1.0);
        if (wgt <= 0.0) {
          continue;
        }
        float a = lab_fetch_Image(texel + ivec2(dx, dy)).a;
        mx = max(mx, a * wgt);
        mn = min(mn, 1.0 - (1.0 - a) * wgt);
      }
    }
    float s_out = (o_pos != 0) ? clamp(mx - a0, 0.0, 1.0) : 0.0;
    float s_in = (o_pos != 1) ? clamp(a0 - mn, 0.0, 1.0) : 0.0;
    vec4 sin_l = o_col * s_in;
    vec4 sout_l = o_col * s_out;
    vec4 res;
    if (o_only != 0) {
      res = sin_l + sout_l;
    }
    else {
      res = img + sout_l * (1.0 - img.a);
      res = sin_l + res * (1.0 - sin_l.a);
    }
    out_Color = res;
"""


def _rim_weights(radius):
    """Offsets inside the anti-aliased disc of ``radius``: (r_max, r2_full, rim) where ``r_max`` is
    the largest offset component, ``r2_full`` the squared radius of the full-weight (w = 1) disc
    (-1 if none) and ``rim`` a list of (dx, dy, w) with 0 < w < 1."""
    f32 = np.float32
    radius = f32(radius)
    r_max = int(math.ceil(float(radius) + RIM))
    span = np.arange(-r_max, r_max + 1)
    dx, dy = np.meshgrid(span, span)
    d2 = (dx * dx + dy * dy).astype(f32)
    w = np.clip(radius + f32(RIM) - np.sqrt(d2), f32(0.0), f32(1.0))
    full = w >= 1.0
    r2_full = int(d2[full].max()) if full.any() else -1
    rim = [(int(x), int(y), float(v)) for x, y, v in zip(dx[(w > 0) & ~full], dy[(w > 0) & ~full],
                                                           w[(w > 0) & ~full])]
    return r_max, r2_full, rim


def morph_disc(a, r2, is_max):
    """Grey-scale dilation (max) or erosion (min) of the (H, W) array ``a`` over the disc of offsets
    ``dx^2 + dy^2 <= r2``, clamped edges. Exact (only comparisons)."""
    op = np.maximum if is_max else np.minimum
    h, w = a.shape
    r = math.isqrt(r2)
    if r == 0:
        return a.copy()
    p = np.pad(a, r, mode="edge")
    hx = {dy: math.isqrt(r2 - dy * dy) for dy in range(-r, r + 1)}
    cur = p[:, r:r + w].copy()       # horizontal window of half width 0
    result = None
    for half in range(0, r + 1):
        if half > 0:
            cur = op(cur, op(p[:, r - half:r - half + w], p[:, r + half:r + half + w]))
        for dy, hh in hx.items():
            if hh == half:
                rows = cur[r + dy:r + dy + h]
                result = rows.copy() if result is None else op(result, rows)
    return result


def morph_soft(a, radius):
    """(dilation, erosion) of the (H, W) float32 array ``a`` with the anti-aliased disc."""
    f32 = np.float32
    r_max, r2_full, rim = _rim_weights(radius)
    h, w = a.shape
    dil = morph_disc(a, max(r2_full, 0), True)
    ero = morph_disc(a, max(r2_full, 0), False)
    if rim:
        p = np.pad(a, r_max, mode="edge")
        for dx, dy, wt in rim:
            sh = p[r_max + dy:r_max + dy + h, r_max + dx:r_max + dx + w]
            wt = f32(wt)
            dil = np.maximum(dil, sh * wt)
            ero = np.minimum(ero, f32(1.0) - (f32(1.0) - sh) * wt)
    return dil, ero


class CompositorNodeLabEdgeStylise(LabNode, bpy.types.CompositorNode):
    '''XDoG ink lines, Sobel edges, or a stroke outline around the alpha channel'''
    bl_idname = "CompositorNodeLabEdgeStylise"
    bl_label = "Edge Stylise"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Image", "COLOR"),
    ]

    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default='XDOG')
    # XDoG
    sigma: FloatProperty(name="Sigma", default=1.0, min=0.05, max=24.0,
                         description="Radius of the fine Gaussian (pixels)")
    k: FloatProperty(name="K", default=1.6, min=1.0, max=10.0,
                     description="Ratio of the coarse to the fine Gaussian")
    tau: FloatProperty(name="Tau", default=0.98, min=0.0, max=2.0,
                       description="Weight of the coarse Gaussian (below 1 also keeps the paper white)")
    phi: FloatProperty(name="Phi", default=20.0, min=0.0, max=1000.0,
                       description="Sharpness of the ink transition (large = hard threshold)")
    epsilon: FloatProperty(name="Epsilon", default=0.0, min=-1.0, max=1.0,
                           description="Threshold of the difference of Gaussians")
    ink: FloatVectorProperty(name="Ink", subtype='COLOR', size=4, min=0.0, soft_max=1.0,
                             default=(0.0, 0.0, 0.0, 1.0))
    paper: FloatVectorProperty(name="Paper", subtype='COLOR', size=4, min=0.0, soft_max=1.0,
                               default=(1.0, 1.0, 1.0, 1.0))
    # Sobel
    gain: FloatProperty(name="Gain", default=2.0, min=0.0, max=100.0,
                        description="Scale of the gradient magnitude")
    colored: BoolProperty(name="Color", default=False,
                          description="Per-channel gradient magnitude instead of luma")
    invert: BoolProperty(name="Invert", default=False, description="Dark edges on white")
    # Outline
    width: FloatProperty(name="Width", default=4.0, min=0.0, max=MAX_WIDTH, subtype='PIXEL',
                         description="Stroke width in pixels")
    position: EnumProperty(name="Position", items=_POS_ITEMS, default='OUTSIDE')
    stroke: FloatVectorProperty(name="Stroke Color", subtype='COLOR', size=4, min=0.0,
                                soft_max=1.0, default=(0.0, 0.0, 0.0, 1.0))
    stroke_only: BoolProperty(name="Stroke Only", default=False,
                              description="Output only the stroke, without the image")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        if self.mode == 'XDOG':
            for name in ("sigma", "k", "tau", "phi", "epsilon", "ink", "paper"):
                layout.prop(self, name)
        elif self.mode == 'SOBEL':
            layout.prop(self, "gain")
            layout.prop(self, "colored")
            layout.prop(self, "invert")
        else:
            layout.prop(self, "width")
            layout.prop(self, "position", text="")
            layout.prop(self, "stroke")
            layout.prop(self, "stroke_only")

    # -- parameters --------------------------------------------------------
    def _xdog(self):
        f32 = np.float32
        sigma = float(f32(min(max(self.sigma, 0.05), 24.0)))
        k = float(f32(min(max(self.k, 1.0), 10.0)))
        return dict(
            sigma=sigma, sigma2=float(f32(f32(sigma) * f32(k))), tau=float(f32(self.tau)),
            phi=float(f32(self.phi)), eps=float(f32(self.epsilon)),
            ink=tuple(float(f32(v)) for v in self.ink),
            paper=tuple(float(f32(v)) for v in self.paper))

    def _outline(self):
        f32 = np.float32
        width = min(max(float(self.width), 0.0), MAX_WIDTH)
        pos = {'OUTSIDE': 1, 'INSIDE': 0, 'CENTER': 2}[self.position]
        radius = float(f32(width / 2.0 if pos == 2 else width))
        return dict(pos=pos, radius=radius, r=int(math.ceil(radius + RIM)),
                    col=tuple(float(f32(v)) for v in self.stroke), only=bool(self.stroke_only))

    # -- CPU ---------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        img = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                             default=(0.5, 0.5, 0.5, 1.0)), np.float32)
        if self.mode == 'XDOG':
            out[...] = self._cpu_xdog(img)
        elif self.mode == 'SOBEL':
            out[...] = self._cpu_sobel(img)
        else:
            out[...] = self._cpu_outline(img)

    def _cpu_xdog(self, img):
        f32 = np.float32
        p = self._xdog()
        luma = (img[..., 0] * _LUMA[0] + img[..., 1] * _LUMA[1] + img[..., 2] * _LUMA[2])
        luma = luma[..., None]
        g1 = np_sampling.blur_gaussian(luma, p["sigma"], "CLAMP")[..., 0]
        g2 = np_sampling.blur_gaussian(luma, p["sigma2"], "CLAMP")[..., 0]
        d = g1 - f32(p["tau"]) * g2
        eps = f32(p["eps"])
        arg = np.clip(f32(p["phi"]) * (d - eps), f32(-20.0), f32(20.0))
        v = np.where(d >= eps, f32(1.0), f32(1.0) + np.tanh(arg))
        v = np.clip(v, f32(0.0), f32(1.0)).astype(f32)[..., None]
        ink = np.asarray(p["ink"], f32)
        paper = np.asarray(p["paper"], f32)
        return ink + (paper - ink) * v

    def _cpu_sobel(self, img):
        f32 = np.float32
        gx, gy = np_sampling.sobel(img[..., :3], "CLAMP")
        gain = f32(self.gain)
        if self.colored:
            m = np.sqrt(gx * gx + gy * gy) * gain
        else:
            lx = gx[..., 0] * _LUMA[0] + gx[..., 1] * _LUMA[1] + gx[..., 2] * _LUMA[2]
            ly = gy[..., 0] * _LUMA[0] + gy[..., 1] * _LUMA[1] + gy[..., 2] * _LUMA[2]
            m = (np.sqrt(lx * lx + ly * ly) * gain)[..., None]
            m = np.broadcast_to(m, m.shape[:2] + (3,))
        if self.invert:
            m = f32(1.0) - m
        res = np.ones(img.shape, f32)
        res[..., :3] = m
        return res

    def _cpu_outline(self, img):
        f32 = np.float32
        p = self._outline()
        a = np.ascontiguousarray(img[..., 3])
        a0 = a
        zero = np.zeros_like(a)
        mx, mn = morph_soft(a, p["radius"])
        s_out = np.clip(mx - a0, f32(0.0), f32(1.0)) if p["pos"] != 0 else zero
        s_in = np.clip(a0 - mn, f32(0.0), f32(1.0)) if p["pos"] != 1 else zero
        col = np.asarray(p["col"], f32)
        sin_l = col * s_in[..., None]
        sout_l = col * s_out[..., None]
        if p["only"]:
            return sin_l + sout_l
        res = img + sout_l * (f32(1.0) - img[..., 3:4])
        return sin_l + res * (f32(1.0) - sin_l[..., 3:4])

    # -- GPU ---------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        src = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        if self.mode == 'XDOG':
            p = self._xdog()
            w, h = int(dst.width), int(dst.height)
            luma = gl_sampling.scratch(w, h, "edge_luma")
            b1 = gl_sampling.scratch(w, h, "edge_b1")
            b2 = gl_sampling.scratch(w, h, "edge_b2")
            gl_sampling.kernel(_LUMA_BODY, {"L": luma}, {"Image": src})
            gl_sampling.gaussian_blur(luma, b1, p["sigma"], 0, role="edge_x1")
            gl_sampling.gaussian_blur(luma, b2, p["sigma2"], 0, role="edge_x2")
            gl_sampling.kernel(
                _XDOG_BODY, {"Color": dst}, {"B1": b1, "B2": b2},
                uniforms={"x_tau": ("float", p["tau"]), "x_phi": ("float", p["phi"]),
                          "x_eps": ("float", p["eps"]), "x_ink": ("vec4", p["ink"]),
                          "x_paper": ("vec4", p["paper"])})
        elif self.mode == 'SOBEL':
            gl_sampling.kernel(
                _SOBEL_BODY, {"Color": dst}, {"Image": src},
                uniforms={"s_gain": ("float", float(np.float32(self.gain))),
                          "s_color": ("int", int(self.colored)),
                          "s_invert": ("int", int(self.invert))})
        else:
            p = self._outline()
            gl_sampling.kernel(
                _OUTLINE_BODY, {"Color": dst}, {"Image": src},
                uniforms={"o_r": ("int", p["r"]), "o_rad": ("float", p["radius"]),
                          "o_pos": ("int", p["pos"]), "o_only": ("int", int(p["only"])),
                          "o_col": ("vec4", p["col"])})


NODE_CLASSES = [CompositorNodeLabEdgeStylise]
