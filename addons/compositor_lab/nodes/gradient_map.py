# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Gradient Map: map a per-pixel value (luminance, OKLab lightness, a channel, max or alpha)
through a colour gradient: one of several curated presets or up to 6 custom stops.

t = source value of the straight (un-premultiplied) colour, clamped to [0, 1] (optionally
reversed). The gradient is evaluated stop by stop: for stops sorted by position, ``t`` at or
below the first position gives the first stop colour, at or above the last position the last.
Interpolation between stops is Linear, Smooth (smoothstep on the segment) or Constant (the left
stop's colour). Colours are interpolated in scene-linear RGB or in OKLab (converted back to RGB,
clamped to >= 0). Preset colours are authored in sRGB and converted to linear.

Output = Fac mix of the input and the mapped colour; with Preserve Alpha the mapped colour takes
the input alpha (premultiplied), otherwise the result is opaque.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty

from ..lib import gpu as lab_gpu, np_color
from ..lib.node import In, LabNode, Out

MENU = "Filter"

F32 = np.float32
MAX_STOPS = 6


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _preset(*hexes, positions=None):
    n = len(hexes)
    pos = positions or [i / (n - 1) for i in range(n)]
    return [(float(p), _hex(h)) for p, h in zip(pos, hexes)]


# (identifier, label, [(position, sRGB colour)])
PRESETS = {
    'INFERNO': ("Inferno", _preset("#000004", "#420a68", "#932667", "#dd513a", "#fca50a",
                                    "#fcffa4")),
    'VIRIDIS': ("Viridis", _preset("#440154", "#3b528b", "#21908d", "#5dc863", "#fde725")),
    'SUNSET': ("Sunset", _preset("#10062b", "#5b1a85", "#d3407a", "#ff9a5a", "#ffe9b0")),
    'OCEAN': ("Ocean", _preset("#00122b", "#023e7d", "#0a8fa0", "#8fd6c4", "#f1faee")),
    'FIRE': ("Fire", _preset("#000000", "#6a0f00", "#d8360a", "#ffb01f", "#ffffff")),
    'ICE_FIRE': ("Ice and Fire", _preset("#0b3d91", "#6fb1e0", "#f2f2f2", "#f0a060", "#b3200f")),
    'SEPIA': ("Sepia", _preset("#1b1008", "#6b4a2b", "#c9a374", "#f6e7c8")),
}

_PRESET_ITEMS = [(k, v[0], "") for k, v in PRESETS.items()] + [
    ('CUSTOM', "Custom", "Up to 6 colour stops")]
_SOURCE_ITEMS = [
    ('LUMINANCE', "Luminance", "Rec. 709 luminance of the linear colour"),
    ('LIGHTNESS', "Lightness", "OKLab lightness"),
    ('RED', "Red", ""),
    ('GREEN', "Green", ""),
    ('BLUE', "Blue", ""),
    ('VALUE', "Value", "Maximum of R, G, B"),
    ('ALPHA', "Alpha", ""),
]
_INTERP_ITEMS = [
    ('LINEAR', "Linear", ""),
    ('SMOOTH', "Smooth", "Smoothstep between stops"),
    ('CONSTANT', "Constant", "Hold each stop's colour until the next stop"),
]
_SPACE_ITEMS = [
    ('LINEAR_RGB', "Linear RGB", "Interpolate scene-linear colours"),
    ('OKLAB', "OKLab", "Interpolate in OKLab (perceptually smoother)"),
]
_IDX = {name: [i[0] for i in items].index for name, items in
        (("src", _SOURCE_ITEMS), ("interp", _INTERP_ITEMS), ("space", _SPACE_ITEMS))}

_DEFAULT_STOPS = (((0.0, 0.0, 0.0), 0.0), ((0.8, 0.05, 0.02), 0.5), ((1.0, 1.0, 1.0), 1.0),
                  ((1.0, 1.0, 1.0), 0.75), ((1.0, 1.0, 1.0), 0.85), ((1.0, 1.0, 1.0), 1.0))

_SEG = """
    if (g_n > %(n)d) {
      float f = (g_i%(k)d > 0.0) ? clamp((t - g_s%(k)d.w) * g_i%(k)d, 0.0, 1.0) :
                                   ((t >= g_s%(n)d.w) ? 1.0 : 0.0);
      if (g_interp == 2) {
        f = (t >= g_s%(n)d.w) ? 1.0 : 0.0;
      }
      else if (g_interp == 1) {
        f = f * f * (3.0 - 2.0 * f);
      }
      col = col * (1.0 - f) + g_s%(n)d.rgb * f;
    }
"""

_BODY = """
    vec4 c = in_Image(texel);
    float a = c.a;
    vec3 s = (a > 0.0) ? c.rgb / a : vec3(0.0);
    float t = 0.0;
    if (g_src == 0) { t = lab_luma(s); }
    else if (g_src == 1) { t = lab_linear_to_oklab(max(s, vec3(0.0))).x; }
    else if (g_src == 2) { t = s.r; }
    else if (g_src == 3) { t = s.g; }
    else if (g_src == 4) { t = s.b; }
    else if (g_src == 5) { t = max(s.r, max(s.g, s.b)); }
    else { t = a; }
    t = clamp(t, 0.0, 1.0);
    if (g_rev != 0) {
      t = 1.0 - t;
    }
    vec3 col = g_s0.rgb;
@SEGMENTS@    if (g_space == 1) {
      col = max(lab_oklab_to_linear(col), vec3(0.0));
    }
    vec4 res = (g_alpha != 0) ? vec4(col * a, a) : vec4(col, 1.0);
    float fac = in_Fac(texel);
    out_Color = c * (1.0 - fac) + res * fac;
""".replace("@SEGMENTS@", "".join(_SEG % {"k": k, "n": k + 1} for k in range(MAX_STOPS - 1)))


class CompositorNodeLabGradientMap(LabNode, bpy.types.CompositorNode):
    '''Map luminance (or a channel) through a colour gradient: presets or up to 6 custom stops'''
    bl_idname = "CompositorNodeLabGradientMap"
    bl_label = "Gradient Map"

    SOCKETS = [
        In("Fac", "FACTOR", 1.0),
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Color", "COLOR"),
    ]

    preset: EnumProperty(name="Preset", items=_PRESET_ITEMS, default='INFERNO')
    source: EnumProperty(name="Source", items=_SOURCE_ITEMS, default='LUMINANCE')
    interpolation: EnumProperty(name="Interpolation", items=_INTERP_ITEMS, default='LINEAR')
    color_space: EnumProperty(name="Blend In", items=_SPACE_ITEMS, default='OKLAB')
    reverse: BoolProperty(name="Reverse", default=False)
    preserve_alpha: BoolProperty(name="Preserve Alpha", default=True)
    stop_count: IntProperty(name="Stops", default=3, min=2, max=MAX_STOPS)
    stop0_color: FloatVectorProperty(name="Color 1", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=_DEFAULT_STOPS[0][0])
    stop0_pos: FloatProperty(name="Position 1", default=0.0, min=0.0, max=1.0)
    stop1_color: FloatVectorProperty(name="Color 2", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=_DEFAULT_STOPS[1][0])
    stop1_pos: FloatProperty(name="Position 2", default=0.5, min=0.0, max=1.0)
    stop2_color: FloatVectorProperty(name="Color 3", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=_DEFAULT_STOPS[2][0])
    stop2_pos: FloatProperty(name="Position 3", default=1.0, min=0.0, max=1.0)
    stop3_color: FloatVectorProperty(name="Color 4", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=_DEFAULT_STOPS[3][0])
    stop3_pos: FloatProperty(name="Position 4", default=0.75, min=0.0, max=1.0)
    stop4_color: FloatVectorProperty(name="Color 5", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=_DEFAULT_STOPS[4][0])
    stop4_pos: FloatProperty(name="Position 5", default=0.85, min=0.0, max=1.0)
    stop5_color: FloatVectorProperty(name="Color 6", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=_DEFAULT_STOPS[5][0])
    stop5_pos: FloatProperty(name="Position 6", default=1.0, min=0.0, max=1.0)

    def draw_buttons(self, context, layout):
        layout.prop(self, "preset", text="")
        layout.prop(self, "source")
        layout.prop(self, "interpolation", text="")
        layout.prop(self, "color_space", text="")
        row = layout.row()
        row.prop(self, "reverse")
        row.prop(self, "preserve_alpha", text="Alpha")
        if self.preset == 'CUSTOM':
            layout.prop(self, "stop_count")
            for i in range(self.stop_count):
                row = layout.row(align=True)
                row.prop(self, "stop%d_color" % i, text="")
                row.prop(self, "stop%d_pos" % i, text="")

    # -- shared -------------------------------------------------------------
    def stops(self):
        """Sorted [(position, linear rgb)] of the active gradient."""
        if self.preset == 'CUSTOM':
            st = [(float(getattr(self, "stop%d_pos" % i)),
                   [float(v) for v in getattr(self, "stop%d_color" % i)])
                  for i in range(self.stop_count)]
        else:
            st = [(p, list(np_color.srgb_to_linear(np.array(c, F32)))) for p, c in
                  PRESETS[self.preset][1]]
        st.sort(key=lambda s: s[0])
        return st

    def _params(self):
        st = self.stops()
        pos = np.array([s[0] for s in st], F32)
        col = np.array([s[1] for s in st], F32)
        space = _IDX["space"](self.color_space)
        if space == 1:
            col = np_color.linear_to_oklab(col)
        return dict(pos=pos, col=col, src=_IDX["src"](self.source),
                    interp=_IDX["interp"](self.interpolation), space=space,
                    reverse=bool(self.reverse), alpha=bool(self.preserve_alpha))

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        c = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                           default=(0.5, 0.5, 0.5, 1.0)))
        fac = self.in_image_array(inputs, "Fac", ctx.shape, 1, default=1.0)
        out[...] = gradient_map(c, fac, **self._params())

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        p = self._params()
        n = len(p["pos"])
        stop = lambda i: [float(v) for v in p["col"][min(i, n - 1)]] + [float(p["pos"][min(i, n - 1)])]
        uniforms = {"g_n": ("int", n), "g_src": ("int", p["src"]), "g_interp": ("int", p["interp"]),
                    "g_space": ("int", p["space"]), "g_rev": ("int", int(p["reverse"])),
                    "g_alpha": ("int", int(p["alpha"]))}
        for i in range(MAX_STOPS):
            uniforms["g_s%d" % i] = ("vec4", stop(i))
        inv = _inv_widths(p["pos"])
        for i in range(MAX_STOPS - 1):
            uniforms["g_i%d" % i] = ("float", float(inv[i]) if i < len(inv) else 0.0)
        lab_gpu.pointwise(
            _BODY, {"Color": dst},
            inputs={
                "Image": ("color", self.in_texture_or_value(inputs, "Image",
                                                            (0.5, 0.5, 0.5, 1.0))),
                "Fac": ("float", self.in_texture_or_value(inputs, "Fac", 1.0)),
            },
            uniforms=uniforms, libs=("color",))


def _inv_widths(pos):
    """1 / segment width (float32 division), 0 for zero-width segments."""
    d = (pos[1:] - pos[:-1]).astype(F32)
    return np.where(d > 0, F32(1.0) / np.where(d > 0, d, F32(1.0)), F32(0.0)).astype(F32)


def gradient_map(c, fac, pos, col, src, interp, space, reverse, alpha):
    """Reference / CPU implementation on a premultiplied (H, W, 4) float32 image. ``col`` holds the
    stop colours already in the interpolation space."""
    a = c[..., 3]
    s = np.where((a > 0)[..., None], c[..., :3] / np.where(a > 0, a, F32(1.0))[..., None],
                 F32(0.0)).astype(F32)
    if src == 0:
        t = np_color.luma(s)
    elif src == 1:
        t = np_color.linear_to_oklab(np.maximum(s, F32(0.0)))[..., 0]
    elif src in (2, 3, 4):
        t = s[..., src - 2]
    elif src == 5:
        t = s.max(axis=-1)
    else:
        t = a
    t = np.clip(t, 0, 1).astype(F32)
    if reverse:
        t = F32(1.0) - t
    inv = _inv_widths(pos)
    res = np.broadcast_to(col[0], c.shape[:2] + (3,)).astype(F32)
    for k in range(len(pos) - 1):
        p0, p1 = pos[k], pos[k + 1]
        if inv[k] > 0:
            f = np.clip((t - p0) * inv[k], 0, 1).astype(F32)
        else:
            f = (t >= p1).astype(F32)
        if interp == 2:
            f = (t >= p1).astype(F32)
        elif interp == 1:
            f = f * f * (F32(3.0) - F32(2.0) * f)
        f = f[..., None]
        res = res * (F32(1.0) - f) + col[k + 1] * f
    if space == 1:
        res = np.maximum(np_color.oklab_to_linear(res), F32(0.0))
    if alpha:
        out = np.concatenate([res * a[..., None], a[..., None]], axis=-1)
    else:
        out = np.concatenate([res, np.ones_like(a)[..., None]], axis=-1)
    fac = np.asarray(fac, F32)
    return (c * (F32(1.0) - fac) + out * fac).astype(F32)


NODE_CLASSES = [CompositorNodeLabGradientMap]
