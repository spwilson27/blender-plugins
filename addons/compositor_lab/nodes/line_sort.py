# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Line Sort: reorder whole rows (or columns) of the image by a per-line statistic.

Each line gets one integer key; the lines are stably sorted by (band, key, index) and moved as a
unit, so the output is a permutation of the input's lines (pixels are never altered).

Statistics (on the colours as given, premultiplied; values are clamped to [0, 4] and quantised to
1/1024 before they are summed, so the sums are exact integers independent of summation order):
  Luminance / Hue / Saturation / Red / Green / Blue / Alpha: the line's mean (= sum, every line
      has the same length) of that per-pixel value. Hue is the plain mean of HSV hue in [0, 1)
      (no circular averaging); luminance is Rec. 709.
  Variance: variance of the luminance along the line, computed in float32 from the exact integer
      sums with a fixed sequence of rounded operations.
Descending sorts by the bitwise complement of the key; ties keep their original order (stable).
``Band Size`` K > 0 sorts only within consecutive groups of K lines (0 = the whole image).

GPU: three dispatches. (1) one thread per line sums its pixels with integer arithmetic and writes
the key; (2) one thread per line computes its rank = band start + number of lines of the band that
sort before it (parallel counting / rank sort, O(K) per line, exactly the stable order); (3) every
pixel is written to its line's new position. CPU and GPU compute identical keys and ranks, so the
outputs are bit-identical.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty

from ..lib import distance, glsl, gpu as lab_gpu
from ..lib.node import In, LabNode, Out

MENU = "Utility"

F32 = np.float32
U32 = np.uint32

_STATS = [
    ('LUMINANCE', "Mean Luminance", "Rec. 709 luminance"),
    ('HUE', "Mean Hue", "Mean HSV hue"),
    ('SATURATION', "Mean Saturation", "Mean HSV saturation"),
    ('VARIANCE', "Luminance Variance", "Variance of the luminance along the line"),
    ('RED', "Sum of Red", ""),
    ('GREEN', "Sum of Green", ""),
    ('BLUE', "Sum of Blue", ""),
    ('ALPHA', "Sum of Alpha", ""),
]
_STAT_INDEX = {"LUMINANCE": 0, "HUE": 1, "SATURATION": 2, "VARIANCE": 3, "RED": 4, "GREEN": 5,
               "BLUE": 6, "ALPHA": 7}
_VAR = 3

_GLSL_COMMON = r'''
float ls_stat_value(vec4 c, int stat)
{
  if (stat == 0 || stat == 3) {
    float a = lab_rnd32(0.2126 * c.r);
    float b = lab_rnd32(0.7152 * c.g);
    float d = lab_rnd32(0.0722 * c.b);
    return lab_rnd32(a + b) + d;
  }
  if (stat == 4) {
    return c.r;
  }
  if (stat == 5) {
    return c.g;
  }
  if (stat == 6) {
    return c.b;
  }
  if (stat == 7) {
    return c.a;
  }
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float delta = mx - mn;
  if (stat == 2) {
    return (mx > 0.0) ? lab_div_exact(delta, mx) : 0.0;
  }
  if (!(delta > 0.0)) {
    return 0.0;
  }
  float h;
  if (mx == c.r) {
    h = lab_div_exact(c.g - c.b, delta);
    if (h < 0.0) {
      h = h + 6.0;
    }
  }
  else if (mx == c.g) {
    h = lab_div_exact(c.b - c.r, delta) + 2.0;
  }
  else {
    h = lab_div_exact(c.r - c.g, delta) + 4.0;
  }
  h = lab_div_exact(h, 6.0);
  return (h >= 1.0) ? 0.0 : h;
}

uint ls_quant(float x)
{
  if (!(x >= 0.0)) {
    x = 0.0;
  }
  x = min(x, 4.0);
  return uint(floor(x * 1024.0 + 0.5));
}
'''

_GLSL_KEYS = r'''
void main()
{
  int line = int(gl_GlobalInvocationID.x);
  if (line >= ls_lines) {
    return;
  }
  uint s1 = 0u;
  uint hi = 0u;
  uint lo = 0u;
  for (int i = 0; i < ls_len; ++i) {
    ivec2 p = (ls_vert != 0) ? ivec2(line, i) : ivec2(i, line);
    uint q = ls_quant(ls_stat_value(texelFetch(src, p, 0), ls_stat));
    s1 += q;
    if (ls_stat == 3) {
      uint q2 = q * q;
      hi += q2 >> 12u;
      lo += q2 & 4095u;
    }
  }
  uint key = s1;
  if (ls_stat == 3) {
    float nf = float(ls_len);
    float mean = lab_div_exact(float(s1), nf);
    float s2 = lab_rnd32(lab_rnd32(float(hi) * 4096.0) + float(lo));
    float e2 = lab_div_exact(s2, nf);
    float mm = lab_rnd32(mean * mean);
    float v = lab_rnd32(e2 - mm);
    if (!(v > 0.0)) {
      v = 0.0;
    }
    key = floatBitsToUint(v);
  }
  imageStore(keys, ivec2(line, 0), uvec4(key, 0u, 0u, 0u));
}
'''

_GLSL_RANK = r'''
void main()
{
  int line = int(gl_GlobalInvocationID.x);
  if (line >= ls_lines) {
    return;
  }
  int band = (ls_band > 0) ? ls_band : ls_lines;
  int lo = (line / band) * band;
  int hi = min(lo + band, ls_lines);
  uint k = imageLoad(keys, ivec2(line, 0)).x;
  if (ls_desc != 0) {
    k = ~k;
  }
  int cnt = 0;
  for (int j = lo; j < hi; ++j) {
    uint kj = imageLoad(keys, ivec2(j, 0)).x;
    if (ls_desc != 0) {
      kj = ~kj;
    }
    if (kj < k || (kj == k && j < line)) {
      ++cnt;
    }
  }
  imageStore(ranks, ivec2(line, 0), uvec4(uint(lo + cnt), 0u, 0u, 0u));
}
'''

_GLSL_SCATTER = r'''
void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= ls_w || p.y >= ls_h) {
    return;
  }
  int line = (ls_vert != 0) ? p.x : p.y;
  int r = int(imageLoad(ranks, ivec2(line, 0)).x);
  ivec2 q = (ls_vert != 0) ? ivec2(r, p.y) : ivec2(p.x, r);
  imageStore(dst, q, texelFetch(src, p, 0));
}
'''

_INT_UNIFORMS = ("ls_lines", "ls_len", "ls_vert", "ls_stat", "ls_band", "ls_desc", "ls_w", "ls_h",
                 "lab_zero")
_KEY_LOCAL = (64, 1, 1)


def _build(kind, fmt):
    import gpu

    scatter = kind == "scatter"
    info = lab_gpu.create_info((16, 16, 1) if scatter else _KEY_LOCAL)
    uimg = dict(qualifiers={'READ', 'WRITE'})
    if kind in ("keys", "scatter"):
        info.sampler(0, 'FLOAT_2D', "src")
    if kind == "keys":
        info.image(0, 'RGBA32UI', 'UINT_2D', "keys", **uimg)
    elif kind == "rank":
        info.image(0, 'RGBA32UI', 'UINT_2D', "keys", **uimg)
        info.image(1, 'RGBA32UI', 'UINT_2D', "ranks", **uimg)
    else:
        info.image(0, 'RGBA32UI', 'UINT_2D', "ranks", **uimg)
        info.image(1, fmt, 'FLOAT_2D', "dst", qualifiers={'WRITE'})
    for name in _INT_UNIFORMS:
        info.push_constant('INT', name)
    body = {"keys": _GLSL_KEYS, "rank": _GLSL_RANK, "scatter": _GLSL_SCATTER}[kind]
    head = glsl.resolve("exact") + _GLSL_COMMON if kind == "keys" else ""
    info.compute_source(head + body)
    return gpu.shader.create_from_info(info)


def _shader(kind, fmt):
    return lab_gpu.get_shader(("line_sort", kind, fmt), lambda: _build(kind, fmt))


def _set_ints(shader, values):
    for name in _INT_UNIFORMS:
        lab_gpu._set(shader.uniform_int, name, int(values.get(name, 0)))


# ---------------------------------------------------------------------------
# numpy twin
# ---------------------------------------------------------------------------

def stat_values(c, stat):
    """Per-pixel statistic (float32, same operation order as ``ls_stat_value``)."""
    c = np.asarray(c, F32)
    r, g, b, a = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    if stat in (0, 3):
        return (F32(0.2126) * r + F32(0.7152) * g) + F32(0.0722) * b
    if stat == 4:
        return r
    if stat == 5:
        return g
    if stat == 6:
        return b
    if stat == 7:
        return a
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    delta = mx - mn
    pos = delta > 0
    if stat == 2:
        return np.where(mx > 0, delta / np.where(mx > 0, mx, F32(1)), F32(0)).astype(F32)
    d = np.where(pos, delta, F32(1))
    hr = (g - b) / d
    hr = np.where(hr < 0, hr + F32(6), hr)
    hg = (b - r) / d + F32(2)
    hb = (r - g) / d + F32(4)
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)).astype(F32) / F32(6)
    h = np.where(h >= 1, F32(0), h)
    return np.where(pos, h, F32(0)).astype(F32)


def quantise(x):
    x = np.where(x >= 0, x, F32(0))
    x = np.minimum(x, F32(4))
    return np.floor(x * F32(1024) + F32(0.5)).astype(np.int64)


def line_keys(c, stat, vertical):
    """uint32 key per line (a row, or a column if ``vertical``) of an (H, W, 4) image."""
    q = quantise(stat_values(c, stat))
    axis = 0 if vertical else 1
    s1 = q.sum(axis=axis)
    if stat != _VAR:
        return s1.astype(U32)
    q2 = q * q
    hi = (q2 >> 12).sum(axis=axis)
    lo = (q2 & 4095).sum(axis=axis)
    n = F32(q.shape[axis])
    mean = s1.astype(F32) / n
    s2 = hi.astype(F32) * F32(4096) + lo.astype(F32)
    e2 = s2 / n
    v = e2 - mean * mean
    v = np.where(v > 0, v, F32(0)).astype(F32)
    return v.view(U32)


def line_order(keys, desc, band):
    """order[o] = index of the input line placed at output position o (stable)."""
    n = keys.shape[0]
    k = ~keys if desc else keys
    band = band if band > 0 else n
    idx = np.arange(n)
    return np.lexsort((idx, k, idx // band))


class CompositorNodeLabLineSort(LabNode, bpy.types.CompositorNode):
    '''Sort whole rows or columns of the image by a statistic (mean luminance, hue, variance...)'''
    bl_idname = "CompositorNodeLabLineSort"
    bl_label = "Line Sort"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Image", "COLOR"),
    ]
    PROPS = ["direction", "statistic", "descending", "band_size"]

    direction: EnumProperty(name="Sort", default='ROWS', items=[
        ('ROWS', "Rows", "Reorder rows (horizontal lines)"),
        ('COLUMNS', "Columns", "Reorder columns (vertical lines)"),
    ])
    statistic: EnumProperty(name="By", items=_STATS, default='LUMINANCE')
    descending: BoolProperty(name="Descending", default=False)
    band_size: IntProperty(name="Band Size", default=0, min=0, soft_max=256, subtype='PIXEL',
                           description="Sort only within groups of this many lines (0 = all)")

    def draw_buttons(self, context, layout):
        layout.prop(self, "direction", expand=True)
        layout.prop(self, "statistic", text="")
        layout.prop(self, "descending")
        layout.prop(self, "band_size")

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        a = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                           default=(0.5, 0.5, 0.5, 1.0)))
        vertical = self.direction == 'COLUMNS'
        keys = line_keys(a, _STAT_INDEX[self.statistic], vertical)
        order = line_order(keys, self.descending, self.band_size)
        out[...] = a[:, order] if vertical else a[order]

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        src = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        if not lab_gpu.is_texture(src):
            vals = lab_gpu._as_floats(src, 4, fill_alpha=True)
            dst.clear(format='FLOAT', value=vals)
            return
        import gpu

        w, h = int(dst.width), int(dst.height)
        vertical = self.direction == 'COLUMNS'
        lines, length = (w, h) if vertical else (h, w)
        keys = distance.scratch(lines, 1, "ls_keys", "RGBA32UI")
        ranks = distance.scratch(lines, 1, "ls_ranks", "RGBA32UI")
        ints = dict(ls_lines=lines, ls_len=length, ls_vert=int(vertical),
                    ls_stat=_STAT_INDEX[self.statistic], ls_band=int(self.band_size),
                    ls_desc=int(self.descending), ls_w=w, ls_h=h, lab_zero=0)
        groups = (lines + _KEY_LOCAL[0] - 1) // _KEY_LOCAL[0]

        sh = _shader("keys", None)
        _set_ints(sh, ints)
        sh.uniform_sampler("src", src)
        sh.image("keys", keys)
        gpu.compute.dispatch(sh, groups, 1, 1)

        sh = _shader("rank", None)
        _set_ints(sh, ints)
        sh.image("keys", keys)
        sh.image("ranks", ranks)
        gpu.compute.dispatch(sh, groups, 1, 1)

        sh = _shader("scatter", dst.format)
        _set_ints(sh, ints)
        sh.uniform_sampler("src", src)
        sh.image("ranks", ranks)
        sh.image("dst", dst)
        lab_gpu.dispatch_grid(sh, w, h, (16, 16))


NODE_CLASSES = [CompositorNodeLabLineSort]
