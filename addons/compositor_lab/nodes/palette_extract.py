# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Palette Extract: k-means colour palette of an image.

Pipeline (identical on CPU and GPU):

1. Sampling. The image is point-sampled on a grid of at most 128 x 128 samples (sample ``i`` of
   ``n`` along an axis of ``s`` pixels reads pixel ``(i * s + s // 2) // n``; images up to 128 px
   are used completely). On the GPU this is a compute pass and the (small) grid is read back.
2. K-means on the host (numpy, float64), on the un-premultiplied colour of the samples with
   weight = alpha (transparent samples are ignored), in OKLab (default) or linear RGB: k-means++
   seeding from a ``numpy.random.Generator(Seed)``, then at most 24 Lloyd iterations. **The
   clustering itself therefore runs on the CPU in both modes**; the GPU does the sampling and the
   per-pixel work below. Because both modes see the same samples the palette is bit-identical.
3. Palette colours are the alpha-weighted mean *linear* colour of each cluster, sorted by total
   weight (largest first). Fewer distinct colours than ``Colors`` give fewer palette entries:
   unused ``Color N`` outputs are black (alpha 1).
4. Per pixel (GPU compute / numpy): ``Quantized`` = the palette entry nearest in the chosen colour
   space (ties: lowest index), keeping the pixel's alpha (premultiplied); ``Swatch`` = the palette
   as equal-width vertical bars across the frame (black if the palette is empty).
"""

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty

from ..lib import gpu as lab_gpu, np_color
from ..lib.node import In, LabNode, Out

MENU = "Utility"

F32 = np.float32
MAX_COLORS = 8
MAX_SAMPLES = 128
ITERATIONS = 24


def sample_grid(w, h):
    """(ix, iy) pixel indices of the sample grid for a w x h image."""
    gw, gh = min(w, MAX_SAMPLES), min(h, MAX_SAMPLES)
    ix = (np.arange(gw, dtype=np.int64) * w + w // 2) // gw
    iy = (np.arange(gh, dtype=np.int64) * h + h // 2) // gh
    return ix, iy


def to_space(rgb, space):
    rgb = np.asarray(rgb, F32)
    return np_color.linear_to_oklab(rgb) if space == 'OKLAB' else rgb


def kmeans_palette(samples, k, space, seed):
    """samples: (N, 4) premultiplied RGBA. Returns (n, 3) float32 linear palette colours, sorted
    by weight descending (n <= k; empty if nothing is visible)."""
    s = np.asarray(samples, F32).reshape(-1, 4)
    s = s[s[:, 3] > 0]
    if s.shape[0] == 0:
        return np.zeros((0, 3), F32)
    wgt = s[:, 3].astype(np.float64)
    lin = (s[:, :3] / s[:, 3:4]).astype(F32)
    pts = to_space(lin, space).astype(np.float64)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    total = wgt.sum()

    def pick(p):
        cum = np.cumsum(p)
        return int(min(np.searchsorted(cum, rng.random() * cum[-1], side="right"),
                       len(p) - 1))

    centres = [pts[pick(wgt)]]
    d2 = ((pts - centres[0]) ** 2).sum(axis=1)
    while len(centres) < k:
        p = wgt * d2
        if p.sum() <= 1e-12 * total:
            break
        centres.append(pts[pick(p)])
        d2 = np.minimum(d2, ((pts - centres[-1]) ** 2).sum(axis=1))
    cen = np.array(centres)

    def assign(c):
        return np.argmin(((pts[:, None, :] - c[None, :, :]) ** 2).sum(axis=2), axis=1)

    for _ in range(ITERATIONS):
        lab = assign(cen)
        new = cen.copy()
        for j in range(len(cen)):
            m = lab == j
            wj = wgt[m].sum()
            if wj > 0:
                new[j] = (pts[m] * wgt[m, None]).sum(axis=0) / wj
        moved = np.abs(new - cen).max()
        cen = new
        if moved < 1e-9:
            break
    lab = assign(cen)
    colours, weights = [], []
    for j in range(len(cen)):
        m = lab == j
        wj = wgt[m].sum()
        if wj > 0:
            colours.append((lin[m].astype(np.float64) * wgt[m, None]).sum(axis=0) / wj)
            weights.append(wj)
    order = np.argsort(-np.array(weights), kind="stable")
    return np.array(colours, np.float64)[order].astype(F32)


def quantize_cpu(img, palette, space):
    """numpy twin of the GPU quantize body. img (H, W, 4) premultiplied."""
    n = len(palette)
    out = np.zeros_like(img)
    if n == 0:
        return out
    a = img[..., 3:4]
    with np.errstate(all="ignore"):
        c = np.where(a > 0, img[..., :3] / np.where(a > 0, a, F32(1.0)), F32(0.0)).astype(F32)
    q = to_space(c, space)
    pal_space = to_space(palette, space)
    best = np.zeros(img.shape[:2], np.int64)
    bd = np.full(img.shape[:2], 1e30, F32)
    for i in range(n):
        d = q - pal_space[i]
        dd = (d * d).sum(axis=-1, dtype=F32)
        better = dd < bd
        best[better] = i
        bd = np.where(better, dd, bd)
    vis = a[..., 0] > 0
    out[..., :3] = np.where(vis[..., None], palette[best] * a, F32(0.0))
    out[..., 3] = np.where(vis, a[..., 0], F32(0.0))
    return out


def swatch_cpu(shape, palette):
    h, w = shape
    out = np.zeros((h, w, 4), F32)
    out[..., 3] = 1.0
    n = len(palette)
    if n:
        idx = np.minimum((np.arange(w) * n) // w, n - 1)
        out[..., :3] = palette[idx][None, :, :]
    return out


_QUANT_BODY = """
    vec4 s = in_Image(texel);
    float a = s.a;
    vec3 c = (a > 0.0) ? s.rgb / a : vec3(0.0);
    vec3 q = (pe_space != 0) ? lab_linear_to_oklab(c) : c;
    int best = 0;
    float bd = 1.0e30;
    for (int i = 0; i < pe_n; i++) {
      vec3 d = q - in_Pal(ivec2(i, 1)).rgb;
      float dd = dot(d, d);
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    out_Quantized = (a > 0.0 && pe_n > 0) ? vec4(in_Pal(ivec2(best, 0)).rgb * a, a) : vec4(0.0);
"""

_SWATCH_BODY = """
    int bar = min((texel.x * pe_n) / res.x, pe_n - 1);
    out_Swatch = vec4(in_Pal(ivec2(bar, 0)).rgb, 1.0);
"""

_GRID_BODY = """
    ivec2 sp = ivec2((texel.x * pe_sw + pe_sw / 2) / res.x, (texel.y * pe_sh + pe_sh / 2) / res.y);
    out_Samples = in_Image(sp);
"""

_COLOR_NAMES = ["Color %d" % (i + 1) for i in range(MAX_COLORS)]


class CompositorNodeLabPaletteExtract(LabNode, bpy.types.CompositorNode):
    '''Extract a palette of up to 8 dominant colours with k-means'''
    bl_idname = "CompositorNodeLabPaletteExtract"
    bl_label = "Palette Extract"

    SOCKETS = [
        In("Image", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        Out("Swatch", "COLOR"),
        Out("Quantized", "COLOR"),
    ] + [Out(n, "COLOR", single=True) for n in _COLOR_NAMES]
    PROPS = ["count", "space", "seed"]

    count: IntProperty(name="Colors", default=5, min=1, max=MAX_COLORS,
                       description="Number of palette colours (clusters)")
    space: EnumProperty(
        name="Space", default='OKLAB',
        items=[('OKLAB', "OKLab", "Cluster in perceptual OKLab space"),
               ('LINEAR', "Linear RGB", "Cluster in scene-linear RGB")])
    seed: IntProperty(name="Seed", default=0, soft_min=0, soft_max=1000,
                    description="Seed of the k-means++ initialisation")

    def _write_colors(self, outputs, palette):
        for i, name in enumerate(_COLOR_NAMES):
            arr = self.out_single(outputs, name)
            if arr is not None:
                arr[:] = (tuple(float(v) for v in palette[i]) + (1.0,)) if i < len(palette) \
                    else (0.0, 0.0, 0.0, 1.0)

    def _palette_texture(self, palette):
        import gpu

        data = np.zeros((2, MAX_COLORS, 4), F32)
        n = len(palette)
        if n:
            data[0, :n, :3] = palette
            data[1, :n, :3] = to_space(palette, self.space)
        buf = gpu.types.Buffer('FLOAT', data.size, data.ravel().tolist())
        return gpu.types.GPUTexture((MAX_COLORS, 2), format='RGBA32F', data=buf)

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        v = inputs.get("Image")
        if v is None or isinstance(v, (int, float, bool, tuple, list)):
            img = self.in_image_array(inputs, "Image", ctx.shape if ctx.has_context else (1, 1),
                                      4, default=(0.0, 0.0, 0.0, 1.0))
        else:
            a = np.asarray(v)
            img = self.in_image_array(inputs, "Image", a.shape[:2], 4)
        h, w = img.shape[:2]
        ix, iy = sample_grid(w, h)
        palette = kmeans_palette(img[iy[:, None], ix[None, :]], self.count, self.space, self.seed)
        self._write_colors(outputs, palette)
        sw = self.out_array(outputs, "Swatch")
        if sw is not None:
            sw[...] = swatch_cpu(sw.shape[:2], palette)
        q = self.out_array(outputs, "Quantized")
        if q is not None:
            src = self.in_image_array(inputs, "Image", q.shape[:2], 4,
                                      default=(0.0, 0.0, 0.0, 1.0))
            q[...] = quantize_cpu(np.ascontiguousarray(src), palette, self.space)

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        import gpu

        sw = self.out_texture(outputs, "Swatch")
        q = self.out_texture(outputs, "Quantized")
        src = self.in_texture_or_value(inputs, "Image", (0.0, 0.0, 0.0, 1.0))
        if lab_gpu.is_texture(src):
            w, h = int(src.width), int(src.height)
        elif sw is not None or q is not None:
            w, h = int((sw or q).width), int((sw or q).height)
        else:
            w, h = 1, 1
        gw, gh = min(w, MAX_SAMPLES), min(h, MAX_SAMPLES)
        grid = gpu.types.GPUTexture((gw, gh), format='RGBA32F')
        lab_gpu.pointwise(_GRID_BODY, {"Samples": grid}, inputs={"Image": ("color", src)},
                          uniforms={"pe_sw": ("int", w), "pe_sh": ("int", h)})
        samples = np.array(grid.read().to_list(), F32).reshape(gh, gw, 4)
        palette = kmeans_palette(samples, self.count, self.space, self.seed)
        self._write_colors(outputs, palette)
        if sw is None and q is None:
            return
        pal = self._palette_texture(palette)
        uniforms = {"pe_n": ("int", len(palette)), "pe_space": ("int", int(self.space == 'OKLAB'))}
        if sw is not None:
            lab_gpu.pointwise(_SWATCH_BODY, {"Swatch": sw}, inputs={"Pal": ("color", pal)},
                              uniforms={"pe_n": uniforms["pe_n"]})
        if q is not None:
            lab_gpu.pointwise(_QUANT_BODY, {"Quantized": q},
                              inputs={"Image": ("color", src), "Pal": ("color", pal)},
                              uniforms=uniforms, libs=("color",))


NODE_CLASSES = [CompositorNodeLabPaletteExtract]
