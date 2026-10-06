# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Auto Levels: contrast stretch from the image's own percentiles.

The statistics are taken on the straight (un-premultiplied) colour of every pixel with alpha > 0.
Per channel ``y = (c - lo) / (hi - lo)`` where ``lo`` / ``hi`` are the ``Low`` / ``High``
percentiles (histogram of 256 bins over [channel min, channel max], interpolated inside the bin:
percentiles are accurate to (max - min) / 256). Modes:

* Luminance (default): ``lo`` / ``hi`` come from the Rec. 709 luminance and are used for all three
  channels, so hues are preserved (a monotone, shared affine map).
* Per channel: each of R, G, B gets its own ``lo`` / ``hi`` (also removes colour casts).

``Clamp`` limits the result to [0, 1]. ``Auto Gamma`` then solves (bisection on the histogram's
bin centres) for the gamma that makes the mean of the stretched, clamped values ``y ** gamma``
equal ``Target`` (per channel in per-channel mode); gamma needs non-negative values so it also
clamps below 0. If the target cannot be reached (e.g. a flat image) gamma stays 1.
``Fac`` blends the original (0) and the corrected image (1); alpha is unchanged.
Degenerate ranges (hi <= lo) leave that channel untouched. Parameters are computed on the host
from the reductions in ``lib/reduce.py`` (CPU: numpy, GPU: compute passes) and applied per pixel.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from ..lib import gpu as lab_gpu, reduce
from ..lib.node import In, LabNode, Out

MENU = "Utility"

F32 = np.float32

_BODY = """
    vec4 s = in_Image(texel);
    float a = s.a;
    vec3 c = (a > 0.0) ? s.rgb / a : vec3(0.0);
    vec3 y = (c - al_lo) * al_scale;
    if (al_clamp != 0) {
      y = clamp(y, vec3(0.0), vec3(1.0));
    }
    if (al_use_gamma != 0) {
      y = pow(max(y, vec3(0.0)), al_gamma);
    }
    vec3 r = y * a;
    out_Color = vec4(s.rgb + (r - s.rgb) * in_Fac(texel), s.a);
"""


def apply_cpu(img, fac, lo, scale, gamma, clamp, use_gamma):
    """numpy twin of ``_BODY``. lo / scale / gamma: float32 3-vectors."""
    a = img[..., 3:4]
    with np.errstate(all="ignore"):
        c = np.where(a > 0, img[..., :3] / np.where(a > 0, a, F32(1.0)), F32(0.0)).astype(F32)
        y = (c - lo) * scale
        if clamp:
            y = np.clip(y, F32(0.0), F32(1.0))
        if use_gamma:
            y = np.power(np.maximum(y, F32(0.0)), gamma)
        r = y * a
        out = np.empty_like(img)
        out[..., :3] = img[..., :3] + (r - img[..., :3]) * fac
    out[..., 3] = img[..., 3]
    return out.astype(F32)


class CompositorNodeLabAutoLevels(LabNode, bpy.types.CompositorNode):
    '''Stretch contrast so the low / high percentiles of the image map to black / white'''
    bl_idname = "CompositorNodeLabAutoLevels"
    bl_label = "Auto Levels"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("Fac", "FACTOR", 1.0),
        Out("Color", "COLOR"),
    ]
    PROPS = ["mode", "low", "high", "clamp", "auto_gamma", "target"]

    mode: EnumProperty(
        name="Mode", default='LUMINANCE',
        items=[('LUMINANCE', "Luminance", "One stretch from the luminance, shared by R, G and B"),
               ('PER_CHANNEL', "Per Channel", "Stretch R, G and B independently")])
    low: FloatProperty(name="Low", default=1.0, min=0.0, max=100.0, subtype='PERCENTAGE',
                       description="Percentile that becomes black")
    high: FloatProperty(name="High", default=99.0, min=0.0, max=100.0, subtype='PERCENTAGE',
                        description="Percentile that becomes white")
    clamp: BoolProperty(name="Clamp", default=True, description="Limit the result to 0..1")
    auto_gamma: BoolProperty(name="Auto Gamma", default=False,
                             description="Adjust gamma so the mean luminance reaches the target")
    target: FloatProperty(name="Target", default=0.5, min=0.01, max=0.99,
                          description="Mean luminance wanted after Auto Gamma")

    # -- analysis (shared) --------------------------------------------------
    def _solve_gamma(self, hist, cmin, cmax, plo, phi):
        """Gamma g with mean(clip((v - plo) / (phi - plo)) ** g) = target, from the histogram
        (bin centres), by bisection (the mean falls monotonically with g). 1 if unreachable."""
        def stretch(v):
            return np.clip((v - plo) / (phi - plo), 0.0, 1.0)

        def mean_at(g):
            return reduce.hist_mean(hist, cmin, cmax, lambda v: stretch(v) ** g)

        lo_g, hi_g = 0.05, 20.0
        if not mean_at(hi_g) < self.target < mean_at(lo_g):
            return 1.0
        for _ in range(48):
            mid = 0.5 * (lo_g + hi_g)
            if mean_at(mid) > self.target:
                lo_g = mid
            else:
                hi_g = mid
        return 0.5 * (lo_g + hi_g)

    def params(self, stats_fn, hist_fn):
        """-> (lo, scale, gamma): float32 3-vectors; identity if the image has no pixels."""
        lo = np.zeros(3)
        scale = np.ones(3)
        gamma = np.ones(3)
        st = stats_fn(True)
        if st.count == 0:
            return lo.astype(F32), scale.astype(F32), gamma.astype(F32)
        linked = self.mode == 'LUMINANCE'
        chans = [3] if linked else [0, 1, 2]
        for ch in chans:
            cmin = float(F32(st.lmin if ch == 3 else st.min[ch]))
            cmax = float(F32(st.lmax if ch == 3 else st.max[ch]))
            hist = hist_fn(ch, cmin, cmax)
            plo = reduce.percentile(hist, cmin, cmax, self.low)
            phi = reduce.percentile(hist, cmin, cmax, self.high)
            sel = slice(0, 3) if linked else slice(ch, ch + 1)
            if phi > plo:
                lo[sel] = plo
                scale[sel] = 1.0 / (phi - plo)
                if self.auto_gamma:
                    gamma[sel] = self._solve_gamma(hist, cmin, cmax, plo, phi)
        return lo.astype(F32), scale.astype(F32), gamma.astype(F32)

    # -- CPU ----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Color")
        if out is None:
            return
        img = np.ascontiguousarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                                       default=(0.5, 0.5, 0.5, 1.0)))
        fac = self.in_image_array(inputs, "Fac", ctx.shape, 1, default=1.0)
        lo, scale, gamma = self.params(
            lambda straight: reduce.stats_cpu(img, straight),
            lambda ch, a, b: reduce.hist_cpu(img, ch, a, b, True))
        out[...] = apply_cpu(img, fac, lo, scale, gamma, self.clamp, self.auto_gamma)

    # -- GPU ----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Color")
        if dst is None:
            return
        src = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        if lab_gpu.is_texture(src):
            lo, scale, gamma = self.params(
                lambda straight: reduce.stats_gpu(src, straight),
                lambda ch, a, b: reduce.hist_gpu(src, ch, a, b, True))
        else:
            img = self.in_image_array(inputs, "Image", (1, 1), 4, default=(0.5, 0.5, 0.5, 1.0))
            lo, scale, gamma = self.params(
                lambda straight: reduce.stats_cpu(img, straight),
                lambda ch, a, b: reduce.hist_cpu(img, ch, a, b, True))
        lab_gpu.pointwise(
            _BODY, {"Color": dst},
            inputs={"Image": ("color", src),
                    "Fac": ("float", self.in_texture_or_value(inputs, "Fac", 1.0))},
            uniforms={
                "al_lo": ("vec3", tuple(float(v) for v in lo)),
                "al_scale": ("vec3", tuple(float(v) for v in scale)),
                "al_gamma": ("vec3", tuple(float(v) for v in gamma)),
                "al_clamp": ("int", int(self.clamp)),
                "al_use_gamma": ("int", int(self.auto_gamma)),
            })


NODE_CLASSES = [CompositorNodeLabAutoLevels]
