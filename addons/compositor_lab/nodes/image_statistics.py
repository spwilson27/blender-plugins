# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Image Statistics: single-value outputs describing the whole image.

Statistics are of the colour values as stored (scene-linear, premultiplied), over every pixel.
Outputs (F3 single values): Min / Max / Mean (colours, per channel, alpha 1), Luminance Mean,
Std Dev (population standard deviation of Rec. 709 luminance) and Percentile (the ``Percentile``
property, in percent, of the luminance, from a 256-bin histogram over [luma min, luma max] with
interpolation inside the bin: the error is below (max - min) / 256).

CPU: float64 accumulation. GPU: ``lib/reduce.py`` (tile passes, float32 sums with Chan merging).
Min / Max agree exactly; means agree to ~1e-6 relative.
"""

import bpy
import numpy as np
from bpy.props import FloatProperty

from ..lib import gpu as lab_gpu, reduce
from ..lib.node import In, LabNode, Out

MENU = "Utility"

F32 = np.float32


def source_array(node, inputs):
    """The Image input as a (H, W, 4) array (single values become 1x1)."""
    v = inputs.get("Image")
    if v is None or isinstance(v, (int, float, bool, tuple, list)):
        return node.in_image_array(inputs, "Image", (1, 1), 4, default=(0.0, 0.0, 0.0, 1.0))
    a = np.asarray(v)
    return node.in_image_array(inputs, "Image", a.shape[:2], 4)


class CompositorNodeLabImageStatistics(LabNode, bpy.types.CompositorNode):
    '''Min, max, mean, luminance mean, standard deviation and percentile of an image'''
    bl_idname = "CompositorNodeLabImageStatistics"
    bl_label = "Image Statistics"

    SOCKETS = [
        In("Image", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        Out("Min", "COLOR", single=True),
        Out("Max", "COLOR", single=True),
        Out("Mean", "COLOR", single=True),
        Out("Luminance Mean", "FLOAT", single=True),
        Out("Std Dev", "FLOAT", single=True),
        Out("Percentile", "FLOAT", single=True),
    ]
    PROPS = ["percentile"]

    percentile: FloatProperty(name="Percentile", default=50.0, min=0.0, max=100.0,
                              subtype='PERCENTAGE',
                              description="Percentile of the luminance for the Percentile output")

    def _write(self, outputs, st, hist_fn):
        def put(name, vals):
            arr = self.out_single(outputs, name)
            if arr is not None:
                arr[:] = vals

        put("Min", (st.min[0], st.min[1], st.min[2], 1.0))
        put("Max", (st.max[0], st.max[1], st.max[2], 1.0))
        put("Mean", (st.mean[0], st.mean[1], st.mean[2], 1.0))
        put("Luminance Mean", [st.lmean])
        put("Std Dev", [st.lstd])
        if self.out_single(outputs, "Percentile") is not None:
            if st.count == 0:
                put("Percentile", [0.0])
            else:
                lo, hi = float(F32(st.lmin)), float(F32(st.lmax))
                put("Percentile", [reduce.percentile(hist_fn(lo, hi), lo, hi, self.percentile)])

    def cpu(self, inputs, outputs, ctx):
        img = source_array(self, inputs)
        self._write(outputs, reduce.stats_cpu(img),
                    lambda lo, hi: reduce.hist_cpu(img, 3, lo, hi))

    def gpu(self, inputs, outputs, ctx):
        tex = inputs.get("Image")
        if not lab_gpu.is_texture(tex):
            return self.cpu(inputs, outputs, ctx)       # a single value: nothing to reduce
        self._write(outputs, reduce.stats_gpu(tex),
                    lambda lo, hi: reduce.hist_gpu(tex, 3, lo, hi))


NODE_CLASSES = [CompositorNodeLabImageStatistics]
