# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Time Displace / Slit-scan: every pixel shows the input from a different moment.

The node keeps the last ``History Frames`` (N) input frames and builds each output pixel from the
frame ``delay`` frames ago, where the delay comes from the pixel's place:

* **Slit-scan Rows**     delay grows linearly with the row (bottom 0 .. top N); Columns: with the column.
* **Displacement Map**   delay = luminance of the ``Map`` input x N.
* **Radial**             delay grows with the distance from the centre (0 at the centre, N at the
                         farthest corner).

``Amount`` (0..1) scales the delay range (delay = t x N x Amount, ``t`` in 0..1); ``Direction``
reverses ``t`` (1 - t). Without ``Blend Frames`` the delay is rounded to whole frames; with it the two
neighbouring frames are mixed linearly. Outputs: ``Image`` and ``Delay`` (the delay actually applied,
divided by N).

State: the history *is* the state. ``lib/np_history.HistoryRing`` (CPU) / ``glsl/history.GpuHistoryRing``
(GPU, a texture array) hold frames by frame number, ``N + 1`` of them (the current frame and N older
ones), optionally box-downscaled (``Downscale`` 1, 2, 4) and stored as half floats. The current
frame is always read at full resolution. A lookup of a frame that is not stored uses the nearest
older stored frame, or the oldest stored one: before the history has filled (and on a still render)
everything clamps to the oldest available frame. ``Memory Cap`` limits the history; ``N`` is
reduced to fit and the node says so. See ``lib/state.py`` for the stream semantics; this node only
needs ``RESET`` (first frame, start frame) from them. The history is addressed by frame number, so
scrubbing back ``k`` frames keeps the older frames that are still stored (the ``k`` oldest delays clamp)
and going forward again loses nothing; a jump beyond the stored window restarts the history from the
requested frame. The frame cache is not used (``cache_frames`` 0, so no history copies).
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty

from ..lib import gpu as lab_gpu, np_history
from ..lib.glsl import history as glsl_history
from ..lib.node import In, LabNode, Out, StatefulNode
from ..lib import state as lab_state

MENU = "Simulate"

F32 = np.float32
MAX_FRAMES = 120            # hard cap on History Frames
MODES = ('ROWS', 'COLUMNS', 'MAP', 'RADIAL')

_MODE_ITEMS = [
    ('ROWS', "Slit-scan Rows", "Each row shows a different delay (linear across the height)"),
    ('COLUMNS', "Slit-scan Columns", "Each column shows a different delay (linear across the width)"),
    ('MAP', "Displacement Map", "Per-pixel delay = luminance of the Map input x History Frames"),
    ('RADIAL', "Radial", "Delay grows with the distance from the centre"),
]
_DIR_ITEMS = [
    ('POSITIVE', "Up / Right / Outward / Bright",
     "Delay grows upwards (rows), to the right (columns), away from the centre (radial), "
     "with the map brightness (map)"),
    ('NEGATIVE', "Down / Left / Inward / Dark", "The opposite"),
]
_SCALE_ITEMS = [('1', "Full", "Store the history at full resolution"),
                ('2', "1/2", "Store the history at half the width and height"),
                ('4', "1/4", "Store the history at a quarter of the width and height")]
_PREC_ITEMS = [('HALF', "Half Float", "16-bit floats: half the memory"),
               ('FULL', "Full Float", "32-bit floats: exact")]

_BODY = """
    float x = 0.0;
    if (d_mode == 0) {
        x = texelFetch(s_Delays, ivec2(texel.y, 0), 0).r;
    } else if (d_mode == 1) {
        x = texelFetch(s_Delays, ivec2(texel.x, 0), 0).r;
    } else {
        float t;
        if (d_mode == 2) {
            vec4 m = texelFetch(s_Map, clamp(texel, ivec2(0), textureSize(s_Map, 0) - ivec2(1)), 0);
            t = m.r * 0.2126 + m.g * 0.7152 + m.b * 0.0722;
        } else {
            vec2 dv = vec2(texel) + vec2(0.5) - d_centre;
            t = sqrt(dv.x * dv.x + dv.y * dv.y) * d_inv_maxd;
        }
        t = clamp(t, 0.0, 1.0);
        if (d_invert != 0) {
            t = 1.0 - t;
        }
        x = t * d_scale;
    }
    x = clamp(x, 0.0, float(d_nmax));
    int d0;
    float fr = 0.0;
    if (d_interp != 0) {
        d0 = int(floor(x));
        fr = x - float(d0);
    } else {
        d0 = int(floor(x + 0.5));
    }
    d0 = clamp(d0, 0, d_nmax);
    int s0 = int(texelFetch(s_Slots, ivec2(d0, 0), 0).r + 0.5);
    vec4 c = lab_hist_sample(s0, texel);
    float shown = float(d0);
    if (d_interp != 0) {
        shown = x;
        if (fr > 0.0) {
            int d1 = min(d0 + 1, d_nmax);
            int s1 = int(texelFetch(s_Slots, ivec2(d1, 0), 0).r + 0.5);
            vec4 c1 = lab_hist_sample(s1, texel);
            c = c + (c1 - c) * fr;
        }
    }
    out_Image = c;
    out_Delay = vec4(shown * d_invn);
"""


class CompositorNodeLabTimeDisplace(StatefulNode, LabNode, bpy.types.CompositorNode):
    '''Time displace / slit-scan: each row, column or pixel shows the input from a different past frame'''
    bl_idname = "CompositorNodeLabTimeDisplace"
    bl_label = "Time Displace / Slit-scan"

    SOCKETS = [
        In("Image", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("Map", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("History Frames", "INT", 30),
        In("Amount", "FACTOR", 1.0),
        In("Center X", "FACTOR", 0.5),
        In("Center Y", "FACTOR", 0.5),
        Out("Image", "COLOR"),
        Out("Delay", "FLOAT"),
    ]
    PROPS = ["mode", "direction", "interpolate", "downscale", "precision", "memory_mb"]

    mode: EnumProperty(name="Mode", items=_MODE_ITEMS, default='ROWS')
    direction: EnumProperty(name="Direction", items=_DIR_ITEMS, default='POSITIVE')
    interpolate: BoolProperty(
        name="Blend Frames", default=True,
        description="Mix the two neighbouring frames for a fractional delay (off: round the delay "
                    "to whole frames, giving visible steps)")
    downscale: EnumProperty(name="Downscale", items=_SCALE_ITEMS, default='1',
                            description="Store the history at a reduced size (the current frame "
                                        "stays sharp)")
    precision: EnumProperty(name="Precision", items=_PREC_ITEMS, default='HALF',
                            description="Storage precision of the history")
    memory_mb: IntProperty(
        name="Memory Cap (MB)", default=640, min=1, soft_max=8192,
        description="Largest history (CPU or GPU memory); History Frames is reduced to fit")

    def draw_buttons(self, context, layout):
        self.draw_props(layout)
        for s in lab_state.streams_for(self.state_key()):
            msg = s.message
            if msg:
                layout.label(text=msg, icon='ERROR')
                break
        self.draw_state_buttons(layout)

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)

    # -- parameters ----------------------------------------------------------
    def _spec(self, inputs, ctx, gpu_ring):
        """History layout for this evaluation: ring capacity (N + 1 limited by the memory cap)."""
        n = min(max(self.in_int(inputs, "History Frames", 30), 0), MAX_FRAMES)
        s = int(self.downscale)
        half = self.precision == 'HALF'
        per_layer = np_history.layer_bytes(ctx.size, s, 2 if half else 4)
        cap, limited = np_history.fit_capacity(n + 1, per_layer, int(self.memory_mb) * np_history.MB)
        msg = ""
        if limited:
            msg = "History limited to %d frames (memory cap %d MB, asked %d)" % (
                cap - 1, self.memory_mb, n)
        return {"capacity": cap, "downscale": s, "half": half, "message": msg, "asked": n}

    def _params(self, inputs, ctx, nmax):
        w, h = ctx.size
        amount = min(max(self.in_float(inputs, "Amount", 1.0), 0.0), 1.0)
        cx = self.in_float(inputs, "Center X", 0.5) * w
        cy = self.in_float(inputs, "Center Y", 0.5) * h
        far = max(abs(cx), abs(w - cx)), max(abs(cy), abs(h - cy))
        maxd = max(float(np.hypot(*far)), 1e-6)
        return {
            "nmax": nmax,
            "scale": F32(nmax) * F32(amount),
            "invn": F32(1.0 / nmax) if nmax else F32(0.0),
            "invert": self.direction == 'NEGATIVE',
            "interp": bool(self.interpolate),
            "centre": (F32(cx), F32(cy)),
            "inv_maxd": F32(1.0 / maxd),
        }

    def _line_delays(self, p, n):
        """Delay in frames per row / column index (float32), for the slit-scan modes."""
        t = np.arange(n, dtype=F32) / F32(max(n - 1, 1))
        if p["invert"]:
            t = F32(1.0) - t
        return (np.clip(t, F32(0.0), F32(1.0)) * p["scale"]).astype(F32)

    # -- shared ring management -----------------------------------------------
    def _ring(self, ctx, spec, make):
        """(plan, ring) for this evaluation: the stream's ring, or a fresh one on a reset or when
        the layout changed (History Frames, Downscale, Precision, size: the signature)."""
        plan = self.advance(ctx, signature=make(None), mutable=True)
        ring = plan.run(lambda: make(True), None)
        plan.stream.message = spec["message"]
        if spec["message"]:
            ctx.report(spec["message"], 'WARNING')
        return plan, ring

    # -- CPU -----------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        out_d = self.out_array(outputs, "Delay")
        h, w = ctx.shape
        spec = self._spec(inputs, ctx, False)
        dtype = np.float16 if spec["half"] else np.float32

        def make(real):
            if real is None:
                return (spec["capacity"], ctx.size, spec["downscale"], np.dtype(dtype).str)
            return np_history.HistoryRing(spec["capacity"], ctx.size, spec["downscale"], dtype)

        plan, ring = self._ring(ctx, spec, make)
        img = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                             default=(0.0, 0.0, 0.0, 1.0)), F32)
        frame = int(round(ctx.frame))
        ring.push(frame, img)
        if out is None and out_d is None:
            return

        nmax = ring.capacity - 1
        p = self._params(inputs, ctx, nmax)
        if self.mode == 'ROWS':
            x = self._line_delays(p, h)[:, None]
        elif self.mode == 'COLUMNS':
            x = self._line_delays(p, w)[None, :]
        else:
            if self.mode == 'MAP':
                m = np.asarray(self.in_image_array(inputs, "Map", ctx.shape, 4,
                                                   default=(0.5, 0.5, 0.5, 1.0)), F32)
                t = m[..., 0] * F32(0.2126) + m[..., 1] * F32(0.7152) + m[..., 2] * F32(0.0722)
            else:
                dx = (np.arange(w, dtype=F32) + F32(0.5))[None, :] - p["centre"][0]
                dy = (np.arange(h, dtype=F32) + F32(0.5))[:, None] - p["centre"][1]
                t = np.sqrt(dx * dx + dy * dy) * p["inv_maxd"]
            t = np.clip(t, F32(0.0), F32(1.0))
            if p["invert"]:
                t = F32(1.0) - t
            x = t * p["scale"]
        x = np.clip(x, F32(0.0), F32(nmax))
        if p["interp"]:
            d0 = np.floor(x).astype(np.int32)
            fr = x - d0.astype(F32)
        else:
            d0 = np.floor(x + F32(0.5)).astype(np.int32)
            fr = None
        d0 = np.clip(d0, 0, nmax)
        table = ring.slot_table(frame, nmax)
        cur = ring.slot_of(frame)
        if out is not None:
            res = ring.gather(table[d0], (h, w), img, cur)
            if fr is not None and np.any(fr > 0):
                d1 = np.minimum(d0 + 1, nmax)
                res1 = ring.gather(table[d1], (h, w), img, cur)
                res = res + (res1 - res) * fr[..., None]
            out[...] = res
        if out_d is not None:
            shown = x if p["interp"] else d0.astype(F32)
            out_d[..., 0] = np.broadcast_to(shown * p["invn"], (h, w))

    # -- GPU -----------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        dst_d = self.out_texture(outputs, "Delay")
        w, h = ctx.size
        spec = self._spec(inputs, ctx, True)
        fmt = 'RGBA16F' if spec["half"] else 'RGBA32F'

        def make(real):
            if real is None:
                return (spec["capacity"], ctx.size, spec["downscale"], fmt)
            return glsl_history.GpuHistoryRing(spec["capacity"], ctx.size, spec["downscale"], fmt)

        plan, ring = self._ring(ctx, spec, make)
        image = self.in_texture_or_value(inputs, "Image", (0.0, 0.0, 0.0, 1.0))
        if not lab_gpu.is_texture(image):
            image = lab_gpu.const_texture(image)
        frame = int(round(ctx.frame))
        ring.push(frame, image)
        if dst is None and dst_d is None:
            return

        nmax = ring.capacity - 1
        p = self._params(inputs, ctx, nmax)
        mode = MODES.index(self.mode)
        if mode == 0:
            delays = self._line_delays(p, h)
        elif mode == 1:
            delays = self._line_delays(p, w)
        else:
            delays = np.zeros(1, F32)
        table = ring.slot_table(frame, nmax)
        m = self.in_texture_or_value(inputs, "Map", (0.5, 0.5, 0.5, 1.0))
        if not lab_gpu.is_texture(m):
            m = lab_gpu.const_texture(m)
        body = _BODY
        if dst is None:
            body = body.replace("    out_Image = c;\n", "")
        if dst_d is None:
            body = body.replace("    out_Delay = vec4(shown * d_invn);\n", "")
        ring.compose(
            body, {"Image": dst, "Delay": dst_d},
            samplers={"Delays": glsl_history.table_texture(delays),
                      "Slots": glsl_history.table_texture(table), "Map": m},
            uniforms={
                "d_mode": ("int", mode),
                "d_invert": ("int", int(p["invert"])),
                "d_interp": ("int", int(p["interp"])),
                "d_nmax": ("int", nmax),
                "d_scale": ("float", float(p["scale"])),
                "d_invn": ("float", float(p["invn"])),
                "d_centre": ("vec2", tuple(float(v) for v in p["centre"])),
                "d_inv_maxd": ("float", float(p["inv_maxd"])),
            },
            current=image, current_slot=ring.slot_of(frame))


NODE_CLASSES = [CompositorNodeLabTimeDisplace]
