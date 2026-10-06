# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Frame-history ring buffer on the GPU: a 2D texture array, one layer per stored frame.
Numpy twin: ``lib/np_history.py`` (same slot / delay semantics, same downscale arithmetic).

    ring = GpuHistoryRing(capacity=31, size=(w, h), downscale=2, fmt="RGBA16F")
    ring.push(frame, texture_or_colour)             # box-downscales into layer frame % capacity
    table = ring.slot_table(frame, max_delay)       # numpy, see np_history.FrameIndex
    ring.compose(body, {"Image": dst}, samplers={"Slots": table_texture(table)},
                 uniforms={...}, current=texture_or_colour, current_slot=ring.slot_of(frame))

``compose`` compiles (cached) and dispatches a per-pixel compute shader over the output size. The
body is GLSL inside ``main`` with ``texel`` (ivec2), ``res`` (ivec2), a ``vec4 out_<Name>`` per
output, the uniforms and ``s_<Name>`` for each extra 2D sampler, and

``vec4 lab_hist_sample(int slot, ivec2 px)``  the stored frame in ``slot`` at output pixel ``px``
    (bilinear when downscaled; the newest frame, ``current_slot``, is read from the full-resolution
    ``current`` input instead, so it stays sharp).

This module is not one of the ``glsl.resolve`` modules: array samplers need a hand-built shader.
"""

import numpy as np

from .. import gpu as lab_gpu
from ..np_history import FrameIndex, layer_bytes, stored_size

DEPS = ()

_ITEMSIZE = {"RGBA32F": 4, "RGBA16F": 2}

SOURCE = r'''
/* ---- history.py ----------------------------------------------------------- */
vec4 lab_hist_tap(int slot, ivec2 p)
{
  return texelFetch(s_hist, ivec3(clamp(p, ivec2(0), ivec2(h_sw - 1, h_sh - 1)), slot), 0);
}

vec4 lab_hist_stored(int slot, ivec2 px)
{
  if (h_s == 1) {
    return lab_hist_tap(slot, px);
  }
  vec2 f = (vec2(px) + vec2(0.5)) * h_inv_s - vec2(0.5);
  vec2 f0 = floor(f);
  vec2 t = f - f0;
  ivec2 i0 = ivec2(f0);
  vec4 c00 = lab_hist_tap(slot, i0);
  vec4 c10 = lab_hist_tap(slot, i0 + ivec2(1, 0));
  vec4 c01 = lab_hist_tap(slot, i0 + ivec2(0, 1));
  vec4 c11 = lab_hist_tap(slot, i0 + ivec2(1, 1));
  vec4 bot = c00 + (c10 - c00) * t.x;
  vec4 top = c01 + (c11 - c01) * t.x;
  return bot + (top - bot) * t.y;
}

vec4 lab_hist_sample(int slot, ivec2 px)
{
  if (slot == h_cur_slot) {
    return texelFetch(s_cur, clamp(px, ivec2(0), textureSize(s_cur, 0) - ivec2(1)), 0);
  }
  return lab_hist_stored(slot, px);
}
'''

_PUSH_SOURCE = r'''
void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= h_sw || p.y >= h_sh) {
    return;
  }
  ivec2 hi = textureSize(s_src, 0) - ivec2(1);
  vec4 acc = vec4(0.0);
  for (int j = 0; j < h_s; j++) {
    for (int i = 0; i < h_s; i++) {
      acc += texelFetch(s_src, clamp(p * h_s + ivec2(i, j), ivec2(0), hi), 0);
    }
  }
  acc = acc * h_inv;
  imageStore(o_hist, ivec3(p, h_layer), acc);
}
'''

_COPY_SOURCE = r'''
void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= h_sw || p.y >= h_sh) {
    return;
  }
  imageStore(o_hist, ivec3(p, h_layer), texelFetch(s_hist, ivec3(p, h_layer), 0));
}
'''


def table_texture(values):
    """A (n, 1) R32F texture holding ``values`` (floats / small ints); read it with
    ``texelFetch(s, ivec2(i, 0), 0).r``."""
    import gpu

    vals = [float(v) for v in np.asarray(values).ravel()]
    buf = gpu.types.Buffer('FLOAT', len(vals), vals)
    return gpu.types.GPUTexture((len(vals), 1), format='R32F', data=buf)


def _info(local=True):
    return lab_gpu.create_info(lab_gpu.LOCAL_SIZE if local else None)


def _push_shader(fmt):
    def factory():
        info = _info()
        info.sampler(0, 'FLOAT_2D', "s_src")
        info.image(0, fmt, 'FLOAT_2D_ARRAY', "o_hist", qualifiers={'WRITE'})
        for n in ("h_sw", "h_sh", "h_s", "h_layer"):
            info.push_constant('INT', n)
        info.push_constant('FLOAT', "h_inv")
        return lab_gpu.compile_shader(info, _PUSH_SOURCE, "history push " + fmt)

    return lab_gpu.get_shader(("hist_push", fmt), factory)


def _copy_shader(fmt):
    def factory():
        info = _info()
        info.sampler(0, 'FLOAT_2D_ARRAY', "s_hist")
        info.image(0, fmt, 'FLOAT_2D_ARRAY', "o_hist", qualifiers={'WRITE'})
        for n in ("h_sw", "h_sh", "h_layer"):
            info.push_constant('INT', n)
        return lab_gpu.compile_shader(info, _COPY_SOURCE, "history copy " + fmt)

    return lab_gpu.get_shader(("hist_copy", fmt), factory)


class GpuHistoryRing(FrameIndex):
    """The last ``capacity`` frames in a ``GPUTexture`` array (``fmt`` RGBA32F or RGBA16F)."""

    def __init__(self, capacity, size, downscale=1, fmt="RGBA16F"):
        import gpu

        super().__init__(capacity)
        self.size = (int(size[0]), int(size[1]))
        self.downscale = int(downscale)
        self.fmt = fmt
        self.sw, self.sh = stored_size(self.size, self.downscale)
        self.tex = gpu.types.GPUTexture((self.sw, self.sh), layers=self.capacity, format=fmt)

    @property
    def key(self):
        return (self.capacity, self.size, self.downscale, self.fmt)

    @property
    def nbytes(self):
        return self.capacity * layer_bytes(self.size, self.downscale, _ITEMSIZE[self.fmt])

    def copy(self):
        r = GpuHistoryRing(self.capacity, self.size, self.downscale, self.fmt)
        shader = _copy_shader(self.fmt)
        for layer in range(self.capacity):
            shader.uniform_sampler("s_hist", self.tex)
            shader.image("o_hist", r.tex)
            lab_gpu.set_uniform(shader, "h_sw", "int", self.sw)
            lab_gpu.set_uniform(shader, "h_sh", "int", self.sh)
            lab_gpu.set_uniform(shader, "h_layer", "int", layer)
            lab_gpu.dispatch_grid(shader, self.sw, self.sh)
        r.frames[...] = self.frames
        return r

    def clear(self):
        self.frames[...] = -(1 << 40)

    def push(self, frame, src):
        """Store ``src`` (a full-resolution texture or a single colour) as frame ``frame``."""
        if not lab_gpu.is_texture(src):
            src = lab_gpu.const_texture(src)
        shader = _push_shader(self.fmt)
        shader.uniform_sampler("s_src", src)
        shader.image("o_hist", self.tex)
        lab_gpu.set_uniform(shader, "h_sw", "int", self.sw)
        lab_gpu.set_uniform(shader, "h_sh", "int", self.sh)
        lab_gpu.set_uniform(shader, "h_s", "int", self.downscale)
        lab_gpu.set_uniform(shader, "h_layer", "int", self.slot_of(frame))
        lab_gpu.set_uniform(shader, "h_inv", "float", 1.0 / (self.downscale * self.downscale))
        lab_gpu.dispatch_grid(shader, self.sw, self.sh)
        self.mark(frame)

    def compose(self, body, outputs, samplers=None, uniforms=None, current=None, current_slot=-1,
                functions=""):
        """Run ``body`` (GLSL, see the module docstring) for every pixel of ``outputs``."""
        samplers = samplers or {}
        uniforms = dict(uniforms or {})
        out_items = sorted(((n, t) for n, t in outputs.items() if t is not None),
                           key=lambda kv: kv[0])
        if not out_items:
            return
        w, h = int(out_items[0][1].width), int(out_items[0][1].height)
        if not lab_gpu.is_texture(current):
            current = lab_gpu.const_texture(current if current is not None else 0.0)
        names = sorted(samplers)
        uniforms.update({
            "h_sw": ("int", self.sw), "h_sh": ("int", self.sh), "h_s": ("int", self.downscale),
            "h_inv_s": ("float", 1.0 / self.downscale), "h_cur_slot": ("int", int(current_slot)),
            "lab_w": ("int", w), "lab_h": ("int", h),
        })
        uspec = tuple((n, uniforms[n][0]) for n in sorted(uniforms))
        formats = tuple((n, t.format) for n, t in out_items)
        key = ("hist_compose", body, functions, tuple(names), uspec, formats)

        def factory():
            info = _info()
            info.sampler(0, 'FLOAT_2D_ARRAY', "s_hist")
            info.sampler(1, 'FLOAT_2D', "s_cur")
            for i, n in enumerate(names):
                info.sampler(2 + i, 'FLOAT_2D', "s_" + lab_gpu.ident(n))
            for slot, (n, fmt) in enumerate(formats):
                info.image(slot, fmt, 'FLOAT_2D', "o_" + lab_gpu.ident(n), qualifiers={'WRITE'})
            for n, t in uspec:
                info.push_constant(lab_gpu.PUSH_TYPES[t], n)
            decl = "".join("  vec4 out_%s = vec4(0.0);\n" % lab_gpu.ident(n) for n, _ in formats)
            store = "".join("  imageStore(o_%s, texel, out_%s);\n" % (lab_gpu.ident(n),
                                                                     lab_gpu.ident(n))
                            for n, _ in formats)
            source = (SOURCE + "\n" + functions + "\n"
                      "void main()\n{\n"
                      "  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);\n"
                      "  ivec2 res = ivec2(lab_w, lab_h);\n"
                      "  if (texel.x >= res.x || texel.y >= res.y) {\n    return;\n  }\n"
                      + decl + "  {\n" + body + "\n  }\n" + store + "}\n")
            return lab_gpu.compile_shader(info, source, "history compose")

        shader = lab_gpu.get_shader(key, factory)
        shader.uniform_sampler("s_hist", self.tex)
        shader.uniform_sampler("s_cur", current)
        for n in names:
            shader.uniform_sampler("s_" + lab_gpu.ident(n), samplers[n])
        for n, t in out_items:
            shader.image("o_" + lab_gpu.ident(n), t)
        for n, (t, v) in uniforms.items():
            lab_gpu.set_uniform(shader, n, t, v)
        lab_gpu.dispatch_grid(shader, w, h)
