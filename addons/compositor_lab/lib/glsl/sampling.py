# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Texture sampling for compute shaders (edge modes, bilinear / bicubic) plus a multi-pass
kernel runner and a separable Gaussian blur. Numpy twin: ``lib/np_sampling.py``.

``gpu.kernel(..., sampling=True)`` (wrapped by ``kernel`` here) is like ``gpu.pointwise`` but
every input is a *sampler* and, for each input ``Name``, the shader gets these functions
(``mode``: 0 clamp, 1 repeat, 2 mirror; ``pos`` in pixels with pixel centres at ``i + 0.5``;
row 0 = bottom)::

    ivec2 lab_size_Name()
    vec4  lab_fetch_Name(ivec2 p)            texelFetch, p clamped to the texture
    vec4  in_Name(ivec2 p)                   same (pointwise-compatible)
    vec4  lab_px_Name(ivec2 p, int mode)     texelFetch with the edge mode
    vec4  lab_bilinear_Name(vec2 pos, int mode)
    vec4  lab_bicubic_Name(vec2 pos, int mode)         Catmull-Rom
    void  lab_sobel_Name(ivec2 p, int mode, out vec4 gx, out vec4 gy)   (np_sampling.sobel)

Single values (unlinked sockets) passed as inputs become cached 1x1 textures, so they behave as
constant images. Besides ``texel`` / ``uv`` / ``res`` / ``out_<Name>`` / ``lab_zero`` (see
``gpu.pointwise``) the body has ``lab_edge_index(i, n, mode)`` and ``lab_cubic_w(t)``.

Multi-pass work: ``scratch(w, h, role)`` (``gpu.scratch``) returns a cached RGBA32F texture; dispatches run in order
(the backend inserts the barriers between them). ``gaussian_blur`` is one such composite.
"""

import math

from .. import gpu as lab_gpu

DEPS = ()

SOURCE = r'''
/* ---- sampling.py ---------------------------------------------------------- */
int lab_edge_index(int i, int n, int mode)
{
  if (mode == 0) {
    return clamp(i, 0, n - 1);
  }
  if (mode == 1) {
    return ((i % n) + n) % n;
  }
  int m = ((i % (2 * n)) + 2 * n) % (2 * n);
  return (m >= n) ? (2 * n - 1 - m) : m;
}

/* Catmull-Rom weights of the taps at -1, 0, 1, 2. */
vec4 lab_cubic_w(float t)
{
  float t2 = t * t;
  float t3 = t2 * t;
  return 0.5 * vec4(-t3 + t2 + t2 - t, 3.0 * t3 - 5.0 * t2 + 2.0, -3.0 * t3 + 4.0 * t2 + t,
                    t3 - t2);
}
'''

_SAMPLER_TEMPLATE = r'''
ivec2 lab_size_NAME()
{
  return textureSize(s_NAME, 0);
}

vec4 lab_fetch_NAME(ivec2 p)
{
  return texelFetch(s_NAME, clamp(p, ivec2(0), textureSize(s_NAME, 0) - ivec2(1)), 0);
}

vec4 in_NAME(ivec2 p)
{
  return lab_fetch_NAME(p);
}

vec4 lab_px_NAME(ivec2 p, int mode)
{
  ivec2 sz = textureSize(s_NAME, 0);
  return texelFetch(s_NAME, ivec2(lab_edge_index(p.x, sz.x, mode),
                                  lab_edge_index(p.y, sz.y, mode)), 0);
}

vec4 lab_bilinear_NAME(vec2 pos, int mode)
{
  vec2 f = clamp(pos, vec2(-1.0e6), vec2(1.0e6)) - vec2(0.5);
  vec2 f0 = floor(f);
  vec2 t = f - f0;
  ivec2 p0 = ivec2(f0);
  vec4 c00 = lab_px_NAME(p0, mode);
  vec4 c10 = lab_px_NAME(p0 + ivec2(1, 0), mode);
  vec4 c01 = lab_px_NAME(p0 + ivec2(0, 1), mode);
  vec4 c11 = lab_px_NAME(p0 + ivec2(1, 1), mode);
  vec4 bot = c00 + (c10 - c00) * t.x;
  vec4 top = c01 + (c11 - c01) * t.x;
  return bot + (top - bot) * t.y;
}

/* Sobel gradients (per pixel, scaled by 1/8) of all channels around p. */
void lab_sobel_NAME(ivec2 p, int mode, out vec4 gx, out vec4 gy)
{
  vec4 c00 = lab_px_NAME(p + ivec2(-1, -1), mode);
  vec4 c10 = lab_px_NAME(p + ivec2(0, -1), mode);
  vec4 c20 = lab_px_NAME(p + ivec2(1, -1), mode);
  vec4 c01 = lab_px_NAME(p + ivec2(-1, 0), mode);
  vec4 c21 = lab_px_NAME(p + ivec2(1, 0), mode);
  vec4 c02 = lab_px_NAME(p + ivec2(-1, 1), mode);
  vec4 c12 = lab_px_NAME(p + ivec2(0, 1), mode);
  vec4 c22 = lab_px_NAME(p + ivec2(1, 1), mode);
  gx = ((c20 + 2.0 * c21 + c22) - (c00 + 2.0 * c01 + c02)) * 0.125;
  gy = ((c02 + 2.0 * c12 + c22) - (c00 + 2.0 * c10 + c20)) * 0.125;
}

vec4 lab_bicubic_NAME(vec2 pos, int mode)
{
  vec2 f = clamp(pos, vec2(-1.0e6), vec2(1.0e6)) - vec2(0.5);
  vec2 f0 = floor(f);
  vec2 t = f - f0;
  ivec2 p0 = ivec2(f0);
  vec4 wx = lab_cubic_w(t.x);
  vec4 wy = lab_cubic_w(t.y);
  vec4 acc = vec4(0.0);
  for (int j = 0; j < 4; ++j) {
    vec4 row = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
      row += lab_px_NAME(p0 + ivec2(i - 1, j - 1), mode) * wx[i];
    }
    acc += row * wy[j];
  }
  return acc;
}
'''

_BLUR_BODY = r'''
    float s2 = 2.0 * g_sigma * g_sigma;
    float wsum = 0.0;
    for (int i = -g_radius; i <= g_radius; ++i) {
      wsum += exp(-float(i * i) / s2);
    }
    vec4 acc = vec4(0.0);
    for (int i = -g_radius; i <= g_radius; ++i) {
      float wgt = exp(-float(i * i) / s2) / wsum;
      ivec2 q = texel + ((g_axis == 0) ? ivec2(i, 0) : ivec2(0, i));
      acc += lab_px_Src(q, g_mode) * wgt;
    }
    out_Dst = acc;
'''

_COPY_BODY = "    out_Dst = lab_fetch_Src(texel);\n"

scratch = lab_gpu.scratch
const_texture = lab_gpu.const_texture
clear_cache = lab_gpu.clear_cache


def sampler_source(names):
    """GLSL of the sampler functions (see the module docstring) for the input names."""
    return "\n".join(_SAMPLER_TEMPLATE.replace("NAME", lab_gpu.ident(n)) for n in names)


def kernel(body, outputs, samplers, uniforms=None, libs=(), local_size=lab_gpu.LOCAL_SIZE,
           functions=""):
    """Compatibility wrapper of ``gpu.kernel(..., sampling=True)``: ``samplers`` is name ->
    GPUTexture or single value (every input is a "color" input)."""
    lab_gpu.kernel(body, outputs, {n: ("color", v) for n, v in samplers.items()}, uniforms,
                   libs, functions, local_size, sampling=True)


def gaussian_blur(src, dst, sigma, mode=0, role="blur"):
    """Separable Gaussian blur of the texture ``src`` into ``dst`` (kernel radius ceil(3 sigma),
    like ``np_sampling.blur_gaussian``). ``src`` must not be ``dst``. sigma <= 0 copies."""
    if sigma <= 0.0:
        kernel(_COPY_BODY, {"Dst": dst}, {"Src": src})
        return
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    tmp = scratch(dst.width, dst.height, role + "_tmp")
    for s, d, axis in ((src, tmp, 0), (tmp, dst, 1)):
        kernel(_BLUR_BODY, {"Dst": d}, {"Src": s},
               uniforms={"g_sigma": ("float", float(sigma)), "g_radius": ("int", radius),
                         "g_axis": ("int", axis), "g_mode": ("int", int(mode))})
