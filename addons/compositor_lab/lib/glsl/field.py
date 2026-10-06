# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Line Integral Convolution (GLSL). Numpy twin: ``lib/np_field.py``.

This module is written for ``gpu.pointwise`` shaders that bind two inputs:

``Fld``    the vector field (RGBA32F texture, ``.xy``)
``Image``  the image to smear (a placeholder texture when ``fl_has_img`` is 0)

and these uniforms: ``int fl_has_img`` (0: smear white noise instead of ``in_Image``),
``int fl_seed``, ``float fl_density`` (sparse impulse density for the streamline output).
They are read through ``s_Fld`` / ``s_Image`` directly (``pointwise`` defines ``in_*`` only after
this source). When no Image is linked, bind any texture as a placeholder. Positions are in pixel
coordinates (pixel centres at x + 0.5), row 0 = bottom.
"""

DEPS = ("hash",)

SOURCE = r'''
/* ---- field.py ------------------------------------------------------------- */
vec4 lab_fl_fld(ivec2 p)
{
  return texelFetch(s_Fld, clamp(p, ivec2(0), textureSize(s_Fld, 0) - ivec2(1)), 0);
}

vec4 lab_fl_img(ivec2 p)
{
  return texelFetch(s_Image, clamp(p, ivec2(0), textureSize(s_Image, 0) - ivec2(1)), 0);
}

vec4 lab_fl_src(ivec2 p)
{
  if (fl_has_img != 0) {
    return lab_fl_img(p);
  }
  ivec2 q = clamp(p, ivec2(0), ivec2(lab_w - 1, lab_h - 1));
  float n = lab_rand(q, uint(fl_seed));
  return vec4(n, n, n, 1.0);
}

float lab_fl_imp(ivec2 p)
{
  ivec2 q = clamp(p, ivec2(0), ivec2(lab_w - 1, lab_h - 1));
  return (lab_rand(q, uint(fl_seed) + 7919u) < fl_density) ? 1.0 : 0.0;
}

/* Bilinear taps (clamp to edge) at pixel-coordinate position pc. */
void lab_fl_taps(vec2 pc, out ivec2 i0, out vec2 t)
{
  vec2 u = clamp(pc - vec2(0.5), vec2(-0.5), vec2(float(lab_w), float(lab_h)) - vec2(0.5));
  vec2 f = floor(u);
  t = u - f;
  i0 = ivec2(f);
}

vec2 lab_fl_field(vec2 pc)
{
  ivec2 i0;
  vec2 t;
  lab_fl_taps(pc, i0, t);
  vec2 a = lab_fl_fld(i0).xy;
  vec2 b = lab_fl_fld(i0 + ivec2(1, 0)).xy;
  vec2 c = lab_fl_fld(i0 + ivec2(0, 1)).xy;
  vec2 d = lab_fl_fld(i0 + ivec2(1, 1)).xy;
  return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

/* Bilinear sample of the source colour (c) and the impulse noise (imp). */
void lab_fl_sample(vec2 pc, out vec4 c, out float imp)
{
  ivec2 i0;
  vec2 t;
  lab_fl_taps(pc, i0, t);
  ivec2 i1 = i0 + ivec2(1, 0);
  ivec2 i2 = i0 + ivec2(0, 1);
  ivec2 i3 = i0 + ivec2(1, 1);
  c = mix(mix(lab_fl_src(i0), lab_fl_src(i1), t.x), mix(lab_fl_src(i2), lab_fl_src(i3), t.x), t.y);
  imp = mix(mix(lab_fl_imp(i0), lab_fl_imp(i1), t.x), mix(lab_fl_imp(i2), lab_fl_imp(i3), t.x),
            t.y);
}

/* LIC at a pixel. len: steps per direction, stp: pixels per step, kern: 0 box, 1 triangle,
 * winv: 1 / (len + 1). Returns the smeared colour and the smeared impulses. */
void lab_fl_lic(ivec2 texel, int len, float stp, int kern, float winv, out vec4 col,
                out float streak)
{
  const float EPS2 = 1e-12;
  vec2 p0 = vec2(texel) + vec2(0.5);
  vec4 acc;
  float iacc;
  lab_fl_sample(p0, acc, iacc);
  float wsum = 1.0;
  for (int s = 0; s < 2; s++) {
    float dir = (s == 0) ? 1.0 : -1.0;
    vec2 p = p0;
    for (int i = 1; i <= len; i++) {
      vec2 v1 = lab_fl_field(p);
      float l1 = v1.x * v1.x + v1.y * v1.y;
      if (l1 < EPS2) {
        break;
      }
      v1 = v1 * (1.0 / sqrt(l1));
      float hs = 0.5 * dir * stp;
      vec2 v2 = lab_fl_field(p + hs * v1);
      float l2 = v2.x * v2.x + v2.y * v2.y;
      v2 = (l2 < EPS2) ? v1 : v2 * (1.0 / sqrt(l2));
      p = p + (dir * stp) * v2;
      float wt = (kern == 1) ? (1.0 - float(i) * winv) : 1.0;
      vec4 c;
      float im;
      lab_fl_sample(p, c, im);
      acc += c * wt;
      iacc += im * wt;
      wsum += wt;
    }
  }
  col = acc / wsum;
  streak = iacc / wsum;
}
'''
