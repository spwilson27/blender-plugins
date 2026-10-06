# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Exact Euclidean distance transform (GLSL). Numpy twin and Python driver: ``lib/distance.py``.

Two ``gpu.pointwise`` bodies (``PASS1_BODY``, ``PASS2_BODY``) compute the *squared* distance of
every pixel to the nearest feature pixel with separable exact passes limited to a window of
``d_r`` pixels (a pixel whose true distance is <= ``d_r`` gets the exact integer value; anything
further is clamped to ``d_cap = (d_r + 1)^2``). Squared distances are integers, exactly
representable in float32 up to 2^24, so CPU and GPU agree bit for bit.

Helpers (SOURCE) compare a squared distance ``d2`` with a float32 distance threshold without a
square root: ``lab_d_ge/gt/le/lt(d2, a)`` mean ``d >= a`` etc. for d = sqrt(d2) >= 0.
"""

DEPS = ()

SOURCE = r'''
/* ---- distance.py ---------------------------------------------------------- */
bool lab_dt_feat(float v, float thr, int pol)
{
  return (v >= thr) == (pol > 0);
}

bool lab_d_ge(float d2, float a)
{
  return (a <= 0.0) || (d2 >= a * a);
}

bool lab_d_gt(float d2, float a)
{
  return (a < 0.0) || (d2 > a * a);
}

bool lab_d_le(float d2, float b)
{
  return (b >= 0.0) && (d2 <= b * b);
}

bool lab_d_lt(float d2, float b)
{
  return (b > 0.0) && (d2 < b * b);
}
'''

# Uniforms: d_r (int window radius), d_pol (int, +1: feature where value >= d_thr, else where
# value < d_thr), d_thr (float), d_cap (float). Input F (float): the feature image. Output G.
PASS1_BODY = r'''
    float best = 1.0e9;
    for (int d = 0; d <= d_r; ++d) {
      bool hit = false;
      int y0 = texel.y - d;
      int y1 = texel.y + d;
      if (y0 >= 0 && lab_dt_feat(in_F(ivec2(texel.x, y0)), d_thr, d_pol)) {
        hit = true;
      }
      if (d > 0 && y1 < res.y && lab_dt_feat(in_F(ivec2(texel.x, y1)), d_thr, d_pol)) {
        hit = true;
      }
      if (hit) {
        best = float(d * d);
        break;
      }
    }
    out_G = vec4(best, 0.0, 0.0, 1.0);
'''

# Input G (float, from pass 1), output D = min(squared distance, d_cap).
PASS2_BODY = r'''
    float best = 1.0e9;
    for (int d = 0; d <= d_r; ++d) {
      float dd = float(d * d);
      if (dd >= best) {
        break;
      }
      int x0 = texel.x - d;
      int x1 = texel.x + d;
      if (x0 >= 0) {
        best = min(best, in_G(ivec2(x0, texel.y)) + dd);
      }
      if (d > 0 && x1 < res.x) {
        best = min(best, in_G(ivec2(x1, texel.y)) + dd);
      }
    }
    out_D = vec4(min(best, d_cap), 0.0, 0.0, 1.0);
'''
