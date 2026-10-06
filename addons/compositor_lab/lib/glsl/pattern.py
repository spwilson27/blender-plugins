# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Voronoi cells and tilings (GLSL). Numpy twin: ``lib/np_pattern.py`` (same formulas).

``lab_voronoi`` expects a scrambled seed ``ss = lab_pcg(seed)``. Pattern modes (``lab_pattern``):
0 stripes, 1 checker, 2 dots, 3 hex, 4 truchet arcs, 5 truchet diagonals, 6 moire, 7 rings.
"""

DEPS = ("hash",)

SOURCE = r'''
/* ---- pattern.py ----------------------------------------------------------- */

/* metric: 0 euclidean (returns the SQUARED distance), 1 manhattan, 2 chebyshev */
float lab_vor_d(vec2 d, int metric)
{
  if (metric == 1) {
    return abs(d.x) + abs(d.y);
  }
  if (metric == 2) {
    return max(abs(d.x), abs(d.y));
  }
  return d.x * d.x + d.y * d.y;
}

/* Feature point offset from the cell centre, jitter applied. See np_pattern.site_offset. */
vec2 lab_vor_offset(ivec2 c, uint ss, float jitter, float cphi, float sphi, bool anim)
{
  vec2 r = lab_hash3_u01(ivec3(c, 0), ss).xy - vec2(0.5);
  if (anim) {
    vec2 rb = lab_hash3_u01(ivec3(c, 1), ss).xy - vec2(0.5);
    r = r * cphi + rb * sphi;
  }
  return r * jitter;
}

/* Nearest / second nearest feature point over the 3x3 neighbourhood. */
void lab_voronoi(vec2 p, int metric, float jitter, float cphi, float sphi, bool anim, uint ss,
                 out float f1, out float f2, out ivec2 cell, out vec2 site)
{
  vec2 fl = floor(p);
  vec2 f = p - fl;
  ivec2 ci = ivec2(fl);
  float d1 = 1e10;
  float d2 = 1e10;
  ivec2 w1 = ivec2(0);
  vec2 s1 = vec2(0.0);
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 o = ivec2(dx, dy);
      vec2 pt = vec2(o) + vec2(0.5) + lab_vor_offset(ci + o, ss, jitter, cphi, sphi, anim);
      float d = lab_vor_d(pt - f, metric);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        w1 = o;
        s1 = pt;
      }
      else if (d < d2) {
        d2 = d;
      }
    }
  }
  f1 = (metric == 0) ? sqrt(d1) : d1;
  f2 = (metric == 0) ? sqrt(d2) : d2;
  cell = ci + w1;
  site = fl + s1;
}

vec3 lab_vor_rgb(ivec2 cell, uint ss)
{
  return lab_hash3_u01(ivec3(cell, 2), ss);
}

float lab_vor_id(ivec2 cell, uint ss)
{
  return lab_hash3_u01(ivec3(cell, 3), ss).x;
}

float lab_vor_border(float f1, float f2, float width, float k)
{
  if (width <= 0.0) {
    return 0.0;
  }
  float inv = 1.0 / (2.0 * k);
  return clamp(0.5 + (width - (f2 - f1)) * inv, 0.0, 1.0);
}

/* ---- tilings: signed distance (positive inside colour B) ---- */
float lab_pat_aa(float s, float inv_w)
{
  return clamp(0.5 + s * inv_w, 0.0, 1.0);
}

float lab_pat_stripes_s(float u, float duty)
{
  if (duty <= 0.0) {
    return -1.0;
  }
  if (duty >= 1.0) {
    return 1.0;
  }
  float c = (u - floor(u)) - 0.5 * duty;
  c = c - floor(c + 0.5);
  return 0.5 * duty - abs(c);
}

float lab_pat_checker_s(float u, float v)
{
  float fu = floor(u);
  float fv = floor(v);
  int n = int(fu) + int(fv);
  float e = 0.5 - max(abs((u - fu) - 0.5), abs((v - fv) - 0.5));
  return ((n & 1) == 1) ? e : -e;
}

float lab_pat_dots_s(float u, float v, float duty)
{
  if (duty <= 0.0) {
    return -1.0;
  }
  float dx = (u - floor(u)) - 0.5;
  float dy = (v - floor(v)) - 0.5;
  return 0.5 * duty - sqrt(dx * dx + dy * dy);
}

float lab_pat_hex_s(float u, float v, float duty)
{
  const float H = 1.7320508;
  const float INV_H = 0.57735027;
  const float HALF_H = 0.8660254;
  float ax = floor(u) + 0.5;
  float ay = (floor(v * INV_H) + 0.5) * H;
  float qx = u - 0.5;
  float qy = v - HALF_H;
  float bx = floor(qx) + 0.5 + 0.5;
  float by = (floor(qy * INV_H) + 0.5) * H + HALF_H;
  vec2 h1 = vec2(u - ax, v - ay);
  vec2 h2 = vec2(u - bx, v - by);
  vec2 h = ((h1.x * h1.x + h1.y * h1.y) < (h2.x * h2.x + h2.y * h2.y)) ? h1 : h2;
  vec2 a = abs(h);
  float e = 0.5 - max(a.x, 0.5 * a.x + HALF_H * a.y);
  return e - (1.0 - duty) * 0.5;
}

bool lab_pat_truchet_flip(float cu, float cv, uint ss)
{
  return lab_u01(lab_hash3(ivec3(int(cu), int(cv), 0), ss).x) < 0.5;
}

float lab_pat_truchet_s(int arc, float u, float v, float duty, uint ss)
{
  float cu = floor(u);
  float cv = floor(v);
  float fx = u - cu;
  float fy = v - cv;
  float lw = duty * 0.25;
  if (arc != 0) {
    if (lab_pat_truchet_flip(cu, cv, ss)) {
      fx = 1.0 - fx;
    }
    float d1 = abs(sqrt(fx * fx + fy * fy) - 0.5);
    float gx = fx - 1.0;
    float gy = fy - 1.0;
    float d2 = abs(sqrt(gx * gx + gy * gy) - 0.5);
    return lw - min(d1, d2);
  }
  /* Diagonals: round-capped segments, so strokes join across tile corners. */
  float best = -1e10;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float px = fx - float(dx);
      float py = fy - float(dy);
      if (lab_pat_truchet_flip(cu + float(dx), cv + float(dy), ss)) {
        px = 1.0 - px;
      }
      float t = clamp((px + py) * 0.5, 0.0, 1.0);
      float qx = px - t;
      float qy = py - t;
      best = max(best, lw - sqrt(qx * qx + qy * qy));
    }
  }
  return best;
}

/* Coverage of colour B. inv_w: 1 / edge ramp width (pattern units); (mc, ms): cos/sin of the
 * second grating angle (moire). */
float lab_pattern(int mode, vec2 uv, float duty, float inv_w, uint ss, float mc, float ms)
{
  float u = uv.x;
  float v = uv.y;
  if (mode == 0) {
    return lab_pat_aa(lab_pat_stripes_s(u, duty), inv_w);
  }
  if (mode == 1) {
    return lab_pat_aa(lab_pat_checker_s(u, v), inv_w);
  }
  if (mode == 2) {
    return lab_pat_aa(lab_pat_dots_s(u, v, duty), inv_w);
  }
  if (mode == 3) {
    return lab_pat_aa(lab_pat_hex_s(u, v, duty), inv_w);
  }
  if (mode == 4 || mode == 5) {
    return lab_pat_aa(lab_pat_truchet_s((mode == 4) ? 1 : 0, u, v, duty, ss), inv_w);
  }
  if (mode == 6) {
    float u2 = u * mc + v * ms;
    return lab_pat_aa(lab_pat_stripes_s(u, duty), inv_w) *
           lab_pat_aa(lab_pat_stripes_s(u2, duty), inv_w);
  }
  return lab_pat_aa(lab_pat_stripes_s(sqrt(u * u + v * v), duty), inv_w);
}
'''
