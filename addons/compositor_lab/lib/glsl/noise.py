# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Procedural noise (GLSL). Numpy twin: ``lib/np_noise.py`` (same hash, gradients, formulas).

All basis functions take a float position ``p`` and a ``uint seed`` and return a *signed* value,
nominally in [-1, 1]. ``lab_noise_field`` maps the result to [0, 1].

Noise types (``type`` argument): 0 value, 1 perlin, 2 simplex, 3 worley F1, 4 worley F2-F1.
"""

DEPS = ("hash",)

# Scale factors that bring the nominal output range to [-1, 1]; measured in tests/lab/test_lib.py.
SIMPLEX2_SCALE = 70.0
SIMPLEX3_SCALE = 32.0

SOURCE = r'''
/* ---- noise.py ------------------------------------------------------------- */
vec3 lab_fade3(vec3 t)
{
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

/* ---- value noise (3D) ---- */
float lab_vcorner(ivec3 i, uint ss)
{
  return lab_u01(lab_hash3(i, ss).x) * 2.0 - 1.0;
}

float lab_value3(vec3 p, uint seed)
{
  uint ss = lab_pcg(seed);
  vec3 fl = floor(p);
  vec3 u = lab_fade3(p - fl);
  ivec3 i = ivec3(fl);
  float c000 = lab_vcorner(i, ss);
  float c100 = lab_vcorner(i + ivec3(1, 0, 0), ss);
  float c010 = lab_vcorner(i + ivec3(0, 1, 0), ss);
  float c110 = lab_vcorner(i + ivec3(1, 1, 0), ss);
  float c001 = lab_vcorner(i + ivec3(0, 0, 1), ss);
  float c101 = lab_vcorner(i + ivec3(1, 0, 1), ss);
  float c011 = lab_vcorner(i + ivec3(0, 1, 1), ss);
  float c111 = lab_vcorner(i + ivec3(1, 1, 1), ss);
  float x00 = c000 + (c100 - c000) * u.x;
  float x10 = c010 + (c110 - c010) * u.x;
  float x01 = c001 + (c101 - c001) * u.x;
  float x11 = c011 + (c111 - c011) * u.x;
  float y0 = x00 + (x10 - x00) * u.y;
  float y1 = x01 + (x11 - x01) * u.y;
  return y0 + (y1 - y0) * u.z;
}

/* ---- gradient (Perlin) noise (3D) ---- */
/* Perlin's improved-noise gradient set: 12 cube-edge directions (+4 repeats), picked from the
 * low 4 bits of the hash. Returns dot(gradient, d). */
float lab_grad3(uint h, vec3 d)
{
  uint k = (h >> 16u) & 15u;
  float u = (k < 8u) ? d.x : d.y;
  float v = (k < 4u) ? d.y : ((k == 12u || k == 14u) ? d.x : d.z);
  return (((k & 1u) == 0u) ? u : -u) + (((k & 2u) == 0u) ? v : -v);
}

float lab_gcorner(ivec3 i, uint ss, vec3 d)
{
  return lab_grad3(lab_hash3(i, ss).x, d);
}

float lab_perlin3(vec3 p, uint seed)
{
  uint ss = lab_pcg(seed);
  vec3 fl = floor(p);
  vec3 f = p - fl;
  vec3 u = lab_fade3(f);
  ivec3 i = ivec3(fl);
  float c000 = lab_gcorner(i, ss, f);
  float c100 = lab_gcorner(i + ivec3(1, 0, 0), ss, f - vec3(1.0, 0.0, 0.0));
  float c010 = lab_gcorner(i + ivec3(0, 1, 0), ss, f - vec3(0.0, 1.0, 0.0));
  float c110 = lab_gcorner(i + ivec3(1, 1, 0), ss, f - vec3(1.0, 1.0, 0.0));
  float c001 = lab_gcorner(i + ivec3(0, 0, 1), ss, f - vec3(0.0, 0.0, 1.0));
  float c101 = lab_gcorner(i + ivec3(1, 0, 1), ss, f - vec3(1.0, 0.0, 1.0));
  float c011 = lab_gcorner(i + ivec3(0, 1, 1), ss, f - vec3(0.0, 1.0, 1.0));
  float c111 = lab_gcorner(i + ivec3(1, 1, 1), ss, f - vec3(1.0, 1.0, 1.0));
  float x00 = c000 + (c100 - c000) * u.x;
  float x10 = c010 + (c110 - c010) * u.x;
  float x01 = c001 + (c101 - c001) * u.x;
  float x11 = c011 + (c111 - c011) * u.x;
  float y0 = x00 + (x10 - x00) * u.y;
  float y1 = x01 + (x11 - x01) * u.y;
  return y0 + (y1 - y0) * u.z;
}

/* ---- simplex noise (3D, Gustavson) ---- */
float lab_scorner3(ivec3 i, uint ss, vec3 d)
{
  float t = 0.6 - (d.x * d.x + d.y * d.y + d.z * d.z);
  if (t <= 0.0) {
    return 0.0;
  }
  t = t * t;
  return t * t * lab_grad3(lab_hash3(i, ss).x, d);
}

float lab_simplex3(vec3 p, uint seed)
{
  const float F3 = 1.0 / 3.0;
  const float G3 = 1.0 / 6.0;
  uint ss = lab_pcg(seed);
  float s = (p.x + p.y + p.z) * F3;
  vec3 fl = floor(p + vec3(s));
  ivec3 i = ivec3(fl);
  float t = (fl.x + fl.y + fl.z) * G3;
  vec3 x0 = p - (fl - vec3(t));
  /* Simplex corner ordering (branch free, Ashima/Gustavson). */
  float gx = (x0.x >= x0.y) ? 1.0 : 0.0;
  float gy = (x0.y >= x0.z) ? 1.0 : 0.0;
  float gz = (x0.z >= x0.x) ? 1.0 : 0.0;
  vec3 i1 = vec3(min(gx, 1.0 - gz), min(gy, 1.0 - gx), min(gz, 1.0 - gy));
  vec3 i2 = vec3(max(gx, 1.0 - gz), max(gy, 1.0 - gx), max(gz, 1.0 - gy));
  vec3 x1 = x0 - i1 + vec3(G3);
  vec3 x2 = x0 - i2 + vec3(2.0 * G3);
  vec3 x3 = x0 - vec3(1.0) + vec3(3.0 * G3);
  float n = lab_scorner3(i, ss, x0) +
            lab_scorner3(i + ivec3(i1), ss, x1) +
            lab_scorner3(i + ivec3(i2), ss, x2) +
            lab_scorner3(i + ivec3(1, 1, 1), ss, x3);
  return n * 32.0;
}

/* ---- simplex noise (2D) ---- */
float lab_grad2(uint h, vec2 d)
{
  uint k = (h >> 16u) & 7u;
  if (k < 4u) {
    return (((k & 1u) == 0u) ? d.x : -d.x) + (((k & 2u) == 0u) ? d.y : -d.y);
  }
  if (k < 6u) {
    return ((k & 1u) == 0u) ? d.x : -d.x;
  }
  return ((k & 1u) == 0u) ? d.y : -d.y;
}

float lab_scorner2(ivec2 i, uint ss, vec2 d)
{
  float t = 0.5 - (d.x * d.x + d.y * d.y);
  if (t <= 0.0) {
    return 0.0;
  }
  t = t * t;
  return t * t * lab_grad2(lab_hash3(ivec3(i, 0), ss).x, d);
}

float lab_simplex2(vec2 p, uint seed)
{
  const float F2 = 0.36602540378;
  const float G2 = 0.21132486540;
  uint ss = lab_pcg(seed);
  float s = (p.x + p.y) * F2;
  vec2 fl = floor(p + vec2(s));
  ivec2 i = ivec2(fl);
  float t = (fl.x + fl.y) * G2;
  vec2 x0 = p - (fl - vec2(t));
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec2 x1 = x0 - i1 + vec2(G2);
  vec2 x2 = x0 - vec2(1.0) + vec2(2.0 * G2);
  float n = lab_scorner2(i, ss, x0) +
            lab_scorner2(i + ivec2(i1), ss, x1) +
            lab_scorner2(i + ivec2(1, 1), ss, x2);
  return n * 70.0;
}

/* ---- Worley (cellular) noise (3D) ----
 * One jittered feature point per unit cell, 3x3x3 neighbourhood. Returns (F1, F2) distances.
 * jitter 0 = points at cell centres, 1 = fully random inside the cell. */
vec2 lab_worley3(vec3 p, uint seed, float jitter)
{
  uint ss = lab_pcg(seed);
  vec3 fl = floor(p);
  vec3 f = p - fl;
  ivec3 ci = ivec3(fl);
  float f1 = 8.0;
  float f2 = 8.0;
  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        vec3 r = lab_hash3_u01(ci + ivec3(dx, dy, dz), ss);
        vec3 pt = vec3(float(dx), float(dy), float(dz)) + vec3(0.5) + (r - vec3(0.5)) * jitter;
        vec3 d = pt - f;
        float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (d2 < f1) {
          f2 = f1;
          f1 = d2;
        }
        else if (d2 < f2) {
          f2 = d2;
        }
      }
    }
  }
  return vec2(sqrt(f1), sqrt(f2));
}

/* ---- selectable basis ---- */
float lab_basis(int type, vec3 p, uint seed, float jitter)
{
  if (type == 0) {
    return lab_value3(p, seed);
  }
  if (type == 1) {
    return lab_perlin3(p, seed);
  }
  if (type == 2) {
    return lab_simplex3(p, seed);
  }
  vec2 w = lab_worley3(p, seed, jitter);
  if (type == 3) {
    return clamp(2.0 * w.x - 1.0, -1.0, 1.0);
  }
  return clamp(2.0 * (w.y - w.x) - 1.0, -1.0, 1.0);
}

/* ---- fBm / ridged ---- */
/* Octave i samples at p * freq + i * (1.7, 9.2, 4.3) so octaves do not share lattice points.
 * Ridged: (1 - |n|)^2 per octave. Result is normalised by the amplitude sum (signed, ~[-1, 1]). */
float lab_fbm(int type, vec3 p, uint seed, float jitter, int octaves, float lacunarity,
              float gain, bool ridged)
{
  float sum = 0.0;
  float amp = 1.0;
  float norm = 0.0;
  float freq = 1.0;
  for (int i = 0; i < octaves; i++) {
    vec3 q = p * freq + vec3(1.7, 9.2, 4.3) * float(i);
    float n = lab_basis(type, q, seed, jitter);
    if (ridged) {
      n = 1.0 - abs(n);
      n = n * n * 2.0 - 1.0;
    }
    sum += n * amp;
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}

/* ---- domain warp ---- */
vec3 lab_warp(int type, vec3 p, uint seed, float jitter, float amount)
{
  if (amount == 0.0) {
    return p;
  }
  float wx = lab_basis(type, p, seed + 101u, jitter);
  float wy = lab_basis(type, p + vec3(5.2, 1.3, 3.7), seed + 202u, jitter);
  float wz = lab_basis(type, p + vec3(2.9, 7.1, 6.3), seed + 303u, jitter);
  return p + vec3(wx, wy, wz) * amount;
}

/* ---- curl of a scalar basis (2D, central differences) ---- */
vec2 lab_curl2(int type, vec3 p, uint seed, float jitter, float eps)
{
  float dx = lab_basis(type, p + vec3(eps, 0.0, 0.0), seed, jitter) -
             lab_basis(type, p - vec3(eps, 0.0, 0.0), seed, jitter);
  float dy = lab_basis(type, p + vec3(0.0, eps, 0.0), seed, jitter) -
             lab_basis(type, p - vec3(0.0, eps, 0.0), seed, jitter);
  float k = 0.5 / eps;
  return vec2(dy * k, -dx * k);
}

/* ---- complete field used by the Noise node: warp + fBm, mapped to [0, 1] ---- */
float lab_noise_field(int type, vec3 p, uint seed, float jitter, int octaves, float lacunarity,
                      float gain, bool ridged, float warp)
{
  vec3 q = lab_warp(type, p, seed, jitter, warp);
  float n = lab_fbm(type, q, seed, jitter, octaves, lacunarity, gain, ridged);
  return clamp(0.5 + 0.5 * n, 0.0, 1.0);
}
'''
