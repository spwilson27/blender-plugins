# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""PCG integer hashing / random numbers (GLSL). Numpy twin: ``lib/np_noise.py``.

Integer maths wraps identically in numpy (uint32) and GLSL, so hashes agree bit for bit.
"""

DEPS = ()

SOURCE = r'''
/* ---- hash.py -------------------------------------------------------------- */
uint lab_pcg(uint v)
{
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

uvec3 lab_pcg3d(uvec3 v)
{
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

/* `sseed` is a scrambled seed: lab_pcg(seed). Scramble once per noise call. */
uvec3 lab_hash3(ivec3 p, uint sseed)
{
  return lab_pcg3d(uvec3(p) + uvec3(sseed, sseed ^ 0x68E31DA4u, sseed ^ 0xB5297A4Du));
}

/* Uniform float, 0 <= x < 1, from the top 24 bits (exact in float32). */
float lab_u01(uint h)
{
  return float(h >> 8u) * (1.0 / 16777216.0);
}

vec3 lab_hash3_u01(ivec3 p, uint sseed)
{
  uvec3 h = lab_hash3(p, sseed);
  return vec3(lab_u01(h.x), lab_u01(h.y), lab_u01(h.z));
}

/* Convenience white noise for a pixel: uniform, 0 <= x < 1. */
float lab_rand(ivec2 texel, uint seed)
{
  return lab_u01(lab_hash3(ivec3(texel, 0), lab_pcg(seed)).x);
}
'''
