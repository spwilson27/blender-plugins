# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Dither thresholds (GLSL). Numpy twin: ``lib/np_dither.py``.

``lab_dither_threshold(mode, texel, seed)`` returns a threshold t in [0, 1) per pixel; use it
as ``floor(v * (levels - 1) + t) / (levels - 1)``. Mode 0 (none) gives 0.5, i.e. plain rounding.
Modes: 0 none, 1/2/3 ordered Bayer 2x2/4x4/8x8, 4 interleaved low-discrepancy (R2, blue-noise
like), 5 white-noise hash (per-pixel random, seeded). All integer maths: bit exact vs numpy.
"""

from .. import np_dither

DEPS = ("hash",)

SOURCE = r'''
/* ---- dither.py ------------------------------------------------------------ */
uint lab_bayer_index(uvec2 p, int levels)
{
  uint v = 0u;
  for (int i = 0; i < levels; i++) {
    uint xb = (p.x >> uint(i)) & 1u;
    uint yb = ((p.x ^ p.y) >> uint(i)) & 1u;
    v = (v << 2u) | (yb << 1u) | xb;
  }
  return v;
}

float lab_dither_threshold(int mode, ivec2 texel, uint seed)
{
  if (mode >= 1 && mode <= 3) {
    int levels = mode;
    uint idx = lab_bayer_index(uvec2(texel) & uvec2(uint((1 << levels) - 1)), levels);
    return (float(idx) + 0.5) * (1.0 / float(1 << (2 * levels)));
  }
  if (mode == 4) {
    uint u = uint(texel.x) * __R2_A__u + uint(texel.y) * __R2_B__u + lab_pcg(seed);
    return lab_u01(u);
  }
  if (mode == 5) {
    return lab_rand(texel, seed);
  }
  return 0.5;
}
'''.replace("__R2_A__", str(np_dither.R2_A)).replace("__R2_B__", str(np_dither.R2_B))
