# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Exact-math helpers for bit-exact agreement with numpy (needs the ``lab_zero`` uniform, which
``gpu.pointwise`` always provides, set to 0).

Metal's fast-math contracts ``a * b + c`` into ``fma`` and divides approximately, so a value that
numpy rounds twice can differ in the last bit; that flips ties and comparisons. ``lab_rnd32`` makes
the compiler round to float32 at that point; ``lab_div_exact`` is a correctly rounded division.
Only use these where exactness matters (sorting keys, thresholds); they cost a few instructions.
"""

DEPS = ()

SOURCE = r'''
/* ---- exact.py ------------------------------------------------------------- */
float lab_rnd32(float x)
{
  return intBitsToFloat(floatBitsToInt(x) ^ lab_zero);
}

float lab_div_exact(float a, float b)
{
  float y = lab_rnd32(1.0 / b);
  float e = lab_rnd32(fma(-b, y, 1.0));
  y = lab_rnd32(fma(y, e, y));
  float q = lab_rnd32(a * y);
  float r = lab_rnd32(fma(-b, q, a));
  return fma(r, y, q);
}
'''
