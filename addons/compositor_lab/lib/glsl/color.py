# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Colour space helpers (GLSL). Numpy twin: ``lib/np_color.py``.

Everything operates on *straight* (un-premultiplied) scene-linear RGB unless noted.
"""

DEPS = ()

# Linear sRGB -> OKLab (Bjorn Ottosson). The inverse matrices are the published ones.
SOURCE = r'''
/* ---- color.py ------------------------------------------------------------- */
float lab_mod(float x, float y)
{
  return x - y * floor(x / y);
}

float lab_luma(vec3 c)
{
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

float lab_srgb_to_linear1(float c)
{
  float a = abs(c);
  float r = (a <= 0.04045) ? a / 12.92 : pow((a + 0.055) / 1.055, 2.4);
  return (c < 0.0) ? -r : r;
}

float lab_linear_to_srgb1(float c)
{
  float a = abs(c);
  float r = (a <= 0.0031308) ? a * 12.92 : 1.055 * pow(a, 1.0 / 2.4) - 0.055;
  return (c < 0.0) ? -r : r;
}

vec3 lab_srgb_to_linear(vec3 c)
{
  return vec3(lab_srgb_to_linear1(c.x), lab_srgb_to_linear1(c.y), lab_srgb_to_linear1(c.z));
}

vec3 lab_linear_to_srgb(vec3 c)
{
  return vec3(lab_linear_to_srgb1(c.x), lab_linear_to_srgb1(c.y), lab_linear_to_srgb1(c.z));
}

/* HSV: h in [0, 1), s, v. */
vec3 lab_rgb_to_hsv(vec3 c)
{
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float d = mx - mn;
  float h = 0.0;
  if (d > 0.0) {
    if (mx == c.r) {
      h = (c.g - c.b) / d;
      if (h < 0.0) {
        h += 6.0;
      }
    }
    else if (mx == c.g) {
      h = (c.b - c.r) / d + 2.0;
    }
    else {
      h = (c.r - c.g) / d + 4.0;
    }
    h = h / 6.0;
  }
  float s = (mx > 0.0) ? d / mx : 0.0;
  return vec3(h, s, mx);
}

float lab_hsv_chan(float h6, float off)
{
  return clamp(abs(lab_mod(h6 + off, 6.0) - 3.0) - 1.0, 0.0, 1.0);
}

vec3 lab_hsv_to_rgb(vec3 hsv)
{
  float h6 = (hsv.x - floor(hsv.x)) * 6.0;
  vec3 k = vec3(lab_hsv_chan(h6, 0.0), lab_hsv_chan(h6, 4.0), lab_hsv_chan(h6, 2.0));
  return hsv.z * (vec3(1.0) + (k - vec3(1.0)) * hsv.y);
}

float lab_cbrt(float x)
{
  if (x == 0.0) {
    return 0.0;
  }
  float r = pow(abs(x), 1.0 / 3.0);
  return (x < 0.0) ? -r : r;
}

/* Linear sRGB <-> OKLab (L, a, b). */
vec3 lab_linear_to_oklab(vec3 c)
{
  float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  l = lab_cbrt(l);
  m = lab_cbrt(m);
  s = lab_cbrt(s);
  return vec3(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
              1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
              0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

vec3 lab_oklab_to_linear(vec3 lab)
{
  float l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  l = l * l * l;
  m = m * m * m;
  s = s * s * s;
  return vec3(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
              -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
              -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

/* OKLCh: (L, C, h) with h in radians (-pi, pi]. */
vec3 lab_oklab_to_oklch(vec3 lab)
{
  return vec3(lab.x, sqrt(lab.y * lab.y + lab.z * lab.z), atan(lab.z, lab.y));
}

vec3 lab_oklch_to_oklab(vec3 lch)
{
  return vec3(lch.x, lch.y * cos(lch.z), lch.y * sin(lch.z));
}

vec3 lab_linear_to_oklch(vec3 c)
{
  return lab_oklab_to_oklch(lab_linear_to_oklab(c));
}

vec3 lab_oklch_to_linear(vec3 lch)
{
  return lab_oklab_to_linear(lab_oklch_to_oklab(lch));
}
'''
