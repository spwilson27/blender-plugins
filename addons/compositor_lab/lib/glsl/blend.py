# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Blend modes (GLSL). Numpy twin: ``lib/np_blend.py``.

Terminology (W3C Compositing): ``cb`` is the backdrop (input A), ``cs`` the source (input B,
drawn on top). ``MODES`` fixes the integer used by the GLSL ``mode`` argument.
"""

DEPS = ("color",)

MODES = (
    "NORMAL", "MULTIPLY", "SCREEN", "OVERLAY", "SOFT_LIGHT", "SOFT_LIGHT_PEGTOP", "HARD_LIGHT",
    "VIVID_LIGHT", "LINEAR_LIGHT", "PIN_LIGHT", "HARD_MIX", "COLOR_DODGE", "COLOR_BURN",
    "LINEAR_DODGE", "LINEAR_BURN", "SUBTRACT", "DIVIDE", "DIFFERENCE", "EXCLUSION", "DARKEN",
    "LIGHTEN", "DARKER_COLOR", "LIGHTER_COLOR", "HUE", "SATURATION", "COLOR", "LUMINOSITY",
)

# Modes whose formulas are only defined on [0, 1]: operands and result are clamped to [0, 1].
BOUNDED = frozenset({
    "SCREEN", "OVERLAY", "SOFT_LIGHT", "SOFT_LIGHT_PEGTOP", "HARD_LIGHT", "VIVID_LIGHT",
    "LINEAR_LIGHT", "PIN_LIGHT", "HARD_MIX", "COLOR_DODGE", "COLOR_BURN", "EXCLUSION",
})

# Chroma below this is treated as "no hue" by the OKLCh modes.
CHROMA_EPS = 1e-5

SOURCE = r'''
/* ---- blend.py ------------------------------------------------------------- */
float lab_bl_dodge(float cb, float cs)
{
  if (cb <= 0.0) {
    return 0.0;
  }
  if (cs >= 1.0) {
    return 1.0;
  }
  return min(1.0, cb / (1.0 - cs));
}

float lab_bl_burn(float cb, float cs)
{
  if (cb >= 1.0) {
    return 1.0;
  }
  if (cs <= 0.0) {
    return 0.0;
  }
  return 1.0 - min(1.0, (1.0 - cb) / cs);
}

float lab_bl_sep(int mode, float cb, float cs)
{
  if (mode == 0) { return cs; }
  if (mode == 1) { return cb * cs; }
  if (mode == 2) { return cb + cs - cb * cs; }
  if (mode == 3) {
    return (cb <= 0.5) ? 2.0 * cb * cs : 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs);
  }
  if (mode == 4) {
    if (cs <= 0.5) {
      return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
    }
    float d = (cb <= 0.25) ? ((16.0 * cb - 12.0) * cb + 4.0) * cb : sqrt(cb);
    return cb + (2.0 * cs - 1.0) * (d - cb);
  }
  if (mode == 5) { return (1.0 - 2.0 * cs) * cb * cb + 2.0 * cs * cb; }
  if (mode == 6) {
    return (cs <= 0.5) ? 2.0 * cb * cs : 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs);
  }
  if (mode == 7) {
    return (cs <= 0.5) ? lab_bl_burn(cb, 2.0 * cs) : lab_bl_dodge(cb, 2.0 * cs - 1.0);
  }
  if (mode == 8) { return cb + 2.0 * cs - 1.0; }
  if (mode == 9) {
    return (cs <= 0.5) ? min(cb, 2.0 * cs) : max(cb, 2.0 * cs - 1.0);
  }
  if (mode == 10) { return (cb + cs >= 1.0) ? 1.0 : 0.0; }
  if (mode == 11) { return lab_bl_dodge(cb, cs); }
  if (mode == 12) { return lab_bl_burn(cb, cs); }
  if (mode == 13) { return cb + cs; }
  if (mode == 14) { return cb + cs - 1.0; }
  if (mode == 15) { return cb - cs; }
  if (mode == 16) {
    if (cs > 0.0) {
      return cb / cs;
    }
    return (cb > 0.0) ? 1.0 : 0.0;
  }
  if (mode == 17) { return abs(cb - cs); }
  if (mode == 18) { return cb + cs - 2.0 * cb * cs; }
  if (mode == 19) { return min(cb, cs); }
  return max(cb, cs);  /* 20 */
}

/* Non-separable modes. Hue / Saturation / Color / Luminosity are the OKLCh component swaps
 * (L, C, h), done in OKLab Cartesian form: no trigonometry, same result. */
vec3 lab_bl_nonsep(int mode, vec3 cb, vec3 cs)
{
  if (mode == 21) {
    return (lab_luma(cs) < lab_luma(cb)) ? cs : cb;
  }
  if (mode == 22) {
    return (lab_luma(cs) > lab_luma(cb)) ? cs : cb;
  }
  vec3 ob = lab_linear_to_oklab(cb);
  vec3 os = lab_linear_to_oklab(cs);
  float chb = sqrt(ob.y * ob.y + ob.z * ob.z);
  float chs = sqrt(os.y * os.y + os.z * os.z);
  vec3 r;
  if (mode == 23) {
    /* Hue of B, chroma and lightness of A. */
    vec2 ab = (chs > 1e-5) ? vec2(os.y, os.z) * (chb / chs) : vec2(0.0);
    r = vec3(ob.x, ab.x, ab.y);
  }
  else if (mode == 24) {
    /* Chroma of B, hue and lightness of A. */
    vec2 ab = (chb > 1e-5) ? vec2(ob.y, ob.z) * (chs / chb) : vec2(0.0);
    r = vec3(ob.x, ab.x, ab.y);
  }
  else if (mode == 25) {
    /* Hue and chroma of B, lightness of A. */
    r = vec3(ob.x, os.y, os.z);
  }
  else {
    /* Lightness of B, hue and chroma of A. */
    r = vec3(os.x, ob.y, ob.z);
  }
  return max(lab_oklab_to_linear(r), vec3(0.0));
}

/* Straight-colour blend B(cb, cs). */
vec3 lab_blend_rgb(int mode, vec3 cb, vec3 cs)
{
  if (mode >= 21) {
    return lab_bl_nonsep(mode, cb, cs);
  }
  bool bounded = (mode >= 2 && mode <= 12) || mode == 18;
  if (bounded) {
    cb = clamp(cb, vec3(0.0), vec3(1.0));
    cs = clamp(cs, vec3(0.0), vec3(1.0));
  }
  vec3 r = vec3(lab_bl_sep(mode, cb.x, cs.x), lab_bl_sep(mode, cb.y, cs.y),
                lab_bl_sep(mode, cb.z, cs.z));
  if (bounded) {
    r = clamp(r, vec3(0.0), vec3(1.0));
  }
  return r;
}

/* Premultiplied RGBA blend with W3C source-over compositing, then mix by fac:
 *   cb = A.rgb / A.a, cs = B.rgb / B.a (straight)
 *   rgb' = (1 - aB) A.rgb + (1 - aA) B.rgb + aA aB B(cb, cs)      a' = aA + aB - aA aB
 *   result = A + (premul(rgb', a') - A) * fac */
vec4 lab_blend_pm(int mode, vec4 a, vec4 b, float fac)
{
  float aa = a.a;
  float ab = b.a;
  vec3 cb = (aa > 0.0) ? a.rgb / aa : vec3(0.0);
  vec3 cs = (ab > 0.0) ? b.rgb / ab : vec3(0.0);
  vec3 bl = lab_blend_rgb(mode, cb, cs);
  vec3 rgb = a.rgb * (1.0 - ab) + b.rgb * (1.0 - aa) + bl * (aa * ab);
  vec4 full = vec4(rgb, aa + ab - aa * ab);
  return a + (full - a) * fac;
}
'''
