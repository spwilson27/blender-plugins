# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Gray-Scott reaction-diffusion bodies for ``gpu.kernel(..., sampling=True)``. Numpy twin:
``lib/np_rd.py`` (same maths and operation order; see there for the equations).

State: RG32F texture, (U, V). Bodies (all expect ``libs=("hash",)``, ``sampling=True``):

* ``INIT_BODY``  input ``Seed``; writes ``State``. Uniforms: ``d_has_seed``, ``d_sratio`` (seed
  size / simulation size), ``d_seed`` (int), ``d_density``, ``d_noise``.
* ``STEP_BODY``  inputs ``Prev``, ``FMap``, ``KMap``; writes ``State``. Uniforms: ``d_edge``
  (0 clamp, 1 repeat), ``d_feed``, ``d_kill``, ``d_du``, ``d_dv``, ``d_dt``, ``d_has_f``,
  ``d_has_k`` (map is a texture), ``d_fratio`` / ``d_kratio`` (map size / simulation size).
* ``output_body(v, u, color)``  input ``State``; samples bilinearly at ``d_up`` (simulation size /
  output size) and writes the used ones of ``V``, ``U``, ``Color`` (``d_a``, ``d_b``, ``d_low``,
  ``d_inv``).
"""

DEPS = ("hash",)

SOURCE = ""

INIT_BODY = r'''
    bool seeded = false;
    if (d_has_seed != 0) {
      vec4 c = lab_bilinear_Seed((vec2(texel) + vec2(0.5)) * d_sratio, 0);
      float lum = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
      seeded = lum > 0.5;
    }
    else {
      uint ss = lab_pcg(uint(d_seed));
      ivec2 cell = texel / 8;
      uvec3 h = lab_hash3(ivec3(cell, 0), ss);
      if (lab_u01(h.z) < d_density) {
        ivec2 ctr = cell * 8 + ivec2(2) + ivec2(int(h.x & 3u), int(h.y & 3u));
        ivec2 dd = ctr - texel;
        seeded = (dd.x * dd.x + dd.y * dd.y) <= 5;
      }
    }
    if (seeded) {
      out_State = vec4(0.5, 1.0, 0.0, 0.0);
    }
    else {
      uint ss2 = lab_pcg(uint(d_seed));
      float n = lab_u01(lab_hash3(ivec3(texel, 1), ss2).x) * d_noise;
      out_State = vec4(1.0, n, 0.0, 0.0);
    }
'''

STEP_BODY = r'''
    vec2 c = lab_fetch_Prev(texel).rg;
    vec2 orth = lab_px_Prev(texel + ivec2(0, -1), d_edge).rg
              + lab_px_Prev(texel + ivec2(0, 1), d_edge).rg
              + lab_px_Prev(texel + ivec2(-1, 0), d_edge).rg
              + lab_px_Prev(texel + ivec2(1, 0), d_edge).rg;
    vec2 diag = lab_px_Prev(texel + ivec2(-1, -1), d_edge).rg
              + lab_px_Prev(texel + ivec2(1, -1), d_edge).rg
              + lab_px_Prev(texel + ivec2(-1, 1), d_edge).rg
              + lab_px_Prev(texel + ivec2(1, 1), d_edge).rg;
    vec2 lap = 0.2 * orth + 0.05 * diag - c;
    float f = d_feed;
    float k = d_kill;
    if (d_has_f != 0) {
      f = f * lab_bilinear_FMap((vec2(texel) + vec2(0.5)) * d_fratio, 0).r;
    }
    if (d_has_k != 0) {
      k = k * lab_bilinear_KMap((vec2(texel) + vec2(0.5)) * d_kratio, 0).r;
    }
    float uvv = c.x * c.y * c.y;
    float du = d_du * lap.x - uvv + f * (1.0 - c.x);
    float dv = d_dv * lap.y + uvv - (f + k) * c.y;
    out_State = vec4(clamp(c.x + d_dt * du, 0.0, 1.0), clamp(c.y + d_dt * dv, 0.0, 1.0), 0.0, 0.0);
'''


def output_body(v, u, color):
    s = "    vec2 st = lab_bilinear_State((vec2(texel) + vec2(0.5)) * d_up, d_edge).rg;\n"
    if v:
        s += "    out_V = vec4(st.y);\n"
    if u:
        s += "    out_U = vec4(st.x);\n"
    if color:
        s += ("    float t = clamp((st.y - d_low) * d_inv, 0.0, 1.0);\n"
              "    t = t * t * (3.0 - 2.0 * t);\n"
              "    out_Color = d_a + (d_b - d_a) * t;\n")
    return s
