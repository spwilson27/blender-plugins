# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Cellular automata helpers (GLSL). Numpy twin: ``lib/np_ca.py``.

A cell is an RG32F texel: ``r`` = state (0 dead, 1 alive, 2.. dying), ``g`` = age. All state logic
is integer maths, so it matches numpy exactly. Needs ``in_Prev`` (the state texture) in the shader.
"""

DEPS = ()

SOURCE = r'''
/* ---- ca.py ---------------------------------------------------------------- */
int lab_ca_state(ivec2 p)
{
  return int(in_Prev(p).r + 0.5);
}

/* Alive (state 1) neighbours of `p` in a grid of size `g`. */
int lab_ca_count(ivec2 p, ivec2 g, int moore, int wrap)
{
  int cnt = 0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      if (moore == 0 && dx != 0 && dy != 0) {
        continue;
      }
      ivec2 q = p + ivec2(dx, dy);
      if (wrap != 0) {
        q = (q + g) % g;
      }
      else if (q.x < 0 || q.y < 0 || q.x >= g.x || q.y >= g.y) {
        continue;
      }
      if (lab_ca_state(q) == 1) {
        cnt++;
      }
    }
  }
  return cnt;
}

/* One generation of the cell at `p`: returns (state, age). */
vec2 lab_ca_step(ivec2 p, ivec2 g, int birth, int survive, int states, int moore, int wrap)
{
  vec4 c = in_Prev(p);
  int s = int(c.r + 0.5);
  int n = lab_ca_count(p, g, moore, wrap);
  int ns;
  if (s == 0) {
    ns = ((birth >> n) & 1);
  }
  else if (s == 1) {
    ns = (((survive >> n) & 1) != 0) ? 1 : ((states > 2) ? 2 : 0);
  }
  else {
    ns = (s + 1 >= states) ? 0 : s + 1;
  }
  float age = 0.0;
  if (ns != 0 && s != 0) {
    age = min(c.g + 1.0, 65535.0);
  }
  return vec2(float(ns), age);
}
'''
