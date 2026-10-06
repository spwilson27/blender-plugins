# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""GPU parallel reduction kernels (compute shader bodies). Driver: ``lib/reduce.py``.

No atomics are used: one thread reduces one tile, and a chain of passes folds the tile results
until one texel is left. Each kernel is a template: ``@TILE@`` and ``@NBINS@`` are substituted
by the driver, and ``lib/glsl/color.py`` (for ``lab_luma``) is prepended.

stats (thread per TILE x TILE tile of the source)         -> 5 tiles textures
    o_min  RGBA32F  per-channel minimum          o_max  RGBA32F  per-channel maximum
    o_mean RGBA32F  per-channel mean             o_cnt  R32F     pixel count
    o_lum  RGBA32F  (luma min, luma max, luma mean, luma sum of squared deviations)
reduce (same inputs/outputs; merges TILE x TILE texels, mean / M2 merged with Chan's formula)
hist   (thread per tile; private histogram -> one row of R32UI per tile) and hist_reduce
(thread per bin, sums the rows).

``straight != 0``: pixels with alpha <= 0 are skipped and the others are un-premultiplied.
"""

DEPS = ("color",)
SOURCE = ""

_PIXEL = r'''
bool lab_valid(vec4 s)
{
  return straight == 0 || s.a > 0.0;
}

vec4 lab_conv(vec4 s)
{
  return (straight != 0) ? vec4(s.rgb / s.a, s.a) : s;
}
'''

STATS = _PIXEL + r'''
void main()
{
  ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
  ivec2 dim = textureSize(src, 0);
  ivec2 nt = (dim + ivec2(@TILE@ - 1)) / ivec2(@TILE@);
  if (tile.x >= nt.x || tile.y >= nt.y) {
    return;
  }
  ivec2 p0 = tile * @TILE@;
  ivec2 p1 = min(p0 + ivec2(@TILE@), dim);
  vec4 mn = vec4(1.0e30);
  vec4 mx = vec4(-1.0e30);
  float lmn = 1.0e30;
  float lmx = -1.0e30;
  vec4 sum = vec4(0.0);
  float lsum = 0.0;
  float n = 0.0;
  vec4 c;
  for (int y = p0.y; y < p1.y; y++) {
    for (int x = p0.x; x < p1.x; x++) {
      vec4 s = texelFetch(src, ivec2(x, y), 0);
      if (!lab_valid(s)) {
        continue;
      }
      c = lab_conv(s);
      float l = lab_luma(c.rgb);
      mn = min(mn, c);
      mx = max(mx, c);
      lmn = min(lmn, l);
      lmx = max(lmx, l);
      sum += c;
      lsum += l;
      n += 1.0;
    }
  }
  vec4 mean = (n > 0.0) ? sum / n : vec4(0.0);
  float lmean = (n > 0.0) ? lsum / n : 0.0;
  float m2 = 0.0;
  for (int y = p0.y; y < p1.y; y++) {
    for (int x = p0.x; x < p1.x; x++) {
      vec4 s = texelFetch(src, ivec2(x, y), 0);
      if (!lab_valid(s)) {
        continue;
      }
      c = lab_conv(s);
      float d = lab_luma(c.rgb) - lmean;
      m2 += d * d;
    }
  }
  imageStore(o_min, tile, mn);
  imageStore(o_max, tile, mx);
  imageStore(o_mean, tile, mean);
  imageStore(o_lum, tile, vec4(lmn, lmx, lmean, m2));
  imageStore(o_cnt, tile, vec4(n, 0.0, 0.0, 0.0));
}
'''

REDUCE = r'''
void main()
{
  ivec2 o = ivec2(gl_GlobalInvocationID.xy);
  ivec2 dim = textureSize(s_cnt, 0);
  ivec2 nt = (dim + ivec2(@TILE@ - 1)) / ivec2(@TILE@);
  if (o.x >= nt.x || o.y >= nt.y) {
    return;
  }
  ivec2 p0 = o * @TILE@;
  ivec2 p1 = min(p0 + ivec2(@TILE@), dim);
  vec4 mn = vec4(1.0e30);
  vec4 mx = vec4(-1.0e30);
  float lmn = 1.0e30;
  float lmx = -1.0e30;
  vec4 mean = vec4(0.0);
  float lmean = 0.0;
  float m2 = 0.0;
  float n = 0.0;
  for (int y = p0.y; y < p1.y; y++) {
    for (int x = p0.x; x < p1.x; x++) {
      ivec2 p = ivec2(x, y);
      float nb = texelFetch(s_cnt, p, 0).r;
      if (nb <= 0.0) {
        continue;
      }
      mn = min(mn, texelFetch(s_min, p, 0));
      mx = max(mx, texelFetch(s_max, p, 0));
      vec4 lm = texelFetch(s_lum, p, 0);
      lmn = min(lmn, lm.x);
      lmx = max(lmx, lm.y);
      float ntot = n + nb;
      float f = nb / ntot;
      mean += (texelFetch(s_mean, p, 0) - mean) * f;
      float dl = lm.z - lmean;
      m2 += lm.w + dl * dl * n * f;
      lmean += dl * f;
      n = ntot;
    }
  }
  imageStore(o_min, o, mn);
  imageStore(o_max, o, mx);
  imageStore(o_mean, o, mean);
  imageStore(o_lum, o, vec4(lmn, lmx, lmean, m2));
  imageStore(o_cnt, o, vec4(n, 0.0, 0.0, 0.0));
}
'''

HIST = _PIXEL + r'''
void main()
{
  ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
  ivec2 dim = textureSize(src, 0);
  ivec2 nt = (dim + ivec2(tile_size - 1)) / ivec2(tile_size);
  if (tile.x >= nt.x || tile.y >= nt.y) {
    return;
  }
  uint h[@NBINS@];
  for (int i = 0; i < @NBINS@; i++) {
    h[i] = 0u;
  }
  ivec2 p0 = tile * tile_size;
  ivec2 p1 = min(p0 + ivec2(tile_size), dim);
  vec4 c;
  for (int y = p0.y; y < p1.y; y++) {
    for (int x = p0.x; x < p1.x; x++) {
      vec4 s = texelFetch(src, ivec2(x, y), 0);
      if (!lab_valid(s)) {
        continue;
      }
      c = lab_conv(s);
      float v = (chan == 3) ? lab_luma(c.rgb) : c[chan];
      float bf = floor((v - lo) * inv);
      int b = int(clamp(bf, 0.0, float(@NBINS@ - 1)));
      h[b] += 1u;
    }
  }
  int row = tile.y * nt.x + tile.x;
  for (int i = 0; i < @NBINS@; i++) {
    imageStore(o_tiles, ivec2(i, row), uvec4(h[i], 0u, 0u, 0u));
  }
}
'''

HIST_REDUCE = r'''
void main()
{
  int bin = int(gl_GlobalInvocationID.x);
  if (bin >= @NBINS@) {
    return;
  }
  uint sum = 0u;
  for (int r = 0; r < rows; r++) {
    sum += imageLoad(o_tiles, ivec2(bin, r)).x;
  }
  imageStore(o_hist, ivec2(bin, 0), uvec4(sum, 0u, 0u, 0u));
}
'''
