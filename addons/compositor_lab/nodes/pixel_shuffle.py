# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Pixel Shuffle: seeded, resolution-independent scrambling of pixels, blocks or pixel pairs.

Every output pixel *gathers* its colour from a source pixel, so the result is always a permutation
of the input. The permutations are computed per pixel (no tables) from a Feistel network over the
index of the pixel inside its block, with the shared PCG hash (``lib/glsl/hash.py`` /
``lib/np_noise.py``) as round function and cycle walking to get an exact permutation of any
domain size. CPU and GPU run the same integer maths: results are identical.

Modes
  Pixels in Blocks: the pixels of every NxN block (edge blocks are smaller) are permuted by a
      different permutation per block.
  Swap Pairs: the image is cut into blocks of (Radius + 1)^2 pixels at a seeded offset; inside
      each block the pixels are randomly paired and the pairs swap. Pixels end up at most
      ``Radius`` pixels away (Chebyshev). ``Iterations`` repeats this with new block offsets.
  Shuffle Blocks: the whole NxN blocks (only complete ones; the right / top remainder stays) are
      permuted.

``Amount`` is the fraction of elements affected (pixels of a block, pairs, or blocks): 0 is the
identity. In the shuffle modes the affected elements (those with the lowest Feistel rank) are
moved along one random cycle, so every affected element changes place. The fraction is
quantised to 1/1024. ``Animate`` adds ``floor(time * rate)`` to the seed.
"""

import math
import sys
import types

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from ..lib import glsl, gpu as lab_gpu, np_noise
from ..lib.node import In, LabNode, Out

MENU = "Utility"

U32 = np.uint32
ROUNDS = 5
WALK = 256
_GOLD = 0x9E3779B1

_GLSL = r'''
/* ---- pixel_shuffle: Feistel permutations ---------------------------------- */
uint ps_f(uint x, uint key, uint r)
{
  return lab_pcg(x ^ lab_pcg(key + r * 0x9E3779B1u));
}

uint ps_fwd(uint x, uint h, uint key)
{
  uint m = (1u << h) - 1u;
  uint l = x >> h;
  uint r = x & m;
  for (uint i = 0u; i < 5u; ++i) {
    uint t = l ^ (ps_f(r, key, i) & m);
    l = r;
    r = t;
  }
  return (l << h) | r;
}

uint ps_inv(uint x, uint h, uint key)
{
  uint m = (1u << h) - 1u;
  uint l = x >> h;
  uint r = x & m;
  for (int i = 4; i >= 0; --i) {
    uint t = r ^ (ps_f(l, key, uint(i)) & m);
    r = l;
    l = t;
  }
  return (l << h) | r;
}

uint ps_half_bits(uint M)
{
  uint bits = (M <= 1u) ? 1u : uint(findMSB(M - 1u)) + 1u;
  return (bits + 1u) / 2u;
}

/* Random permutation of [0, M) (cycle walking), i < M. */
uint ps_tau(uint i, uint M, uint h, uint key)
{
  uint x = ps_fwd(i, h, key);
  for (int k = 0; k < 256 && x >= M; ++k) {
    x = ps_fwd(x, h, key);
  }
  return x;
}

uint ps_tau_inv(uint t, uint M, uint h, uint key)
{
  uint x = ps_inv(t, h, key);
  for (int k = 0; k < 256 && x >= M; ++k) {
    x = ps_inv(x, h, key);
  }
  return x;
}

/* round(M * aq / 1024) without overflow */
uint ps_count(uint M, uint aq)
{
  return (M >> 10u) * aq + (((M & 1023u) * aq + 512u) >> 10u);
}

/* Source index of element i: the r = count(M, aq) elements with the lowest rank tau move along
 * one random cycle (rank + 1), the others stay. */
uint ps_cyc(uint i, uint M, uint key, uint aq)
{
  uint r = ps_count(M, aq);
  if (r < 2u) {
    return i;
  }
  uint h = ps_half_bits(M);
  uint t = ps_tau(i, M, h, key);
  if (t >= r) {
    return i;
  }
  uint t2 = t + 1u;
  if (t2 == r) {
    t2 = 0u;
  }
  return ps_tau_inv(t2, M, h, key);
}

ivec2 ps_block_pixels(ivec2 p, ivec2 res, int N, uint aq, uint ss)
{
  ivec2 b = p / N;
  ivec2 o = b * N;
  ivec2 ext = min(ivec2(N), res - o);
  uint M = uint(ext.x * ext.y);
  uint key = lab_hash3(ivec3(b, 1), ss).x;
  uint i = uint((p.y - o.y) * ext.x + (p.x - o.x));
  uint j = ps_cyc(i, M, key, aq);
  return o + ivec2(int(j) % ext.x, int(j) / ext.x);
}

ivec2 ps_blocks(ivec2 p, ivec2 res, int N, uint aq, uint ss)
{
  ivec2 nb = res / N;
  ivec2 b = p / N;
  if (b.x >= nb.x || b.y >= nb.y) {
    return p;
  }
  uint M = uint(nb.x * nb.y);
  uint key = lab_hash3(ivec3(0, 0, 3), ss).x;
  uint i = uint(b.y * nb.x + b.x);
  uint j = ps_cyc(i, M, key, aq);
  ivec2 sb = ivec2(int(j) % nb.x, int(j) / nb.x);
  return sb * N + (p - b * N);
}

ivec2 ps_swap_once(ivec2 p, ivec2 res, int S, int it, uint aq, uint ss)
{
  uint k0 = lab_hash3(ivec3(it, 0, 7), ss).x;
  ivec2 off = ivec2(int(lab_pcg(k0) % uint(S)), int(lab_pcg(k0 ^ 0x68E31DA4u) % uint(S)));
  ivec2 b = (p + off) / S;
  ivec2 lo = max(b * S - off, ivec2(0));
  ivec2 hi = min(b * S - off + ivec2(S), res);
  ivec2 ext = hi - lo;
  uint M = uint(ext.x * ext.y);
  uint key = lab_hash3(ivec3(b, 100 + it), ss).x;
  uint i = uint((p.y - lo.y) * ext.x + (p.x - lo.x));
  uint h = ps_half_bits(M);
  uint t = ps_tau(i, M, h, key);
  uint t2 = t ^ 1u;
  if (t2 >= M) {
    return p;
  }
  uint pair = t >> 1u;
  if ((lab_pcg(key ^ (pair * 0x9E3779B1u)) >> 22u) >= aq) {
    return p;
  }
  uint j = ps_tau_inv(t2, M, h, key);
  return lo + ivec2(int(j) % ext.x, int(j) / ext.x);
}

ivec2 ps_source(ivec2 p, ivec2 res, int mode, int N, int iters, uint aq, uint ss)
{
  if (mode == 0) {
    return ps_block_pixels(p, res, N, aq, ss);
  }
  if (mode == 2) {
    return ps_blocks(p, res, N, aq, ss);
  }
  for (int it = 0; it < iters; ++it) {
    p = ps_swap_once(p, res, N, it, aq, ss);
  }
  return p;
}
'''


def _register_glsl():
    """Make the node-local GLSL available to ``gpu.pointwise(libs=("pixel_shuffle",))``
    (``glsl.resolve`` imports ``lib/glsl/<name>``; this module is registered under that name)."""
    name = glsl.__name__ + ".pixel_shuffle"
    mod = types.ModuleType(name)
    mod.DEPS = ("hash",)
    mod.SOURCE = _GLSL
    sys.modules[name] = mod


_register_glsl()

_BODY = """
    out_Image = in_Image(ps_source(texel, res, ps_mode, ps_n, ps_iters, uint(ps_aq), uint(ps_ss)));
"""

_MODES = [
    ('BLOCK_PIXELS', "Pixels in Blocks", "Shuffle the pixels inside every NxN block"),
    ('SWAP_PAIRS', "Swap Pairs", "Swap random pairs of pixels within a radius"),
    ('BLOCKS', "Shuffle Blocks", "Shuffle whole NxN blocks"),
]
_MODE_INDEX = {'BLOCK_PIXELS': 0, 'SWAP_PAIRS': 1, 'BLOCKS': 2}


# ---------------------------------------------------------------------------
# numpy twin (uint32 arrays, 1-D)
# ---------------------------------------------------------------------------

def _f(x, key, r):
    rc = U32((r * _GOLD) & 0xFFFFFFFF)
    return np_noise.pcg(x ^ np_noise.pcg(key + rc))


def _fwd(x, h, key):
    m = (U32(1) << h) - U32(1)
    l, r = x >> h, x & m
    for i in range(ROUNDS):
        l, r = r, l ^ (_f(r, key, i) & m)
    return (l << h) | r


def _inv(x, h, key):
    m = (U32(1) << h) - U32(1)
    l, r = x >> h, x & m
    for i in range(ROUNDS - 1, -1, -1):
        l, r = r ^ (_f(l, key, i) & m), l
    return (l << h) | r


def _half_bits(m):
    e = np.frexp(m.astype(np.float64) - 1.0)[1]
    bits = np.where(m <= 1, 1, e)
    return ((bits + 1) // 2).astype(U32)


def _walk(fn, x, m, h, key):
    x = fn(x, h, key)
    for _ in range(WALK):
        bad = np.nonzero(x >= m)[0]
        if bad.size == 0:
            break
        x[bad] = fn(x[bad], h[bad], key[bad])
    return x


def tau(i, m, h, key):
    return _walk(_fwd, i, m, h, key)


def tau_inv(t, m, h, key):
    return _walk(_inv, t, m, h, key)


def count(m, aq):
    aq = U32(aq)
    return (m >> U32(10)) * aq + (((m & U32(1023)) * aq + U32(512)) >> U32(10))


def cyc(i, m, key, aq):
    """Vectorised ps_cyc: source index for each element index ``i`` (< m), arrays of one shape."""
    r = count(m, aq)
    out = i.copy()
    active = np.nonzero(r >= 2)[0]
    if active.size == 0:
        return out
    ia, ma, ka, ra = i[active], m[active], key[active], r[active]
    ha = _half_bits(ma)
    t = tau(ia, ma, ha, ka)
    sel = np.nonzero(t < ra)[0]
    if sel.size:
        t2 = t[sel] + U32(1)
        t2 = np.where(t2 == ra[sel], U32(0), t2).astype(U32)
        out[active[sel]] = tau_inv(t2, ma[sel], ha[sel], ka[sel])
    return out


def _hash(bx, by, z, ss):
    return np_noise.hash3(bx.astype(np.int32), by.astype(np.int32), int(z), ss)[0]


def _grid(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return x.ravel().astype(np.int64), y.ravel().astype(np.int64)


def source_block_pixels(w, h, n, aq, ss):
    x, y = _grid(w, h)
    bx, by = x // n, y // n
    ox, oy = bx * n, by * n
    ex, ey = np.minimum(n, w - ox), np.minimum(n, h - oy)
    m = (ex * ey).astype(U32)
    key = _hash(bx, by, 1, ss)
    i = ((y - oy) * ex + (x - ox)).astype(U32)
    j = cyc(i, m, key, aq).astype(np.int64)
    return ox + j % ex, oy + j // ex


def source_blocks(w, h, n, aq, ss):
    x, y = _grid(w, h)
    nbx, nby = w // n, h // n
    bx, by = x // n, y // n
    inside = (bx < nbx) & (by < nby)
    sx, sy = x.copy(), y.copy()
    if nbx * nby >= 2 and inside.any():
        idx = np.nonzero(inside)[0]
        m = np.full(idx.size, nbx * nby, U32)
        k = _hash(np.zeros(1, np.int64), np.zeros(1, np.int64), 3, ss)[0]
        key = np.full(idx.size, k, U32)
        i = (by[idx] * nbx + bx[idx]).astype(U32)
        j = cyc(i, m, key, aq).astype(np.int64)
        sx[idx] = (j % nbx) * n + (x[idx] - bx[idx] * n)
        sy[idx] = (j // nbx) * n + (y[idx] - by[idx] * n)
    return sx, sy


def swap_once(x, y, w, h, s, it, aq, ss):
    k0 = int(_hash(np.array([it]), np.zeros(1, np.int64), 7, ss)[0])
    ox = int(np_noise.scramble_seed(k0) % s)
    oy = int(np_noise.scramble_seed(k0 ^ 0x68E31DA4) % s)
    bx, by = (x + ox) // s, (y + oy) // s
    lox, loy = np.maximum(bx * s - ox, 0), np.maximum(by * s - oy, 0)
    ex = np.minimum(bx * s - ox + s, w) - lox
    ey = np.minimum(by * s - oy + s, h) - loy
    m = (ex * ey).astype(U32)
    key = _hash(bx, by, 100 + it, ss)
    i = ((y - loy) * ex + (x - lox)).astype(U32)
    hh = _half_bits(m)
    t = tau(i, m, hh, key)
    t2 = t ^ U32(1)
    ok = t2 < m
    pair = t >> U32(1)
    ok &= (np_noise.pcg(key ^ (pair * U32(_GOLD))) >> U32(22)) < U32(aq)
    nx, ny = x.copy(), y.copy()
    idx = np.nonzero(ok)[0]
    if idx.size:
        j = tau_inv(t2[idx], m[idx], hh[idx], key[idx]).astype(np.int64)
        nx[idx] = lox[idx] + j % ex[idx]
        ny[idx] = loy[idx] + j // ex[idx]
    return nx, ny


def source_swap(w, h, s, iters, aq, ss):
    x, y = _grid(w, h)
    for it in range(iters):
        x, y = swap_once(x, y, w, h, s, it, aq, ss)
    return x, y


class CompositorNodeLabPixelShuffle(LabNode, bpy.types.CompositorNode):
    '''Shuffle pixels inside blocks, swap pixel pairs or shuffle whole blocks (seeded)'''
    bl_idname = "CompositorNodeLabPixelShuffle"
    bl_label = "Pixel Shuffle"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        Out("Image", "COLOR"),
    ]
    PROPS = ["mode", "block_size", "radius", "iterations", "amount", "seed", "animate", "rate"]

    mode: EnumProperty(name="Mode", items=_MODES, default='BLOCK_PIXELS')
    block_size: IntProperty(name="Block Size", default=8, min=1, max=512, soft_max=64,
                            subtype='PIXEL', description="Block edge in pixels")
    radius: IntProperty(name="Radius", default=8, min=1, max=512, soft_max=64, subtype='PIXEL',
                        description="Maximum distance (Chebyshev) a pixel can move")
    iterations: IntProperty(name="Iterations", default=1, min=1, max=8,
                            description="Number of swap passes (each with new block offsets)")
    amount: FloatProperty(name="Amount", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                          description="Fraction of pixels / pairs / blocks affected")
    seed: IntProperty(name="Seed", default=0)
    animate: BoolProperty(name="Animate", default=False,
                          description="Change the seed over time")
    rate: FloatProperty(name="Rate", default=12.0, min=0.0, soft_max=60.0,
                        description="Seed changes per second when animated")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        if self.mode == 'SWAP_PAIRS':
            layout.prop(self, "radius")
            layout.prop(self, "iterations")
        else:
            layout.prop(self, "block_size")
        layout.prop(self, "amount")
        layout.prop(self, "seed")
        row = layout.row(align=True)
        row.prop(self, "animate", toggle=True)
        if self.animate:
            row.prop(self, "rate", text="")

    def _params(self, ctx):
        seed = int(self.seed)
        if self.animate:
            seed += int(math.floor(ctx.time * self.rate))
        ss = np_noise.scramble_seed(seed & 0xFFFFFFFF)
        mode = _MODE_INDEX[self.mode]
        n = int(self.radius) + 1 if mode == 1 else int(self.block_size)
        return dict(mode=mode, n=max(n, 1), iters=int(self.iterations),
                    aq=min(max(int(round(self.amount * 1024)), 0), 1024), ss=ss)

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        h, w = ctx.shape
        a = self.in_image_array(inputs, "Image", ctx.shape, 4, default=(0.5, 0.5, 0.5, 1.0))
        p = self._params(ctx)
        if p["aq"] == 0:
            out[...] = a
            return
        with np.errstate(all="ignore"):
            if p["mode"] == 0:
                sx, sy = source_block_pixels(w, h, p["n"], p["aq"], p["ss"])
            elif p["mode"] == 2:
                sx, sy = source_blocks(w, h, p["n"], p["aq"], p["ss"])
            else:
                sx, sy = source_swap(w, h, p["n"], p["iters"], p["aq"], p["ss"])
        out[...] = np.asarray(a)[sy.reshape(h, w), sx.reshape(h, w)]

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        p = self._params(ctx)
        src = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        ss = p["ss"] - (1 << 32) if p["ss"] >= (1 << 31) else p["ss"]
        lab_gpu.pointwise(
            _BODY, {"Image": dst}, inputs={"Image": ("color", src)},
            uniforms={"ps_mode": ("int", p["mode"]), "ps_n": ("int", p["n"]),
                      "ps_iters": ("int", p["iters"]), "ps_aq": ("int", p["aq"]),
                      "ps_ss": ("int", ss)},
            libs=("hash", "pixel_shuffle"))


NODE_CLASSES = [CompositorNodeLabPixelShuffle]
