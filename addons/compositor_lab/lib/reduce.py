# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Image reductions: min / max / mean / luminance variance and histograms, CPU and GPU.

    st = reduce.stats_cpu(img)              # img: (H, W, 4) float32 -> Stats
    st = reduce.stats_gpu(texture)          # same result from a GPUTexture (RGBA32F or RGBA16F)
    h = reduce.hist_cpu(img, chan, lo, hi)  # counts over NBINS equal bins of [lo, hi]
    h = reduce.hist_gpu(texture, chan, lo, hi)
    reduce.percentile(h, lo, hi, 99.0)      # value below which 99 % of the pixels lie

``chan``: 0..2 = r, g, b, 3 = luminance (Rec. 709). With ``straight=True`` pixels with alpha <= 0
are ignored and the rest are un-premultiplied first (statistics of the straight colour).

GPU: no atomics. ``stats_gpu`` runs a tile pass (one thread per 16x16 tile) and a chain of merge
passes until one texel is left (means and variances are merged with Chan's formula, so there is no
catastrophic cancellation); ``hist_gpu`` has one thread per tile build a private histogram, then
one thread per bin sums the tiles. Only the final 1x1 / NBINS x 1 textures are read back.
Float32 accumulation order differs from numpy, so means agree to ~1e-6 relative (min / max /
count / histogram counts are exact, except that a pixel exactly on a bin edge may land in the
neighbouring bin when the GPU divides ``rgb / a`` with a different rounding).
"""

import numpy as np

from . import glsl as _glsl
from . import gpu as lab_gpu
from .glsl import reduce as _src

F32 = np.float32

NBINS = 256
TILE = 16


class Stats:
    """count: number of pixels counted; min / max / mean: 4-vectors (r, g, b, a);
    lmin / lmax / lmean / lvar: luminance minimum, maximum, mean and (population) variance."""

    def __init__(self, count, mn, mx, mean, lmin, lmax, lmean, lvar):
        self.count = int(count)
        self.min = np.asarray(mn, np.float64)
        self.max = np.asarray(mx, np.float64)
        self.mean = np.asarray(mean, np.float64)
        self.lmin = float(lmin)
        self.lmax = float(lmax)
        self.lmean = float(lmean)
        self.lvar = float(lvar)

    @property
    def lstd(self):
        return float(np.sqrt(max(self.lvar, 0.0)))

    def __repr__(self):
        return "Stats(n=%d, min=%s, max=%s, mean=%s, luma mean=%.6g std=%.6g)" % (
            self.count, self.min, self.max, self.mean, self.lmean, self.lstd)


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F32)


def _pixels(img, straight):
    """(N, 4) float32 pixels that count (straight colour if requested)."""
    px = np.asarray(img, dtype=F32).reshape(-1, 4)
    if straight:
        px = px[px[:, 3] > 0]
        px = np.concatenate([px[:, :3] / px[:, 3:4], px[:, 3:4]], axis=1)
    return px


def _luma(px):
    return _LUMA[0] * px[:, 0] + _LUMA[1] * px[:, 1] + _LUMA[2] * px[:, 2]


def stats_cpu(img, straight=False):
    px = _pixels(img, straight)
    n = px.shape[0]
    if n == 0:
        return Stats(0, np.zeros(4), np.zeros(4), np.zeros(4), 0, 0, 0, 0)
    lum = _luma(px)
    l64 = lum.astype(np.float64)
    return Stats(n, px.min(axis=0), px.max(axis=0), px.mean(axis=0, dtype=np.float64),
                 lum.min(), lum.max(), l64.mean(), l64.var())


def _bins_cpu(px, chan, lo, hi):
    v = _luma(px) if chan == 3 else px[:, chan]
    inv = hist_scale(lo, hi)
    b = np.floor((v - F32(lo)) * F32(inv))
    return np.clip(b, 0, NBINS - 1).astype(np.int64)


def hist_scale(lo, hi):
    """float32 bins-per-unit factor used by both backends (0 if the range is empty)."""
    lo, hi = F32(lo), F32(hi)
    if not hi > lo:
        return 0.0
    return float(F32(NBINS) / (hi - lo))


def hist_cpu(img, chan, lo, hi, straight=False):
    px = _pixels(img, straight)
    if px.shape[0] == 0:
        return np.zeros(NBINS, np.int64)
    return np.bincount(_bins_cpu(px, chan, lo, hi), minlength=NBINS).astype(np.int64)


def percentile(hist, lo, hi, p):
    """p-th percentile (0..100) from a histogram over [lo, hi], interpolated inside the bin.
    p <= 0 gives lo and p >= 100 gives hi."""
    hist = np.asarray(hist, np.float64)
    total = hist.sum()
    lo, hi = float(lo), float(hi)
    if total <= 0 or not hi > lo or p <= 0:
        return lo
    if p >= 100:
        return hi
    target = p / 100.0 * total
    cum = np.cumsum(hist)
    b = int(min(np.searchsorted(cum, target, side="left"), NBINS - 1))
    prev = cum[b - 1] if b > 0 else 0.0
    frac = (target - prev) / hist[b] if hist[b] > 0 else 0.0
    return float(min(max(lo + (b + frac) * (hi - lo) / NBINS, lo), hi))


def hist_mean(hist, lo, hi, transform=None):
    """Mean of the bin centres (optionally ``transform(centres)``) weighted by the counts."""
    hist = np.asarray(hist, np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    centres = lo + (np.arange(NBINS) + 0.5) * (hi - lo) / NBINS
    if transform is not None:
        centres = transform(centres)
    return float((hist * centres).sum() / total)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

_scratch = {}


def _tex(size, fmt, index=0):
    """Cached scratch texture (``index`` distinguishes several textures of one size)."""
    import gpu

    key = (size, fmt, index)
    tex = _scratch.get(key)
    if tex is None:
        if len(_scratch) > 64:
            _scratch.clear()
        tex = gpu.types.GPUTexture(size, format=fmt)
        _scratch[key] = tex
    return tex


def _read(tex):
    """GPUTexture -> numpy array (h, w, channels)."""
    arr = np.array(tex.read().to_list())
    w, h = tex.width, tex.height
    return arr.reshape(h, w, -1)


_set = lab_gpu.set_if_present


def _nt(w, h, tile=TILE):
    return (w + tile - 1) // tile, (h + tile - 1) // tile


def _kernel(name, source, samplers, images, consts, local_size):
    """Cached compute shader. images: [(name, fmt, kind, qualifiers)]."""
    key = ("lab_reduce", name, NBINS, TILE)

    def factory():
        src = _glsl.resolve(*_src.DEPS) + "\n" + source.replace("@TILE@", str(TILE)).replace(
            "@NBINS@", str(NBINS))
        info = lab_gpu.create_info(local_size)
        for i, s in enumerate(samplers):
            info.sampler(i, 'FLOAT_2D', s)
        for i, (n, fmt, kind, q) in enumerate(images):
            info.image(i, fmt, kind, n, qualifiers=set(q))
        for n, t in consts:
            info.push_constant(t, n)
        return lab_gpu.compile_shader(info, src, "reduce " + name)

    return lab_gpu.get_shader(key, factory)


_STATS_OUT = [("o_min", "RGBA32F"), ("o_max", "RGBA32F"), ("o_mean", "RGBA32F"),
              ("o_lum", "RGBA32F"), ("o_cnt", "RGBA32F")]


def _level_textures(w, h):
    return [_tex((w, h), "RGBA32F", i) for i in range(len(_STATS_OUT))]


def stats_gpu(src, straight=False):
    """Statistics of a GPUTexture (see module docstring). Result equals ``stats_cpu``."""
    import gpu

    w, h = int(src.width), int(src.height)
    out_imgs = [(n, "RGBA32F", "FLOAT_2D", ("WRITE",)) for n, _ in _STATS_OUT]
    stats = _kernel("stats", _src.STATS, ["src"], out_imgs, [("straight", "INT")], (8, 8, 1))
    reduce_ = _kernel("reduce", _src.REDUCE,
                      ["s_min", "s_max", "s_mean", "s_lum", "s_cnt"], out_imgs, [], (8, 8, 1))
    tw, th = _nt(w, h)
    cur = _level_textures(tw, th)
    _set(stats.uniform_sampler, "src", src)
    _set(stats.uniform_int, "straight", int(bool(straight)))
    for (name, _), tex in zip(_STATS_OUT, cur):
        stats.image(name, tex)
    gpu.compute.dispatch(stats, (tw + 7) // 8, (th + 7) // 8, 1)
    while (tw, th) != (1, 1):
        nw, nh = _nt(tw, th)
        nxt = _level_textures(nw, nh)
        for sname, tex in zip(("s_min", "s_max", "s_mean", "s_lum", "s_cnt"), cur):
            _set(reduce_.uniform_sampler, sname, tex)
        for (name, _), tex in zip(_STATS_OUT, nxt):
            reduce_.image(name, tex)
        gpu.compute.dispatch(reduce_, (nw + 7) // 8, (nh + 7) // 8, 1)
        cur, tw, th = nxt, nw, nh
    mn, mx, mean, lum, cnt = (_read(t)[0, 0] for t in cur)
    n = int(round(float(cnt[0])))
    if n == 0:
        return Stats(0, np.zeros(4), np.zeros(4), np.zeros(4), 0, 0, 0, 0)
    return Stats(n, mn, mx, mean, lum[0], lum[1], lum[2], float(lum[3]) / n)


def hist_gpu(src, chan, lo, hi, straight=False):
    import gpu

    w, h = int(src.width), int(src.height)
    tile = 32
    while _nt(w, h, tile)[0] * _nt(w, h, tile)[1] > 16000:
        tile *= 2
    ntx, nty = _nt(w, h, tile)
    rows = ntx * nty
    tiles = _tex((NBINS, rows), "R32UI")
    hist = _tex((NBINS, 1), "R32UI")
    k1 = _kernel("hist", _src.HIST, ["src"], [("o_tiles", "R32UI", "UINT_2D", ("WRITE",))],
                 [("straight", "INT"), ("chan", "INT"), ("tile_size", "INT"), ("lo", "FLOAT"),
                  ("inv", "FLOAT")], (8, 8, 1))
    k2 = _kernel("hist_reduce", _src.HIST_REDUCE, [],
                 [("o_tiles", "R32UI", "UINT_2D", ("READ",)),
                  ("o_hist", "R32UI", "UINT_2D", ("WRITE",))],
                 [("rows", "INT")], (64, 1, 1))
    _set(k1.uniform_sampler, "src", src)
    _set(k1.uniform_int, "straight", int(bool(straight)))
    _set(k1.uniform_int, "chan", int(chan))
    _set(k1.uniform_int, "tile_size", tile)
    _set(k1.uniform_float, "lo", float(F32(lo)))
    _set(k1.uniform_float, "inv", hist_scale(lo, hi))
    k1.image("o_tiles", tiles)
    gpu.compute.dispatch(k1, (ntx + 7) // 8, (nty + 7) // 8, 1)
    k2.image("o_tiles", tiles)
    k2.image("o_hist", hist)
    _set(k2.uniform_int, "rows", rows)
    gpu.compute.dispatch(k2, (NBINS + 63) // 64, 1, 1)
    return _read(hist).reshape(-1).astype(np.int64)


def clear_cache():
    _scratch.clear()
