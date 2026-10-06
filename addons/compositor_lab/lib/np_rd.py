# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Gray-Scott reaction-diffusion on the CPU (numpy). GPU twin: ``glsl/rd.py``.

One iteration (U, V in [0, 1], all float32)::

    lap(X) = 0.2 (4 orthogonal neighbours) + 0.05 (4 diagonal neighbours) - X        (9-point)
    U' = clip(U + dt (Du lap(U) - U V V + F (1 - U)), 0, 1)
    V' = clip(V + dt (Dv lap(V) + U V V - (F + k) V), 0, 1)

The 9-point Laplacian is nearly isotropic (round spots instead of diamond ones). Its most
negative eigenvalue is -1.6, so the explicit step is stable while ``D dt < 1.25``; the clip keeps
the fields bounded (and NaN free) for any finite parameters regardless.

Edges: ``WRAP`` (the field tiles) or ``CLAMP`` (zero flux: the border pixel is repeated).
Images are (H, W) float32, row 0 = bottom.
"""

import numpy as np

from . import np_noise, np_sampling

F32 = np.float32

W_ORTH = F32(0.2)
W_DIAG = F32(0.05)
CELL = 8                    # noise seeding: one possible blob per CELL x CELL cell
BLOB_R2 = 5                 # blob radius^2 in simulation pixels (centre offset 2..5 in the cell)
SEED_U, SEED_V = F32(0.5), F32(1.0)
LUMA = (F32(0.2126), F32(0.7152), F32(0.0722))

# name -> (feed, kill): classic Gray-Scott parameter sets for Du = 1, Dv = 0.5
PRESETS = {
    'CORAL': (0.0545, 0.062),
    'MITOSIS': (0.0367, 0.0649),
    'WORMS': (0.078, 0.061),
    'SPOTS': (0.035, 0.065),
    'MAZE': (0.029, 0.057),
    'BUBBLES': (0.098, 0.057),
}


def sim_size(w, h, scale):
    """Simulation size for an output of w x h: ceil(size / scale), at least 1."""
    s = max(1, int(scale))
    return max(1, -(-int(w) // s)), max(1, -(-int(h) // s))


def luminance(rgb):
    return rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]


def resample_coords(dst_w, dst_h, src_w, src_h):
    """Source pixel coordinates (xs, ys: full (dst_h, dst_w) float32 arrays) of the centres of the
    destination pixels, for bilinear sampling of a src_w x src_h image."""
    rx = F32(src_w / dst_w)
    ry = F32(src_h / dst_h)
    xs = (np.arange(dst_w, dtype=F32) + F32(0.5)) * rx
    ys = (np.arange(dst_h, dtype=F32) + F32(0.5)) * ry
    return (np.ascontiguousarray(np.broadcast_to(xs[None, :], (dst_h, dst_w))),
            np.ascontiguousarray(np.broadcast_to(ys[:, None], (dst_h, dst_w))))


def resample(img, dst_w, dst_h, mode="CLAMP"):
    """Bilinear resample of an (H, W, C) image to dst_w x dst_h (identity when the size matches)."""
    h, w = img.shape[:2]
    if (w, h) == (dst_w, dst_h):
        return np.asarray(img, F32)
    xs, ys = resample_coords(dst_w, dst_h, w, h)
    return np_sampling.sample_bilinear(np.asarray(img, F32), xs, ys, mode)


def blob_mask(sw, sh, seed, density):
    """Random blobs (boolean (sh, sw)): per CELL x CELL cell a hash decides (with probability
    ``density``) whether it holds a blob and where its centre is. Integer maths only, so the GPU
    twin agrees exactly."""
    ss = np_noise.scramble_seed(seed)
    y, x = np.mgrid[0:sh, 0:sw].astype(np.int32)
    cx, cy = x // CELL, y // CELL
    hx, hy, hz = np_noise.hash3(cx, cy, 0, ss)
    # the density is compared in float32 exactly like the shader (float(h >> 8) / 2^24 < d)
    on = np_noise.u01(hz) < F32(density)
    ox = (cx * CELL + 2 + (hx & np_noise.U32(3)).astype(np.int32)) - x
    oy = (cy * CELL + 2 + (hy & np_noise.U32(3)).astype(np.int32)) - y
    return on & (ox * ox + oy * oy <= BLOB_R2)


def init_state(mask, noise, seed):
    """(U, V) from a boolean seed mask: U = 0.5, V = 1 inside, U = 1 and V = ``noise`` * white
    noise outside."""
    sh, sw = mask.shape
    ss = np_noise.scramble_seed(seed)
    y, x = np.mgrid[0:sh, 0:sw].astype(np.int32)
    n = np_noise.u01(np_noise.hash3(x, y, 1, ss)[0]) * F32(noise)
    u = np.where(mask, SEED_U, F32(1.0)).astype(F32)
    v = np.where(mask, SEED_V, n).astype(F32)
    return u, v


def _fill(p, edge):
    if edge == 'WRAP':
        p[0, 1:-1] = p[-2, 1:-1]
        p[-1, 1:-1] = p[1, 1:-1]
        p[:, 0] = p[:, -2]
        p[:, -1] = p[:, 1]
    else:
        p[0, 1:-1] = p[1, 1:-1]
        p[-1, 1:-1] = p[-2, 1:-1]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]


def _lap(p):
    orth = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    diag = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    return W_ORTH * orth + W_DIAG * diag - p[1:-1, 1:-1]


def iterate(u, v, n, feed, kill, du, dv, dt, edge="WRAP"):
    """Run ``n`` iterations; ``feed`` / ``kill`` are scalars or (H, W) float32 arrays. Returns new
    (U, V) arrays (the inputs are not modified)."""
    h, w = u.shape
    pu = np.empty((h + 2, w + 2), F32)
    pv = np.empty((h + 2, w + 2), F32)
    pu[1:-1, 1:-1] = u
    pv[1:-1, 1:-1] = v
    du, dv, dt = F32(du), F32(dv), F32(dt)
    fk = feed + kill
    one = F32(1.0)
    for _ in range(int(n)):
        _fill(pu, edge)
        _fill(pv, edge)
        uu = pu[1:-1, 1:-1]
        vv = pv[1:-1, 1:-1]
        lu = _lap(pu)
        lv = _lap(pv)
        uvv = uu * vv * vv
        nu = du * lu - uvv + feed * (one - uu)
        nv = dv * lv + uvv - fk * vv
        nu = np.clip(uu + dt * nu, 0.0, 1.0)
        nv = np.clip(vv + dt * nv, 0.0, 1.0)
        pu[1:-1, 1:-1] = nu
        pv[1:-1, 1:-1] = nv
    return pu[1:-1, 1:-1].copy(), pv[1:-1, 1:-1].copy()


def colorize(v, color_a, color_b, low, inv):
    """Two-colour gradient of V: smoothstep((v - low) * inv) between the (premultiplied) colours."""
    t = np.clip((v - F32(low)) * F32(inv), 0.0, 1.0).astype(F32)
    t = t * t * (F32(3.0) - F32(2.0) * t)
    a = np.asarray(color_a, F32)
    b = np.asarray(color_b, F32)
    return (a + (b - a) * t[..., None]).astype(F32)
