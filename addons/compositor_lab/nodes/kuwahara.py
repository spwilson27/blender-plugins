# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Anisotropic Kuwahara: painterly smoothing that keeps edges sharp and follows image structure
(Kyprianidis, Kang, Doellner 2009, "Image and Video Abstraction by Anisotropic Kuwahara Filtering").

Algorithm (CPU and GPU do the same arithmetic):

1. Structure tensor ``(E, F, G) = (gx.gx, gx.gy, gy.gy)`` from Sobel gradients of RGB (all three
   channels summed), smoothed with a Gaussian of sigma 2 (clamped edges).
2. Orientation and anisotropy: ``A = (l1 - l2) / (l1 + l2)`` from the eigenvalues; the kernel's
   major axis is the *tangent* (along the edge). ``A`` below 1e-5 counts as isotropic.
3. Elliptical kernel with semi-axes ``a = r (1 + k A)`` (along the edge) and ``b = r / (1 + k A)``
   (``k`` = Anisotropy; 0 gives a circle). Every integer offset inside the ellipse is a sample.
4. 8 sectors (45 degrees each). A sample at normalised position ``v`` (``|v| <= 1``) has weight
   ``max(0, v.d_i)^4 / |v|^4 * exp(-3.125 |v|^2) (1 - |v|^2)`` in sector ``i`` (``d_i`` = unit vector of the
   sector's axis): a smooth angular falloff and a Gaussian radial one that vanishes at the ellipse border; the centre sample counts
   fully in all sectors.
5. Per sector the weighted mean ``m_i`` (RGBA) and the RGB standard deviation
   ``s_i = sqrt(sum_c var_c)``; the output is ``sum a_i m_i / sum a_i`` with
   ``a_i = 1 / (1 + (255 s_i)^q)`` (``q`` = Sharpness; 0 averages all sectors equally).

The output is a convex combination of input pixels, so it stays inside the input's per-channel
range, and a flat image is unchanged. Sampling uses integer offsets (no interpolation), clamped
edges. Alpha is filtered like a colour channel (premultiplied) but ignored by the variance.

CPU path: vectorised over image tiles, looping over the kernel offsets; roughly
``(2 r + 1)^2`` tile passes, so keep the radius modest on large images (the GPU path is the
intended one). GPU path: a tensor pass, a separable blur (2 passes) and the filter pass.
"""

import math

import bpy
import numpy as np

from ..lib import np_sampling
from ..lib.glsl import sampling as gl_sampling
from ..lib.node import In, LabNode, Out

MENU = "Filter"

TENSOR_SIGMA = 2.0
MAX_RADIUS = 16.0
MAX_ANISOTROPY = 2.0
MAX_SHARPNESS = 16.0
SECTORS = 8
_S = np.float32(0.70710678)

_TENSOR_BODY = """
    vec4 gx, gy;
    lab_sobel_Image(texel, 0, gx, gy);
    out_T = vec4(dot(gx.rgb, gx.rgb), dot(gx.rgb, gy.rgb), dot(gy.rgb, gy.rgb), 0.0);
"""

_FILTER_BODY = """
    vec4 T = lab_fetch_T(texel);
    float E = T.x;
    float F = T.y;
    float G = T.z;
    float tr = E + G;
    float dd = E - G;
    float disc = sqrt(max(dd * dd + 4.0 * F * F, 0.0));
    float A = 0.0;
    vec2 tdir = vec2(1.0, 0.0);
    if (tr > 1.0e-20 && disc > 1.0e-5 * tr) {
      A = disc / tr;
      float c2 = dd / disc;
      float cs = sqrt(max(0.5 * (1.0 + c2), 0.0));
      float sn = sqrt(max(0.5 * (1.0 - c2), 0.0));
      if (F < 0.0) {
        sn = -sn;
      }
      tdir = vec2(-sn, cs);
    }
    vec2 ndir = vec2(-tdir.y, tdir.x);
    float sc = 1.0 + k_aniso * A;
    float ea = k_radius * sc;
    float eb = k_radius / sc;

    float S0[8];
    float S2[8];
    vec4 S1[8];
    for (int i = 0; i < 8; ++i) {
      S0[i] = 0.0;
      S2[i] = 0.0;
      S1[i] = vec4(0.0);
    }
    for (int dy = -k_ext; dy <= k_ext; ++dy) {
      for (int dx = -k_ext; dx <= k_ext; ++dx) {
        vec2 o = vec2(float(dx), float(dy));
        vec2 v = vec2(dot(o, tdir) / ea, dot(o, ndir) / eb);
        float l2 = dot(v, v);
        if (l2 > 1.0) {
          continue;
        }
        float w[8];
        if (dx == 0 && dy == 0) {
          for (int i = 0; i < 8; ++i) {
            w[i] = 1.0;
          }
        }
        else {
          float s = exp(-3.125 * l2) * (1.0 - l2) / (l2 * l2);
          vec4 p = vec4(v.x, (v.x + v.y) * 0.70710678, v.y, (v.y - v.x) * 0.70710678);
          vec4 wp = max(p, vec4(0.0));
          vec4 wn = max(-p, vec4(0.0));
          wp = wp * wp;
          wn = wn * wn;
          wp = wp * wp * s;
          wn = wn * wn * s;
          for (int i = 0; i < 4; ++i) {
            w[i] = wp[i];
            w[i + 4] = wn[i];
          }
        }
        vec4 c = lab_fetch_Image(texel + ivec2(dx, dy));
        float c2 = dot(c.rgb, c.rgb);
        for (int i = 0; i < 8; ++i) {
          S0[i] += w[i];
          S1[i] += w[i] * c;
          S2[i] += w[i] * c2;
        }
      }
    }
    vec4 acc = vec4(0.0);
    float asum = 0.0;
    for (int i = 0; i < 8; ++i) {
      vec4 m = S1[i] / S0[i];
      float var = max(S2[i] / S0[i] - dot(m.rgb, m.rgb), 0.0);
      float base = max(255.0 * sqrt(var), 1.0e-20);
      float al = 1.0 / (1.0 + min(pow(base, k_q), 1.0e30));
      acc += m * al;
      asum += al;
    }
    out_Color = acc / asum;
"""


class CompositorNodeLabKuwahara(LabNode, bpy.types.CompositorNode):
    '''Anisotropic Kuwahara filter: painterly smoothing that follows edges and keeps them sharp'''
    bl_idname = "CompositorNodeLabKuwahara"
    bl_label = "Kuwahara"

    SOCKETS = [
        In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)),
        In("Radius", "FLOAT", 8.0),
        In("Sharpness", "FLOAT", 8.0),
        In("Anisotropy", "FLOAT", 1.0),
        Out("Image", "COLOR"),
    ]

    # -- parameters --------------------------------------------------------
    def _params(self, inputs):
        f32 = np.float32
        radius = min(max(self.in_float(inputs, "Radius", 4.0), 0.0), MAX_RADIUS)
        q = min(max(self.in_float(inputs, "Sharpness", 8.0), 0.0), MAX_SHARPNESS)
        k = min(max(self.in_float(inputs, "Anisotropy", 1.0), 0.0), MAX_ANISOTROPY)
        radius, q, k = float(f32(radius)), float(f32(q)), float(f32(k))
        ext = int(math.ceil(radius * (1.0 + k)))
        return radius, q, k, ext

    # -- CPU ---------------------------------------------------------------
    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")
        if out is None:
            return
        img = np.asarray(self.in_image_array(inputs, "Image", ctx.shape, 4,
                                             default=(0.5, 0.5, 0.5, 1.0)), np.float32)
        radius, q, k, _ = self._params(inputs)
        out[...] = img if radius <= 0.0 else kuwahara_np(img, radius, q, k)

    # -- GPU ---------------------------------------------------------------
    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        src = self.in_texture_or_value(inputs, "Image", (0.5, 0.5, 0.5, 1.0))
        radius, q, k, ext = self._params(inputs)
        if radius <= 0.0:
            gl_sampling.kernel("    out_Color = lab_fetch_Image(texel);\n", {"Color": dst},
                               {"Image": src})
            return
        w, h = int(dst.width), int(dst.height)
        t_raw = gl_sampling.scratch(w, h, "kuwahara_t")
        t_blur = gl_sampling.scratch(w, h, "kuwahara_tb")
        gl_sampling.kernel(_TENSOR_BODY, {"T": t_raw}, {"Image": src})
        gl_sampling.gaussian_blur(t_raw, t_blur, TENSOR_SIGMA, 0, role="kuwahara_blur")
        gl_sampling.kernel(
            _FILTER_BODY, {"Color": dst}, {"Image": src, "T": t_blur},
            uniforms={"k_radius": ("float", radius), "k_q": ("float", q),
                      "k_aniso": ("float", k), "k_ext": ("int", ext)})


# ---------------------------------------------------------------------------
# numpy implementation
# ---------------------------------------------------------------------------

def structure_orientation(img, sigma=TENSOR_SIGMA):
    """Smoothed structure tensor of the RGB channels -> (A, tx, ty): anisotropy in [0, 1] and the
    unit tangent (along the edge) per pixel."""
    f32 = np.float32
    gx, gy = np_sampling.sobel(img[..., :3], "CLAMP")
    tensor = np.stack([np.sum(gx * gx, axis=2), np.sum(gx * gy, axis=2),
                       np.sum(gy * gy, axis=2)], axis=2)
    tensor = np_sampling.blur_gaussian(tensor, sigma, "CLAMP")
    e, f, g = tensor[..., 0], tensor[..., 1], tensor[..., 2]
    tr = e + g
    dd = e - g
    disc = np.sqrt(np.maximum(dd * dd + f32(4.0) * f * f, f32(0.0)))
    valid = (tr > f32(1e-20)) & (disc > f32(1e-5) * tr)
    safe_disc = np.where(valid, disc, f32(1.0))
    safe_tr = np.where(valid, tr, f32(1.0))
    aniso = np.where(valid, disc / safe_tr, f32(0.0)).astype(f32)
    c2 = dd / safe_disc
    cs = np.sqrt(np.maximum(f32(0.5) * (f32(1.0) + c2), f32(0.0)))
    sn = np.sqrt(np.maximum(f32(0.5) * (f32(1.0) - c2), f32(0.0)))
    sn = np.where(f < 0, -sn, sn)
    tx = np.where(valid, -sn, f32(1.0)).astype(f32)
    ty = np.where(valid, cs, f32(0.0)).astype(f32)
    return aniso, tx, ty


def kuwahara_np(img, radius, q, k, tile_pixels=6144):
    """Anisotropic Kuwahara of an (H, W, 4) float32 image (see the module docstring)."""
    f32 = np.float32
    h, w = img.shape[:2]
    aniso, tx, ty = structure_orientation(img)
    sc = f32(1.0) + f32(k) * aniso
    ea = f32(radius) * sc
    eb = f32(radius) / sc
    ext = int(math.ceil(float(radius) * (1.0 + float(k))))
    ext = min(ext, int(math.ceil(float(ea.max()))))
    # Planar channels (r, g, b, a, |rgb|^2) padded by the clamped border.
    planes = np.concatenate([np.moveaxis(img, 2, 0),
                             np.sum(img[..., :3] * img[..., :3], axis=2)[None]], axis=0)
    padded = np.pad(planes, ((0, 0), (ext, ext), (ext, ext)), mode="edge")
    out = np.empty_like(img)
    rows = max(1, tile_pixels // w)
    s = _S
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        n = y1 - y0
        ttx, tty = tx[y0:y1], ty[y0:y1]
        tea, teb = ea[y0:y1], eb[y0:y1]
        amax = float(tea.max())
        s0 = np.zeros((SECTORS, n, w), f32)
        s1 = np.zeros((SECTORS, 5, n, w), f32)
        wts = np.empty((SECTORS, n, w), f32)
        tmp = np.empty((SECTORS, 5, n, w), f32)
        p4 = np.empty((4, n, w), f32)
        for dy in range(-ext, ext + 1):
            for dx in range(-ext, ext + 1):
                if dx * dx + dy * dy > amax * amax:
                    continue
                if dx == 0 and dy == 0:
                    wts[...] = 1.0
                else:
                    vx = (f32(dx) * ttx + f32(dy) * tty) / tea
                    vy = (f32(dx) * (-tty) + f32(dy) * ttx) / teb
                    l2 = vx * vx + vy * vy
                    inside = l2 <= f32(1.0)
                    if not inside.any():
                        continue
                    l2s = np.where(inside, l2, f32(1.0))
                    scale = np.where(inside, np.exp(f32(-3.125) * l2s) * (f32(1.0) - l2s) / (l2s * l2s), f32(0.0))
                    p4[0] = vx
                    p4[1] = (vx + vy) * s
                    p4[2] = vy
                    p4[3] = (vy - vx) * s
                    wts[:4] = np.maximum(p4, f32(0.0))
                    wts[4:] = np.maximum(-p4, f32(0.0))
                    wts *= wts
                    wts *= wts
                    wts *= scale
                c = padded[:, ext + y0 + dy:ext + y1 + dy, ext + dx:ext + dx + w]
                s0 += wts
                np.multiply(wts[:, None], c[None], out=tmp)
                s1 += tmp
        m = s1 / s0[:, None]                       # (8, 5, n, w): means of r, g, b, a, |rgb|^2
        var = np.maximum(m[:, 4] - (m[:, 0] * m[:, 0] + m[:, 1] * m[:, 1] + m[:, 2] * m[:, 2]),
                         f32(0.0))
        base = np.maximum(f32(255.0) * np.sqrt(var), f32(1e-20))
        al = f32(1.0) / (f32(1.0) + np.minimum(np.power(base, f32(q)), f32(1e30)))
        acc = np.sum(m[:, :4] * al[:, None], axis=0)
        out[y0:y1] = np.moveaxis(acc / np.sum(al, axis=0)[None], 0, 2)
    return out


NODE_CLASSES = [CompositorNodeLabKuwahara]
