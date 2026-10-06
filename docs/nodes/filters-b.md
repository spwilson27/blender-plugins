# Filters B (Add > Lab > Filter)

Kuwahara, Displace / Glass, Liquify, Edge Stylise. All colours are premultiplied scene-linear,
row 0 = bottom. Every node has a vectorised numpy CPU path and a GPU compute path. Tests:
`tests/lab/test_kuwahara.py`, `test_displace.py`, `test_liquify.py`, `test_edge_stylise.py` (with
`tests/lab/imgfx_helpers.py`: test images and independent float64 references).

## New library modules

* `lib/np_sampling.py` (CPU) and `lib/glsl/sampling.py` (GPU), twins:
  * edge modes `CLAMP` / `REPEAT` / `MIRROR` (index 0 / 1 / 2; mirror duplicates the border pixel),
    `edge_index`, `pad`
  * `sample_bilinear`, `sample_bicubic` (Catmull-Rom), `sample`; continuous pixel coordinates with
    centres at `i + 0.5` (sampling at `(x + 0.5, y + 0.5)` is exact)
  * `sobel` (per-pixel gradients, divided by 8), `gaussian_weights`, `blur_axis`, `blur_gaussian`
    (separable, radius `ceil(3 sigma)`)
  * GPU: `kernel(body, outputs, samplers, uniforms, libs)` is a `pointwise` variant where every input
    is a sampler and gets `lab_fetch_N`, `lab_px_N(p, mode)`, `lab_bilinear_N(pos, mode)`,
    `lab_bicubic_N(pos, mode)`, `lab_sobel_N(p, mode, gx, gy)`, `lab_size_N()` (N = input name);
    unlinked values become cached 1x1 textures. `scratch(w, h, role)` gives cached RGBA32F scratch
    textures (per thread); `gaussian_blur(src, dst, sigma, mode, role)` is a two-pass blur.

## Kuwahara (`CompositorNodeLabKuwahara`)

Inputs: Image, Radius (default 4, 0..16), Sharpness q (8, 0..16), Anisotropy (1, 0..2). Output: Image.

Anisotropic generalised Kuwahara after Kyprianidis et al.: Sobel structure tensor of RGB, Gaussian
smoothed (sigma 2); anisotropy `A = (l1 - l2) / (l1 + l2)` and edge tangent from the eigen
decomposition (`A < 1e-5` counts as isotropic); elliptical kernel with semi-axes `r (1 + k A)` along
the edge and `r / (1 + k A)` across; 8 sectors with weights `max(0, cos)^4 * exp(-3.125 |v|^2) *
(1 - |v|^2)` (the last factor makes the border smooth, so rounding cannot flip border samples); per
sector weighted mean (RGBA) and RGB deviation `s`; result `sum a_i m_i / sum a_i` with
`a_i = 1 / (1 + (255 s_i)^q)`. Integer sample offsets, clamped edges. Flat images are unchanged,
step edges stay sharp, output is a convex combination of inputs (within the per-channel range).
GPU: tensor pass, 2-pass blur, filter pass. CPU: tiled numpy, loops over kernel offsets only.

Tolerances: CPU vs float64 per-pixel reference 2e-4 (observed 6e-5); CPU vs GPU 5e-4 (observed 1e-4
worst, typically 1e-5): the orientation and the `pow(255 s, q)` weights amplify last-bit differences.
CPU speed: 4 s for 1080p noisy at radius 4 (cost grows with `(2 r)^2`); GPU 0.1 s.

## Displace / Glass (`CompositorNodeLabDisplace`)

Inputs: Image, Map (default 0.5 grey), Strength (20; may be an image), Dispersion (0). Props: Mode
(Offset / Glass), Edges (Clamp / Repeat / Mirror), Interpolation (Bilinear / Bicubic).

Sample position `p + offset * s_c` (pull). Offset mode: `offset = (Map.rg - 0.5) * 2 * Strength`
pixels. Glass: `offset = -Strength * gain * grad(luma(Map))` (Sobel/8, `gain = max(w, h) / 100`, so
resolution independent). Dispersion d scales the offset per channel `s = (1 - d, 1, 1 + d)` for R, G,
B (alpha follows G); 0 uses a single sample. Zero strength is an exact identity; a constant map is
an exact translation (tested with integer offsets for all edge modes, both backends). Bicubic may
overshoot. A map of another size is clamp-resampled (nearest) on both backends.

Tolerances: vs float64 reference 2e-5 (offset; observed 3e-6), 1e-4 (glass; observed 2e-5); CPU vs
GPU 5e-5 (observed 2.2e-5 worst, glass on a noisy image).

## Liquify (`CompositorNodeLabLiquify`)

Inputs: Image, Center X / Y (0.5; normalised, y up), Radius (0.4), Strength (0.5), Falloff (1).
Props: Mode (Twirl / Pinch-Bulge), Aspect Correction (on), Interpolation.

`t = |d| / Radius`, weight `f = 1 - smoothstep(1 - Falloff, 1, t)`. Twirl rotates the source position by
`Strength * pi * f`; Pinch / Bulge scales it by `1 - Strength * f` (Strength clamped to [-4, 1]; positive
bulges). With aspect correction distances are in pixels and Radius is a fraction of the height.
Pixels outside the radius, strength 0 and radius 0 are bit-exact identities. Tests also check the
radial mass distribution under a twirl (<0.3 % change) and that bulge / pinch grow / shrink a disc.
Tolerances: vs float64 reference 2e-5 (observed 2e-6); CPU vs GPU 5e-5 (observed 1.4e-5).

## Edge Stylise (`CompositorNodeLabEdgeStylise`)

Input: Image. Output: Image. Props (shown per mode): Mode (XDoG / Sobel / Outline).

* **XDoG**: Sigma (1), K (1.6), Tau (0.98), Phi (20), Epsilon (0), Ink and Paper colours (black /
  white). `D = G(sigma) L - Tau G(K sigma) L` on Rec. 709 luma; `v = 1` if `D >= eps` else
  `1 + tanh(Phi (D - eps))`; output `Ink + (Paper - Ink) v`, always in [0, 1] for the default colours.
  Flat image: paper. Needs Tau < 1 or Epsilon <= 0 for that.
* **Sobel**: Gain (2), Color (per-channel instead of luma), Invert. Magnitude of the Sobel/8 gradient.
  Alpha 1.
* **Outline**: Width px (4, max 32), Position (Outside / Inside / Center), Stroke Color
  (premultiplied), Stroke Only. Coverage from dilation / erosion of the alpha channel with an
  anti-aliased disc (offset weight `clamp(r + 0.85 - |o|, 0, 1)`, clamped edges). Outside is drawn
  under the image, inside over it. Measured width on a disc: within 0.4 px of Width for Width 2..12.
  CPU: exact row-wise max for the full-weight disc plus a loop over the (about `2 pi r`) rim offsets.

Tolerances: XDoG vs float64 2e-4 (observed 1.4e-4 at Phi 1000, otherwise 4e-6); Sobel 1e-5 (1.5e-7);
Outline 1e-5 (1.5e-7); CPU vs GPU XDoG 2e-4 (observed 4e-6), others 1e-5.
