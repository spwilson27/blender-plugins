# Generators (menu: Add > Lab > Generate)

Three generator nodes. With no image linked they use the render-sized domain (F1); a linked Image
or Vector defines the domain. All colours are premultiplied scene-linear; row 0 is the bottom.
Scalar sockets read the socket value only (a linked image falls back to the default on CPU and GPU
alike). New library modules: `lib/np_pattern.py` + `lib/glsl/pattern.py` (Voronoi sites, tilings),
`lib/np_field.py` + `lib/glsl/field.py` (vector fields, LIC).

## Voronoi / Mosaic (`CompositorNodeLabVoronoi`)

Inputs: Image (optional), Scale (cells across the larger side, 8), Jitter (0..1, 1), Border Width
(cell units, measured as F2 - F1, 0.06), Border Color, Fill Color (Edges mode), Offset X/Y, Phase,
Speed (0), Seed. Properties: Mode (Cells, F1 Distance, Edges, Mosaic), Metric (Euclidean,
Manhattan, Chebyshev). Outputs: Color; Value (cell id in Cells/Mosaic, F1 in F1 mode, clamped
F2 - F1 in Edges); Border (coverage 0..1).

* One jittered site per unit cell (3x3 neighbourhood), PCG hash shared bit for bit between numpy and
  GLSL, so the CPU and GPU pick the same cells. Cells mode: flat random colour per cell. Mosaic: each
  cell takes the Image colour at the pixel containing its site (random cell colours if no Image).
* Border coverage = clamp(0.5 + (width - (F2 - F1)) / (2k), 0, 1), k = cell units per pixel (about
  one pixel of anti-aliasing); cell fills themselves are not anti-aliased.
* Animation: `site = centre + Jitter * (ra cos(phi) + rb sin(phi))`, phi = 2 pi (Phase + time *
  Speed), two random vectors per cell; phi = 0 is the static pattern.
* The 3x3 window makes F1 exact; F2 is exact for Euclidean/Chebyshev and wrong for ~0.02% of pixels
  for Manhattan (tested, only affects borders/Edges there).
* Tolerances: CPU vs GPU max abs difference ~1e-6 (tests use 1e-5, <= 0.02% of pixels flipping).
  CPU 1080p: ~0.3 s.

**Ranges** (soft range | clamp):

* Scale 0..100 (internally >= 0.001), Jitter 0..1 (internally clamped), Border Width 0..1
  (internally >= 0), Offset X/Y -10..10, Phase -10..10, Speed -10..10, Seed 0..1000. No lib clamps.

## Pattern (`CompositorNodeLabPattern`)

Inputs: Scale (periods across the larger side, 10), Rotation (degrees, about the image centre),
Offset X/Y (pattern units; applied after rotation), Duty (0.5), Softness (0), Moire Angle (4 deg),
Seed, Color A (black), Color B (white). Property: Pattern (Stripes, Checker, Dots, Hex Grid, Truchet
Arcs, Truchet Diagonals, Moire, Rings). Outputs: Color = mix(A, B, Mask); Mask (coverage of B).

* Coverage is a linear ramp of the signed distance to the nearest edge, width = 1 pixel +
  Softness / 2 periods, so straight axis-aligned edges are exactly box filtered. Duty = stripe width,
  dot diameter, hex cell fill, truchet line width (0.25 * Duty); Moire multiplies two gratings, the
  second rotated by Moire Angle; Rings are centred on the pattern origin (image centre at Offset 0).
  Truchet tiles flip by a hash of (cell, Seed); diagonal truchet strokes have round caps that join
  across tile corners.
* Tolerances: CPU vs GPU max abs ~6e-6 (9.8e-5 with Offset -123: float32 spacing). Versus a 6x6
  supersampled float64 binary reference the mean absolute difference is <= 0.015 (max 0.27 at
  corners/thin features).

**Ranges** (soft range | clamp):

* Scale 0..200 (internally >= 0.001), Rotation -180..180 (degrees), Offset X/Y -10..10, Duty 0..1
  and Softness 0..1 (internally clamped), Moire Angle -180..180 (degrees), Seed 0..1000. No lib clamps.

## Flow Field (`CompositorNodeLabFlowField`)

Inputs: Image (smeared; grey white noise if unlinked), Vector (RG field for the Vector source,
default (1, 0, 0)), Length (steps per direction, 20, clamped 0..256), Step (pixels per step, 1),
Scale (curl noise scale, 3), Rotate (degrees applied to the field), Phase, Speed (animates the
curl noise), Density (impulse density for Streaks, 0.05), Seed. Properties: Source (Curl Noise,
Image Gradient, Image Contour = gradient rotated 90 deg, Vector Image), Noise (Perlin, Simplex,
Value), Kernel (Box, Triangle). Outputs: Color (LIC), Streaks (grey), Field (RG = unit direction *
0.5 + 0.5, B = raw magnitude).

* Stateless: the field is evaluated once per pixel into a scratch RGBA32F texture (GPU) / array
  (CPU), then each pixel integrates its streamline both ways with a midpoint (RK2) step, bilinear
  field and image sampling (clamp to edge), unit speed (only the direction matters). A streamline
  stops where the squared field length is < 1e-12; weights are renormalised, so a constant image
  stays constant and a uniform horizontal field is exactly a 1D box (or triangle) blur of length
  2 * Length + 1 (tested to 2e-7).
* Streaks applies the same integration to sparse impulse noise (Density) and returns
  clamp(mean * 0.5 / Density, 0, 1): hair-like streamlines. Without an Image, Image Gradient /
  Contour have a zero field (the noise is returned unchanged).
* CPU: numpy, gathers on a thread pool (1080p, Length 20: ~2.6 s). GPU: two dispatches
  (`lib/gpu.pointwise` twice); the second reads the first's output texture (works on Metal).
* CPU vs GPU: Field output and smooth fields agree to ~1e-6; where a streamline passes a point of
  near-zero interpolated field (noisy gradient images, curl extrema) rounding differences amplify:
  mean abs diff <= 1.3e-4, <= 1.8% of pixels over 5e-4, max 0.02 colour (0.25 for the high-contrast
  Streaks of a random-noise gradient image). Tests assert exactly that.

**Ranges** (soft range | clamp):

* Length 0..256 (internally clamped to the same), Step -10..10, Scale 0..50, Rotate -180..180
  (degrees), Phase -10..10, Speed -10..10, Density 0..1 (internally clamped), Seed 0..1000. No lib
  clamps.

## Notes

* GLSL gotchas found (Blender's MSL translation rejects forward declarations of functions and the
  identifier `kernel`, even in comments) are collected in `addons/compositor_lab/lib/README.md`.
  Lib code that needs an input may call `in_<Name>`: `gpu.kernel` declares the accessors before the
  libs.

## Noise ranges

Socket / property | soft range | clamp

* Scale: 0..50, none; Octaves: 1..16, clamped to 1..16; Lacunarity: 1..4, none; Gain: 0..1, none
* Warp: 0..4, none; Randomness: 0..1, clamped to 0..1
* Offset X / Y: -100..100, none; Phase: -10..10, none; Speed: -5..5, none; Seed: 0..1000, none
