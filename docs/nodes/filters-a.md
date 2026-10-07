# Filters A: Posterize+, Gradient Map, Halftone, Glitch

All four nodes live in the Lab ▸ Filter menu, work on premultiplied scene-linear colour, keep
alpha, and have a vectorised numpy CPU path and a GPU compute path (`lib/gpu.pointwise`). Sockets
named in the tables are inputs; every node has a `Fac` mix and a `Color` output. Unlinked `Image`
inputs use their default colour; scalar sockets (`Levels`, `Seed`, ...) are single values.

## Shared library additions

* `lib/glsl/dither.py` / `lib/np_dither.py`: dither thresholds. `lab_dither_threshold(mode, texel,
  seed)` (GLSL) and `np_dither.threshold(mode, (H, W), seed)` return t in [0, 1) per pixel; use as
  `floor(v * (levels - 1) + t) / (levels - 1)`. Modes: 0 none (t = 0.5, plain rounding), 1/2/3
  Bayer 2x2/4x4/8x8 (bit-interleaved index, `(i + 0.5) / n^2`), 4 R2 low-discrepancy sequence in
  32-bit fixed point (blue-noise-like, negative neighbour correlation), 5 PCG white hash
  (`lab_rand`). All integer maths, so thresholds are bit-identical on both backends.

## Posterize+ (`CompositorNodeLabPosterize`)

Inputs: `Fac`, `Image`, `Levels` (INT, default 4, clamped to 2..65536). Properties: Per Channel +
Levels (R, G, B), Lightness Only, Gamma (2.2), Dither (None, Bayer 2/4/8, Blue-ish, Random), Dither
Amount, Seed.

Works on straight colour clamped to [0, 1]: `x = c^(1/gamma)`, `q = floor(x (N-1) + t) / (N-1)`,
`c' = q^gamma` (`t = 0.5 + (threshold - 0.5) * amount`). So without dither a channel has at most N
values; with N huge the node is the identity (within 3e-5, gamma 2.2 within 3e-4); dithering keeps
the local mean (every Bayer matrix cell block averages to the input within half a matrix step).
Lightness Only quantises OKLab L (gamma ignored) and keeps a, b; results outside the RGB gamut are
clamped to >= 0. Output is mixed `c (1 - fac) + result fac`, which makes Fac 1 and Fac 0 exact.

CPU vs GPU: max difference 3e-8 (1e-5 tolerance); 6e-6 in lightness mode (5e-5 tolerance); up to
0.05 % of pixels may differ by a quantiser step if a value sits within rounding of a step edge
(observed: none).

**Ranges** (soft range | clamp):

* Fac 0..1 | none. Levels 2..256 | 2..65536. Per-channel levels 2..256 (hard 2..65536). Gamma 0.1..5 (hard).
  Dither Amount 0..1 (hard). Seed 0..1000 (hard min 0).

## Gradient Map (`CompositorNodeLabGradientMap`)

Inputs: `Fac`, `Image`. Properties: Preset (Inferno, Viridis, Sunset, Ocean, Fire, Ice and Fire,
Sepia, Custom), Source (Luminance, OKLab Lightness, R, G, B, Value = max, Alpha), Interpolation
(Linear, Smooth, Constant), Blend In (Linear RGB, OKLab), Reverse, Preserve Alpha, and for Custom
a stop count (2..6) with `stopN_color` (COLOR, scene linear) and `stopN_pos`.

`t = clamp(source of the straight colour)`; stops are sorted by position; `t` at or below the first
position gives the first stop, at or above the last position the last. Smooth uses smoothstep per
segment, Constant holds the left stop. Preset colours are authored in sRGB and converted to linear.
OKLab interpolation converts the stops once on the CPU and the result back (clamped >= 0). Note that
a black to white 2-stop map equals luminance in Linear RGB only; in OKLab it is monotone but
follows L cubed. Preserve Alpha off gives an opaque result. Coincident stops are fine (step).

CPU vs GPU: max difference 1.5e-6 (tolerance 1e-5); constant-interpolation edges may flip for
<= 0.2 % of pixels (observed none).

**Ranges** (soft range | clamp):

* Fac 0..1 | none. Stops 2..6, stop positions 0..1 and colours 0..1 (hard).

## Halftone (`CompositorNodeLabHalftone`)

Inputs: `Fac`, `Image`. Properties: Mode (CMYK Dots, Mono Dots, Lines, Cross-hatch), Shape (Round,
Square, Diamond; dot modes), Cell Size (px), Softness (anti-aliasing width, 0 = hard), Angle (mono,
lines, hatch), CMYK angles (15/75/0/45 degrees default) and inks, Paper colour (white), Ink colour.

Each screen is a rotated lattice; the spot function maps the position in the cell to a threshold t
uniformly distributed over [0, 1] (the cumulative dot area, correct even for overlapping round
dots), a pixel is inked by `clamp((d - t) / w + 0.5, 0, 1)` where `w` is the tone change across one
pixel. So a flat field of darkness d (1 - luminance in linear light) has mean coverage d (measured
within 0.018 for round/square/diamond at 0, 30 and 45 degrees) and the output repeats with the
cell period (at 0/90 degrees). Dot tone is sampled at the cell centre (clean dots); line tone at the
pixel's projection onto the line centre. CMYK separates `k = 1 - max`, `c = (max - r) / max`, ...
and multiplies the inks `paper * prod(1 - cov_i (1 - ink_i))`, so flat primaries are reproduced
exactly and mixtures within 0.05 on average. Cross-hatch: four line layers at angle, +90, +45,
+135 that fade in at darkness 0, .25, .5, .75 (each at most 60 % of its cell). Alpha kept.

CPU 1080p CMYK: 0.5 s including image IO. CPU vs GPU: mean abs difference 2.3e-4 worst case and
at most 0.07 % of pixels differ (up to ~0.9) because a pixel centre within float rounding of a
cell boundary reads another cell's tone, or the anti-aliasing ramp (division by a tiny width)
amplifies rounding; tolerance 2e-3 with 0.2 % of pixels allowed.

**Ranges** (soft range | clamp):

* Fac 0..1 | none. Cell Size 2..100 (hard 2..400). Softness 0..8 (hard). Angles -pi..pi (radians).

## Glitch (`CompositorNodeLabGlitch`)

Inputs: `Fac`, `Image`, `Seed`, `Speed` (glitch steps per second, 8), `Split` (px, 6), `Shift`
(max block shift px, 48), `Density` (fraction of blocks that move, 0.2), `Jitter` (max row shift
px, 6). Properties: RGB Split (angle, flicker), Block Displacement (block width/height, 96 x 24),
Scanline Jitter (row density 0.25), Bit Crush (bits, off by default).

All decisions are PCG hashes of integers (block cell or row, step, per-effect sub-seed) with
integer pixel shifts, so CPU and GPU make identical random decisions and copy identical pixels
(measured exactly equal, difference 0). `step = floor(time * Speed)` (0 without an evaluation
context or Speed 0), so the same seed and time render identically and a new step (or seed) changes
the pattern. Order: block shift (wraps horizontally), row jitter (wraps), RGB split (R at +offset,
B at -offset, clamped; alpha = max), bit-crush (premultiplied RGB clamped to [0, 1], rounded to
2^bits - 1 levels), Fac mix. Crush CPU vs GPU: 1.2e-7.

**Ranges** (soft range | clamp):

* Fac 0..1, Density 0..1 | none. Seed 0..1000 | none. Speed -60..60, Split -100..100 | none.
  Shift 0..500, Jitter 0..100 | none (clamped to 0..width-1 internally).
* Properties: Angle -pi..pi; Flicker and Row Density 0..1 (hard); Block Width/Height 1..512 (hard
  max 4096); Bits 1..16 (hard).
