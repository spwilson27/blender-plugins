# Utility A: Expression, Image Statistics, Auto Levels, Palette Extract

All four nodes are in the Lab > Utility menu. Colours are premultiplied scene-linear, row 0 is the
bottom of the image. New library modules: `lib/expr.py`, `lib/reduce.py`, `lib/glsl/reduce.py`.

## Expression (`CompositorNodeLabExpression`)

Inputs: `A`, `B`, `C`, `D` (colours, default black). Output: `Color`. Property: `expression` (text).

A per-pixel formula in a whitelisted subset of Python syntax. It is parsed with `ast`, checked,
lowered to a typed IR and compiled twice, to a tree of numpy closures (CPU) and to a GLSL block of
SSA temporaries (GPU). Nothing is passed to `eval` / `exec`.

* Variables: `a b c d` (vec4 colours; unlinked: the socket value), `x y` (pixel centre, texel + 0.5),
  `u v` (x / w, y / h), `w h` (size), `t` (context time, seconds), `frame`, constants `pi tau`.
* Operators: `+ - * / % **`, unary `- + not`, comparisons (scalars, chains allowed, give 1.0/0.0),
  `and` / `or` (1.0/0.0), `a if c else b`.
* Functions: `sin cos tan asin acos atan sqrt abs floor ceil fract exp log sign`, `atan2 min max pow
  step`, `clamp mix smoothstep`, `length dot normalize luma`, `noise(x, y[, z])` (Perlin of the Lab
  noise library, 0..1), `vec(...)` (constructor, `vec(f)` splats to vec3).
* Swizzles: `.r .g .b .a .rgb .bgr .xy .xyzw ...` (rgba or xyzw letters, 1 to 4 components).
* Types: scalar, vec2, vec3, vec4; scalars broadcast. The result must be scalar (grey, alpha 1),
  vec3 (alpha 1) or vec4 (as is).
* Definitions made identical on both backends: `sqrt(max(x,0))`, `log(max(x,1e-30))`, `asin`/`acos`
  clamp their argument, `%` is floor-mod, `mix = a + (b-a)t`, `normalize(0) = 0`, `x ** n` for a
  literal integer `|n| <= 16` is repeated multiplication (negative bases fine), other powers are
  `pow(max(x,0), y)`. Division by zero is undefined (inf/nan).
* Rejected with an `ExprError` (shown as the node's info message, output black): any other syntax
  (lambda, comprehension, subscript, strings, `//`, bit operators, keyword arguments ...), unknown
  names, attribute access other than swizzles, calls to anything outside the whitelist, type
  mismatches, expressions over 2000 characters / 1500 nodes / 60 levels deep.
* Compiled code is cached by expression string (`lib/expr.compile_expr`); GPU shaders are cached by
  the generated body by `lib/gpu.pointwise`.

Tolerances (tests): CPU node vs float64 reference 2e-5; GPU vs CPU 2e-4 (GPU trig/exp/pow are
approximate), `noise()` 6e-4.

Ranges: no numeric sockets (colour inputs only); no numeric properties.

## Image Statistics (`CompositorNodeLabImageStatistics`)

Input: `Image`. Single-value (F3) outputs: `Min`, `Max`, `Mean` (colours, per channel, alpha 1),
`Luminance Mean`, `Std Dev` (population standard deviation of Rec. 709 luminance), `Percentile`.
Property: `percentile` (0..100) of the luminance.

Statistics are of the stored (premultiplied) values over all pixels. The percentile comes from a
256-bin histogram over [luminance min, max] with interpolation inside the bin (error below
range/256; 0 gives the minimum, 100 the maximum). CPU: numpy with float64 accumulation. GPU: no
atomics: a tile pass (one thread per 16x16 tile, 5 scratch textures) and merge passes until 1x1
(mean/variance combined with Chan's formula), then one small readback; the histogram uses one
thread per tile with a private histogram and a per-bin sum pass. An unlinked input is a 1x1 image.
Tolerances: Min/Max exact; means 3e-6 relative (observed 5e-7); CPU vs numpy 2e-7.

Ranges: no numeric sockets; Percentile 0..100 (hard).

## Auto Levels (`CompositorNodeLabAutoLevels`)

Inputs: `Image`, `Fac`. Output: `Color`. Properties: `mode` (Luminance / Per Channel), `low`
(default 1 %), `high` (99 %), `clamp` (on), `auto_gamma` (off), `target` (0.5).

Contrast stretch `y = (c - lo) / (hi - lo)` of the straight colour, with `lo`/`hi` the Low/High
percentiles of pixels with alpha > 0 (histogram, as above). Luminance mode takes them from the
luminance and uses one affine map for R, G and B (hue preserving); Per Channel stretches each
channel on its own. `auto_gamma` solves (bisection on the histogram) for the gamma that makes the
mean of the stretched, clamped values equal `target`. `Fac` mixes with the original; alpha is
unchanged; a degenerate range (hi <= lo, flat or empty image) leaves the image untouched. The
parameters are computed on the host from the reductions (CPU: numpy, GPU: compute passes), the
per-pixel apply is a numpy / GLSL pointwise kernel. Tolerances: CPU vs independent numpy 2.3e-7
observed (test 1e-5); GPU vs CPU 3.6e-7 observed (test 2e-6). Output percentiles land on 0/1
within 0.02 (one bin of the stretched range).

Ranges (soft | clamp): Fac 0..1 | none; properties Low / High 0..100 and Target 0.01..0.99 (hard).

## Palette Extract (`CompositorNodeLabPaletteExtract`)

Input: `Image`. Outputs: `Swatch` and `Quantized` (images), `Color 1` .. `Color 8` (single-value
colours; unused ones are black with alpha 1). Properties: `count` (1..8, default 5), `space` (OKLab
default / Linear RGB), `seed`.

1. The image is point-sampled on a grid of at most 128 x 128 samples (sample i of n over s pixels
   reads pixel `(i*s + s//2)//n`). On the GPU this is a compute pass, then a small readback.
2. K-means runs on the host in numpy (float64) in **both** modes (k-means++ seeding with a seeded
   `numpy.random.Generator`, at most 24 Lloyd iterations, weight = alpha, transparent samples
   ignored). Documented deviation from a fully GPU-side clustering: the clustering sees at most
   16k samples, and CPU and GPU get bit-identical palettes.
3. Palette colours are the alpha-weighted mean linear colour of each cluster, sorted by population
   (largest first). Fewer distinct colours than `count` give fewer entries.
4. `Quantized`: every pixel becomes the nearest palette colour in the chosen space (ties: lowest
   index), alpha kept (premultiplied). `Swatch`: equal-width bars of the palette (black if empty).
   Both are pointwise GPU kernels / numpy; the palette reaches the shader in a 8x2 RGBA32F texture.

Observed CPU vs GPU: palette, Swatch and Quantized are bit-identical in the tests.

Ranges: no numeric sockets; Colors 1..8 (hard); Seed soft 0..1000, no clamp.
