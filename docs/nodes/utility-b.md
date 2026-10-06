# Compositor Lab: Utility B nodes

Menu: Add > Lab > Utility. Sources: `addons/compositor_lab/nodes/{seamless_tile,mask_tools,pixel_shuffle,line_sort}.py`,
shared code in `lib/distance.py` and `lib/glsl/distance.py`. Tests: `tests/lab/test_<node>.py`
(suites `lab_seamless_tile`, `lab_mask_tools`, `lab_pixel_shuffle`, `lab_line_sort`).

## Seamless Tile

Makes an image tile without a visible seam. Output has the input's size.

- Inputs: `Image` (colour, default 50% grey). Output: `Image`.
- Properties: `Mode` (Offset Blend / Mirror), `Axes` (Both / Horizontal / Vertical),
  `Blend Width` (0.01 to 1, default 0.5; Offset Blend only).
- Offset Blend: the image is cross-faded with a copy rolled by half its size (that copy is continuous
  across the border, the original is continuous in the middle). Per axis `C = lerp(roll_x(A), A, mx(x))`,
  then `D = lerp(roll_y(C), C, my(y))`, fused into four samples. `m` is a smoothstep of the distance to the
  nearest border, reaching 1 at `Blend Width` x half the size. The region where both masks are 1 is the
  input unchanged; premultiplied colours blend linearly.
- Mirror: `x -> min(x, W - 1 - x)` (same for y). The output is mirror symmetric, opposite edges are
  identical; the right / top half of the input is not used.
- CPU / GPU: float32 maths; GPU differs by fma contraction only (observed max 1.8e-7, test tolerance 2e-6).
  Mirror is exact.

## Mask Tools

Threshold, grow / shrink, feather, outline and invert, with exact Euclidean distances.

- Input: `Mask` (colour socket, default white; a float output linked to it is its own value, alpha 1).
  Output: `Mask` (float).
- Properties: `Value` (Value = red / Alpha / Luminance), `Threshold` on/off with `Low`, `High`, `Softness`,
  `Grow` (pixels, negative shrinks), `Edge` (None / Feather / Outline) with `Feather` (width) or `Width` +
  `Position` (Centered / Outside / Inside), `Invert`.
- Pipeline: value -> threshold (`low <= v <= high`; softness replaces each edge by a smoothstep of that
  width) -> morphology on the binary mask `m >= 0.5` -> invert.
  - Edge None: grow by r keeps every pixel within distance r of an inside pixel (a pixel becomes the exact
    lattice disc `dx^2 + dy^2 <= r^2`); shrink keeps pixels whose distance to the nearest outside pixel is
    > r (dual of grow, so shrink then grow is an opening: anti-extensive and idempotent).
  - Feather: signed distance `s = 0.5 - d_in` (inside) / `d_out - 0.5` (outside), minus the grow amount;
    result `smoothstep(0.5 - s / width)`; compact support, exactly 1 / 0 beyond width/2 (not a Gaussian).
  - Outline: pixels with `lo < s <= hi` (centred: +-width/2; outside: 0..width; inside: -width..0), binary.
    A 1 px inside outline is the 1 px ring; outside with width 1 includes diagonal neighbours (d <= 1.5).
- The image border is not an edge. Radius is capped at 1024 px (feather width 2048).
- CPU: `lib/distance.edt_sq`, Felzenszwalb-Huttenlocher run on all rows / columns at once with numpy
  (exact, 0.4 s for 1920x1080). GPU: two separable exact passes with a search window sized to the
  needed distance (`lib/glsl/distance.py`), squared distances are integers in float32, clamped to
  `(R+1)^2` identically on both backends. All binary results are bit-identical on CPU and GPU (tests:
  every mode, odd sizes down to 4x4). Soft threshold / feather agree to 2e-5 (sqrt, smoothstep rounding;
  observed far smaller). Luminance uses the rounded-product trick from `glsl/exact.py` so thresholds agree.

## Pixel Shuffle

Seeded scrambling; the output is always a permutation of the input pixels.

- Inputs: `Image`. Output: `Image`.
- Properties: `Mode`, `Block Size` (Pixels in Blocks / Shuffle Blocks), `Radius` + `Iterations` (Swap Pairs),
  `Amount` (0..1), `Seed`, `Animate` + `Rate` (seed += floor(time * rate); uses the F2 context time).
- Pixels in Blocks: permutation of the pixels of each NxN block (edge blocks smaller), different per block.
  Swap Pairs: blocks of (Radius+1)^2 at a seeded offset, pixels randomly paired inside, pairs swap (an
  involution; Chebyshev displacement <= Radius); `Iterations` repeats with new offsets. Shuffle Blocks:
  whole NxN blocks permuted (incomplete right / top blocks stay).
- Amount: fraction of pixels / pairs / blocks affected, quantised to 1/1024; 0 is the identity. In the shuffle
  modes the `round(amount * M)` elements with the lowest Feistel rank move along one random cycle (so every
  affected element changes place; no fixed points among them).
- Implementation: each output pixel gathers; permutations are 5-round Feistel networks (round function = the
  shared PCG hash, seeds via `lab_hash3`) with cycle walking, computed per pixel (no tables). The GLSL is
  node-local (registered as `lib/glsl/pixel_shuffle` in `sys.modules`, see "lib gaps" in the report).
  CPU and GPU run the same integer maths: bit-identical.

## Line Sort

Reorders whole rows or columns by a per-line statistic.

- Inputs: `Image`. Output: `Image`.
- Properties: `Sort` (Rows / Columns), `By`, `Descending`, `Band Size` (0 = whole image, else sort within
  consecutive groups of K lines).
- Statistics: mean luminance (Rec. 709), mean HSV hue (plain mean in [0, 1), no circular averaging), mean
  saturation, luminance variance, sum of red / green / blue / alpha. Colours are used as stored
  (premultiplied), clamped to [0, 4] and quantised to 1/1024 per pixel before summing, so sums are exact
  integers independent of summation order (keys equal sums, every line has equal length). The variance key
  is a float32 computed from exact integer sums by a fixed sequence of rounded operations (both backends).
- Ordering: stable by (band, key, index); descending uses the complemented key and keeps original order for
  ties.
- GPU (3 dispatches, scratch RGBA32UI textures): one thread per line computes the key; one thread per line
  computes its rank = band start + number of lines in the band that sort before it (parallel counting
  sort, O(K) per line, exactly the stable order; used instead of a bitonic network since it needs no
  inter-thread synchronisation); every pixel is scattered to its line's rank. Output is bit-identical to
  the CPU (`np.lexsort`), tested for all statistics, both axes, bands, ties and odd sizes.
