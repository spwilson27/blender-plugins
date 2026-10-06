# Simulate (menu: Add > Lab > Simulate)

Stateful nodes: they carry state from one evaluation to the next. How the state is keyed, cached
and reset is described in [lib/README.md](../../addons/compositor_lab/lib/README.md#stateful-nodes)
and `docs/plan-stateful-nodes.md`. Colours are premultiplied scene-linear; row 0 is the bottom.
Every simulate node has a **Reset** button (clears its state), and in the sidebar **Max Catch-up**
(frames it may step in one evaluation after a jump) and **Cached Frames** (scrub-back cache).

## Feedback / Trails (`CompositorNodeLabFeedback`)

Inputs: Image, Amount (factor, 0.9), Decay (0.97), Zoom (1), Rotation (degrees, 0), Offset X / Y
(pixels, 0), Hue Shift (degrees, 0). Properties: Blend (Mix or any Blend Modes+ mode), Edges
(Clamp, Repeat, Mirror), Pre-roll (0). Output: Image.

```
out_n = blend(input_n, T(out_{n-1}) * Decay, Amount)      Blend = Mix: input + (prev - input) * Amount
```

* `T` samples the previous output (bilinear) at the inverse of: zoom about the image centre,
  counter-clockwise rotation, offset (positive X moves the content right, positive Y up), then
  rotates the hue (a rotation of RGB about the grey axis, so alpha is untouched). With no zoom,
  rotation and offset the previous output is used as is; with Hue Shift 0 no hue rotation happens.
* Mix lerps all four premultiplied channels, so alpha trails too. Other blend modes use
  `lab_blend_pm` with the input as backdrop and the faded previous output as source: Add / Screen
  give light trails, Zoom 1.02 with a little Rotation gives the video-feedback tunnel.
* Closed form (static input, no transform, Mix): `out_n = s + (a d)^n (in - s)` with
  `s = in (1 - a) / (1 - a d)`, `n` = frames since the start (tested on CPU and GPU).
* First frame / start frame: outputs the input. **Pre-roll** N runs N feedback iterations on the
  input at that point, so a single still render shows converged trails (the stateless fallback).
* State: the previous output (RGBA32F, numpy on CPU, `GPUTexture` on GPU), one stream per
  (node, evaluation kind, size, device). Re-rendering a frame is idempotent and live edits show
  without stepping; scrubbing back restores cached frames; jumps catch up (the input of the
  requested frame feeds every step) up to Max Catch-up, then hold the last state.
* Tolerances: CPU vs GPU about 1e-6 per step; the tests allow 2e-4 over the first 5 frames with
  zoom, rotation, hue and blend modes.
