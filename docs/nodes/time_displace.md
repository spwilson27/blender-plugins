# Time Displace / Slit-scan (`CompositorNodeLabTimeDisplace`, menu Simulate)

Every row, column or pixel of the output shows the input from a different moment: the node keeps
the last frames of its input and reads each pixel from the frame `delay` frames ago. Slit-scan of a
moving bar draws sine waves, a radial delay gives ripples that travel out from the centre. Colours
are premultiplied scene-linear; row 0 is the bottom. Stateful: see
[lib/README.md](../../addons/compositor_lab/lib/README.md#stateful-nodes) and
`docs/plan-stateful-nodes.md`.

## Interface

Inputs: **Image**, **Map** (colour, used by Displacement Map; unlinked: 0.5 grey), **History Frames**
N (integer, 30; clamped to 0..120), **Amount** (factor, 1), **Center X / Y** (factors, 0.5; Radial).
Outputs: **Image**, **Delay** (float: the delay actually applied divided by N, 0..1).
Properties: Mode, Direction, Blend Frames (on), Downscale (Full, 1/2, 1/4), Precision (Half Float),
Memory Cap (MB, 640). **Reset** button. (Max Catch-up and Cached Frames of the common stateful sidebar are not used,
see State.)

| Mode | Delay of a pixel (`t` in 0..1, delay = `t x N x Amount`) |
|---|---|
| Slit-scan Rows | `t = row / (height - 1)`: bottom row 0, top row N (Direction *Down*: reversed) |
| Slit-scan Columns | `t = column / (width - 1)`: left 0, right N (Direction *Left*: reversed) |
| Displacement Map | `t` = luminance (Rec. 709 weights) of the Map pixel, clamped to 0..1 (*Dark*: `1 - t`) |
| Radial | `t` = distance from the centre / distance to the farthest corner (*Inward*: `1 - t`) |

`Direction` is a single enum: its first item reads Up / Right / Outward / Bright, the second the
opposite. Without **Blend Frames** the delay is rounded to whole frames (`floor(x + 0.5)`), which
shows steps in slit-scan; with it the two neighbouring frames are mixed linearly (`d = 3.5` is the
mean of the frames 3 and 4 back).

## History and state

* The history *is* the state: `N + 1` frames (the current one and N older), addressed by **frame
  number** (frame `f` lives in slot `f mod (N + 1)`), as a numpy array on the CPU
  (`lib/np_history.py`) and a 2D `GPUTexture` array on the GPU (`lib/glsl/history.py`, `layers=`
  works in this build), one layer per frame. The helper pair is reusable by other nodes.
* Frames enter as they are evaluated. The current frame is always read from the input at full
  resolution and precision (delay 0 is exactly the input); older frames come from the history.
* **Before the history fills, and after a jump,** a frame that is not stored is replaced by the
  nearest *older* stored frame in the window, or the oldest stored frame: frame 1 of a render shows
  the input, frame 2 clamps everything older than one frame to frame 1, and so on. A skipped frame
  (a jump of a few frames forward) holds the last frame seen. A still render with no history is
  therefore the input (the stateless fallback; a static input stays static whatever the delays).
* **Re-render** of the same frame replaces that frame in the history and is idempotent; a live edit
  of any parameter (Amount, Mode, ...) on the same frame recomputes immediately without stepping.
  Changing History Frames, Downscale, Precision or the render size rebuilds the history (it starts
  again from the current frame).
* **Scrub back:** frames newer than the requested one are ignored but kept, so going back `k`
  frames and forward again loses nothing; the delays that would reach before the oldest stored frame
  (the `k` oldest) clamp to it. Going back beyond the stored window restarts the history at that
  frame (a still image of the input, then it fills again). **Frames at or before the scene start
  frame, and the Reset button, clear the history.** After a forward jump beyond N frames, history
  also restarts from the requested frame (frames older than the window are never used).
* The frame cache of `lib/state.py` is not used (it would copy the whole history per frame), so
  *Cached Frames* does nothing here; the ring buffer is addressed by frame number instead. Streams
  are per (node, evaluation kind, size, device) as usual. The history counts towards the registry
  budget (`lib/state.py`: 1 GB over all streams).

## Memory

One layer is `ceil(w / s) x ceil(h / s) x 4` values (2 bytes with Half Float, 4 with Full Float).
At 1080p that is 16.6 MB per frame at Half / Full resolution, 4.1 MB at 1/2, 1 MB at 1/4. **Memory
Cap** (default 640 MB) bounds the whole ring; if `N + 1` layers do not fit, N is reduced to what
fits (the delay range is spread over the shorter history), and the node shows *History limited to
M frames* in its panel (`stream.message`). History Frames is capped at 120 whatever the memory.
Downscale stores each frame as the mean of `s x s` blocks (edge pixels replicated) and reads it
back with a bilinear sample, so smooth content is hardly changed while sharp detail of old frames
softens; the current frame stays sharp.

## Verification (tests/lab/test_time_displace.py, CPU and GPU)

* Frame `f` = constant `f/100` plus a spatial gradient. Slit-scan row `r` equals row `r` of frame
  `current - delay(r)` bit for bit (Full Float, no blending); same for columns and both
  directions; constant Map `v` gives frame `current - round(v N)`; blended delays against an
  independent float64 reference (`2e-6`; radial `2e-5`); clamping before the history fills.
* Ring helper against naive loops: box downscale, bilinear read (odd sizes, 1/2 and 1/4), slot
  resolution with gaps, and the memory fit.
* Downscale: through renders the gradient + constant frames survive 1/2 and 1/4 (sizes 32x24,
  40x28, 33x25) within `1e-5` away from the border (2 blocks); Half Float within `1e-3`.
* Semantics: sequential steps, re-render and parameter tweak, scrub back (inside and beyond the
  window), forward jump, skipped frames, scene start frame, Reset, rebuild on History Frames change.
* Memory cap (1 MB: capacity, bytes, message, delay range scaled), hard cap 120.
* CPU vs GPU: exact (0) for rounded rows / map, `<= 1e-6` with Blend Frames and radial (sqrt / FMA
  differences); with Half Float the GPU store truncates where the CPU rounds, so they differ by at
  most one half-float ulp (`2^-11` relative), tested with `rtol 1.1e-3`.
* Robustness: nothing linked, sizes 4x4 .. 33x17, every mode and Downscale, History Frames 0 / 1 / 2.

Performance at 1080p (Half Float, 30 frames, Blend Frames; including a ~48-56 ms render and
readback that a pass-through also pays): about +14 ms per frame on the CPU for slit-scan rows,
+52 ms for a Displacement Map at 1/2 (four taps per frame), +42 ms radial at Full Float; on the GPU
about +1 to +20 ms. The first frame also allocates the ring.
