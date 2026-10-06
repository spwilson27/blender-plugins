# Reaction-Diffusion (`CompositorNodeLabReactionDiffusion`, Simulate)

Gray-Scott reaction-diffusion: two fields U and V that grow patterns (spots, coral, worms, mazes)
a few iterations per frame. A stateful node: see [simulate.md](simulate.md) for the shared
behaviour (Reset button, Max Catch-up, Cached Frames, repeats, scrubbing, jumps).

Inputs: Seed (image, optional), Feed (0.0545), Kill (0.062), Du (1.0), Dv (0.5), dt (1.0),
Iterations per Frame (20), Feed Map, Kill Map (value 1, or an image). Outputs: V (float),
U (float), Color (V through a gradient). Properties: Preset, Scale (2), Edges (Wrap), Seed (1),
Density (0.12), Noise (0.02), Pre-roll (0), Color A / Color B, Low / High (V range of the
gradient, 0.05 / 0.35).

## Equations

```
lap(X) = 0.2 (4 orthogonal neighbours) + 0.05 (4 diagonal neighbours) - X          9-point
U' = clip(U + dt (Du lap(U) - U V V + F (1 - U)), 0, 1)
V' = clip(V + dt (Dv lap(V) + U V V - (F + k) V), 0, 1)
```

* The 9-point Laplacian is nearly isotropic (round spots, not diamonds). Its most negative
  eigenvalue is -1.6, so the explicit step is stable while `D dt < 1.25`. The clip to [0, 1] after
  every iteration keeps the fields bounded for any finite parameters (parameters are clamped:
  F, k to [0, 1]; Du, Dv, dt to [0, 5]; iterations to [0, 1000]), so there are never NaN or Inf.
* **Edges**: Wrap (the fields tile, so the result is a seamless texture) or Clamp (the border pixel
  repeats: no flux through the edge).
* Per frame the node runs *Iterations per Frame* iterations. One GPU dispatch per iteration, ping-pong
  between scratch textures.

## Presets

The Preset enum writes the Feed / Kill sockets (edit a socket afterwards to go custom; *Custom*
leaves them alone). All for Du = 1, Dv = 0.5:

| Preset | Feed | Kill | Look |
|---|---|---|---|
| Coral | 0.0545 | 0.062 | branching coral / labyrinth |
| Mitosis | 0.0367 | 0.0649 | blobs that split |
| Worms | 0.078 | 0.061 | short wandering strands |
| Spots | 0.035 | 0.065 | stable round spots |
| Maze | 0.029 | 0.057 | labyrinth stripes |
| Bubbles | 0.098 | 0.057 | sparse rings and dots |

## Seeding

The first frame (and any frame at or before the scene start) seeds the fields:

* **Seed image linked**: where its luminance (Rec. 709 on the premultiplied RGB, bilinearly
  resampled to the simulation size) is above 0.5, `V = 1, U = 0.5`; elsewhere `U = 1` and `V` is
  white noise times *Noise*.
* **Seed unlinked**: hashed random blobs. The simulation is cut into 8x8 cells; each cell holds a
  disc (radius ~2 simulation pixels) with probability *Density*, at a hashed position. *Seed* is the
  hash seed (the same `lab_hash3` PCG hash as the noise nodes, integer maths, so CPU and GPU produce
  identical blobs). Very small simulations may get no blob: raise the density.
* **Pre-roll** N: after seeding, runs N frames' worth of iterations (N x Iterations per Frame) so
  that a single still render shows a developed pattern (the stateless fallback). 0 = frame 1 shows
  the seed.

## Feed / Kill maps

`F(x) = Feed * FeedMap(x)`, `k(x) = Kill * KillMap(x)` (red channel, bilinearly resampled to the
simulation size, edge clamped). Unlinked, the socket value scales the parameter uniformly, so 1
changes nothing. Values around 0.9 to 1.1 give smooth transitions between pattern types across the
image; gradients through the Coral / Mitosis region are the interesting range.

## Scale and output

The simulation runs at `ceil(size / Scale)` (state is stored at that size: 8 bytes per pixel), the
outputs are bilinearly upsampled with the Edges mode. Scale 1 is exact (no resampling). Changing
Scale restarts the state. Larger Scale is faster and gives larger features (the length scale of
the pattern is fixed in simulation pixels).

`Color = mix(Color A, Color B, smoothstep(clamp((V - Low) / (High - Low), 0, 1)))`, computed after
upsampling. V seldom exceeds about 0.5 in patterns, hence the default range 0.05 to 0.35.

## State

(U, V) at simulation size: two float32 arrays on the CPU, one RG32F texture on the GPU, one stream
per (node, evaluation kind, size, device). Re-rendering a frame is idempotent, live edits show
without stepping, scrubbing back restores cached frames, jumps catch up (every skipped frame runs
its full iteration count), beyond Max Catch-up the state is held.

## Tests and tolerances (`tests/lab/test_reaction_diffusion.py`)

* An independent float64 numpy Gray-Scott in the test (index gathers, not shared with `lib/np_rd.py`)
  matches the node over the first frames (up to 220 iterations) within 5e-6 on CPU and GPU, for
  Wrap and Clamp edges, other Du / Dv / dt, per-pixel and uniform Feed / Kill maps, the U output and
  pre-roll. Observed: about 1e-6 at 60 iterations, 2.6e-6 at 100 (float32 rounding).
* CPU vs GPU over the first 5 frames: observed 6e-7 or better, tolerance 5e-6 (image seed, Clamp with
  maps, noise seeds, Scale 2). Noise seeding is bit-identical.
* Statistics after 29 frames (chaotic divergence allowed): the pattern coverage (fraction of V > 0.15)
  agrees within 0.10 (observed equal to 3 decimals) and mean V within 0.05.
* Presets reach non-trivial patterns (coverage ranges per preset, std > 0.02) after 40 frames.
* Extreme parameters (feed/kill 0 and 1, out-of-range, 5000 iterations, huge dt), odd sizes
  (7x5, 33x17, 4x4 with Scale 3): finite and in [0, 1].
* Semantics per `lib/state.py`: step, repeat, parameter tweak, restore, catch-up, hold, start-frame
  reset, Reset operator, duplicate independence, state at simulation size.

Performance at 1920x1080, 20 iterations per frame (Mac mini, M-series): CPU 250 ms/frame at Scale 1,
210 ms at Scale 2; GPU 55 ms at Scale 1, 30 ms at Scale 2.
