# Plan: Compositor Lab, the stateful nodes

Stateful nodes carry information from one evaluation to the next. Examples are simulations that
advance one step per frame, and effects that read earlier frames. This builds on the stateless
library (`docs/plan-stateless-nodes.md`).

## The problem

The compositor evaluates a node from scratch every time, and it evaluates the same tree in
several places that are independent and can interleave:

| Evaluation | Trigger | Device | Thread |
|---|---|---|---|
| Render (F12, `-f`, `-a`) | render | per scene setting | render thread / main (`-b`) |
| Backdrop / viewer | tree edits, frame change | per scene setting | job thread |
| Viewport compositor | redraw | GPU | main |
| Sequencer strip modifier | strip render | per scene setting | sequencer |

Additional constraints:
- Each evaluation runs on a **copy** of the node tree, so neither the node pointer nor the Python
  object stays stable.
- Frames can arrive **in any order**: playback, scrubbing, jumps, and re-renders of the same frame.

A stateful node therefore needs three things:
1. A stable identity.
2. State kept separately for each evaluation stream, so the viewport doesn't step the render's
   simulation.
3. Rules for what happens when frames don't arrive one after another.

## Design

### F4 (framework): evaluation kind

Add `context.kind` to the evaluation context. It is one of `'RENDER'`, `'BACKDROP'`, `'VIEWPORT'`
or `'SEQUENCER'`, derived from the compositor `Context` subclass. Also add
`context.is_animation_playing`, which the compositor job already knows. This is a small C++
change in the same place as F2. Without it, Python can't tell interleaved streams apart.

### Stable node identity (pure Python)

- **Stable ID:** a hidden `StringProperty` `lab_uid`, set in `init()` to a UUID.
- **Copies:** regenerated in `copy()`, so duplicated nodes get their own state.
- **Survival:** it's stored in the .blend, so state keys survive copy-on-evaluation, renames and
  reloads.

### `lib/state.py`, the simulation cache

```python
sim = self.state(ctx)          # StateStream for (lab_uid, ctx.kind, size, use_gpu)
step = sim.advance(ctx.frame)  # decides how this frame relates to the stored state
```

`advance()` follows simulation-zone semantics:

| Situation | Behaviour |
|---|---|
| frame ≤ scene start frame, or first evaluation, or size/device changed | **reset**: init state from inputs |
| frame == last + 1 | **step** once |
| frame == last (re-evaluation, e.g. a parameter tweak) | **repeat**: recompute from the state *before* the last step, so live edits show without running the simulation forward |
| frame in the per-stream frame cache | **restore** the cached state (scrubbing back) |
| forward jump of k > 1 | **catch up**: step k times up to `max_catch_up` (a property), then hold |
| anything else (backward to an uncached frame) | **reset and simulate** from start, bounded by `max_catch_up`, else hold with an info message |

Implementation details:
- **State storage:** numpy arrays on CPU, `GPUTexture` on GPU.
- **Frame cache:** a ring buffer with a memory budget, by default the last 32 frames or 256 MB per
  node, with least-recently-used eviction across all nodes.
- **Cleanup:** `bpy.app.handlers` `load_pre` clears all state, and node deletion is detected
  lazily, so unused streams expire.
- **Reset button:** each stateful node draws a **Reset** button, an operator that clears its
  streams and tags the tree.
- **Thread safety:** state is only touched with the GIL held, inside evaluate. The cache is a plain
  dict keyed per stream.

### Nodes

| Node | State | Per-step work | Notes |
|---|---|---|---|
| **Reaction-Diffusion** | U, V fields (RG32F) | N Gray-Scott iterations per frame (feed, kill, Du, Dv, dt); presets for coral, mitosis, worms and spots | Seeded from an image or mask input (V where the mask is set) or a noise seed. Optional feed/kill maps from inputs. Output: V, U, and a coloured view. |
| **Cellular Automata** | cell grid (R8) | one or more generations per frame; rules in B/S notation (Life `B3/S23`, HighLife, Seeds…) or Generations-style decay | Seeded from a thresholded input. Output: cells plus an "age" heatmap. |
| **Feedback / Trails** | previous output | `out = mix(input, transform(prev) * decay, amount)`, where transform covers zoom, rotate, offset and hue shift (video-feedback tunnel) | Optional "blend mode" from `np_blend`/`glsl.blend`. |
| **Time Displace / Slit-scan** | ring buffer of the last N input frames | each pixel or row picks a frame from the buffer via a displacement map, or row index for slit-scan | Memory-heavy: N × frame size. A downscale option and a hard cap apply. Frames enter the buffer as they are evaluated. |

Each node also has a **stateless fallback**: when the stream can't advance, for example on a
single still render with no history, it runs N iterations in one evaluation. Reaction-diffusion
and automata then still produce a still image.

## Testing

Background tests render frame sequences with `scene.frame_set` plus `bpy.ops.render.render` (the
RENDER stream). Coverage:

- **Determinism:** the same frames in the same order give identical output; CPU and GPU agree
  within tolerance for the first frames, since chaotic systems diverge later. Long runs are
  compared statistically.
- **Simulation semantics:**
  - Sequential frames step.
  - Re-rendering a frame is idempotent.
  - Scrubbing back restores from the cache.
  - Jumps catch up.
  - The start frame resets.
  - Duplicated nodes have independent state.
  - The memory budget evicts.
- **Correctness against references:**
  - Gray-Scott against an independent numpy implementation.
  - Life patterns (blinker period 2, glider translation after 4 generations).
  - Feedback as a closed form for decay.
  - Slit-scan row r equals frame r's row.
- **Streams are independent:** a GUI test where backdrop evaluations interleaved with renders
  don't disturb each other (F4).
- **Memory:** the caps hold, and `load_pre` clears.

## Execution

1. **F4** (agent → my review → build → sync to the mini).
2. **`lib/state.py`** with its semantics tests, plus **Feedback / Trails** as the reference node
   (one agent).
3. In parallel (3 agents): Reaction-Diffusion, Cellular Automata, Time Displace / Slit-scan.
4. Full suite on the mini, a gallery including animated strips (frames 1, 10, 50), docs, commit
   and push.
