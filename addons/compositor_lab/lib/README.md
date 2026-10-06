# Compositor Lab library

How to write a node for the Compositor Lab package, and the library pieces it is built on. The
nodes live in `nodes/<name>.py`; the package `__init__.py` discovers them (a broken module is
logged and skipped).

## Layout

| Path | Purpose |
|---|---|
| `lib/node.py` | `LabNode` base class, `In` / `Out` socket specs, `Ctx` |
| `lib/gpu.py` | `kernel` / `pointwise` builder, shader cache, uniform binding, scratch textures |
| `lib/state.py` | state streams for stateful nodes: frame cache, memory budget, Reset operator |
| `lib/errors.py` | log of evaluation errors (read by the tests), `LabShaderError` |
| `lib/glsl/*.py` | GLSL as Python strings: `hash`, `color`, `noise`, `blend`, `exact`, `pattern`, `field`, `dither`, `sampling`, `reduce`, `distance` |
| `lib/np_*.py`, `lib/distance.py`, `lib/reduce.py`, `lib/expr.py` | numpy twins of the GLSL, and the CPU sides of the multi-pass helpers |

Rule: a node's maths lives once per backend. If more than one node needs it, it goes in `lib/`
(GLSL in `lib/glsl/<name>.py`, numpy twin in `lib/np_<name>.py`).

## Writing a node

```python
import bpy
import numpy as np
from bpy.props import FloatProperty

from ..lib import gpu as lab_gpu
from ..lib.node import In, LabNode, Out

MENU = "Filter"            # "Filter", "Generate", "Simulate" or "Utility": the Add > Lab submenu

class CompositorNodeLabGain(LabNode, bpy.types.CompositorNode):
    '''Multiply the image by a gain'''                  # tooltip
    bl_idname = "CompositorNodeLabGain"
    bl_label = "Gain"

    SOCKETS = [In("Image", "COLOR", (0.5, 0.5, 0.5, 1.0)), In("Gain", "FLOAT", 2.0),
               Out("Image", "COLOR")]
    PROPS = ["clamp"]                                   # drawn by draw_buttons, in order

    clamp: FloatProperty(name="Clamp", default=1.0)

    def cpu(self, inputs, outputs, ctx):
        out = self.out_array(outputs, "Image")          # (H, W, 4) float32, None if unused
        if out is None:
            return
        img = self.in_image_array(inputs, "Image", ctx.shape)
        gain = self.in_float(inputs, "Gain", 2.0)
        out[...] = np.minimum(img * np.float32(gain), np.float32(self.clamp))

    def gpu(self, inputs, outputs, ctx):
        dst = self.out_texture(outputs, "Image")
        if dst is None:
            return
        lab_gpu.pointwise(
            "    out_Image = min(in_Image(texel) * gain, vec4(clamp_max));\n",
            {"Image": dst},
            inputs={"Image": ("color", self.in_texture_or_value(inputs, "Image", (0.5,) * 3 + (1.0,)))},
            uniforms={"gain": ("float", self.in_float(inputs, "Gain", 2.0)),
                      "clamp_max": ("float", self.clamp)})

NODE_CLASSES = [CompositorNodeLabGain]                  # required
```

* **`SOCKETS`**: `In(name, type, default, hide_value=False)` and `Out(name, type, single=False)`;
  types `FLOAT`, `FACTOR`, `COLOR`, `VECTOR`, `INT`, `BOOL`. **`PROPS`**: names of the properties to
  draw (`(name, {kwargs})` for `layout.prop` arguments, `None` for a separator); override
  `draw_buttons` for anything conditional. Properties are normal `bpy.props` annotations.
* **`cpu` / `gpu`** receive `(inputs, outputs, ctx)`. `ctx` has `frame`, `fps`, `time` (seconds),
  `size` (w, h), `shape` (h, w), `use_gpu`. Without the F2 context feature of the build, `ctx`
  defaults to frame 0 / time 0 / the output size. `evaluate_cpu` / `evaluate_gpu` are provided.
  Any exception in them is recorded in `lib/errors.py` and re-raised: Blender shows it as the
  node's info message and writes default outputs, and the tests fail on it.
* **Outputs**: `out_array(outputs, name)` (CPU, writable, `None` if unused) and
  `out_texture(outputs, name)` (GPU). Always skip unused outputs. Float outputs have one channel.
* **Single-value outputs** (F3): `Out("Mean", "FLOAT", single=True)`, written with
  `self.out_single(outputs, "Mean")[:] = ...` on both backends (a 1-D array of the channel count).
* **Inputs**: `in_float` / `in_int` / `in_color` give a *single value* (the default if the socket is
  linked to an image, identically on both backends). Per-pixel inputs: `in_image_array(inputs,
  name, ctx.shape, channels)` on the CPU (broadcasts single values, adapts channels, clamp-resamples
  other sizes) and `in_texture_or_value(inputs, name, default)` on the GPU (a `GPUTexture` or a
  single value; `lab_gpu.pointwise` accepts both). Colours are premultiplied scene-linear; row 0 is
  the bottom.
* A node with no image inputs is a *generator*: its outputs have the render size (F1).
* Seeds / time: derive per-frame values from `ctx.time` / `ctx.frame` on the CPU **and** pass the
  same value to the GPU as a uniform, so both paths agree.

## Stateful nodes

Nodes that carry state between evaluations (feedback, simulations, history buffers). The compositor
evaluates a copy of the tree from independent places (render, backdrop, viewport, sequencer, see
`ctx.kind`) with frames in any order, so the state lives in `lib/state.py`, not on the node.
`nodes/feedback.py` is the reference implementation.

```python
from ..lib.node import LabNode, StatefulNode

class CompositorNodeLabThing(StatefulNode, LabNode, bpy.types.CompositorNode):   # mixin order matters
    PROPS = [...]
    def draw_buttons(self, context, layout):
        self.draw_props(layout)
        self.draw_state_buttons(layout)                 # the Reset button

    def cpu(self, inputs, outputs, ctx):
        plan = self.advance(ctx)                        # StateStream for this node + evaluation
        state = plan.run(init=lambda: make_state(inputs), step=lambda s: step(s, inputs))
        out[...] = state
```

* **Identity.** `StatefulNode` adds a hidden `lab_uid` (UUID, set in `init()`, regenerated in
  `copy()`, which Blender calls only when the user duplicates the node). ID properties survive the
  evaluation copies of the tree, so `lab_uid` keys the state; `self.state_key()` falls back to the node
  name for nodes made before it existed. It also adds the `max_catch_up` and `cache_frames`
  properties (drawn in the sidebar) used by `self.advance(ctx)`.
* **Streams.** `self.state(ctx)` returns the `StateStream` for `(lab_uid, ctx.kind, ctx.size,
  ctx.use_gpu)`; a new key (other size, device or evaluation kind) simply starts from reset. On
  builds without F4 the kind is `'UNKNOWN'`, the scene start frame defaults to 1 (`ctx.kind`,
  `ctx.is_animation_playing`, `ctx.frame_start`, `ctx.frame_end`).
* **`advance(frame, start_frame, max_catch_up, cache_frames)`** returns a `Plan`
  (`plan.kind`, `plan.state`, `plan.init`, `plan.steps`, `plan.message`). `plan.run(init, step)`
  executes it and commits the result: `init()` builds the state from the inputs, `step(state)` returns
  the next state; both return **new** payloads. Do not mutate `plan.state` or a committed payload.
  Do custom flows with `plan.commit(state, pre)`.

  | Situation | `plan.kind` | What `run` does |
  |---|---|---|
  | first evaluation, or frame <= scene start | `RESET` | `init()` |
  | frame == last + 1 | `STEP` | one step |
  | frame == last (re-render, parameter tweak) | `REPEAT` | one step from the state *before* the last step |
  | frame in the frame cache | `RESTORE` | the cached state (a copy), no step |
  | forward jump of k <= max_catch_up | `CATCH_UP` | k steps (all fed the requested frame's inputs) |
  | backward to an uncached frame f, f - start <= max_catch_up | `RESIM` | `init()`, then f - start steps |
  | any other jump | `HOLD` | the current state unchanged (`plan.message` says why; not cached); the stream moves to the requested frame, so sequential frames carry on |

  Frames are rounded to integers. The input of the skipped frames is not available, so a catch-up of a
  node that reads time-varying inputs is an approximation.
* **State payloads** are numpy arrays (CPU), `gpu.types.GPUTexture` (GPU), or tuples / lists / dicts of
  them. The frame cache stores copies (GPU: a copy kernel, `state.gpu_copy`). On the GPU allocate a
  fresh texture per step (`gpu.types.GPUTexture(ctx.size, format=...)`) and write the result
  to the node's output with a copy kernel; kernels may write several outputs at once.
* **Memory.** Per stream: `cache_frames` (default 32) frames and 256 MB; over all streams 1 GB, by
  least-recently-used eviction (cached frames first, then idle streams); streams unused for 30 minutes
  expire. `state.configure(cache_frames=, stream_bytes=, global_bytes=, max_catch_up=, ttl=)` changes
  the defaults. `load_pre` clears everything; the `compositor_lab.reset_state` operator clears one
  node's streams by `uid` and tags the tree.
* **Stateless fallback.** A node should offer a way to produce a meaningful still without history
  (Feedback: *Pre-roll* iterations in `init()`; simulations: N iterations from the seed).
* **Tests.** Render frame sequences with `scene.frame_set(f)` + `H.render(scene)`; look at
  `state.streams_for(node.lab_uid)[0].last_kind` to assert which branch ran. See
  `tests/lab/test_state.py` (semantics with fake payloads) and `tests/lab/test_feedback.py`.

## The GPU kernel builder

`lab_gpu.kernel(body, outputs, inputs, uniforms, libs, functions, local_size, sampling)` builds,
caches, binds and dispatches a compute shader over the output size. `pointwise` is the same without
`sampling`. See the docstring in `lib/gpu.py` for the full contract; in short:

* `body` is GLSL inside `main`. You get `texel`, `uv`, `res`, `in_<Name>(ivec2 p)` per input,
  a `vec4 out_<Name>` per output, your uniforms by name, and `lab_zero`.
* `inputs`: `name -> (kind, value)`, kind `"float"`, `"vec3"` or `"color"` (the type `in_<Name>`
  returns). A texture is read with `texelFetch` (clamped to the texture), a single value is a push
  constant. A single-channel texture (R8 / R16F / R32F) reads as grey: `vec3(r)` / `vec4(r, r, r, 1)`
  (`"float"` reads `.r`), the same as `LabNode.in_image_array` on the CPU.
* `uniforms`: `name -> (type, value)` with type `int`, `float`, `vec2`, `vec3`, `vec4`.
* `libs`: `lib/glsl` modules and their dependencies, each included once.
* `functions`: node-local GLSL (helper functions, constants) placed **after** the accessors and
  libs and **before** `main`, so helpers may call `in_<Name>`, lib functions and read uniforms.
  Order in the final source: (sampling module) / accessors, libs, functions, main. Lib code may
  therefore call `in_<Name>` too (it is declared first).
* `sampling=True`: every input becomes a sampler (single values become cached 1x1 textures) with
  `lab_fetch_`, `lab_px_` (edge modes), `lab_bilinear_`, `lab_bicubic_` and `lab_sobel_<Name>`
  helpers, see `lib/glsl/sampling.py`. `lib/glsl/sampling.kernel(body, outputs, samplers, ...)` is
  the compatibility wrapper.
* Multi-pass nodes: `lab_gpu.scratch(w, h, role, fmt)` gives cached per-thread textures (use a
  different `role` for every texture that is alive at once); call `kernel` repeatedly, later
  dispatches see earlier results (back-to-back dispatches need no extra synchronisation).
  `sampling.gaussian_blur`, `distance.gpu_edt_sq` and `reduce.*` are such composites.
* Hand-written shaders (several kernels, image load/store, shared memory): build a
  `lab_gpu.create_info(local_size)`, add resources, and compile with
  `lab_gpu.compile_shader(info, source, "name")`; cache with `lab_gpu.get_shader(key, factory)`;
  bind with `set_uniform` / `set_if_present` (which ignores uniforms the compiler dropped) and
  `dispatch_grid`.

### Errors

`compile_shader` (used by `kernel`) raises `LabShaderError` with the compiler's message, so the
node fails visibly (info message, default outputs). The same text is recorded in `lib/errors.py`;
`tests/lab/harness.py` (`render`, hence `render_node` / `render_generator`) fails the test for any
recorded error unless it is called with `allow_errors=True` (then see `H.LAST_ERRORS`).

## GLSL / numpy twin convention

Each `lib/glsl/<name>.py` defines `SOURCE` (GLSL string), `DEPS` (module names) and, if useful,
documentation of its API. Its numpy twin is `lib/np_<name>.py` with the same function names and
the same float32 operation order, so the CPU and GPU agree as tightly as the float maths allows.
Every GLSL function that a twin implements is checked in `tests/lab/test_lib.py` with a probe
node that evaluates both. Per node, tests assert CPU vs GPU agreement with a stated tolerance,
node-specific behaviour, and robustness (unlinked inputs, odd sizes). Nodes keep module-local GLSL in
a string and pass it as `functions=` (do not register fake modules).

## Metal gotchas

Blender translates GLSL to Metal Shading Language. What has bitten so far:

* **No forward declarations.** Define each function before its first use (a declaration without a
  body is rejected). That is also why `functions` come after libs.
* **`kernel` is a reserved word**, even inside a GLSL comment or a string that ends up in the
  source. (The Python function is called `kernel`; the GLSL must never contain the word.)
* **Read-write images need a fence.** When a shader stores to an image and reads it back (or other
  invocations read it), call `imageFence(img)` after the stores and `barrier()` (on Metal a
  threadgroup barrier with texture memory scope). `memoryBarrierImage()` does not exist in MSL.
  See `addons/pixel_sort_node.py` and `nodes/line_sort.py`.
* **Fast-math.** Metal contracts `a * b + c` to `fma` and divides approximately. Where a value must
  match numpy bit for bit (sort keys, thresholds, tie-breaks), use `glsl/exact.py`
  (`lab_rnd32`, `lab_div_exact`; needs the `lab_zero` uniform, which `kernel` always sets).
* **`texelFetch` on single-channel formats** (R16F / R32F) returns `(r, 0, 0, 1)`, not grey. Use
  `.r`, or declare the input kind `"float"` / let the accessor grey it (see above); do not read
  `.g` / `.b`.
* **Back-to-back dispatches are fine.** A second `dispatch` reading the first one's output texture
  is correctly ordered; no manual barrier or readback is needed.
* Integer maths is exact and identical to numpy `uint32`; floating-point `exp` / `sin` / `pow` differ
  in the last bits between GPU and numpy, so compare with a tolerance, and avoid feeding such
  values into discontinuities (floor, comparisons) where the two backends must agree.
