# Plan: Compositor Lab, the stateless nodes

Goal: a collection of experimental compositor nodes (filters, generators, utilities). They're
built on a shared library, and each node has a numpy CPU path and a GPU compute path. Stateful
nodes (feedback across frames) are a later round with its own design.

## 0. Framework changes (Blender fork, `compositor-python-nodes`)

The node ideas need three things the framework lacks today:

| # | Change | Why | Where |
|---|--------|-----|-------|
| F1 | **Generator domain.** When a Python node has no image inputs, its outputs use the compositing domain (render size) instead of 1×1. | Noise and patterns have no image to take a size from. | `PythonNodeOperation::compute_domain()` override |
| F2 | **Evaluation context.** If `evaluate_*` accepts a 4th parameter, pass a `context` object: `frame` (float), `fps`, `time` (seconds), `size` (w, h), `use_gpu`. Existing 3-argument methods are unchanged. | Animated noise and glitch need time, but reading `bpy.context` off the main thread is unsafe. | `bpy_compositor_node.cc`, which checks the method's `__code__.co_argcount` |
| F3 | **Single-value outputs.** A class attribute `single_value_outputs = {"Mean", ...}` makes those outputs single values. Python receives a writable buffer of shape `(channels,)` (CPU and GPU), and C++ stores it with `set_single_value`. | Image Statistics and Palette Extract output numbers and colours that drive other nodes. | `node_composite_python.cc` |

Each change gets tests in the plugin repo, and the contract in `NOD_composite_python.hh` is
updated. The build is synced to the mini once, and agents reuse it.

## 1. Package layout: one add-on, a shared library

Pixel Sort stays a self-contained single file. The new nodes ship as one package add-on,
`addons/compositor_lab/`, installed as a zip:

```
compositor_lab/
  __init__.py          bl_info; discovers nodes/*.py, registers each defensively (one broken
                       module logs an error and doesn't block the others); "Lab" Add-menu
                       categories
  lib/
    node.py            LabNode base: declarative SOCKETS/PROPS spec -> init(), draw_buttons,
                       property update tagging; evaluate dispatch with helpers to read
                       inputs as float / colour / array / texture; output helpers;
                       accepts the F2 context
    gpu.py             shader cache keyed by source + format; CreateInfo builder; pointwise
                       kernel helper (GLSL body -> full compute shader over the output);
                       dispatch helper; sampler/image binding; exact-math helpers
    glsl/              GLSL snippets as Python strings: hash/rng, noise (value, Perlin,
                       simplex, Worley, fBm, ridged, domain warp, curl), colour (sRGB<->linear,
                       HSV, OKLab/OKLCh), blend modes, coordinate transforms
    np_*.py            numpy twins of the same snippets (noise, colour, blend) so both paths
                       share one specification
    expr.py            safe expression compiler: Python AST subset -> numpy callable and GLSL
                       source (for the Expression node)
  nodes/
    <one file per node>  each exports NODE_CLASSES and a menu category
```

Rule: each node's math lives once per backend, in `lib/` when more than one node uses it.

## 2. Nodes (stateless)

| Category | Nodes |
|---|---|
| Filters | Blend Modes+, Posterize, Gradient Map, Halftone, Glitch, Kuwahara (anisotropic), Displace/Glass, Liquify (twirl/pinch/bulge), Edge Stylise (XDoG, outline) |
| Generators | Noise, Voronoi/Mosaic, Pattern (stripes, checker, hex, truchet, moiré), Flow Field (LIC/streamlines, integrated within one evaluation) |
| Utility | Expression, Image Statistics (F3), Auto Levels, Palette Extract (F3), Seamless Tile, Mask Tools (grow/shrink/feather via distance transform, range threshold), Pixel Shuffle, Line Sort |

## 3. Testing strategy

- `tests/lab/harness.py` gives each node test what it needs: build a tree, render with the CPU or
  GPU device, read back float EXR, a reference image generator, and a helper that compares CPU
  and GPU results.
- Every node gets three kinds of check:
  1. **Cross-backend:** CPU and GPU output agree. The tolerance is set per node, and it must be
     justified when it isn't tight.
  2. **Behaviour:** node-specific properties, such as posterize producing ≤ N levels per channel,
     tile output wrapping seamlessly, statistics matching numpy, expressions matching numpy
     evaluation, and generators being deterministic for a seed and changing with the seed or time.
  3. **Robustness:** unlinked inputs, odd sizes, and no crash in either mode.
- Framework tests cover F1–F3 directly, plus the install test for the package zip.
- Everything runs on the mini (`tests/run_remote.sh`). Each agent uses its own remote directory
  and runs only its own suites (`run_tests.sh BLENDER [pattern]`).

## 4. Execution

| Wave | Who | Work |
|---|---|---|
| 0 | agent → review, build | F1–F3 in the fork; framework tests; build; sync app to mini; push |
| 1 | agent → review | Package skeleton, `lib/` (node base, gpu, glsl/np colour+noise+blend), harness, run-tests filtering, zip build + install test, plus **Noise** and **Blend Modes+** as reference nodes |
| 2 | 4 agents in parallel, separate files | A: Posterize, Gradient Map, Halftone, Glitch · B: Kuwahara, Displace/Glass, Liquify, Edge Stylise · C: Voronoi/Mosaic, Pattern, Flow Field · D: Expression, Image Statistics, Auto Levels, Palette Extract, Seamless Tile, Mask Tools, Pixel Shuffle, Line Sort |
| 3 | me | Full-suite run on the mini, review, README and docs, commit and push |

Then a design → implement → test round for the stateful nodes: Reaction-Diffusion, Cellular
Automata, Trails/Feedback and Slit-scan.
