# Blender Plugins

Blender add-ons built on **Python compositor nodes**: compositor nodes whose pixel processing is
written in Python, on the CPU with numpy or on the GPU with compute shaders.

> **Requires a custom Blender build.** Stock Blender can't evaluate Python compositor nodes.
> Build the `compositor-python-nodes` branch of
> [spwilson27/blender](https://github.com/spwilson27/blender/tree/compositor-python-nodes)
> (Blender 5.2.2 plus the feature). See [docs/python-compositor-nodes.md](docs/python-compositor-nodes.md)
> for the design.

## Add-ons

### Pixel Sort (`addons/pixel_sort_node.py`)

A **Pixel Sort** compositor node (Add ▸ Filter ▸ Pixel Sort). It finds runs of pixels whose
threshold key lies between *Lower* and *Upper*, then sorts each run along rows or columns.

- **Threshold by / Sort by:** luminance, hue, saturation, value, red, green or blue.
- **Options:** vertical, descending, invert mask.
- **Mask input:** a pixel is sorted only if it passes the threshold test (*Invert Mask* applies
  to that test only) and the mask is above 0.5; runs are split wherever either test fails. An
  unlinked Mask of 0.5 or less returns the input unchanged, above 0.5 has no effect. Nodes saved
  before the Mask socket existed keep working. With the (benchmark-only) serial GPU
  implementation a masked node falls back to the parallel shader.
- **CPU path:** vectorised numpy.
- **GPU path:** parallel bitonic sort in a compute shader, with no GPU↔CPU round trip. Its output
  is bit-identical to the CPU path at full precision.

Install it with *Preferences ▸ Add-ons ▸ Install from Disk*, choosing `addons/pixel_sort_node.py`.

### Compositor Lab (`addons/compositor_lab/`)

A package of 21 experimental nodes (Add ▸ Lab ▸ Filter / Generate / Utility), each with a numpy CPU
path and a GPU compute path, built on a shared library (`lib/`: node base class, GPU kernel
builder, GLSL snippets with numpy twins for hashing, noise, colour, blend modes, sampling,
distance transforms and more). Needs the generator domain and evaluation context features of the
custom build (see [docs/plan-stateless-nodes.md](docs/plan-stateless-nodes.md)). One broken node
module is logged and does not stop the others from registering. How to write a node, the kernel
builder and the GLSL / Metal gotchas: [addons/compositor_lab/lib/README.md](addons/compositor_lab/lib/README.md).

Install: `python3 tools/build_zip.py` writes `dist/compositor_lab.zip`; use *Preferences ▸
Add-ons ▸ Install from Disk* on that zip. `tools/gallery.py` renders every node with its default
settings into `dist/gallery.png` (see below).

Node details (sockets, properties, algorithms, tolerances) are in the linked files under
[docs/nodes/](docs/nodes/).

**Filter**

| Node | What it does | Details |
|---|---|---|
| Blend Modes+ | 27 Photoshop-style blend modes on premultiplied A / B with Fac, including OKLCh Hue / Saturation / Color / Luminosity | this README |
| Posterize+ | Reduce colours to a few levels, with gamma, lightness-only mode and dithering | [filters-a](docs/nodes/filters-a.md#posterize-compositornodelabposterize) |
| Gradient Map | Map luminance or a channel through a preset or custom (up to 6 stops) gradient | [filters-a](docs/nodes/filters-a.md#gradient-map-compositornodelabgradientmap) |
| Halftone | Print halftone: CMYK or mono dots, lines or cross-hatch | [filters-a](docs/nodes/filters-a.md#halftone-compositornodelabhalftone) |
| Glitch | RGB split, block displacement, scanline jitter and bit-crush, animated | [filters-a](docs/nodes/filters-a.md#glitch-compositornodelabglitch) |
| Kuwahara | Anisotropic Kuwahara: painterly smoothing that follows edges | [filters-b](docs/nodes/filters-b.md#kuwahara-compositornodelabkuwahara) |
| Displace / Glass | Displace by a map (offset or glass refraction) with edge modes and dispersion | [filters-b](docs/nodes/filters-b.md#displace--glass-compositornodelabdisplace) |
| Liquify | Twirl and pinch / bulge warps around a point | [filters-b](docs/nodes/filters-b.md#liquify-compositornodelabliquify) |
| Edge Stylise | XDoG ink lines, Sobel edges or an outline around the alpha | [filters-b](docs/nodes/filters-b.md#edge-stylise-compositornodelabedgestylise) |

**Generate**

| Node | What it does | Details |
|---|---|---|
| Noise | Value, Perlin, simplex, Worley (F1, F2-F1); fBm, ridged, domain warp; animated via Phase + time * Speed; Value and per-channel Color outputs | this README |
| Voronoi / Mosaic | Voronoi cells, distance fields, edges and stained-glass mosaic | [generators](docs/nodes/generators.md#voronoi--mosaic-compositornodelabvoronoi) |
| Pattern | Anti-aliased stripes, checker, dots, hex grid, truchet, moire, rings | [generators](docs/nodes/generators.md#pattern-compositornodelabpattern) |
| Flow Field | Line integral convolution and streamlines along a vector field | [generators](docs/nodes/generators.md#flow-field-compositornodelabflowfield) |

**Utility**

| Node | What it does | Details |
|---|---|---|
| Expression | Per-pixel expression (safe Python-syntax subset) on CPU or GPU | [utility-a](docs/nodes/utility-a.md#expression-compositornodelabexpression) |
| Image Statistics | Min, max, mean, luminance mean, standard deviation, percentile (single-value outputs) | [utility-a](docs/nodes/utility-a.md#image-statistics-compositornodelabimagestatistics) |
| Auto Levels | Stretch contrast between low / high percentiles | [utility-a](docs/nodes/utility-a.md#auto-levels-compositornodelabautolevels) |
| Palette Extract | Up to 8 dominant colours with k-means (single-value outputs) | [utility-a](docs/nodes/utility-a.md#palette-extract-compositornodelabpaletteextract) |
| Seamless Tile | Make an image tile seamlessly (offset cross-fade or mirror) | [utility-b](docs/nodes/utility-b.md#seamless-tile) |
| Mask Tools | Threshold, grow / shrink, feather, outline, invert with exact distances | [utility-b](docs/nodes/utility-b.md#mask-tools) |
| Pixel Shuffle | Seeded shuffle of pixels in blocks, swapped pairs or whole blocks | [utility-b](docs/nodes/utility-b.md#pixel-shuffle) |
| Line Sort | Sort whole rows or columns by a statistic (mean luminance, hue, variance...) | [utility-b](docs/nodes/utility-b.md#line-sort) |

Tests: `tests/lab/` (auto-run as suites `lab_*`), for example
`tests/run_remote.sh HOST /path/to/Blender.app blender-test-lab 'lab_*'`. A node that raises
during evaluation (including a GPU shader that does not compile) fails the test that rendered it.

Gallery: `tools/gallery.py` (run it with `Blender -b --factory-startup --python tools/gallery.py`;
it needs a GPU device) renders every Lab node and Pixel Sort at 480x270 with default settings on a
shared test image, generators standalone, into `dist/gallery/*.png` and a labelled contact sheet
`dist/gallery.png`.

## Writing your own node

[`templates/custom_compositor_node.py`](templates/custom_compositor_node.py) is a minimal node that
implements both evaluation paths; for nodes in the Lab package see
[addons/compositor_lab/lib/README.md](addons/compositor_lab/lib/README.md). The short version:

```python
class MyNode(bpy.types.CompositorNode):
    bl_idname = "CompositorNodeMyNode"
    bl_label = "My Node"

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Image")
        self.outputs.new('NodeSocketColor', "Image")

    def evaluate_cpu(self, inputs, outputs):
        # inputs/outputs: dicts by socket identifier. Images are zero-copy buffers of shape
        # (height, width, channels), row 0 at the bottom; unlinked inputs are plain values.
        np.asarray(outputs["Image"])[:] = np.asarray(inputs["Image"])

    def evaluate_gpu(self, inputs, outputs):
        # Images are gpu.types.GPUTexture: bind and dispatch a compute shader.
        ...
```

## Tests

```bash
tests/run_tests.sh /path/to/Blender
```

The GUI tests open Blender windows. To keep them off your own screen, run the suite on another
Mac over SSH. This needs key-based login, and a user logged in to that Mac's desktop:

```bash
tests/run_remote.sh other-mac.local /path/to/Blender.app
```

It syncs the app bundle and the working tree (including uncommitted changes), then runs the suite
there.

The suite covers:
- installing into a clean config
- the reference implementation
- CPU and GPU output compared exactly against numpy
- options re-running the compositor (GUI)
- a thread-safety stress test (GUI)

`tools/bench_gpu.py` benchmarks the CPU and GPU paths.

## License

GPL-2.0-or-later, see [LICENSE](LICENSE).
