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

A package of experimental nodes (Add ▸ Lab ▸ Filter / Generate / Utility), each with a numpy CPU
path and a GPU compute path, built on a shared library (`lib/`: node base class, GPU kernel
helper, GLSL snippets with numpy twins for hashing, noise, colour spaces and blend modes). Needs
the generator domain and evaluation context features of the custom build (see
[docs/plan-stateless-nodes.md](docs/plan-stateless-nodes.md)). One broken node module is logged and
does not stop the others from registering.

Install: `python3 tools/build_zip.py` writes `dist/compositor_lab.zip`; use *Preferences ▸
Add-ons ▸ Install from Disk* on that zip.

Nodes so far:
- **Noise** (Generate): value, Perlin, simplex, Worley F1 and F2-F1; fBm (octaves, lacunarity,
  gain), ridged, domain warp, scale, offset, seed. The third dimension is `Phase + time * Speed`,
  so it animates. Outputs Value and a decorrelated per-channel Color.
- **Blend Modes+** (Filter): 27 modes on premultiplied inputs A (backdrop) and B (source) with
  Fac: Normal, Multiply, Screen, Overlay, Soft Light (W3C and Pegtop), Hard/Vivid/Linear/Pin
  Light, Hard Mix, Color Dodge/Burn, Linear Dodge/Burn, Subtract, Divide, Difference, Exclusion,
  Darken/Lighten, Darker/Lighter Color and OKLCh Hue/Saturation/Color/Luminosity.

Tests: `tests/lab/` (auto-run as suites `lab_*`), for example
`tests/run_remote.sh HOST /path/to/Blender.app blender-test-lab 'lab_*'`.

## Writing your own node

[`templates/custom_compositor_node.py`](templates/custom_compositor_node.py) is a minimal node that
implements both evaluation paths. The short version:

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
