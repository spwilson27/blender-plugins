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

The suite covers:
- installing into a clean config
- the reference implementation
- CPU and GPU output compared exactly against numpy
- options re-running the compositor (GUI)
- a thread-safety stress test (GUI)

`tools/bench_gpu.py` benchmarks the CPU and GPU paths.

## License

GPL-2.0-or-later, see [LICENSE](LICENSE).
