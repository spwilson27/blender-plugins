# Python-evaluated Compositor Nodes — Design

Target: Blender v5.2.2, branch `compositor-python-nodes` of
[spwilson27/blender](https://github.com/spwilson27/blender/tree/compositor-python-nodes).

## Goal

Let a Python add-on register a `bpy.types.CompositorNode` subclass whose output is
computed by Python, live inside the compositor graph, with two evaluation paths:

| Path | Python method | Data handed to Python | Used when |
|------|---------------|-----------------------|-----------|
| CPU  | `evaluate_cpu(self, inputs, outputs)` | zero-copy buffer objects (numpy via `np.asarray`) | CPU compositing, or GPU compositing when the node has no `evaluate_gpu` (GPU⇄CPU round trip) |
| GPU  | `evaluate_gpu(self, inputs, outputs)` | `gpu.types.GPUTexture` wrapping the compositor's own textures | GPU compositing (viewport, backdrop/render with device = GPU) |

Both methods are optional. Today a Python `CompositorNode` has a null
`get_compositor_operation`, which `NodeGroupOperation::get_node_operation`
dereferences unconditionally (`compositor/intern/node_group_operation.cc:~167`).
This patch fixes that as a side effect: every Python compositor node gets an
operation, and nodes without evaluate methods output defaults.

## Python API

```python
class PixelSortNode(bpy.types.CompositorNode):
    bl_idname = "CompositorNodePixelSort"
    bl_label = "Pixel Sort"

    threshold: bpy.props.FloatProperty(default=0.5)

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CompositorNodeTree'

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Image")
        self.inputs.new('NodeSocketFloat', "Mix")
        self.outputs.new('NodeSocketColor', "Image")

    def evaluate_cpu(self, inputs, outputs):
        img = np.asarray(inputs["Image"])     # (H, W, 4) float32, read-only, row 0 = bottom
        out = np.asarray(outputs["Image"])    # (H, W, 4) float32, writable, same domain
        out[:] = img

    def evaluate_gpu(self, inputs, outputs):
        src = inputs["Image"]                 # gpu.types.GPUTexture (sample / imageLoad)
        dst = outputs["Image"]                # gpu.types.GPUTexture (bind as image, write)
        shader.image("dst", dst); shader.uniform_sampler("src", src)
        gpu.compute.dispatch(shader, ceil(w/16), ceil(h/16), 1)
```

Rules:
- `inputs` / `outputs` are dicts keyed by **socket identifier**.
- An input that is a **single value** (unlinked socket, or a linked constant) is
  passed as a Python scalar: `float`, `int`, `bool`, or a `tuple` for vectors and
  colours. Image inputs are passed as buffers (CPU) or textures (GPU).
- Outputs are always images allocated over the operation's compute domain
  (`NodeOperation::compute_domain()`, i.e. the domain of the highest-priority
  image input; 1x1 if all inputs are single values). Unused outputs
  (`!should_compute()`) are omitted from the dict.
- The CPU buffer layout is the compositor's: contiguous row-major
  `(height, width, channels)`, row 0 at the **bottom**, scene-linear,
  colours premultiplied. Float types use format `'f'`, Int types `'i'`, Bool `'?'`.
  Channels: Float 1 (shape is `(H, W)`), Float2 2, Float3 3, Float4 / Color 4, Int 1, Int2 2.
- Unsupported socket types (Menu, String, …) are passed as `None`.
- The method must be **pure**: it runs on the compositor thread with the GIL
  held, possibly not the main thread. Reading `self.<prop>` is fine. Do not write
  RNA, call operators, or touch `bpy.context`.
- If the method raises: the traceback is printed, outputs fall back to the
  defaults, and the message is reported via `Context::set_info_message`.
- `evaluate_gpu` runs with the compositor's GPU context active. It may create
  and cache shaders, bind the given textures, and dispatch compute. It must
  leave no shader bound. The C++ side unbinds and issues a memory barrier
  afterwards anyway.
- Buffers and textures handed to Python stay memory-safe if retained (they hold
  a reference), but their contents are undefined after the call returns.

## C++ architecture

The compositor and nodes modules must not depend on Python, so the dependency is
inverted through a small callback table.

### 1. `source/blender/nodes/NOD_composite_python.hh` (new)

```cpp
namespace blender::nodes::compositor_python {

enum class EvalMode { CPU, GPU };

struct SocketValue {
  StringRefNull identifier;
  compositor::ResultType type;
  enum class Kind { None, Single, Buffer, Texture } kind = Kind::None;
  /* Kind::Single */
  float4 single_float;       /* floats/vectors/colours (unused channels 0) */
  int2   single_int;         /* int, int2, bool, menu */
  /* Kind::Buffer */
  void *data = nullptr; int2 size; int channels; bool writable;
  ImplicitSharingPtr<> sharing;  /* keeps `data` alive when retained by Python; may be null */
  /* Kind::Texture */
  gpu::Texture *texture = nullptr;
};

struct Callbacks {
  /* Does the Python class of this node type define evaluate_cpu / evaluate_gpu? */
  bool (*has_method)(const bke::bNodeType &ntype, EvalMode mode);
  /* Call the method. Returns false and fills r_error when Python raised. */
  bool (*evaluate)(const bNode &node, EvalMode mode,
                   Span<SocketValue> inputs, Span<SocketValue> outputs,
                   std::string &r_error);
};

void set_callbacks(const Callbacks *callbacks);  /* nullptr to clear */
const Callbacks *get_callbacks();

/* Fill in compositor-specific fields of a Python-registered compositor node type. */
void node_type_init(bke::bNodeType &ntype);
}
```

### 2. `source/blender/nodes/composite/node_composite_python.cc` (new)

- `node_type_init` sets `ntype.get_compositor_operation = get_compositor_operation`.
- `class PythonNodeOperation : public compositor::NodeOperation`, `execute()`:
  1. `callbacks = get_callbacks()`. If there are no callbacks, or the class has
     neither method, call `allocate_default_remaining_outputs()` and return.
  2. Choose the mode:
     - `use_gpu() && has(GPU)` → GPU.
     - `has(CPU)` → CPU (with a round trip if `use_gpu()`).
     - Otherwise (only GPU in a CPU context) → `set_info_message("<node>: requires GPU compositing")` and defaults.
  3. **CPU mode.**
     - Inputs: for each input result, a single value becomes `Kind::Single`.
       Otherwise, if `use_gpu()`, use `download_to_cpu()` (keep that temporary
       `Result` alive until the end of execute, then `release()`); else use the
       result directly.
     - Fill `data` / `size` / `channels` from `cpu_data()` and `domain().data_size`,
       and `sharing` from `sharing_info()`.
     - If `sharing` is null (external data such as render passes), copy into
       a new `ImplicitSharedValue`-style owned buffer so Python cannot dangle.
       Use `implicit_sharing::info_for_mem_free` with `MEM_dupallocN`, or an
       equivalent.
     - Outputs: create a temporary CPU `Result` per computed output via
       `context().create_result(type)` and
       `allocate_texture(domain, false, ResultStorageType::CPU)`. Hand it over as
       `Kind::Buffer`, writable.
     - Call `evaluate`. On success:
       - `use_gpu()`: `output.share_data(tmp.upload_to_gpu(true))` and release.
       - CPU: `output.share_data(tmp)`.
       - Release the temporaries.
     - On failure: release, then defaults, and `set_info_message`.
     - Before calling, write zeros to the output buffers so an unwritten output is
       black, not garbage.
  4. **GPU mode.**
     - Inputs: single values become `Kind::Single`; images become
       `Kind::Texture` (`gpu_texture()`).
     - Outputs: `allocate_texture(domain)`, then `Kind::Texture`. Clear them with
       `GPU_texture_clear` to zero.
     - Call `evaluate`. Afterwards call `GPU_shader_unbind()`,
       `GPU_texture_unbind_all()`, `GPU_texture_image_unbind_all()` and
       `GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS
       | GPU_BARRIER_TEXTURE_UPDATE)`.
     - On failure: defaults and info message.
     - Outputs that were allocated but must be defaulted: free them first,
       or simply leave them zero-cleared. Zero-cleared is acceptable.
- Register the file in `source/blender/nodes/composite/CMakeLists.txt`, and
  declare the header in the nodes CMake.

### 3. `makesrna/intern/rna_nodetree.cc`

In `rna_CompositorNode_register`, after `rna_Node_register_base`, call
`blender::nodes::compositor_python::node_type_init(*nt)` before
`bke::node_register_type`. Add the include.

Only `CompositorNode`'s registration is touched. Custom groups keep their own path.

### 4. `source/blender/python/intern/bpy_compositor_node.cc` (+ `.hh`) (new)

- `void BPY_compositor_node_callbacks_register()` and `..._unregister()`. Call
  register from `BPY_python_start` after the `bpy` and `gpu` modules are
  initialised. Call unregister in `BPY_python_end` before finalize. Grep
  `bpy_interface.cc` for the right spots.
- `has_method`:
  - `PyGILState_Ensure`.
  - `cls = (PyObject *)ntype.rna_ext.data`.
  - `PyObject_HasAttrString(cls, mode == CPU ? "evaluate_cpu" : "evaluate_gpu")`.
  - Release.
  - If `rna_ext.data` is null, return false.
- `evaluate`:
  - `PyGILState_Ensure`.
  - Build `PointerRNA ptr = RNA_pointer_create_discrete(&node.owner_tree().id,
    node.typeinfo->rna_ext.srna, &node)` (const_cast), then
    `self = pyrna_struct_CreatePyObject(&ptr)`.
  - Build the `inputs` and `outputs` dicts:
    - `Kind::Single`: `PyFloat`, `PyLong`, `PyBool`, or a tuple (by type).
    - `Kind::Buffer`: a new `CompositorBuffer` object (below).
    - `Kind::Texture`: `BPyGPUTexture_CreatePyObject(texture, true)`. Verify that
      `shared_reference=true` increments the texture ref count so that dealloc
      frees only the ref, not the pool's texture. Read `gpu_py_texture.cc:988`.
    - `Kind::None`: `None`.
  - `PyObject_CallMethod(self, name, "OO", inputs, outputs)`.
  - On exception, capture the message and print the traceback (`PyErr_Print`).
  - Decref and `PyGILState_Release`.
- `CompositorBuffer`: a minimal private `PyTypeObject` (no need to add it to the
  `bpy.types` namespace; name it `bpy_compositor.Buffer` in `tp_name`).
  - Fields: `void *data`, `Py_ssize_t shape[3]`, `int ndim`, `char format[2]`,
    `Py_ssize_t itemsize`, `bool readonly`, and `const ImplicitSharingInfo *sharing`
    (a user is added on creation and removed in dealloc).
  - Implements `bf_getbuffer`, filling `Py_buffer` with C-contiguous strides,
    and rejects `PyBUF_WRITABLE` when readonly.
  - Exposes a read-only `shape` getter.
  - `ndim` is 2 when channels == 1, else 3.
- Add the file to `source/blender/python/intern/CMakeLists.txt`. Include paths
  for nodes and compositor (`../../nodes`, `../../compositor`) may be needed;
  follow how other intern files include `NOD_*` headers.

### 5. Docs, template, tests

- `scripts/templates_py/custom_compositor_node.py`: a minimal template node that
  implements both methods (CPU: numpy invert; GPU: compute-shader invert).
- Tests live in this repository's `tests/` directory.

## F1–F3: generator domain, evaluation context, single value outputs

Three framework additions (see `plan-stateless-nodes.md` section 0). Tests live in
`tests/framework/`.

### F1. Generator domain

The compute domain is still the domain of the image input with the highest priority. If the node
has no such input (no sockets, or all inputs are unlinked / single values), the outputs use the
compositing domain (render resolution) instead of 1x1. Implemented as
`PythonNodeOperation::compute_domain()`, which applies the same criteria as
`Operation::compute_domain` to detect that case. A node with an image input is unchanged.

### F2. Evaluation context

The `context` object is passed as a 4th argument if the 4th positional parameter of
`evaluate_cpu` / `evaluate_gpu` (counting `self`) is named `context`, is required (has no default),
or the function takes `*args`. So `(self, inputs, outputs, context)`, `(..., context=None)`,
`(..., ctx)` and `(self, inputs, outputs, *args)` receive it. A 4th parameter that has a default and
another name (the `_orig=_orig` idiom) is not filled and keeps its default, and keyword-only
parameters don't count. Methods with 3 parameters are called exactly as before.

`context` is a `types.SimpleNamespace`:

| attribute | type | value |
|---|---|---|
| `frame` | float | scene frame including the subframe (`cfra + subframe`) |
| `fps` | float | `frs_sec / frs_sec_base` |
| `time` | float | seconds, `frame / fps` (not offset by the start frame) |
| `size` | tuple `(w, h)` | size of the compute domain in pixels |
| `use_gpu` | bool | true if the compositor evaluates on the GPU (both methods get the same value for a given render) |
| `kind` | str | the evaluation kind, see below: `'RENDER'`, `'BACKDROP'`, `'VIEWPORT'` or `'SEQUENCER'` |
| `is_animation_playing` | bool | true while the animation plays in the UI, see below |
| `frame_start`, `frame_end` | int | the render frame range (`scene.frame_start` / `frame_end`) of the scene the compositor context evaluates |
| `report(message, level='INFO')` | function | report a non-fatal message, see F5 |

`kind` is derived in C++ from the `compositor::Context` subclass (`get_evaluation_kind()`):

| `kind` | evaluation | `is_animation_playing` |
|---|---|---|
| `'RENDER'` | the render pipeline: F12, `-f` / `-a`, `bpy.ops.render.render()` | always `False` |
| `'BACKDROP'` | the interactive compositor job of the node editor backdrop (the render context with the interactive flag) | the state when the job was scheduled (playback, not scrubbing) |
| `'VIEWPORT'` | the viewport compositor draw engine (3D view, `use_compositor`) | the state at draw time (playback, not scrubbing) |
| `'SEQUENCER'` | the compositor modifier of a sequencer strip | always `False` |

These streams are interleaved (the backdrop, viewport and render can all evaluate the same node),
so stateful nodes should include `kind` in their state key.

The values are gathered in C++ (`EvalInfo` in `NOD_composite_python.hh`), so the method never has
to read `bpy.context` off the main thread.

### F5. Non-fatal messages: `context.report`

A method that raises gets default outputs. To report a problem while still producing output, call
`context.report(message, level='INFO')` with `level` `'INFO'` or `'WARNING'` (other levels raise
`ValueError`). After the evaluation, C++ (`PythonNodeOperation::forward_messages`) forwards the
messages:

* as node warnings (Info / Warning) shown on the node in the node editor, through the compositor
  `nodes_evaluation_log()` like the Warning node, when the context has a log (the node editor
  backdrop and render; the viewport and sequencer have none);
* as the info message of the compositor (`Context::set_info_message`, prefixed with the node name),
  the most severe and latest message only; an exception still overrides it.

At most 64 messages are kept per evaluation. `report` is bound to a capsule that owns a per
evaluation collector; the collector is closed when the method returns, so calling a retained
`report` afterwards raises `RuntimeError`. Warnings are not readable from Python, so
`tests/framework/test_report.py` checks the API (CPU and GPU, output still produced, invalid
arguments, use after the call).

### F3. Single value outputs

A class attribute `single_value_outputs` (any iterable of output socket identifiers, for example
a set) makes those outputs single values instead of images:

```python
class MyStats(bpy.types.CompositorNode):
    single_value_outputs = {"Mean", "Palette"}

    def evaluate_cpu(self, inputs, outputs):       # also in evaluate_gpu
        mean = outputs["Mean"]                      # bpy_compositor.Buffer, shape (1,)
        np.asarray(mean)[0] = 0.5
        np.asarray(outputs["Palette"])[:] = (1.0, 0.0, 0.0, 1.0)   # shape (4,)
```

- In both CPU and GPU mode Python gets a writable `bpy_compositor.Buffer` of shape `(channels,)`
  (1 dimension; Float 1, Float2 2, Float3 3, Float4/Color 4, Int 1, Int2 2, Bool 1) instead of an
  image or a `GPUTexture`. The format is `f` (float/color), `i` (int) or `?` (bool).
- The buffer is zero initialized. After a successful call it is stored as the single value of the
  output (Float, Float2, Float3, Float4, Color, Int, Int2, Bool).
- If the method raises, these outputs have the default value like all the others.
- Outputs that nothing consumes are omitted from `outputs` as usual. Identifiers that do not
  exist are ignored.
- A node with only single value outputs and no image inputs is a valid use of F1 (the domain is
  not used by those outputs).

## Threading / GIL notes

- Blender releases the GIL after startup (`bpy_interface.cc:~648`), so
  `PyGILState_Ensure` from a render or job thread works. Drivers do the same.
- If the compositor runs on the main thread from inside Python (for example a
  script calling `bpy.ops.render.render()`), the GIL is already held by that
  thread. `PyGILState_Ensure` is re-entrant, so that is fine.
- Deadlock is possible only if some thread holds the GIL while blocking on a
  compositor running on another thread. Test F12, background `-b -f`, the
  backdrop and the viewport compositor.

## Known limitations (v1)

- No caching: the method runs on every compositor evaluation.
- No multi-function / pixel-node fusion. The node is always a standalone operation.
- Node previews work automatically (`NodeOperation::log_data`).
- Sockets come only from `init()` (`self.inputs.new`). There is no `declare` for
  Python nodes.
