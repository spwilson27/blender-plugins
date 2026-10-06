# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Pixel Sort Node",
    "author": "Sean Wilson",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "Compositor > Add > Filter > Pixel Sort",
    "description": "Pixel Sort compositor node evaluated in Python (CPU numpy / GPU compute)",
    "category": "Compositing",
}

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty

# ---------------------------------------------------------------------------
# Core algorithm (pure numpy, no bpy dependency)
# ---------------------------------------------------------------------------

KEYS = ('LUMA', 'HUE', 'SATURATION', 'VALUE', 'RED', 'GREEN', 'BLUE')


def compute_key(rgb, key):
    """rgb: (..., 3) float array -> (...) float key array."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if key == 'LUMA':
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    if key == 'RED':
        return r
    if key == 'GREEN':
        return g
    if key == 'BLUE':
        return b
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    if key == 'VALUE':
        return mx
    delta = mx - mn
    if key == 'SATURATION':
        return np.where(mx > 0, delta / np.where(mx > 0, mx, 1), 0.0)
    # HUE in [0, 1)
    d = np.where(delta > 0, delta, 1)
    h = np.where(
        mx == r, ((g - b) / d) % 6,
        np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    h = np.where(delta > 0, h / 6.0, 0.0)
    return h % 1.0


def pixel_sort(pixels, mask_key='LUMA', lo=0.25, hi=0.8, sort_key='LUMA',
               vertical=False, reverse=False, invert_mask=False, mask=None):
    """Sort runs of in-threshold pixels.

    pixels: (H, W, C) float array, C >= 3 (row 0 = top or bottom, irrelevant).
    Runs are contiguous pixels along a row (or column if vertical) whose
    mask key lies in [lo, hi]. Each run is sorted by sort_key.
    mask: optional (H, W) bool array (same orientation as pixels); a pixel is
    only sorted where it is True (in addition to the threshold test, which is
    the only one affected by invert_mask). Runs are split where either fails.
    Returns a new array of the same shape.
    """
    img = np.asarray(pixels)
    if vertical:
        img = np.swapaxes(img, 0, 1)
    h, w, c = img.shape
    flat = img.reshape(h * w, c)
    rgb = flat[:, :3]

    mkey = compute_key(rgb, mask_key)
    sel = (mkey >= lo) & (mkey <= hi)
    if invert_mask:
        sel = ~sel
    if mask is not None:
        user = np.asarray(mask, dtype=bool)
        if vertical:
            user = np.swapaxes(user, 0, 1)
        sel &= user.reshape(h * w)
    mask = sel

    idx = np.flatnonzero(mask)
    out = flat.copy()
    if idx.size:
        # A run starts at a masked pixel whose predecessor is unmasked or
        # is the last pixel of the previous row.
        prev = np.empty_like(mask)
        prev[0] = False
        prev[1:] = mask[:-1]
        prev[::w] = False
        label = np.cumsum(mask & ~prev)[idx]

        skey = compute_key(rgb[idx], sort_key)
        if reverse:
            skey = -skey
        # label is non-decreasing over idx, so sorting by (label, key)
        # permutes pixels only within their own run.
        perm = np.lexsort((skey, label))
        out[idx] = flat[idx[perm]]

    out = out.reshape(h, w, c)
    if vertical:
        out = np.swapaxes(out, 0, 1)
    return np.ascontiguousarray(out)

_KEY_ITEMS = [
    ('LUMA', "Luminance", ""),
    ('HUE', "Hue", ""),
    ('SATURATION', "Saturation", ""),
    ('VALUE', "Value", ""),
    ('RED', "Red", ""),
    ('GREEN', "Green", ""),
    ('BLUE', "Blue", ""),
]
_KEY_INDEX = {k[0]: i for i, k in enumerate(_KEY_ITEMS)}

# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

# "parallel": one 256-thread workgroup per line (scan + bitonic sort).
# "serial": legacy one-thread-per-line shell sort (kept for benchmarking only).
_GPU_IMPL = "parallel"

_LOCAL_SIZE = 64
_PAR_LOCAL_SIZE = 256

# One invocation per line (row, or column if vertical).
# The line is first copied from "src" to "dst", then every run of pixels
# whose mask key lies in [lower, upper] is sorted in place in "dst"
# with a shell sort (Knuth gaps) using imageLoad/imageStore. imageFence() is
# required on Metal to read back values this invocation stored (no-op on GLSL).
_GLSL = '''
float rnd32(float x)
{
  /* Opaque int round trip (xor with a uniform zero): the compiler cannot see
   * through it, so the value is rounded to float32 here and Metal's fast-math
   * cannot contract/reassociate the surrounding mul+add into fma(). Needed so
   * keys match numpy bit for bit (ties must stay ties). */
  return intBitsToFloat(floatBitsToInt(x) ^ zero);
}

/* Correctly rounded float32 division (Metal fast-math division is approximate,
 * which would make keys differ from numpy by 1 ulp). Markstein refinement. */
float div_exact(float a, float b)
{
  float y = rnd32(1.0 / b);
  float e = rnd32(fma(-b, y, 1.0));
  y = rnd32(fma(y, e, y));
  float q = rnd32(a * y);
  float r = rnd32(fma(-b, q, a));
  return fma(r, y, q);
}

float key_of(vec3 c, int k)
{
  if (k == 0) {
    /* Round every product separately: Metal fast-math would fuse them into
     * fma(), changing the last bit versus numpy and flipping ties. */
    float a = rnd32(0.2126 * c.r);
    float b = rnd32(0.7152 * c.g);
    float d = rnd32(0.0722 * c.b);
    return rnd32(a + b) + d;
  }
  if (k == 4) {
    return c.r;
  }
  if (k == 5) {
    return c.g;
  }
  if (k == 6) {
    return c.b;
  }
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  if (k == 3) {
    return mx;
  }
  float delta = mx - mn;
  if (k == 2) {
    return (mx > 0.0) ? div_exact(delta, mx) : 0.0;
  }
  /* Hue in [0, 1). */
  if (!(delta > 0.0)) {
    return 0.0;
  }
  float h;
  if (mx == c.r) {
    h = div_exact(c.g - c.b, delta);
    if (h < 0.0) {
      h = h + 6.0; /* numpy float remainder(x, 6) for |x| <= 1. */
    }
  }
  else if (mx == c.g) {
    h = div_exact(c.b - c.r, delta) + 2.0;
  }
  else {
    h = div_exact(c.r - c.g, delta) + 4.0;
  }
  h = div_exact(h, 6.0);
  return (h >= 1.0) ? 0.0 : h;
}

ivec2 pos_of(int line, int i)
{
  return (is_vertical != 0) ? ivec2(line, i) : ivec2(i, line);
}

bool in_mask(vec3 c)
{
  float m = key_of(c, mask_key);
  bool inside = (m >= lower) && (m <= upper);
  return (invert_mask != 0) ? !inside : inside;
}

float sort_value(ivec2 p)
{
  float v = key_of(imageLoad(dst, p).rgb, sort_key);
  return (reverse_order != 0) ? -v : v;
}

void sort_run(int line, int start, int end)
{
  int n = end - start;
  if (n < 2) {
    return;
  }
  int gap = 1;
  while (gap < n / 3) {
    gap = gap * 3 + 1;
  }
  for (; gap >= 1; gap /= 3) {
    for (int i = start + gap; i < end; i++) {
      ivec2 pa = pos_of(line, i);
      vec4 tmp = imageLoad(dst, pa);
      float tk = key_of(tmp.rgb, sort_key);
      if (reverse_order != 0) {
        tk = -tk;
      }
      int j = i;
      while (j - gap >= start) {
        ivec2 pj = pos_of(line, j - gap);
        if (sort_value(pj) <= tk) {
          break;
        }
        imageStore(dst, pos_of(line, j), imageLoad(dst, pj));
        imageFence(dst);
        j -= gap;
      }
      imageStore(dst, pos_of(line, j), tmp);
      imageFence(dst);
    }
  }
}

void main()
{
  int line = int(gl_GlobalInvocationID.x);
  int lines = (is_vertical != 0) ? size_x : size_y;
  int len = (is_vertical != 0) ? size_y : size_x;
  if (line >= lines) {
    return;
  }

  /* Pass 1: copy. */
  for (int i = 0; i < len; i++) {
    ivec2 p = pos_of(line, i);
    imageStore(dst, p, texelFetch(src, p, 0));
  }
  imageFence(dst);

  /* Pass 2: find runs and sort them. */
  int run_start = -1;
  for (int i = 0; i < len; i++) {
    bool m = in_mask(texelFetch(src, pos_of(line, i), 0).rgb);
    if (m) {
      if (run_start < 0) {
        run_start = i;
      }
    }
    else if (run_start >= 0) {
      sort_run(line, run_start, i);
      run_start = -1;
    }
  }
  if (run_start >= 0) {
    sort_run(line, run_start, len);
  }
}
'''

# ---- Parallel implementation ------------------------------------------------
# One workgroup (256 threads) per line. Scratch image (P, lines) RGBA32UI with
#   x = run id (start index of the run), w = scan ping-pong,
#   y = order-preserving sort key bits, z = original lane index.
# Phase 1: per-lane keys/mask/run starts. Phase 2: Hillis-Steele max-scan of x.
# Phase 3: bitonic sort on (x, y, z) -> stable, equals numpy lexsort.
# Phase 4: gather. Synchronisation: imageFence() after stores (Metal needs it
# to make read-write texture stores visible) + barrier() (on Metal this is
# threadgroup_barrier with mem_texture; memoryBarrierImage() does not exist in MSL).
_GLSL_PAR_COMMON = _GLSL.split("ivec2 pos_of")[0]  # key_of()

_GLSL_PAR = _GLSL_PAR_COMMON + '''
ivec2 pos_of(int line, int i)
{
  return (is_vertical != 0) ? ivec2(line, i) : ivec2(i, line);
}

bool in_mask(vec3 c)
{
  float m = key_of(c, mask_key);
  bool inside = (m >= lower) && (m <= upper);
  return (invert_mask != 0) ? !inside : inside;
}

/* User mask: mask_mode 0 = none, 1 = texture (> 0.5 passes), 2 = nothing passes. */
bool user_mask(ivec2 p)
{
  if (mask_mode == 0) {
    return true;
  }
  if (mask_mode == 2) {
    return false;
  }
  return texelFetch(mask_tex, p, 0).r > 0.5;
}

bool pix_mask(int line, int i)
{
  ivec2 p = pos_of(line, i);
  return in_mask(texelFetch(src, p, 0).rgb) && user_mask(p);
}

uint key_bits(vec3 c)
{
  float v = key_of(c, sort_key);
  if (v == 0.0) {
    v = 0.0; /* Normalise -0. */
  }
  uint u = floatBitsToUint(v);
  u = ((u & 0x80000000u) != 0u) ? ~u : (u | 0x80000000u);
  return (reverse_order != 0) ? ~u : u;
}

bool greater3(uvec4 a, uvec4 b)
{
  if (a.x != b.x) {
    return a.x > b.x;
  }
  if (a.y != b.y) {
    return a.y > b.y;
  }
  return a.z > b.z;
}

void sync()
{
  imageFence(scratch);
  barrier();
}

void main()
{
  int line = int(gl_WorkGroupID.y);
  int lane = int(gl_LocalInvocationID.x);
  int len = (is_vertical != 0) ? size_y : size_x;
  int P = padded;

  /* Phase 1. */
  for (int i = lane; i < P; i += 256) {
    uvec4 v;
    if (i < len) {
      vec3 c = texelFetch(src, pos_of(line, i), 0).rgb;
      bool m = pix_mask(line, i);
      uint rs = uint(i);
      if (m && i > 0 && pix_mask(line, i - 1)) {
        rs = 0u;
      }
      v = uvec4(rs, key_bits(c), uint(i), rs);
    }
    else {
      v = uvec4(0xFFFFFFFFu, 0xFFFFFFFFu, uint(i), 0xFFFFFFFFu);
    }
    imageStore(scratch, ivec2(i, line), v);
  }
  sync();

  /* Phase 2: inclusive max-scan of x, ping-pong between x and w. */
  bool from_x = true;
  for (int d = 1; d < P; d <<= 1) {
    for (int i = lane; i < P; i += 256) {
      uvec4 me = imageLoad(scratch, ivec2(i, line));
      uint a = from_x ? me.x : me.w;
      if (i >= d) {
        uvec4 o = imageLoad(scratch, ivec2(i - d, line));
        a = max(a, from_x ? o.x : o.w);
      }
      if (from_x) {
        me.w = a;
      }
      else {
        me.x = a;
      }
      imageStore(scratch, ivec2(i, line), me);
    }
    from_x = !from_x;
    sync();
  }
  if (!from_x) {
    /* Result is in w. */
    for (int i = lane; i < P; i += 256) {
      uvec4 me = imageLoad(scratch, ivec2(i, line));
      me.x = me.w;
      imageStore(scratch, ivec2(i, line), me);
    }
    sync();
  }

  /* Phase 3: bitonic sort. */
  for (int k = 2; k <= P; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = lane; i < P; i += 256) {
        int l = i ^ j;
        if (l > i) {
          uvec4 a = imageLoad(scratch, ivec2(i, line));
          uvec4 b = imageLoad(scratch, ivec2(l, line));
          bool asc = (i & k) == 0;
          if (greater3(a, b) == asc) {
            imageStore(scratch, ivec2(i, line), b);
            imageStore(scratch, ivec2(l, line), a);
          }
        }
      }
      sync();
    }
  }

  /* Phase 4: gather. */
  for (int i = lane; i < len; i += 256) {
    int s = int(imageLoad(scratch, ivec2(i, line)).z);
    imageStore(dst, pos_of(line, i), texelFetch(src, pos_of(line, s), 0));
  }
}
'''

_par_shader_cache = {}
_scratch_cache = {}


def _get_shader_parallel(fmt):
    shader = _par_shader_cache.get(fmt)
    if shader is not None:
        return shader
    import gpu

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "src")
    info.sampler(1, 'FLOAT_2D', "mask_tex")
    info.image(0, fmt, 'FLOAT_2D', "dst", qualifiers={'WRITE'})
    info.image(1, 'RGBA32UI', 'UINT_2D', "scratch", qualifiers={'READ', 'WRITE'})
    for name in ("size_x", "size_y", "padded", "mask_key", "sort_key", "is_vertical",
                 "reverse_order", "invert_mask", "zero", "mask_mode"):
        info.push_constant('INT', name)
    info.push_constant('FLOAT', "lower")
    info.push_constant('FLOAT', "upper")
    info.local_group_size(_PAR_LOCAL_SIZE, 1, 1)
    info.compute_source(_GLSL_PAR)
    shader = gpu.shader.create_from_info(info)
    _par_shader_cache[fmt] = shader
    return shader


def _get_scratch(padded, lines):
    key = (padded, lines)
    tex = _scratch_cache.get(key)
    if tex is None:
        import gpu

        if len(_scratch_cache) >= 3:
            _scratch_cache.clear()
        tex = gpu.types.GPUTexture((padded, lines), format='RGBA32UI')
        _scratch_cache[key] = tex
    return tex


_shader_cache = {}


def _get_shader(fmt):
    """Compile (once per output texture format) the pixel sort compute shader."""
    shader = _shader_cache.get(fmt)
    if shader is not None:
        return shader
    import gpu

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "src")
    info.image(0, fmt, 'FLOAT_2D', "dst", qualifiers={'READ', 'WRITE'})
    for name in ("size_x", "size_y", "mask_key", "sort_key", "is_vertical",
                 "reverse_order", "invert_mask", "zero"):
        info.push_constant('INT', name)
    info.push_constant('FLOAT', "lower")
    info.push_constant('FLOAT', "upper")
    info.local_group_size(_LOCAL_SIZE, 1, 1)
    info.compute_source(_GLSL)
    shader = gpu.shader.create_from_info(info)
    _shader_cache[fmt] = shader
    return shader


def _scalar(value, default):
    """Float socket value: a Python number when unlinked, a buffer when linked."""
    if isinstance(value, (int, float, bool)):
        return float(value)
    try:
        return float(np.asarray(value).mean())
    except Exception:
        return default


def _mask_state(value):
    """Classify the Mask socket value (None if the socket is missing, e.g. nodes
    saved before it existed). Returns ('none'|'zero'|'image', value)."""
    if value is None:
        return 'none', None
    if isinstance(value, (int, float, bool)):
        return ('none' if float(value) > 0.5 else 'zero'), None
    if isinstance(value, (tuple, list)):
        return ('none' if float(value[0]) > 0.5 else 'zero'), None
    return 'image', value


def _mask_to_bool(buf):
    """(H, W) float buffer -> (H, W) bool (channel 0 if the buffer has channels)."""
    m = np.asarray(buf)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0.5


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CompositorNodePixelSort(bpy.types.CompositorNode):
    '''Sort runs of pixels along rows or columns'''
    bl_idname = "CompositorNodePixelSort"
    bl_label = "Pixel Sort"

    mask_key: EnumProperty(name="Threshold By", items=_KEY_ITEMS, default='LUMA')
    sort_key: EnumProperty(name="Sort By", items=_KEY_ITEMS, default='LUMA')
    vertical: BoolProperty(name="Vertical", default=False)
    reverse: BoolProperty(name="Descending", default=False)
    invert_mask: BoolProperty(name="Invert Mask", default=False)

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CompositorNodeTree'

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Image")
        lower = self.inputs.new('NodeSocketFloat', "Lower")
        lower.default_value = 0.25
        upper = self.inputs.new('NodeSocketFloat', "Upper")
        upper.default_value = 0.8
        # Factor subtype = float socket with a 0..1 range (min/max can't be set from Python).
        mask = self.inputs.new('NodeSocketFloatFactor', "Mask")
        mask.default_value = 1.0
        self.outputs.new('NodeSocketColor', "Image")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mask_key", text="Mask")
        layout.prop(self, "sort_key", text="Sort")
        row = layout.row(align=True)
        row.prop(self, "vertical", toggle=True)
        row.prop(self, "reverse", toggle=True)
        layout.prop(self, "invert_mask")

    # -- CPU ---------------------------------------------------------------
    def evaluate_cpu(self, inputs, outputs):
        out = np.asarray(outputs["Image"])  # (H, W, 4) float32, writable
        src = inputs["Image"]
        if isinstance(src, (tuple, list, float, int)):
            # Single value (unlinked / constant): fill the output with it.
            col = np.zeros(4, np.float32)
            vals = np.atleast_1d(np.asarray(src, np.float32))
            col[:min(4, vals.size)] = vals[:4]
            out[...] = col
            return

        img = np.asarray(src)
        # Lower / Upper are Python floats when unlinked. If an image is linked
        # they arrive as buffers; a per-pixel threshold is not supported, so the
        # mean of the buffer is used.
        lo = _scalar(inputs["Lower"], 0.25)
        hi = _scalar(inputs["Upper"], 0.8)
        # Mask: unlinked value <= 0.5 -> copy, > 0.5 -> no masking; linked image ->
        # per-pixel (> 0.5). Missing socket (old files) -> no masking.
        state, mval = _mask_state(inputs.get("Mask"))
        if state == 'zero':
            out[...] = img
            return
        mask = _mask_to_bool(mval) if state == 'image' else None
        out[...] = pixel_sort(
            img, self.mask_key, lo, hi, self.sort_key,
            self.vertical, self.reverse, self.invert_mask, mask)

    # -- GPU ---------------------------------------------------------------
    def _gpu_parallel(self, gpu, src, dst, lo, hi, mask_mode, mask_tex):
        w, h = int(dst.width), int(dst.height)
        length, lines = (h, w) if self.vertical else (w, h)
        padded = 1
        while padded < length:
            padded <<= 1
        shader = _get_shader_parallel(dst.format)
        scratch = _get_scratch(padded, lines)
        shader.uniform_sampler("src", src)
        # Always bound; unused (src as a placeholder) unless mask_mode == 1.
        shader.uniform_sampler("mask_tex", mask_tex if mask_mode == 1 else src)
        shader.uniform_int("mask_mode", mask_mode)
        shader.image("dst", dst)
        shader.image("scratch", scratch)
        shader.uniform_int("size_x", w)
        shader.uniform_int("size_y", h)
        shader.uniform_int("padded", padded)
        shader.uniform_int("mask_key", _KEY_INDEX[self.mask_key])
        shader.uniform_int("sort_key", _KEY_INDEX[self.sort_key])
        shader.uniform_int("is_vertical", int(self.vertical))
        shader.uniform_int("reverse_order", int(self.reverse))
        shader.uniform_int("invert_mask", int(self.invert_mask))
        shader.uniform_int("zero", 0)
        shader.uniform_float("lower", lo)
        shader.uniform_float("upper", hi)
        gpu.compute.dispatch(shader, 1, lines, 1)

    def evaluate_gpu(self, inputs, outputs):
        import gpu

        dst = outputs["Image"]
        src = inputs["Image"]
        if isinstance(src, (tuple, list, float, int)):
            vals = list(np.atleast_1d(np.asarray(src, np.float32)))[:4]
            vals += [0.0] * (4 - len(vals))
            dst.clear(format='FLOAT', value=vals)
            return

        # Linked Lower/Upper arrive as textures; fall back to the socket
        # defaults (0.25 / 0.8) in that case (per-pixel thresholds are not supported).
        lo = _scalar(inputs["Lower"], 0.25)
        hi = _scalar(inputs["Upper"], 0.8)

        # Mask socket (see evaluate_cpu). The serial shader has no mask support, so
        # it is only used when no masking is needed; otherwise the parallel one runs.
        state, mval = _mask_state(inputs.get("Mask"))
        mask_mode = {'none': 0, 'image': 1, 'zero': 2}[state]

        if _GPU_IMPL == "parallel" or mask_mode != 0:
            self._gpu_parallel(gpu, src, dst, lo, hi, mask_mode, mval)
            return

        shader = _get_shader(dst.format)
        shader.uniform_sampler("src", src)
        shader.image("dst", dst)
        shader.uniform_int("size_x", int(dst.width))
        shader.uniform_int("size_y", int(dst.height))
        shader.uniform_int("mask_key", _KEY_INDEX[self.mask_key])
        shader.uniform_int("sort_key", _KEY_INDEX[self.sort_key])
        shader.uniform_int("is_vertical", int(self.vertical))
        shader.uniform_int("reverse_order", int(self.reverse))
        shader.uniform_int("invert_mask", int(self.invert_mask))
        shader.uniform_int("zero", 0)
        shader.uniform_float("lower", lo)
        shader.uniform_float("upper", hi)

        lines = dst.width if self.vertical else dst.height
        groups = (lines + _LOCAL_SIZE - 1) // _LOCAL_SIZE
        gpu.compute.dispatch(shader, groups, 1, 1)


# ---------------------------------------------------------------------------
# Add menu
# ---------------------------------------------------------------------------

_menu_target = None


def _draw_add_item(self, context):
    layout = self.layout
    layout.separator()
    try:
        # NodeMenu subclasses (Blender 5.x) provide node_operator().
        self.node_operator(layout, CompositorNodePixelSort.bl_idname)
    except Exception:
        props = layout.operator("node.add_node", text=CompositorNodePixelSort.bl_label)
        props.type = CompositorNodePixelSort.bl_idname
        if hasattr(props, "use_transform"):
            props.use_transform = True


def _register_menu():
    global _menu_target
    for name in ("NODE_MT_category_compositor_filter", "NODE_MT_add"):
        menu = getattr(bpy.types, name, None)
        if menu is None:
            continue
        try:
            menu.append(_draw_add_item)
            _menu_target = menu
            return
        except Exception as ex:
            print("pixel_sort_node: could not extend", name, ex)


def _unregister_menu():
    global _menu_target
    if _menu_target is not None:
        try:
            _menu_target.remove(_draw_add_item)
        except Exception:
            pass
        _menu_target = None


def register():
    bpy.utils.register_class(CompositorNodePixelSort)
    _register_menu()


def unregister():
    _unregister_menu()
    bpy.utils.unregister_class(CompositorNodePixelSort)
    _shader_cache.clear()
    _par_shader_cache.clear()
    _scratch_cache.clear()


if __name__ == "__main__":
    register()
