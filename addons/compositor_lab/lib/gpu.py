# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""GPU helpers for Lab nodes: the public kernel builder, shader cache, uniform binding, scratch
textures and error reporting. (See ``lib/README.md`` for the big picture.)

``kernel`` turns a GLSL body into a complete compute shader over the output image::

    kernel(
        body='''
            vec4 a = in_A(texel);
            out_Color = a * in_Fac(texel) * gain + helper(texel);
        ''',
        outputs={"Color": outputs["Color"]},              # name -> GPUTexture (None: skipped)
        inputs={"A": ("color", inputs["A"]),              # name -> (kind, value-or-texture)
                "Fac": ("float", inputs["Fac"])},
        uniforms={"gain": ("float", 2.0)},                # name -> (type, value)
        libs=("noise", "blend"),                          # lib/glsl modules (+ dependencies)
        functions='''                                     # node-local GLSL helpers
            vec4 helper(ivec2 p) { return in_A(p + ivec2(1, 0)); }
        ''',
    )

``pointwise`` is the same call without the ``sampling`` option (the name most nodes use). The
shader source is laid out as

1. (``sampling=True`` only) the ``glsl/sampling.py`` module,
2. the accessors ``in_<Name>`` (and, with ``sampling=True``, the ``lab_*_<Name>`` samplers),
3. ``libs`` (so lib code may call ``in_<Name>`` or read ``s_<Name>``),
4. ``functions`` (node-local helpers; may use the accessors, the libs and the push constants),
5. ``main`` with the body.

GLSL for Metal has no forward declarations: define a function before it is used.

Inside the body you have:

``ivec2 texel``  the pixel being computed (row 0 = bottom, like the compositor's buffers)
``vec2 uv``      pixel centre in 0..1
``ivec2 res``    output size
``in_<Name>(ivec2 p)``  the input value at pixel ``p``. Single values (unlinked sockets) are
                 push constants, linked inputs are sampled with ``texelFetch`` (``p`` is clamped
                 to the texture, so ``in_Image(texel + ivec2(1, 0))`` is safe at borders).
                 Returns ``float`` (kind "float"), ``vec3`` ("vec3") or ``vec4`` ("color").
                 Single-channel textures (R8/R16F/R32F) read as grey: ``vec3(r)`` for "vec3",
                 ``vec4(r, r, r, 1)`` for "color" (like ``LabNode.in_image_array``).
``out_<Name>``   a ``vec4`` per output, pre-initialised to 0 and written to the image afterwards.
                 Float outputs (R16F/R32F) use ``.r``.
your uniforms by name; ``lab_zero`` (always 0, for ``glsl/exact.py``).

With ``sampling=True`` every input is a sampler (a single value becomes a cached 1x1 texture)
and ``in_<Name>`` always returns ``vec4``; see ``glsl/sampling.py`` for the extra functions
(edge modes, bilinear / bicubic, Sobel). ``glsl.sampling.kernel`` is a thin wrapper for it.

Shader and push-constant names are prefixed ``lab_``/``u_``/``s_``/``o_`` to avoid clashes.
Shaders are cached by (body, functions, libs, input modes, uniform types, output formats).

Errors: a shader that fails to compile raises ``LabShaderError`` (naming the shader and carrying
the compiler's error). Like every exception in node evaluation it is also recorded in
``lib/errors.py`` (Blender itself only shows it as the node's info message), so tests can fail
on it.

Public helpers besides ``kernel``: ``get_shader``, ``create_info``, ``compile_shader`` (for
hand-written shaders), ``set_uniform`` / ``set_if_present``, ``dispatch_grid``, ``scratch``,
``const_texture``, ``as_floats``, ``ident``, ``is_texture``, ``is_single_channel``,
``PUSH_TYPES``, ``LOCAL_SIZE``.
"""

import re
import threading

from . import errors as _errors
from . import glsl as _glsl
from .errors import LabShaderError

_cache = {}
_tls = threading.local()

INPUT_KINDS = {"float": "float", "vec3": "vec3", "color": "vec4"}
PUSH_TYPES = {"float": 'FLOAT', "int": 'INT', "vec2": 'VEC2', "vec3": 'VEC3', "vec4": 'VEC4'}
SINGLE_CHANNEL_FORMATS = ("R8", "R16", "R16F", "R32F")
LOCAL_SIZE = (16, 16)


def clear_cache():
    """Drop cached shaders and this thread's scratch / constant textures."""
    _cache.clear()
    for name in ("scratch", "const"):
        _local_cache(name).clear()


def get_shader(key, factory):
    """Shader cache: ``factory()`` builds a shader the first time ``key`` is seen."""
    shader = _cache.get(key)
    if shader is None:
        shader = factory()
        _cache[key] = shader
    return shader


def create_info(local_size=None):
    """A GPUShaderCreateInfo with an optional compute local size."""
    import gpu

    info = gpu.types.GPUShaderCreateInfo()
    if local_size:
        info.local_group_size(*local_size)
    return info


def compile_shader(info, source, name="shader"):
    """Set the compute ``source`` on ``info`` and compile it. If the compiler rejects it, the
    error is recorded (``lib/errors.py``) and ``LabShaderError`` is raised."""
    import gpu

    try:
        info.compute_source(source)
        return gpu.shader.create_from_info(info)
    except Exception as ex:
        msg = "GPU shader %r failed to compile: %s" % (name, ex)
        _errors.record(msg)
        raise LabShaderError(msg) from ex


def ident(name):
    """A socket / uniform name as a GLSL identifier (non-word characters become ``_``)."""
    return re.sub(r"\W", "_", name)


def is_texture(value):
    return hasattr(value, "width") and hasattr(value, "height") and not isinstance(
        value, (int, float, tuple, list))


def is_single_channel(tex):
    """True for a GPUTexture with one channel (R8 / R16 / R16F / R32F)."""
    return is_texture(tex) and getattr(tex, "format", None) in SINGLE_CHANNEL_FORMATS


def as_floats(value, n, fill_alpha):
    """A single value (number or sequence) as ``n`` floats: a number is broadcast to rgb (and
    alpha 1 for n == 4); a missing alpha is 1 if ``fill_alpha``, other missing items 0."""
    if isinstance(value, (int, float, bool)):
        vals = [float(value)]
    else:
        vals = [float(v) for v in value]
    if len(vals) == 1 and n > 1:
        vals = vals * min(n, 3) + ([1.0] if n == 4 else [])
    if len(vals) < n:
        vals += [1.0 if (fill_alpha and len(vals) == 3 and n == 4) else 0.0] * (n - len(vals))
    return vals[:n]


def set_if_present(fn, name, *args):
    """Call a ``shader.uniform_*`` binder; tolerate a uniform the compiler removed because the
    shader never uses it."""
    try:
        fn(name, *args)
    except ValueError as ex:
        if "not found" not in str(ex):
            raise


def set_uniform(shader, name, utype, value):
    """Bind a push constant by type name ("int", "float", "vec2".."vec4") or a sampler
    (``utype`` "sampler"); unused uniforms are ignored."""
    if utype == "int":
        set_if_present(shader.uniform_int, name, int(value))
    elif utype == "float":
        set_if_present(shader.uniform_float, name, float(value))
    elif utype == "sampler":
        set_if_present(shader.uniform_sampler, name, value)
    else:
        set_if_present(shader.uniform_float, name, as_floats(value, int(utype[-1]), False))


def dispatch_grid(shader, width, height, local_size=LOCAL_SIZE):
    import gpu

    gx = (int(width) + local_size[0] - 1) // local_size[0]
    gy = (int(height) + local_size[1] - 1) // local_size[1]
    gpu.compute.dispatch(shader, gx, gy, 1)


# ---------------------------------------------------------------------------
# Scratch / constant textures (cached per thread)
# ---------------------------------------------------------------------------

def _local_cache(name):
    cache = getattr(_tls, name, None)
    if cache is None:
        cache = {}
        setattr(_tls, name, cache)
    return cache


def scratch(width, height, role="a", fmt="RGBA32F"):
    """A cached scratch texture (per thread, size, role, format). Use distinct ``role`` strings
    for textures that are alive at the same time within one node evaluation."""
    import gpu

    cache = _local_cache("scratch")
    key = (int(width), int(height), role, fmt)
    tex = cache.get(key)
    if tex is None:
        if len(cache) >= 24:
            cache.clear()
        tex = gpu.types.GPUTexture((int(width), int(height)), format=fmt)
        cache[key] = tex
    return tex


def const_texture(value):
    """A cached 1x1 RGBA32F texture holding a single value (float, 3- or 4-tuple)."""
    import gpu

    key = tuple(float(v) for v in as_floats(value, 4, fill_alpha=True))
    cache = _local_cache("const")
    tex = cache.get(key)
    if tex is None:
        if len(cache) >= 64:
            cache.clear()
        buf = gpu.types.Buffer('FLOAT', 4, list(key))
        tex = gpu.types.GPUTexture((1, 1), format='RGBA32F', data=buf)
        cache[key] = tex
    return tex


# ---------------------------------------------------------------------------
# The kernel builder
# ---------------------------------------------------------------------------

def _accessor(name, kind, tex, single):
    n = ident(name)
    gl = INPUT_KINDS[kind]
    if not tex:
        return "%s in_%s(ivec2 p)\n{\n  return u_%s;\n}\n" % (gl, n, n)
    fetch = "texelFetch(s_%s, clamp(p, ivec2(0), textureSize(s_%s, 0) - ivec2(1)), 0)" % (n, n)
    if single and kind == "vec3":
        expr = "vec3(%s.r)" % fetch
    elif single and kind == "color":
        expr = "vec4(vec3(%s.r), 1.0)" % fetch
    else:
        expr = fetch + {"float": ".r", "vec3": ".rgb", "color": ""}[kind]
    return "%s in_%s(ivec2 p)\n{\n  return %s;\n}\n" % (gl, n, expr)


def build_source(body, libs, input_specs, output_names, functions="", sampling=False):
    """The full compute source. ``input_specs``: [(name, kind, is_texture, is_single_channel)]."""
    parts = []
    lib_names = list(_glsl.resolve_order(*libs))
    if sampling:
        head = _glsl.resolve_order("sampling")
        parts.append(_glsl.resolve(*head))
        lib_names = [n for n in lib_names if n not in head]
        from .glsl import sampling as _sampling
        parts.append(_sampling.sampler_source([s[0] for s in input_specs]))
    else:
        for name, kind, tex, single in input_specs:
            parts.append(_accessor(name, kind, tex, single))
    if lib_names:
        parts.append(_glsl.resolve(*lib_names))
    if functions:
        parts.append(functions)
    decl = "".join("  vec4 out_%s = vec4(0.0);\n" % ident(n) for n in output_names)
    store = "".join("  imageStore(o_%s, texel, out_%s);\n" % (ident(n), ident(n))
                    for n in output_names)
    parts.append(
        "void main()\n{\n"
        "  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);\n"
        "  ivec2 res = ivec2(lab_w, lab_h);\n"
        "  if (texel.x >= res.x || texel.y >= res.y) {\n    return;\n  }\n"
        "  vec2 uv = (vec2(texel) + vec2(0.5)) / vec2(res);\n"
        + decl + "  {\n" + body + "\n  }\n" + store + "}\n")
    return "\n".join(parts)


def _build_kernel(source, input_specs, uniform_specs, outputs, local_size, name):
    info = create_info(local_size)
    slot = 0
    for iname, kind, tex, single in input_specs:
        if tex:
            info.sampler(slot, 'FLOAT_2D', "s_" + ident(iname))
            slot += 1
        else:
            ptype = {"float": 'FLOAT', "vec3": 'VEC3', "color": 'VEC4'}[kind]
            info.push_constant(ptype, "u_" + ident(iname))
    for slot, (oname, fmt) in enumerate(outputs):
        info.image(slot, fmt, 'FLOAT_2D', "o_" + ident(oname), qualifiers={'WRITE'})
    for uname, utype in uniform_specs:
        info.push_constant(PUSH_TYPES[utype], uname)
    for uname in ("lab_w", "lab_h", "lab_zero"):
        info.push_constant('INT', uname)
    return compile_shader(info, source, name)


def kernel(body, outputs, inputs=None, uniforms=None, libs=(), functions="",
           local_size=LOCAL_SIZE, sampling=False):
    """Build (cached), bind and dispatch a per-pixel compute shader. See the module docstring.

    ``outputs``: name -> GPUTexture (None entries skipped; the first by name gives the size).
    ``inputs``: name -> (kind, texture or single value). ``uniforms``: name -> (type, value).
    ``libs``: ``lib/glsl`` module names. ``functions``: GLSL helpers placed after the accessors
    and libs. ``sampling``: every input is a sampler with the ``glsl/sampling.py`` helpers."""
    inputs = inputs or {}
    uniforms = uniforms or {}
    out_items = sorted(((n, t) for n, t in outputs.items() if t is not None),
                       key=lambda kv: kv[0])
    if not out_items:
        return
    w, h = int(out_items[0][1].width), int(out_items[0][1].height)

    values = {}
    input_specs = []
    for iname in sorted(inputs):
        kind, value = inputs[iname]
        if kind not in INPUT_KINDS:
            raise ValueError("input kind %r" % (kind,))
        if sampling and not is_texture(value):
            value = const_texture(value)
        values[iname] = value
        input_specs.append((iname, kind, is_texture(value), is_single_channel(value)))
    input_specs = tuple(input_specs)
    uniform_specs = tuple((n, uniforms[n][0]) for n in sorted(uniforms))
    out_formats = tuple((n, t.format) for n, t in out_items)
    libs = tuple(libs)
    local_size = tuple(local_size)
    key = ("kernel", body, functions, libs, input_specs, uniform_specs, out_formats, local_size,
           bool(sampling))

    def factory():
        source = build_source(body, libs, input_specs, [n for n, _ in out_items], functions,
                              sampling)
        return _build_kernel(source, input_specs, uniform_specs, out_formats, local_size,
                             "kernel: " + ((body.strip().splitlines() or [""])[0].strip()[:60]))

    shader = get_shader(key, factory)

    for iname, kind, tex, single in input_specs:
        if tex:
            set_if_present(shader.uniform_sampler, "s_" + ident(iname), values[iname])
        else:
            n = {"float": 1, "vec3": 3, "color": 4}[kind]
            vals = as_floats(values[iname], n, fill_alpha=True)
            set_if_present(shader.uniform_float, "u_" + ident(iname), vals[0] if n == 1 else vals)
    for uname, (utype, value) in uniforms.items():
        set_uniform(shader, uname, utype, value)
    for oname, tex in out_items:
        shader.image("o_" + ident(oname), tex)
    set_if_present(shader.uniform_int, "lab_w", w)
    set_if_present(shader.uniform_int, "lab_h", h)
    set_if_present(shader.uniform_int, "lab_zero", 0)
    dispatch_grid(shader, w, h, local_size)


def pointwise(body, outputs, inputs=None, uniforms=None, libs=(), local_size=LOCAL_SIZE,
              functions=""):
    """``kernel`` without the sampling helpers (the name most nodes use)."""
    kernel(body, outputs, inputs, uniforms, libs, functions, local_size)
