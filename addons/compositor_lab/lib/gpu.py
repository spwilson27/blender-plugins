# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""GPU helpers for Lab nodes: shader cache, CreateInfo builder, and the ``pointwise`` kernel.

``pointwise`` turns a GLSL body into a complete compute shader over the output image::

    pointwise(
        body='''
            vec4 a = in_A(texel);
            out_Color = a * in_Fac(texel) * gain;
        ''',
        outputs={"Color": outputs["Color"]},              # name -> GPUTexture
        inputs={"A": ("color", inputs["A"]),              # name -> (kind, value-or-texture)
                "Fac": ("float", inputs["Fac"])},
        uniforms={"gain": ("float", 2.0)},                # name -> (type, value)
        libs=("noise", "blend"),                          # lib/glsl modules (+ dependencies)
    )

Inside the body you have:

``ivec2 texel``  the pixel being computed (row 0 = bottom, like the compositor's buffers)
``vec2 uv``      pixel centre in 0..1
``ivec2 res``    output size
``in_<Name>(ivec2 p)``  the input value at pixel ``p``. Single values (unlinked sockets) are
                 push constants, linked inputs are sampled with ``texelFetch`` (``p`` is clamped
                 to the texture, so ``in_Image(texel + ivec2(1, 0))`` is safe at borders).
                 Returns ``float`` (kind "float"), ``vec3`` ("vec3") or ``vec4`` ("color").
``out_<Name>``   a ``vec4`` per output, pre-initialised to 0 and written to the image afterwards.
                 Float outputs (R16F/R32F) use ``.r``.
your uniforms by name; ``lab_zero`` (always 0, for ``glsl/exact.py``).

Shader and push-constant names are prefixed ``lab_``/``u_``/``s_``/``o_`` to avoid clashes.
Shaders are cached by (body, libs, input modes, uniform types, output formats).
"""

import re

from . import glsl as _glsl

_cache = {}

INPUT_KINDS = {"float": "float", "vec3": "vec3", "color": "vec4"}
_PUSH_TYPES = {"float": 'FLOAT', "int": 'INT', "vec2": 'VEC2', "vec3": 'VEC3', "vec4": 'VEC4'}
LOCAL_SIZE = (16, 16)


def clear_cache():
    _cache.clear()


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


def _ident(name):
    return re.sub(r"\W", "_", name)


def is_texture(value):
    return hasattr(value, "width") and hasattr(value, "height") and not isinstance(
        value, (int, float, tuple, list))


def _as_floats(value, n, fill_alpha):
    if isinstance(value, (int, float, bool)):
        vals = [float(value)]
    else:
        vals = [float(v) for v in value]
    if len(vals) == 1 and n > 1:
        vals = vals * min(n, 3) + ([1.0] if n == 4 else [])
    if len(vals) < n:
        vals += [1.0 if (fill_alpha and len(vals) == 3 and n == 4) else 0.0] * (n - len(vals))
    return vals[:n]


def build_pointwise_source(body, libs, input_specs, uniform_specs, output_names):
    """input_specs: [(name, kind, is_texture)], uniform_specs: [(name, type)]."""
    parts = [_glsl.resolve(*libs) if libs else ""]
    for name, kind, tex in input_specs:
        n = _ident(name)
        gl = INPUT_KINDS[kind]
        if tex:
            swz = {"float": ".r", "vec3": ".rgb", "color": ""}[kind]
            parts.append(
                "%s in_%s(ivec2 p)\n{\n  return texelFetch(s_%s, clamp(p, ivec2(0), "
                "textureSize(s_%s, 0) - ivec2(1)), 0)%s;\n}\n" % (gl, n, n, n, swz))
        else:
            parts.append("%s in_%s(ivec2 p)\n{\n  return u_%s;\n}\n" % (gl, n, n))
    decl = "".join("  vec4 out_%s = vec4(0.0);\n" % _ident(n) for n in output_names)
    store = "".join("  imageStore(o_%s, texel, out_%s);\n" % (_ident(n), _ident(n))
                    for n in output_names)
    parts.append(
        "void main()\n{\n"
        "  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);\n"
        "  ivec2 res = ivec2(lab_w, lab_h);\n"
        "  if (texel.x >= res.x || texel.y >= res.y) {\n    return;\n  }\n"
        "  vec2 uv = (vec2(texel) + vec2(0.5)) / vec2(res);\n"
        + decl + "  {\n" + body + "\n  }\n" + store + "}\n")
    return "\n".join(parts)


def _build_pointwise(source, input_specs, uniform_specs, outputs, local_size):
    import gpu

    info = create_info(local_size)
    slot = 0
    for name, kind, tex in input_specs:
        if tex:
            info.sampler(slot, 'FLOAT_2D', "s_" + _ident(name))
            slot += 1
        else:
            ptype = {"float": 'FLOAT', "vec3": 'VEC3', "color": 'VEC4'}[kind]
            info.push_constant(ptype, "u_" + _ident(name))
    for slot, (name, fmt) in enumerate(outputs):
        info.image(slot, fmt, 'FLOAT_2D', "o_" + _ident(name), qualifiers={'WRITE'})
    for name, utype in uniform_specs:
        info.push_constant(_PUSH_TYPES[utype], name)
    for name in ("lab_w", "lab_h", "lab_zero"):
        info.push_constant('INT', name)
    info.compute_source(source)
    return gpu.shader.create_from_info(info)


def _set(fn, name, *args):
    """Bind a uniform; tolerate one the compiler removed because the body never uses it."""
    try:
        fn(name, *args)
    except ValueError as ex:
        if "not found" not in str(ex):
            raise


def dispatch_grid(shader, width, height, local_size=LOCAL_SIZE):
    import gpu

    gx = (int(width) + local_size[0] - 1) // local_size[0]
    gy = (int(height) + local_size[1] - 1) // local_size[1]
    gpu.compute.dispatch(shader, gx, gy, 1)


def pointwise(body, outputs, inputs=None, uniforms=None, libs=(), local_size=LOCAL_SIZE):
    """Build (cached), bind and dispatch a per-pixel compute shader. See the module docstring."""
    inputs = inputs or {}
    uniforms = uniforms or {}
    out_items = [(n, t) for n, t in outputs.items() if t is not None]
    if not out_items:
        return
    out_items.sort(key=lambda kv: kv[0])
    first = out_items[0][1]
    w, h = int(first.width), int(first.height)

    input_specs = []
    for name in sorted(inputs):
        kind, value = inputs[name]
        if kind not in INPUT_KINDS:
            raise ValueError("input kind %r" % (kind,))
        input_specs.append((name, kind, is_texture(value)))
    uniform_specs = [(n, uniforms[n][0]) for n in sorted(uniforms)]
    out_formats = tuple((n, t.format) for n, t in out_items)

    libs = tuple(libs)
    key = ("pointwise", body, libs, tuple(input_specs), tuple(uniform_specs), out_formats,
           tuple(local_size))

    def factory():
        source = build_pointwise_source(body, libs, input_specs, uniform_specs,
                                        [n for n, _ in out_items])
        return _build_pointwise(source, input_specs, uniform_specs, out_formats, local_size)

    shader = get_shader(key, factory)

    for name, kind, tex in input_specs:
        value = inputs[name][1]
        if tex:
            _set(shader.uniform_sampler, "s_" + _ident(name), value)
        else:
            n = {"float": 1, "vec3": 3, "color": 4}[kind]
            vals = _as_floats(value, n, fill_alpha=True)
            _set(shader.uniform_float, "u_" + _ident(name), vals[0] if n == 1 else vals)
    for name, (utype, value) in uniforms.items():
        if utype == "int":
            _set(shader.uniform_int, name, int(value))
        elif utype == "float":
            _set(shader.uniform_float, name, float(value))
        else:
            n = int(utype[-1])
            _set(shader.uniform_float, name, _as_floats(value, n, fill_alpha=False))
    for name, tex in out_items:
        shader.image("o_" + _ident(name), tex)
    _set(shader.uniform_int, "lab_w", w)
    _set(shader.uniform_int, "lab_h", h)
    _set(shader.uniform_int, "lab_zero", 0)
    dispatch_grid(shader, w, h, local_size)
