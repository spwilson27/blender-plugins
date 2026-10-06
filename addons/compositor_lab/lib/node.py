# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""LabNode: base class (mixin) for Compositor Lab nodes.

A node declares its sockets and the properties to draw, and implements ``cpu`` and ``gpu``::

    class CompositorNodeLabThing(LabNode, bpy.types.CompositorNode):
        bl_idname = "CompositorNodeLabThing"
        bl_label = "Thing"
        SOCKETS = [In("Fac", "FACTOR", 1.0), In("A", "COLOR", (0.5, 0.5, 0.5, 1.0)),
                   Out("Color", "COLOR")]
        PROPS = ["mode"]
        mode: EnumProperty(...)          # bpy.props are declared normally

        def cpu(self, inputs, outputs, ctx):
            a = self.in_image_array(inputs, "A", ctx.shape)       # (H, W, 4) float32
            out = self.out_array(outputs, "Color")                # (H, W, 4) writable, or None
            ...

        def gpu(self, inputs, outputs, ctx):
            lab_gpu.pointwise(body, outputs={"Color": self.out_texture(outputs, "Color")}, ...)

``evaluate_cpu`` / ``evaluate_gpu`` accept the optional F2 ``context`` (a 4th argument) and pass
``cpu`` / ``gpu`` a normalised ``Ctx``. Builds that do not pass a context give defaults:
frame 0, time 0, size = output size. Colours are premultiplied scene-linear, row 0 = bottom.

Scalar parameters (``in_float``, ``in_int``, ``in_color``) are single values: if the socket is
linked to an image they fall back to ``default`` on both backends (identically, so CPU and GPU
agree). Per-pixel inputs use ``in_image_array`` (CPU) / ``in_texture_or_value`` (GPU).
"""

from collections import namedtuple

import numpy as np

from . import errors as _errors

# Socket type name -> bpy socket idname.
SOCKET_TYPES = {
    "FLOAT": "NodeSocketFloat",
    "FACTOR": "NodeSocketFloatFactor",
    "COLOR": "NodeSocketColor",
    "VECTOR": "NodeSocketVector",
    "INT": "NodeSocketInt",
    "BOOL": "NodeSocketBool",
}

In = namedtuple("In", "name type default hide_value", defaults=(None, False))
Out = namedtuple("Out", "name type single", defaults=(False,))


class Ctx:
    """Normalised evaluation context. ``has_context`` is False on builds without F2."""

    __slots__ = ("frame", "fps", "time", "size", "use_gpu", "has_context")

    def __init__(self, frame=0.0, fps=24.0, time=0.0, size=(1, 1), use_gpu=False,
                 has_context=False):
        self.frame = float(frame)
        self.fps = float(fps)
        self.time = float(time)
        self.size = (int(size[0]), int(size[1]))
        self.use_gpu = bool(use_gpu)
        self.has_context = has_context

    @property
    def shape(self):
        """(height, width): the output array shape."""
        return (self.size[1], self.size[0])

    def __repr__(self):
        return "Ctx(frame=%g, time=%g, size=%s, use_gpu=%s, has_context=%s)" % (
            self.frame, self.time, self.size, self.use_gpu, self.has_context)


def make_ctx(context, size, use_gpu):
    if context is None:
        return Ctx(size=size, use_gpu=use_gpu)
    fps = float(getattr(context, "fps", 24.0))
    frame = float(getattr(context, "frame", 0.0))
    time = getattr(context, "time", None)
    if time is None:
        time = frame / fps if fps else 0.0
    csize = getattr(context, "size", None)
    return Ctx(frame, fps, time, size if csize is None else csize,
               getattr(context, "use_gpu", use_gpu), True)


def is_single(value):
    return value is None or isinstance(value, (int, float, bool, tuple, list))


def _first_output(outputs):
    for v in outputs.values():
        if v is not None:
            return v
    return None


def _to_floats(value, n, default):
    """Single value -> list of n floats (colour: alpha defaults to 1)."""
    if isinstance(value, (int, float, bool)):
        vals = [float(value)] * (n if n < 4 else 3)
        if n == 4:
            vals.append(1.0)
    else:
        vals = [float(v) for v in value]
    if len(vals) < n:
        vals += [1.0 if (n == 4 and len(vals) == 3) else 0.0] * (n - len(vals))
    return vals[:n]


class LabNode:
    """Mixin for ``bpy.types.CompositorNode`` subclasses. Put it first in the bases."""

    SOCKETS = ()
    PROPS = ()
    single_value_outputs = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # F3: Out(..., single=True) outputs are single values (read with out_single()).
        singles = {s.name for s in cls.SOCKETS if isinstance(s, Out) and s.single}
        if singles:
            cls.single_value_outputs = frozenset(singles)

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CompositorNodeTree'

    # -- UI --------------------------------------------------------------
    def init(self, context):
        for spec in self.SOCKETS:
            if isinstance(spec, In):
                sock = self.inputs.new(SOCKET_TYPES[spec.type], spec.name)
                if spec.default is not None:
                    sock.default_value = spec.default
                if spec.hide_value:
                    sock.hide_value = True
            else:
                self.outputs.new(SOCKET_TYPES[spec.type], spec.name)

    def draw_buttons(self, context, layout):
        self.draw_props(layout)

    def draw_props(self, layout):
        """Draw every property in PROPS: a name, ``(name, {kwargs})`` or ``None`` (separator)."""
        for entry in self.PROPS:
            if entry is None:
                layout.separator()
            elif isinstance(entry, str):
                layout.prop(self, entry)
            else:
                layout.prop(self, entry[0], **entry[1])

    # -- evaluation ------------------------------------------------------
    def evaluate_cpu(self, inputs, outputs, context=None):
        try:
            ctx = make_ctx(context, self.output_size(outputs), False)
            self.cpu(inputs, outputs, ctx)
        except Exception as ex:
            self._record_error("cpu", ex)
            raise

    def evaluate_gpu(self, inputs, outputs, context=None):
        try:
            ctx = make_ctx(context, self.output_size(outputs), True)
            self.gpu(inputs, outputs, ctx)
        except Exception as ex:
            self._record_error("gpu", ex)
            raise

    def _record_error(self, backend, ex):
        """Blender turns an exception in evaluation into the node's info message and default
        outputs; keep a copy in ``lib/errors.py`` so tests can fail on it. (A shader compile
        error is already recorded by ``gpu.compile_shader``, with the GLSL error.)"""
        from .errors import LabShaderError

        if not isinstance(ex, LabShaderError):
            _errors.record("%s %s evaluation failed: %s: %s" % (
                type(self).__name__, backend, type(ex).__name__, ex))

    def cpu(self, inputs, outputs, ctx):
        raise NotImplementedError("%s has no CPU implementation" % type(self).__name__)

    def gpu(self, inputs, outputs, ctx):
        raise NotImplementedError("%s has no GPU implementation" % type(self).__name__)

    # -- outputs ---------------------------------------------------------
    def output_size(self, outputs):
        """(width, height) of the output domain (CPU buffers and GPU textures); single-value
        outputs (F3) are ignored. (1, 1) if there is no image output."""
        singles = self.single_value_outputs
        first = _first_output({k: v for k, v in outputs.items() if k not in singles})
        if first is None:
            return (1, 1)
        if hasattr(first, "width"):
            return (int(first.width), int(first.height))
        shape = np.asarray(first).shape
        return (int(shape[1]), int(shape[0]))

    @staticmethod
    def out_array(outputs, name):
        """Writable (H, W, C) float32 view of an output (C = 1 for float outputs), or None if
        the output is unused."""
        buf = outputs.get(name)
        if buf is None:
            return None
        arr = np.asarray(buf)
        return arr.reshape(arr.shape[0], arr.shape[1], -1)

    @staticmethod
    def out_single(outputs, name):
        """F3: writable 1-D float32/int32 array of shape (channels,) for a single-value output
        (declared with ``Out(name, type, single=True)``), CPU and GPU; None if unused.
        Example: ``self.out_single(outputs, "Mean")[:] = (r, g, b, 1.0)``."""
        buf = outputs.get(name)
        return None if buf is None else np.asarray(buf)

    @staticmethod
    def out_texture(outputs, name):
        """GPUTexture of an output, or None if unused."""
        return outputs.get(name)

    # -- inputs ----------------------------------------------------------
    @staticmethod
    def in_float(inputs, name, default=0.0):
        """Scalar float; ``default`` if the socket is missing or linked to an image."""
        v = inputs.get(name)
        if isinstance(v, (int, float, bool)):
            return float(v)
        if isinstance(v, (tuple, list)) and v:
            return float(v[0])
        return float(default)

    @classmethod
    def in_int(cls, inputs, name, default=0):
        v = inputs.get(name)
        if isinstance(v, (int, float, bool)):
            return int(v)
        return int(default)

    @staticmethod
    def in_color(inputs, name, default=(0.0, 0.0, 0.0, 1.0)):
        """Single colour as a 4-tuple (premultiplied); ``default`` if linked or missing."""
        v = inputs.get(name)
        if isinstance(v, (int, float, bool, tuple, list)):
            return tuple(_to_floats(v, 4, default))
        return tuple(default)

    @staticmethod
    def in_image_array(inputs, name, shape, channels=4, default=None):
        """Input as a (H, W, channels) float32 array (read-only: may be a broadcast view).

        Single values are broadcast; images with a different size are clamp-resampled; channel
        counts are adapted (grey -> RGB, RGB -> RGBA with alpha 1, RGBA -> RGB drops alpha).
        ``default`` is used when the socket is missing."""
        h, w = int(shape[0]), int(shape[1])
        v = inputs.get(name)
        if v is None:
            v = default if default is not None else 0.0
        if is_single(v):
            vals = np.asarray(_to_floats(v, channels, None), dtype=np.float32)
            return np.broadcast_to(vals, (h, w, channels))
        a = np.asarray(v)
        a = a.reshape(a.shape[0], a.shape[1], -1)
        c = a.shape[2]
        if c != channels:
            if c == 1:
                a = np.repeat(a, 3, axis=2)
                c = 3
            if channels == 1:
                a = a[..., :1]
            elif channels == 4 and c == 3:
                a = np.concatenate([a, np.ones(a.shape[:2] + (1,), a.dtype)], axis=2)
            elif channels == 3 and c >= 3:
                a = a[..., :3]
            elif channels == 2:
                a = a[..., :2]
        if a.shape[:2] != (h, w):
            if a.shape[:2] == (1, 1):
                a = np.broadcast_to(a, (h, w, a.shape[2]))
            else:
                yi = np.minimum(np.arange(h), a.shape[0] - 1)
                xi = np.minimum(np.arange(w), a.shape[1] - 1)
                a = a[yi[:, None], xi[None, :]]
        return a

    @staticmethod
    def in_texture_or_value(inputs, name, default=0.0):
        """GPU: a GPUTexture for a linked input, else the single value (float / tuple)."""
        v = inputs.get(name)
        return default if v is None else v
