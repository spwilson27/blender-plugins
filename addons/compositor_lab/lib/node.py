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

import uuid
from collections import namedtuple

import numpy as np
from bpy.props import IntProperty, StringProperty

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

# ``min`` / ``max``: the slider (soft) range of FLOAT / FACTOR / INT inputs. ``clamp``: a hard
# clamp applied to the *single value* by the ``in_*`` helpers on both backends. ``True`` clamps to
# ``min`` / ``max`` (a missing side stays open), ``(lo, hi)`` clamps to that range (and is the soft
# range unless ``min`` / ``max`` are given). FACTOR inputs default to the range 0..1.
In = namedtuple("In", "name type default hide_value min max clamp",
                defaults=(None, False, None, None, None))
Out = namedtuple("Out", "name type single", defaults=(False,))


class Ctx:
    """Normalised evaluation context. ``has_context`` is False on builds without F2."""

    __slots__ = ("frame", "fps", "time", "size", "use_gpu", "has_context", "kind",
                 "is_animation_playing", "frame_start", "frame_end", "_report")

    def __init__(self, frame=0.0, fps=24.0, time=0.0, size=(1, 1), use_gpu=False,
                 has_context=False, kind="UNKNOWN", is_animation_playing=False,
                 frame_start=1, frame_end=250, report=None):
        self.frame = float(frame)
        self.fps = float(fps)
        self.time = float(time)
        self.size = (int(size[0]), int(size[1]))
        self.use_gpu = bool(use_gpu)
        self.has_context = has_context
        # F4 (builds without it: 'UNKNOWN', False, scene range 1..250 placeholders).
        # kind: 'RENDER' | 'BACKDROP' | 'VIEWPORT' | 'SEQUENCER' | 'UNKNOWN'
        self.kind = str(kind)
        self.is_animation_playing = bool(is_animation_playing)
        self.frame_start = int(frame_start)
        self.frame_end = int(frame_end)
        self._report = report

    def report(self, message, level='INFO'):
        """F5: a non-fatal message (``'INFO'`` or ``'WARNING'``) shown on the node / in the info
        bar. Only valid during the evaluate call. A no-op on builds without ``context.report``."""
        if self._report is not None:
            self._report(str(message), level)

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
               getattr(context, "use_gpu", use_gpu), True,
               getattr(context, "kind", "UNKNOWN"),
               getattr(context, "is_animation_playing", False),
               getattr(context, "frame_start", 1),
               getattr(context, "frame_end", 250),
               getattr(context, "report", None))


def resolve_range(spec):
    """(soft_lo, soft_hi, clamp_lo, clamp_hi) of an ``In`` spec; ``None`` for an open side. The
    clamp pair is ``(None, None)`` when the input has no hard clamp."""
    lo, hi = spec.min, spec.max
    if spec.type == "FACTOR":
        lo = 0.0 if lo is None else lo
        hi = 1.0 if hi is None else hi
    c = spec.clamp
    if not c:
        return lo, hi, None, None
    if c is True:
        clo, chi = lo, hi
    else:
        clo, chi = c
        lo = clo if lo is None else lo
        hi = chi if hi is None else hi
    return lo, hi, clo, chi


_CLAMPS = {}     # LabNode subclass -> {socket name: (is_int, lo, hi)}


def _clamp_table(cls):
    table = _CLAMPS.get(cls)
    if table is None:
        table = {}
        for s in cls.SOCKETS:
            if isinstance(s, In) and s.type in ("FLOAT", "FACTOR", "INT"):
                _, _, clo, chi = resolve_range(s)
                if clo is not None or chi is not None:
                    table[s.name] = (s.type == "INT", clo, chi)
        _CLAMPS[cls] = table
    return table


def _clamp_value(entry, v):
    is_int, lo, hi = entry
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return int(v) if is_int else float(v)


def apply_ranges(node):
    """Set the slider range of a Lab node's FLOAT / FACTOR / INT sockets from its ``SOCKETS``
    spec. A no-op on Blender builds whose sockets have no ``min_value`` / ``max_value``."""
    for spec in node.SOCKETS:
        if not isinstance(spec, In) or spec.type not in ("FLOAT", "FACTOR", "INT"):
            continue
        sock = node.inputs.get(spec.name)
        if sock is None or not hasattr(sock, "min_value"):
            continue
        lo, hi, _, _ = resolve_range(spec)
        cast = int if spec.type == "INT" else float
        if lo is not None:
            sock.min_value = cast(lo)
        if hi is not None:
            sock.max_value = cast(hi)


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
        apply_ranges(self)

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
    @classmethod
    def in_float(cls, inputs, name, default=0.0):
        """Scalar float; ``default`` if the socket is missing or linked to an image. Inputs
        declared with ``clamp=`` are clamped (the default too), identically on both backends."""
        v = inputs.get(name)
        if isinstance(v, (int, float, bool)):
            v = float(v)
        elif isinstance(v, (tuple, list)) and v:
            v = float(v[0])
        else:
            v = float(default)
        entry = _clamp_table(cls).get(name)
        return v if entry is None else float(_clamp_value(entry, v))

    @classmethod
    def in_int(cls, inputs, name, default=0):
        """Scalar int (see ``in_float`` for ``clamp=``)."""
        v = inputs.get(name)
        v = int(v) if isinstance(v, (int, float, bool)) else int(default)
        entry = _clamp_table(cls).get(name)
        return v if entry is None else int(_clamp_value(entry, v))

    @classmethod
    def _clamp_single(cls, name, v):
        """Clamp a single scalar value of an input declared with ``clamp=``. Images and
        tuples (colours, vectors) pass through unchanged."""
        entry = _clamp_table(cls).get(name)
        if entry is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            return _clamp_value(entry, v)
        return v

    @staticmethod
    def in_color(inputs, name, default=(0.0, 0.0, 0.0, 1.0)):
        """Single colour as a 4-tuple (premultiplied); ``default`` if linked or missing."""
        v = inputs.get(name)
        if isinstance(v, (int, float, bool, tuple, list)):
            return tuple(_to_floats(v, 4, default))
        return tuple(default)

    @classmethod
    def in_image_array(cls, inputs, name, shape, channels=4, default=None):
        """Input as a (H, W, channels) float32 array (read-only: may be a broadcast view).

        Single values are broadcast; images with a different size are clamp-resampled; channel
        counts are adapted (grey -> RGB, RGB -> RGBA with alpha 1, RGBA -> RGB drops alpha).
        ``default`` is used when the socket is missing. ``clamp=`` applies to a scalar single
        value only; the pixels of a linked image are never clamped."""
        h, w = int(shape[0]), int(shape[1])
        v = inputs.get(name)
        if v is None:
            v = default if default is not None else 0.0
        v = cls._clamp_single(name, v)
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
    def in_is_image(inputs, name):
        """True when the input holds an image (a CPU array / GPUTexture), False when it is a
        single value: an unlinked socket *or* one linked to a single-value output (a Value node,
        a Mean output), which evaluate cannot tell apart and for which the single value is the
        right thing to use. (Whether a link exists is deliberately not exposed: evaluation runs
        on a copy of the tree on other threads, and a linked constant should act as a scalar.)"""
        return not is_single(inputs.get(name))

    @classmethod
    def in_texture_or_value(cls, inputs, name, default=0.0):
        """GPU: a GPUTexture for a linked input, else the single value (float / tuple). As for
        ``in_image_array``, ``clamp=`` applies to a scalar value, never to the texture."""
        v = inputs.get(name)
        return default if v is None else cls._clamp_single(name, v)


class StatefulNode:
    """Mixin for nodes that carry state from one evaluation to the next (see ``lib/state.py``).

    Put it before ``LabNode``: ``class CompositorNodeLabX(StatefulNode, LabNode,
    bpy.types.CompositorNode)``. It adds

    * a hidden ``lab_uid`` StringProperty (a UUID) set in ``init()`` and regenerated in
      ``copy()``. Evaluation runs on copies of the node tree, but ID properties survive those
      copies, so ``lab_uid`` is the stable key of the node's state. ``copy()`` is only called
      for user duplication (verified by ``tests/lab/test_feedback.py``).
    * ``max_catch_up`` and ``cache_frames`` properties (sidebar) used by ``state()``.
    * ``self.state(ctx)`` -> ``StateStream``; ``draw_state_buttons(layout)`` draws Reset.
    """

    lab_uid: StringProperty(name="Lab UID", default="", options={'HIDDEN'})
    max_catch_up: IntProperty(
        name="Max Catch-up", default=64, min=0, soft_max=1000,
        description="Frames the simulation may step in one evaluation after a jump forward "
                    "(or when re-simulating after jumping back); beyond that it holds")
    cache_frames: IntProperty(
        name="Cached Frames", default=32, min=0, soft_max=256,
        description="Frames of state kept for scrubbing back (0 disables the frame cache)")

    def init(self, context):
        super().init(context)
        self.lab_uid = uuid.uuid4().hex

    def copy(self, node):
        # Called when the user duplicates the node (never for evaluation copies).
        self.lab_uid = uuid.uuid4().hex

    def state_key(self):
        """Stable identity: ``lab_uid``; nodes made before it existed fall back to the name."""
        return self.lab_uid or ("name:" + self.name)

    def state(self, ctx):
        """The ``StateStream`` of this node for the evaluation described by ``ctx``."""
        from . import state as lab_state

        return lab_state.get_stream(self.state_key(), ctx.kind, ctx.size, ctx.use_gpu)

    def advance(self, ctx, signature=None, mutable=False):
        """``self.state(ctx).advance(...)`` with this node's start frame and properties.
        ``signature``: reset the state when this value changes (the layout of the state, e.g.
        grid size); ``mutable``: in-place frame-addressed payload (no frame cache, see
        ``lib/state.py``)."""
        plan = self.state(ctx).advance(ctx.frame, ctx.frame_start, self.max_catch_up,
                                       self.cache_frames, signature, mutable)
        if plan.message:
            ctx.report(plan.message, 'WARNING')      # e.g. a hold after a long jump
        return plan

    def draw_state_buttons(self, layout):
        op = layout.operator("compositor_lab.reset_state", text="Reset", icon='FILE_REFRESH')
        op.uid = self.state_key()

    def draw_buttons_ext(self, context, layout):
        self.draw_props(layout)
        layout.prop(self, "max_catch_up")
        layout.prop(self, "cache_frames")
        self.draw_state_buttons(layout)
