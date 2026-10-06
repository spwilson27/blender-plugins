# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""State streams for stateful nodes (simulations, feedback, history buffers).

The compositor evaluates a node from scratch, on a copy of the tree, from several independent
places (render, backdrop, viewport, sequencer) and frames may arrive in any order. A node keeps its
state in a ``StateStream`` keyed by ``(lab_uid, ctx.kind, size, use_gpu)``; ``advance`` decides
how the requested frame relates to the stored state, following simulation-zone semantics::

    plan = node.advance(ctx)                       # = node.state(ctx).advance(frame, start, ...)
    state = plan.run(init=lambda: ..., step=lambda s: ...)    # commits; returns the new state

``plan.kind`` / what to do (``plan.init``: build the state from the inputs first, ``plan.steps``:
then step that many times, ``plan.state``: the state to start from, or None when ``init``):

=============================  ==========================================================
reset (``RESET``)              first evaluation of the stream, or frame <= start frame:
                               ``init`` only (state = init(inputs)). Pre-step state: none.
step (``STEP``)                frame == last + 1: one step from the current state.
repeat (``REPEAT``)            frame == last (a parameter tweak, a re-render): one step from
                               the state *before* the last step, so edits show live without
                               running the simulation forward.
restore (``RESTORE``)          frame in the frame cache: its state, no stepping.
catch up (``CATCH_UP``)        forward jump of k > 1 frames with k <= max_catch_up: k steps.
reset and resimulate           backward to an uncached frame f: ``init`` then f - start steps,
(``RESIM``)                    if that is <= max_catch_up.
hold (``HOLD``)                a jump beyond max_catch_up (either direction): the current
                               state is shown unchanged, with ``plan.message``.
=============================  ==========================================================

Every step (also those of a catch-up and of a re-simulation) is fed the inputs of the *requested*
frame; the inputs of skipped frames are not available. After a hold the stream considers itself at
the requested frame (the state is stale but sequential frames carry on from it); holds are not
cached. Frames are rounded to integers.

State payloads are numpy arrays (CPU) or ``gpu.types.GPUTexture`` (GPU), or tuples / lists / dicts of
them. A payload handed to ``commit`` / returned by a step belongs to the stream: never mutate it
afterwards, and treat ``plan.state`` as read-only (step functions return a new payload). The frame
cache stores *copies* (GPU: a copy kernel).

Memory: each stream keeps at most ``cache_frames`` cached frames and ``stream_bytes``; across all
streams the total stays within ``global_bytes`` by least-recently-used eviction (cached frames
first, then whole idle streams). ``configure(...)`` changes the limits. Streams unused for
``ttl`` seconds expire (lazily, so deleted nodes do not leak), ``clear()`` drops everything (the
``load_pre`` handler) and the Reset operator drops one node's streams by uid.

Thread safety: one lock guards the registry; a stream is only used by one evaluation at a time.
"""

import threading
import time
from collections import OrderedDict

MB = 1024 * 1024

# Defaults per docs/plan-stateful-nodes.md.
CONFIG = {
    "cache_frames": 32,            # per stream
    "stream_bytes": 256 * MB,      # per stream (frame cache + live state)
    "global_bytes": 1024 * MB,     # all streams
    "max_catch_up": 64,
    "ttl": 1800.0,                 # seconds without use before a stream expires (0: never)
}

RESET, STEP, REPEAT, RESTORE, CATCH_UP, RESIM, HOLD = (
    "RESET", "STEP", "REPEAT", "RESTORE", "CATCH_UP", "RESIM", "HOLD")

_lock = threading.RLock()
_streams = OrderedDict()           # key -> StateStream, least recently used first
_tick = [0]
_clock = [time.monotonic]          # replaced by tests


def configure(**kw):
    """Change limits: cache_frames, stream_bytes, global_bytes, max_catch_up, ttl."""
    for k, v in kw.items():
        if k not in CONFIG:
            raise KeyError(k)
        CONFIG[k] = v
    with _lock:
        _enforce_global(None)


def _next_tick():
    _tick[0] += 1
    return _tick[0]


# ---------------------------------------------------------------------------
# Payload helpers (numpy / GPUTexture / containers; tests may use fake payloads)
# ---------------------------------------------------------------------------

_FORMAT_BYTES = {"RGBA32F": 16, "RGBA16F": 8, "RGBA8": 4, "RG32F": 8, "RG16F": 4, "RG8": 2,
                 "R32F": 4, "R16F": 2, "R8": 1, "R32UI": 4, "R32I": 4}


def payload_nbytes(p):
    if p is None:
        return 0
    if isinstance(p, (tuple, list)):
        return sum(payload_nbytes(x) for x in p)
    if isinstance(p, dict):
        return sum(payload_nbytes(x) for x in p.values())
    n = getattr(p, "nbytes", None)
    if n is not None:
        return int(n)
    if hasattr(p, "width") and hasattr(p, "height"):
        return int(p.width) * int(p.height) * _FORMAT_BYTES.get(getattr(p, "format", ""), 16)
    return 0


def gpu_copy(tex):
    """A new GPUTexture holding a copy of ``tex`` (same size and format), via a copy kernel."""
    import gpu

    from . import gpu as lab_gpu

    dst = gpu.types.GPUTexture((int(tex.width), int(tex.height)), format=tex.format)
    lab_gpu.pointwise("    out_Dst = in_Src(texel);\n", {"Dst": dst},
                      inputs={"Src": ("color", tex)})
    return dst


def payload_copy(p):
    if p is None:
        return None
    if isinstance(p, tuple):
        return tuple(payload_copy(x) for x in p)
    if isinstance(p, list):
        return [payload_copy(x) for x in p]
    if isinstance(p, dict):
        return {k: payload_copy(v) for k, v in p.items()}
    if hasattr(p, "copy") and hasattr(p, "nbytes"):
        return p.copy()
    if hasattr(p, "width") and hasattr(p, "height"):
        return gpu_copy(p)
    if hasattr(p, "copy"):
        return p.copy()
    return p


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class Plan:
    """What a node has to do for this evaluation (returned by ``StateStream.advance``)."""

    def __init__(self, stream, frame, kind, state, init, steps, message="", pre=None):
        self.stream = stream
        self.frame = frame
        self.kind = kind
        self.state = state          # payload to start from (read-only), None when ``init``
        self.init = init            # build the state from the inputs first
        self.steps = steps          # then step this many times
        self.message = message      # info text for holds ("" otherwise)
        self.pre = pre              # state before the last step (see ``run``)

    @property
    def hold(self):
        return self.kind == HOLD

    def run(self, init, step):
        """Execute the plan and commit the result: ``init()`` -> state, ``step(state)`` -> new
        state. Returns the final state (the node outputs it)."""
        s = self.state
        pre = self.pre
        if self.init:
            s = init()
            pre = None
        for _ in range(self.steps):
            pre = s
            s = step(s)
        self.commit(s, pre)
        return s

    def commit(self, state, pre=None):
        """Store ``state`` as the stream's state at this frame (``pre``: the state before the last
        step, used by REPEAT). ``run`` calls it; call it yourself for custom flows."""
        self.stream._commit(self, state, pre)


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class StateStream:
    def __init__(self, key):
        self.key = key
        self.last = None            # frame of ``current``
        self.current = None
        self.pre = None
        self.last_kind = None       # kind of the last committed plan (diagnostics)
        self.stale = False          # current is not the state of ``last`` (after a hold)
        self.cache = OrderedDict()  # frame -> [payload, nbytes, tick]; least recently used first
        self.cache_bytes = 0
        self.live_bytes = 0
        self.used = _clock[0]()
        self.max_frames = CONFIG["cache_frames"]

    # -- queries -----------------------------------------------------------
    @property
    def nbytes(self):
        return self.cache_bytes + self.live_bytes

    def cached_frames(self):
        return sorted(self.cache)

    def reset(self):
        """Forget everything: the next evaluation resets from the inputs."""
        with _lock:
            self.last = self.current = self.pre = self.last_kind = None
            self.stale = False
            self.cache.clear()
            self.cache_bytes = self.live_bytes = 0

    # -- the semantics table -----------------------------------------------
    def advance(self, frame, start_frame=1, max_catch_up=None, cache_frames=None):
        with _lock:
            self.used = _clock[0]()
            if self.key in _streams:
                _streams.move_to_end(self.key)
            if max_catch_up is None:
                max_catch_up = CONFIG["max_catch_up"]
            self.max_frames = CONFIG["cache_frames"] if cache_frames is None else int(cache_frames)
            f = int(round(frame))
            start = int(round(start_frame))
            last = self.last
            if last is None or f <= start:
                return Plan(self, f, RESET, None, True, 0)
            if f == last:
                if self.pre is not None and not self.stale:
                    return Plan(self, f, REPEAT, self.pre, False, 1, pre=self.pre)
                return Plan(self, f, RESTORE, self.current, False, 0, pre=self.pre)
            if f == last + 1:
                return Plan(self, f, STEP, self.current, False, 1)
            ent = self.cache.get(f)
            if ent is not None:
                self._touch(f, ent)
                prev = self.cache.get(f - 1)
                return Plan(self, f, RESTORE, payload_copy(ent[0]), False, 0,
                            pre=None if prev is None else prev[0])
            if f > last:
                k = f - last
                if k <= max_catch_up:
                    return Plan(self, f, CATCH_UP, self.current, False, k)
                return self._hold(f, "jumped %d frames forward (max catch-up %d): holding"
                                  % (k, max_catch_up))
            k = f - start
            if k <= max_catch_up:
                return Plan(self, f, RESIM, None, True, k)
            return self._hold(f, "jumped back to uncached frame %d, %d frames from the start "
                                 "(max catch-up %d): holding" % (f, k, max_catch_up))

    def _hold(self, f, msg):
        if self.current is None:
            return Plan(self, f, RESET, None, True, 0)
        return Plan(self, f, HOLD, self.current, False, 0, msg, pre=self.pre)

    # -- storage -------------------------------------------------------------
    def _touch(self, f, ent):
        ent[2] = _next_tick()
        self.cache.move_to_end(f)

    def _commit(self, plan, state, pre):
        with _lock:
            f = plan.frame
            self.last_kind = plan.kind
            if plan.kind == HOLD:
                self.last = f
                self.stale = True
                return
            unchanged = state is self.current
            self.current = state
            self.pre = pre
            self.last = f
            self.stale = self.stale and unchanged
            self.live_bytes = payload_nbytes(state) + payload_nbytes(pre)
            old = self.cache.pop(f, None)
            if old is not None:
                self.cache_bytes -= old[1]
            if self.max_frames > 0:
                cp = payload_copy(state)
                nb = payload_nbytes(cp)
                self.cache[f] = [cp, nb, _next_tick()]
                self.cache_bytes += nb
            self._enforce_stream()
            _enforce_global(self)

    def _evict_oldest(self):
        _, ent = self.cache.popitem(last=False)
        self.cache_bytes -= ent[1]

    def _enforce_stream(self):
        while self.cache and (len(self.cache) > self.max_frames
                              or self.nbytes > CONFIG["stream_bytes"]):
            self._evict_oldest()
        if self.max_frames <= 0:
            self.cache.clear()
            self.cache_bytes = 0


def _total_bytes():
    return sum(s.nbytes for s in _streams.values())


def _enforce_global(active):
    """Evict least-recently-used cached frames across all streams, then idle streams."""
    limit = CONFIG["global_bytes"]
    while _total_bytes() > limit:
        victims = [s for s in _streams.values() if s.cache]
        if victims:
            s = min(victims, key=lambda s: next(iter(s.cache.values()))[2])
            s._evict_oldest()
            continue
        idle = [s for s in _streams.values() if s is not active and s.nbytes]
        if not idle:
            return
        victim = min(idle, key=lambda s: s.used)
        del _streams[victim.key]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_stream(uid, kind, size, use_gpu):
    """The stream for ``(uid, kind, size, use_gpu)`` (created on first use)."""
    key = (str(uid), str(kind), (int(size[0]), int(size[1])), bool(use_gpu))
    with _lock:
        _expire()
        s = _streams.get(key)
        if s is None:
            s = _streams[key] = StateStream(key)
        return s


def _expire():
    ttl = CONFIG["ttl"]
    if not ttl:
        return
    now = _clock[0]()
    for k in [k for k, s in _streams.items() if now - s.used > ttl]:
        del _streams[k]


def clear(uid=None):
    """Drop all streams, or those of one node uid."""
    with _lock:
        for k in [k for k in _streams if uid is None or k[0] == uid]:
            del _streams[k]


def stream_count():
    return len(_streams)


def total_bytes():
    with _lock:
        return _total_bytes()


def streams_for(uid):
    with _lock:
        return [s for k, s in _streams.items() if k[0] == uid]


# ---------------------------------------------------------------------------
# Blender glue: load_pre handler and the Reset operator
# ---------------------------------------------------------------------------

_registered = []


def register():
    import bpy
    from bpy.app.handlers import persistent

    @persistent
    def _on_load_pre(*_args):
        clear()

    class COMPOSITOR_LAB_OT_reset_state(bpy.types.Operator):
        """Clear the stored simulation state of this node (it restarts from its inputs)"""
        bl_idname = "compositor_lab.reset_state"
        bl_label = "Reset State"
        bl_options = {'INTERNAL'}

        uid: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

        def execute(self, context):
            clear(self.uid or None)
            tree = getattr(getattr(context, "space_data", None), "edit_tree", None)
            if tree is None:
                tree = context.scene.compositing_node_group
            if tree is not None:
                tree.update_tag()
            context.scene.update_tag()
            return {'FINISHED'}

    if _registered:
        unregister()
    bpy.utils.register_class(COMPOSITOR_LAB_OT_reset_state)
    bpy.app.handlers.load_pre.append(_on_load_pre)
    _registered[:] = [COMPOSITOR_LAB_OT_reset_state, _on_load_pre]


def unregister():
    import bpy

    if not _registered:
        return
    cls, handler = _registered
    try:
        bpy.app.handlers.load_pre.remove(handler)
    except ValueError:
        pass
    try:
        bpy.utils.unregister_class(cls)
    except Exception:
        pass
    _registered.clear()
    clear()
