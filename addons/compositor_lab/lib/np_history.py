# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Frame-history ring buffer on the CPU (numpy). GPU twin: ``glsl/history.py``.

A ``HistoryRing`` keeps the last ``capacity`` frames of an image stream, addressed by *frame
number*: frame ``f`` lives in slot ``f % capacity``. A frame is "stored" while its slot still holds
it, and a lookup "``d`` frames ago" (the *delay*) resolves against the frames that are stored::

    ring = HistoryRing(capacity=31, size=(w, h), downscale=2, dtype=np.float16)
    ring.push(frame, rgba)                       # (h, w, 4) float32; stores a box-downscaled copy
    table = ring.slot_table(frame, max_delay=30) # table[d] = slot holding the frame ``d`` ago
    img = ring.gather(table[delays], (h, w))     # (h, w, 4) float32, per-pixel slot -> pixels

Semantics of ``slot_table`` (shared with the GPU ring through ``FrameIndex``): the window is the
``capacity`` frames ``frame - capacity + 1 .. frame``. For a delay ``d`` the frame ``frame - d`` is
used if stored; otherwise the nearest *older* stored frame (a skipped frame holds the last one seen);
if none is older (history not filled yet, or after a jump), the oldest stored frame in the window.
Frames newer than ``frame`` (after scrubbing back) are never used, but stay stored, so going back a
few frames and forward again loses nothing (going back ``k`` frames does clamp the ``k`` oldest delays,
whose frames were already overwritten). Pushing the same frame again replaces it (re-render).

Downscaling (``downscale`` 1, 2 or 4) stores ``ceil(w / s) x ceil(h / s)`` pixels, each the mean of an
``s x s`` block of the input (edge pixels replicated beyond the border; summed row by row, then
multiplied by ``1 / s^2``, the operation order of the GLSL twin). ``gather`` reads it back with a
bilinear sample at ``(x + 0.5) / s - 0.5`` (clamped to the stored image), so a linear gradient
survives the round trip exactly away from the borders. With ``s == 1`` nothing is interpolated.
"""

import numpy as np

F32 = np.float32
NO_FRAME = -(1 << 40)
MB = 1024 * 1024


def stored_size(size, downscale):
    """(w, h) of a stored layer for an image of ``size`` (w, h)."""
    s = int(downscale)
    return ((int(size[0]) + s - 1) // s, (int(size[1]) + s - 1) // s)


def layer_bytes(size, downscale, itemsize):
    sw, sh = stored_size(size, downscale)
    return sw * sh * 4 * int(itemsize)


def fit_capacity(wanted, per_layer, budget_bytes):
    """Layers that fit in ``budget_bytes`` (at least 1): ``(capacity, limited)``."""
    wanted = max(int(wanted), 1)
    fit = max(int(budget_bytes) // max(int(per_layer), 1), 1)
    return min(wanted, fit), fit < wanted


def box_downscale(img, s):
    """(h, w, 4) float32 -> (ceil(h/s), ceil(w/s), 4): mean of s x s blocks, edges replicated."""
    s = int(s)
    img = np.asarray(img, F32)
    if s == 1:
        return img
    h, w = img.shape[:2]
    sw, sh = stored_size((w, h), s)
    if sh * s != h or sw * s != w:
        img = np.pad(img, ((0, sh * s - h), (0, sw * s - w), (0, 0)), mode="edge")
    acc = None
    for j in range(s):
        for i in range(s):
            tap = img[j::s, i::s]
            acc = tap.copy() if acc is None else acc + tap
    return (acc * F32(1.0 / (s * s))).astype(F32, copy=False)


class FrameIndex:
    """Which frame each ring slot holds, and the delay -> slot resolution (CPU and GPU rings)."""

    def __init__(self, capacity):
        self.capacity = max(int(capacity), 1)
        self.frames = np.full(self.capacity, NO_FRAME, np.int64)

    def slot_of(self, frame):
        return int(frame) % self.capacity

    def stored_frames(self):
        """Sorted frame numbers currently stored."""
        f = self.frames[self.frames != NO_FRAME]
        return np.sort(f)

    def stored_count(self, frame):
        """Frames stored in the window of ``frame`` (``frame - capacity + 1 .. frame``)."""
        f = self.frames
        return int(np.count_nonzero((f <= frame) & (f > frame - self.capacity)))

    def slot_table(self, frame, max_delay):
        """int32 array ``t`` with ``t[d]`` = slot to read for a delay of ``d`` frames, d = 0..max_delay
        (``max_delay <= capacity - 1``). The frame ``frame`` must have been pushed."""
        frame = int(frame)
        f = self.frames
        valid = np.nonzero((f <= frame) & (f > frame - self.capacity))[0]
        if len(valid) == 0:
            raise ValueError("frame %d is not stored" % frame)
        order = np.argsort(f[valid], kind="stable")
        slots = valid[order]
        fr = f[slots]
        want = frame - np.arange(int(max_delay) + 1, dtype=np.int64)
        idx = np.maximum(np.searchsorted(fr, want, side="right") - 1, 0)
        return slots[idx].astype(np.int32)

    def mark(self, frame):
        self.frames[self.slot_of(frame)] = int(frame)


class HistoryRing(FrameIndex):
    """The last ``capacity`` frames as a numpy array ``(capacity, sh, sw, 4)`` (``dtype`` float32
    or float16)."""

    def __init__(self, capacity, size, downscale=1, dtype=np.float32):
        super().__init__(capacity)
        self.size = (int(size[0]), int(size[1]))
        self.downscale = int(downscale)
        self.dtype = np.dtype(dtype)
        self.sw, self.sh = stored_size(self.size, self.downscale)
        self.data = np.zeros((self.capacity, self.sh, self.sw, 4), self.dtype)
        self._axes = {}

    @property
    def key(self):
        """Settings that define the layout; a ring is only reused when these match."""
        return (self.capacity, self.size, self.downscale, self.dtype.str)

    @property
    def nbytes(self):
        return int(self.data.nbytes)

    def copy(self):
        r = HistoryRing(self.capacity, self.size, self.downscale, self.dtype)
        r.data[...] = self.data
        r.frames[...] = self.frames
        return r

    def clear(self):
        self.frames[...] = NO_FRAME

    def push(self, frame, image):
        """Store ``image`` ((h, w, 4) float32) as frame ``frame`` (replacing it if present)."""
        self.data[self.slot_of(frame)] = box_downscale(image, self.downscale)
        self.mark(frame)

    # -- reading ---------------------------------------------------------------
    def _axis(self, n, stored):
        """Per output coordinate: lower / upper stored index and the lerp weight."""
        s = self.downscale
        key = (n, stored)
        ent = self._axes.get(key)
        if ent is None:
            c = (np.arange(n, dtype=F32) + F32(0.5)) * F32(1.0 / s) - F32(0.5)
            c0 = np.floor(c)
            t = (c - c0).astype(F32)
            i0 = c0.astype(np.int64)
            ent = (np.clip(i0, 0, stored - 1), np.clip(i0 + 1, 0, stored - 1), t)
            self._axes[key] = ent
        return ent

    def gather(self, slots, shape, current=None, current_slot=-1):
        """Pixels of the ring: ``slots`` is an int array broadcastable to ``shape`` (h, w) giving the
        slot to read per output pixel. Returns (h, w, 4) float32. Where ``slots == current_slot`` the
        full-resolution ``current`` image is used instead (the newest frame stays sharp)."""
        h, w = shape
        slots = np.asarray(slots)
        base = slots.astype(np.int64) * (self.sh * self.sw)
        flat = self.data.reshape(-1, 4)
        if self.downscale == 1:
            idx = base + (np.arange(h, dtype=np.int64) * self.sw)[:, None] \
                + np.arange(w, dtype=np.int64)[None, :]
            out = np.take(flat, idx, axis=0).astype(F32, copy=False)
        else:
            xa, xb, tx = self._axis(w, self.sw)
            ya, yb, ty = self._axis(h, self.sh)
            tx = tx[None, :, None]
            ty = ty[:, None, None]

            def tap(yi, xi):
                idx = base + (yi * self.sw)[:, None] + xi[None, :]
                return np.take(flat, idx, axis=0).astype(F32, copy=False)

            c00, c10, c01, c11 = tap(ya, xa), tap(ya, xb), tap(yb, xa), tap(yb, xb)
            bot = c00 + (c10 - c00) * tx
            top = c01 + (c11 - c01) * tx
            out = bot + (top - bot) * ty
        if current is not None:
            m = np.broadcast_to(slots == current_slot, (h, w))
            if m.any():
                out = np.where(m[..., None], np.asarray(current, F32), out)
        return np.ascontiguousarray(out, dtype=F32)
