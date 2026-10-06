# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Time Displace / Slit-scan: independent reference checks (slit-scan rows / columns, displacement
map, radial, frame blending, clamping before the history fills, downscale, precision), the ring
buffer helper against naive loops, stream semantics through renders (sequential, re-render, tweak,
scrub back / jump, start frame, Reset), memory cap, CPU vs GPU, robustness.
   Blender -b --factory-startup --python-exit-code 1 --python test_time_displace.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

import harness as H

H.setup()

from compositor_lab.lib import np_history, state as lab_state  # noqa: E402

NODE = "CompositorNodeLabTimeDisplace"
F32 = np.float32
SIZE = (32, 24)
DEVICES = ("CPU", "GPU")
N = 10


# ---------------------------------------------------------------------------------------------
# Test sequence: frame f = constant f/100 + a spatial gradient (to detect geometry)
# ---------------------------------------------------------------------------------------------
def fimg(f, size=SIZE):
    w, h = size
    x = np.arange(w, dtype=F32)[None, :] / F32(max(w - 1, 1))
    y = np.arange(h, dtype=F32)[:, None] / F32(max(h - 1, 1))
    img = np.empty((h, w, 4), F32)
    img[..., 0] = F32(f) / F32(100) + F32(0.25) * x
    img[..., 1] = F32(0.1) + F32(0.25) * y
    img[..., 2] = F32(0.3)
    img[..., 3] = 1.0
    return img


def noisy(f, size=SIZE):
    return H.test_image(size[0], size[1], seed=1000 + int(f), alpha=True)


class Rig:
    """A tree Image -> Time Displace -> output whose input image is replaced each frame."""

    def __init__(self, device, props=None, inputs=None, size=SIZE, seq=fimg, start=1,
                 map_img=None, out_socket=None):
        lab_state.clear()
        self.scene = H.configure_scene(size, device)
        self.scene.frame_start, self.scene.frame_end = start, 1000
        self.scene.frame_set(start)
        self.size, self.seq, self.device = size, seq, device
        p = {"interpolate": False, "precision": 'FULL'}
        p.update(props or {})
        i = {"History Frames": N}
        i.update(inputs or {})
        images = {"Image": seq(start, size)}
        if map_img is not None:
            images["Map"] = map_img
        self.node = H.build_tree(self.scene, NODE, p, i, images=images, out_socket=out_socket)
        self.src = self.node.inputs["Image"].links[0].from_node
        self.n = 0

    def render(self, f):
        self.scene.frame_set(int(f))
        self.n += 1
        old = self.src.image
        self.src.image = H.make_image("lab_seq_%d" % self.n, self.seq(f, self.size))
        bpy.data.images.remove(old)
        return H.render(self.scene)

    def stream(self):
        s = lab_state.streams_for(self.node.lab_uid)
        return s[0] if s else None

    def ring(self):
        s = self.stream()
        return s.current if s else None


def cleanup():
    H._clear_images()
    lab_state.clear()


# ---------------------------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------------------------
def ref_x(mode, size, n, direction=False, amount=1.0, map_img=None, centre=(0.5, 0.5)):
    """Delay in frames per pixel (float64, (h, w)); ``direction`` True = reversed."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if mode == 'ROWS':
        t = yy / max(h - 1, 1)
    elif mode == 'COLUMNS':
        t = xx / max(w - 1, 1)
    elif mode == 'MAP':
        m = np.asarray(map_img, np.float64)
        t = 0.2126 * m[..., 0] + 0.7152 * m[..., 1] + 0.0722 * m[..., 2]
    else:
        cx, cy = centre[0] * w, centre[1] * h
        far = np.hypot(max(cx, w - cx), max(cy, h - cy))
        t = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / far
    t = np.clip(t, 0.0, 1.0)
    if direction:
        t = 1.0 - t
    return t * n * amount


def ref_image(frames, f, lo, x, interp):
    """Output at frame ``f`` for delay map ``x``: pixel = frames[f - d]; frames older than ``lo`` (the
    oldest available) clamp to ``lo``. ``frames``: {frame: (h, w, 4)}."""
    def get(g):
        return np.asarray(frames[min(max(g, lo), f)], np.float64)

    if not interp:
        d = np.floor(x + 0.5).astype(int)
        out = np.zeros(x.shape + (4,))
        for v in np.unique(d):
            out = np.where((d == v)[..., None], get(f - int(v)), out)
        return out
    d0 = np.floor(x).astype(int)
    fr = (x - d0)[..., None]
    out = np.zeros(x.shape + (4,))
    for v in np.unique(d0):
        m = (d0 == v)[..., None]
        out = np.where(m, get(f - int(v)) * (1 - fr) + get(f - int(v) - 1) * fr, out)
    return out


def run_ref(rig_args, frames_to_render, check, interp=False, mode='ROWS', direction=False,
            amount=1.0, n=N, map_img=None, tol=1e-6, label="", max_frac=0.0, start=1,
            centre=(0.5, 0.5)):
    """Render frames ``start..max(frames_to_render)`` on every device; compare the frames in
    ``check`` with the reference. Returns the CPU renders."""
    res = {}
    for dev in DEVICES:
        rig = Rig(dev, start=start, map_img=map_img, **rig_args)
        seq = rig.seq
        frames = {}
        for f in range(start, max(frames_to_render) + 1):
            out = rig.render(f)
            frames[f] = seq(f, rig.size)
            if f in check:
                x = ref_x(mode, rig.size, n, direction, amount, map_img, centre)
                lo = max(start, f - n)
                ref = ref_image(frames, f, lo, x, interp)
                H.compare("%s %s f=%d" % (dev, label, f), out, ref, atol=tol, max_frac=max_frac,
                          quiet=True)
        res[dev] = out
        cleanup()
    return res


# ---------------------------------------------------------------------------------------------
@H.guard("ring helper")
def test_ring_helper():
    rng = np.random.default_rng(5)
    for size, s in (((13, 9), 1), ((13, 9), 2), ((13, 9), 4), ((16, 8), 4), ((5, 4), 2)):
        w, h = size
        ring = np_history.HistoryRing(4, size, s, np.float32)
        imgs = [rng.random((h, w, 4), dtype=np.float32) for _ in range(6)]
        for f, im in enumerate(imgs):
            ring.push(f, im)
        sw, sh = np_history.stored_size(size, s)
        H.check((ring.sw, ring.sh) == (sw, sh) and ring.data.shape == (4, sh, sw, 4),
                "ring %s s=%d: stored size %dx%d" % (size, s, sw, sh))
        # naive downscale of the newest frame (frame 5, slot 1)
        im = imgs[5].astype(np.float64)
        naive = np.zeros((sh, sw, 4))
        for j in range(sh):
            for i in range(sw):
                acc = np.zeros(4)
                for jj in range(s):
                    for ii in range(s):
                        acc += im[min(j * s + jj, h - 1), min(i * s + ii, w - 1)]
                naive[j, i] = acc / (s * s)
        H.compare("ring %s s=%d box downscale" % (size, s), ring.data[ring.slot_of(5)], naive,
                  atol=1e-6, quiet=True)
        # naive bilinear read of the stored frame 4 (slot 0) at every pixel
        st = ring.data[ring.slot_of(4)].astype(np.float64)
        stored = box = None
        ref = np.zeros((h, w, 4))
        for y in range(h):
            for x in range(w):
                if s == 1:
                    ref[y, x] = st[y, x]
                    continue
                fx, fy = (x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5
                x0, y0 = int(np.floor(fx)), int(np.floor(fy))
                tx, ty = fx - x0, fy - y0

                def at(yy, xx):
                    return st[min(max(yy, 0), sh - 1), min(max(xx, 0), sw - 1)]
                ref[y, x] = ((1 - ty) * ((1 - tx) * at(y0, x0) + tx * at(y0, x0 + 1))
                             + ty * ((1 - tx) * at(y0 + 1, x0) + tx * at(y0 + 1, x0 + 1)))
        got = ring.gather(np.full((h, w), ring.slot_of(4), np.int32), (h, w))
        H.compare("ring %s s=%d bilinear read" % (size, s), got, ref, atol=2e-6, quiet=True)
        # newest-frame override and per-pixel slots
        cur = rng.random((h, w, 4), dtype=np.float32)
        slots = np.where(np.arange(w)[None, :] % 2 == 0, ring.slot_of(5), ring.slot_of(4))
        slots = np.broadcast_to(slots, (h, w))
        got2 = ring.gather(slots, (h, w), cur, ring.slot_of(5))
        evens = (np.arange(w) % 2 == 0)
        H.check(np.array_equal(got2[:, evens], cur[:, evens]) and np.allclose(
            got2[:, ~evens], got[:, ~evens], atol=1e-7), "ring %s s=%d current override" % (size, s))
    # slot resolution
    fi = np_history.FrameIndex(5)
    for f in (10, 11, 12, 14):
        fi.mark(f)
    t = fi.slot_table(14, 4)
    want = [14, 12, 12, 11, 10]    # d=0..4 -> frames 14, 13 (missing: 12), 12, 11, 10
    H.check([int(fi.frames[s]) for s in t] == want, "slot table skips a missing frame: %s" %
            [int(fi.frames[s]) for s in t])
    fi.mark(18)
    t = fi.slot_table(14, 4)
    H.check([int(fi.frames[s]) for s in t] == want, "newer (stale) frames are never used")
    t = fi.slot_table(18, 4)
    H.check([int(fi.frames[s]) for s in t] == [18, 14, 14, 14, 14],
            "after a jump only frames inside the window count: %s" % [int(fi.frames[s]) for s in t])
    cap, lim = np_history.fit_capacity(31, 1000, 10 * 1000)
    H.check((cap, lim) == (10, True) and np_history.fit_capacity(5, 1000, 10 ** 9) == (5, False)
            and np_history.fit_capacity(5, 10 ** 9, 10) == (1, True), "fit_capacity")


@H.guard("slit-scan rows / columns")
def test_slitscan():
    seq = [1, 2, 3, 5, 9, 10, 11, 12, 13, 14, 15, 23, 24, 25]
    for mode in ('ROWS', 'COLUMNS'):
        for rev in (False, True):
            run_ref({"props": {"mode": mode, "direction": 'NEGATIVE' if rev else 'POSITIVE'}},
                    seq, set(seq), mode=mode, direction=rev, label="%s%s" % (mode, " rev" if rev else ""))
    # the headline property, spelled out: row r of the output is row r of frame (f - delay(r))
    rig = Rig("CPU")
    frames = {}
    for f in range(1, 21):
        out = rig.render(f)
        frames[f] = fimg(f)
    h = SIZE[1]
    ok = True
    for r in range(h):
        d = int(np.floor(r / (h - 1) * N + 0.5))
        ok &= np.array_equal(out[r], frames[20 - d][r])
    H.check(ok, "slit-scan row r equals row r of frame (current - delay(r)), bit exact")
    H.check(np.array_equal(out[0], frames[20][0]) and np.array_equal(out[-1], frames[20 - N][-1]),
            "bottom row is the current frame, top row is N frames old")
    cleanup()
    # amount scales the delay range
    run_ref({"inputs": {"Amount": 0.5}}, [1, 12, 13], {12, 13}, amount=0.5, label="amount 0.5")


@H.guard("clamp before fill")
def test_clamp():
    for dev in DEVICES:
        rig = Rig(dev)
        o1 = rig.render(1)
        H.compare("%s frame 1 (empty history) = input" % dev, o1, fimg(1), atol=1e-7, quiet=True)
        o2 = rig.render(2)
        # one frame of history: rows with delay >= 1 clamp to frame 1, delay 0 rows show frame 2
        h = SIZE[1]
        d = np.floor(np.arange(h) / (h - 1) * N + 0.5).astype(int)
        ref = np.stack([fimg(2)[r] if d[r] == 0 else fimg(1)[r] for r in range(h)])
        H.compare("%s frame 2 clamps to the oldest frame" % dev, o2, ref, atol=1e-7, quiet=True)
        for f in range(3, 6):
            rig.render(f)
        o6 = rig.render(6)
        ref = np.stack([fimg(max(6 - d[r], 1))[r] for r in range(h)])
        H.compare("%s frame 6 (partly filled history) clamps" % dev, o6, ref, atol=1e-7, quiet=True)
        cleanup()


@H.guard("displacement map")
def test_map():
    w, h = SIZE
    for dev in DEVICES:
        for v, d in ((0.0, 0), (0.4, 4), (1.0, 10), (0.7, 7)):
            rig = Rig(dev, props={"mode": 'MAP'}, map_img=np.full((h, w, 4), v, np.float32))
            frames = {}
            for f in range(1, 17):
                out = rig.render(f)
                frames[f] = fimg(f)
            H.compare("%s constant map %.1f: whole image = frame current-%d" % (dev, v, d), out,
                      frames[16 - d], atol=1e-7, quiet=True)
            if v == 0.4:
                dl = Rig(dev, props={"mode": 'MAP'}, map_img=np.full((h, w, 4), v, np.float32),
                         out_socket="Delay")
                o = None
                for f in range(1, 4):
                    o = dl.render(f)
                H.compare("%s Delay output = delay / N" % dev, o[..., 0], np.full((h, w), 0.4),
                          atol=1e-6, quiet=True)
            cleanup()
        # reversed direction: dark = long delay
        rig = Rig(dev, props={"mode": 'MAP', "direction": 'NEGATIVE'},
                  map_img=np.full((h, w, 4), 0.3, np.float32))
        for f in range(1, 17):
            out = rig.render(f)
        H.compare("%s reversed map (1 - 0.3) -> delay 7" % dev, out, fimg(16 - 7), atol=1e-7,
                  quiet=True)
        cleanup()
    # varying map with frame blending, independent float64 reference
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = np.stack([xx / w, yy / h, 0.5 * np.ones_like(xx), np.ones_like(xx)], -1)
    m = (m * 0.8 + 0.1 * rng.random((h, w, 4), dtype=np.float32)).astype(np.float32)
    m[..., 3] = 1.0
    run_ref({"props": {"mode": 'MAP', "interpolate": True}}, [1, 6, 14, 15], {6, 14, 15},
            interp=True, mode='MAP', map_img=m, tol=2e-6, label="map + blend")
    run_ref({"props": {"mode": 'MAP'}}, [1, 6, 14, 15], {6, 14, 15},
            interp=False, mode='MAP', map_img=m, tol=1e-6, max_frac=0.002, label="map rounded")


@H.guard("interpolation")
def test_interp():
    run_ref({"props": {"interpolate": True}}, [1, 3, 13, 14], {3, 13, 14}, interp=True,
            tol=2e-6, label="rows + blend")
    run_ref({"props": {"mode": 'COLUMNS', "interpolate": True, "direction": 'NEGATIVE'}},
            [1, 13, 14], {13, 14}, interp=True, mode='COLUMNS', direction=True, tol=2e-6,
            label="columns reversed + blend")
    # a half-way delay mixes two frames evenly
    for dev in DEVICES:
        w, h = SIZE
        rig = Rig(dev, props={"mode": 'MAP', "interpolate": True}, map_img=np.full((h, w, 4), 0.35, np.float32))
        for f in range(1, 13):
            out = rig.render(f)
        ref = 0.5 * fimg(12 - 3) + 0.5 * fimg(12 - 4)
        H.compare("%s delay 3.5 = mean of frames 3 and 4 back" % dev, out, ref, atol=2e-6, quiet=True)
        cleanup()


@H.guard("radial")
def test_radial():
    for centre in ((0.5, 0.5), (0.25, 0.7)):
        run_ref({"props": {"mode": 'RADIAL', "interpolate": True},
                 "inputs": {"Center X": centre[0], "Center Y": centre[1]}},
                [1, 13, 14], {13, 14}, interp=True, mode='RADIAL', tol=2e-5,
                label="radial %s" % (centre,), centre=centre)
    run_ref({"props": {"mode": 'RADIAL', "interpolate": True, "direction": 'NEGATIVE'}},
            [1, 13], {13}, interp=True, mode='RADIAL', direction=True, tol=2e-5,
            label="radial inward")


@H.guard("downscale")
def test_downscale():
    # Constant-per-frame + linear gradient survives box downscale + bilinear read away from the
    # border; the current frame (delay 0) is exact everywhere.
    for size, s in (((32, 24), 2), ((40, 28), 4), ((33, 25), 2)):
        for dev in DEVICES:
            rig = Rig(dev, props={"downscale": str(s), "mode": 'MAP', "interpolate": True},
                      size=size, map_img=np.full((size[1], size[0], 4), 0.5, np.float32))
            for f in range(1, 14):
                out = rig.render(f)
            ref = 0.5 * fimg(13 - 5, size) + 0.5 * fimg(13 - 5, size)   # delay 5.0 exactly
            b = 2 * s
            H.compare("%s downscale 1/%d %s: interior" % (dev, s, size),
                      out[b:-b, b:-b], ref[b:-b, b:-b], atol=1e-5, quiet=True)
            H.check(np.abs(out - ref).max() < 0.2, "%s downscale 1/%d %s: border stays close"
                    % (dev, s, size))
            ring = rig.ring()
            full = np_history.layer_bytes(size, 1, 4) * ring.capacity
            H.check(ring.nbytes <= full / (s * s) * 1.4 + 1,
                    "%s downscale 1/%d stores ~1/%d of the memory (%d vs %d)" % (
                        dev, s, s * s, ring.nbytes, full))
            cleanup()
    # delay 0 is the sharp input even with downscale
    for dev in DEVICES:
        rig = Rig(dev, props={"downscale": '4'}, seq=noisy)
        for f in range(1, 4):
            out = rig.render(f)
        H.compare("%s downscale: the bottom row (delay 0) is the sharp input" % dev,
                  out[0], noisy(3)[0], atol=1e-7, quiet=True)
        cleanup()
    # half precision: within half-float rounding
    for dev in DEVICES:
        rig = Rig(dev, props={"precision": 'HALF'})
        frames = {}
        for f in range(1, 15):
            out = rig.render(f)
            frames[f] = fimg(f)
        h = SIZE[1]
        d = np.floor(np.arange(h) / (h - 1) * N + 0.5).astype(int)
        ref = np.stack([frames[14 - d[r]][r] for r in range(h)])
        H.compare("%s half precision storage" % dev, out, ref, atol=1e-3, quiet=True)
        cleanup()


@H.guard("semantics")
def test_semantics():
    for dev in DEVICES:
        rig = Rig(dev, props={"interpolate": True})
        first = {f: rig.render(f) for f in range(1, 21)}
        H.check(rig.stream().last_kind == "STEP", "%s: sequential frames step" % dev)
        again = rig.render(20)
        H.check(rig.stream().last_kind == "REPEAT", "%s: re-render repeats" % dev)
        H.compare("%s re-render idempotent" % dev, again, first[20], atol=1e-7, quiet=True)
        H.compare("%s re-render idempotent (twice)" % dev, rig.render(20), first[20], atol=1e-7,
                  quiet=True)
        # live edit of a parameter on the same frame: no stepping, history unchanged
        rig.node.inputs["Amount"].default_value = 0.5
        edited = rig.render(20)
        frames = {f: fimg(f) for f in range(1, 21)}
        ref = ref_image(frames, 20, 10, ref_x('ROWS', SIZE, N, False, 0.5), True)
        H.compare("%s parameter tweak on the same frame" % dev, edited, ref, atol=2e-6, quiet=True)
        rig.node.inputs["Amount"].default_value = 1.0
        H.compare("%s tweak undone" % dev, rig.render(20), first[20], atol=1e-7, quiet=True)
        # scrub back inside the stored window: exact, and playback continues
        # (frames 10..20 are stored; going back to 17 loses the oldest ones, they clamp to 10)
        back = rig.render(17)
        frames = {f: fimg(f) for f in range(1, 21)}
        ref = ref_image(frames, 17, 10, ref_x('ROWS', SIZE, N), True)
        H.compare("%s scrub back to 17 inside the window" % dev, back, ref, atol=2e-6, quiet=True)
        near = np.arange(SIZE[1]) / (SIZE[1] - 1) * N <= 17 - 10
        H.compare("%s ... rows with delay <= 7 are exactly as before" % dev, back[near],
                  first[17][near], atol=2e-6, quiet=True)
        ref = ref_image(frames, 18, 10, ref_x('ROWS', SIZE, N), True)
        H.compare("%s ... and on to 18" % dev, rig.render(18), ref, atol=2e-6, quiet=True)
        H.compare("%s ... and to 20 again" % dev, rig.render(20), first[20], atol=2e-6, quiet=True)
        # scrub back beyond the stored window: the history restarts at that frame
        far = rig.render(3)
        H.compare("%s scrub back beyond the window: output = input" % dev, far, fimg(3), atol=1e-7,
                  quiet=True)
        nxt = rig.render(4)
        ref = ref_image({3: fimg(3), 4: fimg(4)}, 4, 3, ref_x('ROWS', SIZE, N), True)
        H.compare("%s ... then history builds from there" % dev, nxt, ref, atol=2e-6, quiet=True)
        # forward jump beyond the window
        jump = rig.render(60)
        H.compare("%s jump forward beyond the window: output = input" % dev, jump, fimg(60),
                  atol=1e-7, quiet=True)
        ref = ref_image({60: fimg(60), 61: fimg(61)}, 61, 60, ref_x('ROWS', SIZE, N), True)
        H.compare("%s ... then history builds from there (jump)" % dev, rig.render(61), ref,
                  atol=2e-6, quiet=True)
        # a small skip holds the last frame seen for the missing ones
        rig.render(62)
        rig.render(63)
        skip = rig.render(66)      # 64, 65 missing
        frames = {f: fimg(f) for f in (60, 61, 62, 63, 66)}
        x = ref_x('ROWS', SIZE, N)
        ref = np.zeros(x.shape + (4,))
        d0 = np.floor(x).astype(int)
        fr = (x - d0)[..., None]

        def look(g):
            older = [k for k in frames if k <= g]
            return frames[max(older) if older else min(frames)]
        for v in np.unique(d0):
            ref = np.where((d0 == v)[..., None], look(66 - v) * (1 - fr) + look(66 - v - 1) * fr, ref)
        H.compare("%s skipped frames hold the last one seen" % dev, skip, ref, atol=2e-6, quiet=True)
        # start frame resets and forgets
        r1 = rig.render(1)
        H.check(rig.stream().last_kind == "RESET", "%s: start frame resets" % dev)
        H.compare("%s start frame outputs the input" % dev, r1, fimg(1), atol=1e-7, quiet=True)
        ref = ref_image({1: fimg(1), 2: fimg(2)}, 2, 1, ref_x('ROWS', SIZE, N), True)
        H.compare("%s restarts from the start frame" % dev, rig.render(2), ref, atol=2e-6, quiet=True)
        # Reset operator
        uid = rig.node.lab_uid
        bpy.ops.compositor_lab.reset_state(uid=uid)
        H.check(lab_state.streams_for(uid) == [], "%s: Reset clears the streams" % dev)
        H.compare("%s evaluation after Reset starts over" % dev, rig.render(3), fimg(3), atol=1e-7,
                  quiet=True)
        # changing History Frames restarts the history (new layout)
        rig.render(4)
        rig.node.inputs["History Frames"].default_value = 5
        out = rig.render(5)
        ref = ref_image({5: fimg(5)}, 5, 5, ref_x('ROWS', SIZE, 5), True)
        H.compare("%s changing History Frames restarts the history" % dev, out, ref, atol=2e-6,
                  quiet=True)
        H.check(rig.ring().capacity == 6, "%s: capacity follows History Frames" % dev)
        cleanup()
    # scene start frame 10
    rig = Rig("CPU", start=10)
    rig.render(10)
    for f in range(11, 16):
        out = rig.render(f)
    frames = {f: fimg(f) for f in range(10, 16)}
    ref = ref_image(frames, 15, 10, ref_x('ROWS', SIZE, N), False)
    H.compare("start frame 10: clamps to frame 10", out, ref, atol=1e-6, quiet=True)
    st = rig.stream()
    if st is not None and st.key[1] == "UNKNOWN":
        H.note("build has no context.frame_start (F4): start frame falls back to 1")
    else:
        r = rig.render(10)
        H.check(rig.stream().last_kind == "RESET", "scene start frame resets")
        H.compare("scene start frame outputs the input", r, fimg(10), atol=1e-7, quiet=True)
    cleanup()


@H.guard("memory")
def test_memory():
    size = (64, 48)
    per = np_history.layer_bytes(size, 1, 4)
    for dev in DEVICES:
        # 1 MB cap: (1 MB // layer) layers, History Frames 30 asked
        rig = Rig(dev, props={"memory_mb": 1}, inputs={"History Frames": 30}, size=size)
        for f in range(1, 40):
            out = rig.render(f)
        cap = (1024 * 1024) // per
        ring = rig.ring()
        H.check(ring.capacity == cap, "%s: capacity limited by the memory cap (%d == %d)" % (
            dev, ring.capacity, cap))
        H.check(ring.nbytes <= 1024 * 1024 and lab_state.total_bytes() <= 1024 * 1024,
                "%s: history bytes %d within the cap" % (dev, ring.nbytes))
        msg = getattr(rig.stream(), "message", "")
        H.check("limited" in msg and str(cap - 1) in msg, "%s: message names the limit: %r" % (dev, msg))
        # the delay range follows the reduced length: top row = frame f - (cap - 1)
        nn = cap - 1
        H.compare("%s top row shows the oldest frame kept (delay %d)" % (dev, nn), out[-1],
                  fimg(39 - nn, size)[-1], atol=1e-7, quiet=True)
        cleanup()
        # roomy cap: no message, hard cap 120
        rig = Rig(dev, inputs={"History Frames": 500}, size=(16, 12))
        rig.render(1)
        H.check(rig.ring().capacity == 121 and getattr(rig.stream(), "message", "x") == "",
                "%s: History Frames hard cap 120 (capacity %d)" % (dev, rig.ring().capacity))
        cleanup()
        # downscale lets more frames fit in the same cap
        rig = Rig(dev, props={"memory_mb": 1, "downscale": '4'}, inputs={"History Frames": 30},
                  size=size)
        rig.render(1)
        H.check(rig.ring().capacity == 31 and rig.ring().nbytes <= 1024 * 1024,
                "%s: 1/4 downscale fits 30 frames in 1 MB" % dev)
        cleanup()
    # global budget: the state registry still accounts the history
    rig = Rig("CPU", size=size)
    rig.render(1)
    H.check(lab_state.total_bytes() == rig.ring().nbytes, "history counted by the state registry")
    cleanup()


@H.guard("cpu vs gpu")
def test_cpu_gpu():
    size = (40, 30)
    w, h = size
    rng = np.random.default_rng(9)
    m = H.test_image(w, h, seed=77)
    cases = [
        ("rows", {"mode": 'ROWS'}, {}),
        ("rows blend", {"mode": 'ROWS', "interpolate": True}, {}),
        ("columns reversed blend", {"mode": 'COLUMNS', "interpolate": True,
                                    "direction": 'NEGATIVE'}, {}),
        ("map blend", {"mode": 'MAP', "interpolate": True}, {}),
        ("map rounded", {"mode": 'MAP'}, {}),
        ("radial blend", {"mode": 'RADIAL', "interpolate": True}, {"Center X": 0.3}),
        ("radial blend 1/2", {"mode": 'RADIAL', "interpolate": True, "downscale": '2'}, {}),
        ("rows 1/4 half", {"mode": 'ROWS', "interpolate": True, "downscale": '4',
                           "precision": 'HALF'}, {}),
        ("map half amount", {"mode": 'MAP', "interpolate": True, "precision": 'HALF'},
         {"Amount": 0.6, "History Frames": 7}),
    ]
    for name, props, inputs in cases:
        outs = {}
        for dev in DEVICES:
            rig = Rig(dev, props=props, inputs=inputs, size=size, seq=noisy,
                      map_img=m if props["mode"] == 'MAP' else None)
            outs[dev] = [rig.render(f) for f in range(1, 16)]
            cleanup()
        worst = 0.0
        bad = 0.0
        for i, (c, g) in enumerate(zip(outs["CPU"], outs["GPU"])):
            d = np.abs(c - g).max(axis=-1)
            worst = max(worst, float(d.max()))
            bad = max(bad, float((d > 1e-5).mean()))
        H.note("cpu vs gpu %s: max diff %.3g, worst frame has %.4f%% pixels over 1e-5" % (
            name, worst, bad * 100))
        # Delay rounding ties / half-float rounding can differ on a few pixels; everything else is
        # equal to float precision.
        # Half storage: the CPU rounds to nearest, the GPU store may truncate, so the two differ by up
        # to one half-float ulp (2^-11 relative, 4.9e-4 for values below 1).
        half = props.get("precision") == 'HALF'
        for i, (c, g) in enumerate(zip(outs["CPU"], outs["GPU"])):
            H.compare("cpu vs gpu %s frame %d" % (name, i + 1), g, c, atol=1e-5,
                      rtol=1.1e-3 if half else 0.0, max_frac=0.0 if half else 0.01, quiet=True)
        H.check(not np.allclose(outs["CPU"][0], outs["CPU"][14], atol=1e-2), "%s: sequence changes" % name)
    # Delay output
    outs = {}
    for dev in DEVICES:
        rig = Rig(dev, props={"mode": 'RADIAL', "interpolate": True}, size=size, seq=noisy,
                  out_socket="Delay")
        outs[dev] = [rig.render(f) for f in range(1, 4)][-1]
        cleanup()
    H.compare("cpu vs gpu Delay output (radial)", outs["GPU"][..., 0], outs["CPU"][..., 0],
              atol=1e-5, quiet=True)
    d = outs["CPU"][..., 0]
    H.check(d.min() < 0.05 and 0.9 < d.max() <= 1.0 + 1e-6, "radial Delay spans 0..1 (%g..%g)"
            % (d.min(), d.max()))


@H.guard("robustness")
def test_robustness():
    # unlinked inputs, odd and tiny sizes, every mode, both devices (H.render fails on any error)
    for dev in DEVICES:
        for size in ((4, 4), (7, 5), (9, 4), (33, 17)):
            for mode in ('ROWS', 'COLUMNS', 'MAP', 'RADIAL'):
                for s in ('1', '2', '4'):
                    H._clear_images()
                    lab_state.clear()
                    scene = H.configure_scene(size, dev)
                    scene.frame_start = 1
                    scene.frame_set(1)
                    node = H.build_tree(scene, NODE, {"mode": mode, "downscale": s,
                                                      "interpolate": mode != 'COLUMNS'},
                                        {"History Frames": 6})      # nothing linked
                    for f in range(1, 5):
                        scene.frame_set(f)
                        out = H.render(scene)
                    H.check(np.isfinite(out).all() and out.shape == (size[1], size[0], 4),
                            "%s %s %s 1/%s: unlinked inputs render" % (dev, size, mode, s))
        # degenerate History Frames
        for n in (0, 1, 2):
            rig = Rig(dev, inputs={"History Frames": n}, seq=noisy)
            for f in range(1, 5):
                out = rig.render(f)
            H.check(np.isfinite(out).all(), "%s History Frames %d renders" % (dev, n))
            if n == 0:
                H.compare("%s History Frames 0 is a pass-through" % dev, out, noisy(4), atol=1e-7,
                          quiet=True)
            cleanup()
        # map with a different size than the image
        rig = Rig(dev, props={"mode": 'MAP'}, map_img=np.full((6, 8, 4), 0.5, np.float32))
        for f in range(1, 9):
            out = rig.render(f)
        # (Blender places a smaller image in the domain itself; the node just gets a full-size map)
        H.check(np.isfinite(out).all() and out.shape == (SIZE[1], SIZE[0], 4),
                "%s map of another size renders" % dev)
        cleanup()
    # the same node on two devices keeps separate streams
    cleanup()


test_ring_helper()
test_slitscan()
test_clamp()
test_map()
test_interp()
test_radial()
test_downscale()
test_semantics()
test_memory()
test_cpu_gpu()
test_robustness()
cleanup()
H.finish()
