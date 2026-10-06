# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""lib/state.py: the StateStream semantics table, caches and budgets, with fake payloads (no
rendering; the module is loaded by path so no Blender is needed, though the suite runs in it).
   Blender -b --factory-startup --python-exit-code 1 --python test_state.py"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, os.pardir, os.pardir, "addons", "compositor_lab", "lib", "state.py")
spec = importlib.util.spec_from_file_location("lab_state_under_test", PATH)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)

FAILURES = []


def check(cond, msg):
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond:
        FAILURES.append(msg)


class P:
    """Fake payload: a number with a byte size; copies are counted."""
    copies = 0

    def __init__(self, v, nbytes=100):
        self.v = v
        self.nbytes = nbytes

    def copy(self):
        P.copies += 1
        return P(self.v, self.nbytes)


def fresh(**cfg):
    S.clear()
    S.CONFIG.update(cache_frames=32, stream_bytes=1 << 40, global_bytes=1 << 40,
                    max_catch_up=64, ttl=0)
    S.CONFIG.update(cfg)
    return S.get_stream("node-a", "RENDER", (8, 8), False)


def run(stream, frame, start=1, inc=1, max_catch_up=None, cache_frames=None, nbytes=100):
    """One evaluation of the fake simulation: init -> 100, step -> v + inc. Returns (plan, v)."""
    plan = stream.advance(frame, start, max_catch_up, cache_frames)
    out = plan.run(lambda: P(100, nbytes), lambda s: P(s.v + inc, nbytes))
    return plan, out.v


def seq(stream, frames, **kw):
    return [run(stream, f, **kw) for f in frames]


# --- reset / step -------------------------------------------------------------------------
st = fresh()
res = seq(st, [1, 2, 3, 4, 5])
check([p.kind for p, _ in res] == ["RESET", "STEP", "STEP", "STEP", "STEP"], "reset then steps")
check([v for _, v in res] == [100, 101, 102, 103, 104], "sequential values")
p, v = run(st, 1)
check(p.kind == "RESET" and v == 100, "frame <= start resets (even when it is the last frame)")
p, v = run(st, 0)
check(p.kind == "RESET" and v == 100, "frame below start resets")
check(run(fresh(), 7)[0].kind == "RESET", "first evaluation resets (also beyond the start)")
st = fresh()
seq(st, [4, 5])
p, v = run(st, 5, start=5)
check(p.kind == "RESET", "start frame is a parameter")

# --- repeat --------------------------------------------------------------------------------
st = fresh()
seq(st, [1, 2, 3, 4])
p, v = run(st, 4)
check(p.kind == "REPEAT" and v == 103, "repeat recomputes from the pre-step state (103)")
p, v = run(st, 4, inc=10)
check(p.kind == "REPEAT" and v == 112, "repeat with a changed parameter shows live (112)")
p, v = run(st, 4)
check(v == 103, "repeat is idempotent after a tweak (103)")
p, v = run(st, 5)
check(p.kind == "STEP" and v == 104, "step continues after repeats")
st = fresh()
run(st, 1)
p, v = run(st, 1)
check(p.kind == "RESET" and v == 100, "repeat of a reset frame is a reset")

# --- restore -------------------------------------------------------------------------------
st = fresh()
seq(st, range(1, 11))
p, v = run(st, 3)
check(p.kind == "RESTORE" and v == 102, "scrub back to a cached frame restores (102)")
p, v = run(st, 4)
check(p.kind == "STEP" and v == 103, "stepping on after a restore continues from it (103)")
p, v = run(st, 8)
check(p.kind == "RESTORE" and v == 107, "forward to a cached frame restores (107)")
st = fresh()
seq(st, range(1, 6))
run(st, 3)
p, v = run(st, 3, inc=5)
check(p.kind == "REPEAT" and v == 106, "repeat on a restored frame steps from the cached previous frame (101 + 5)")

# --- catch up / hold -----------------------------------------------------------------------
st = fresh()
seq(st, [1, 2, 3])
p, v = run(st, 7)
check(p.kind == "CATCH_UP" and p.steps == 4 and v == 106, "forward jump catches up (106)")
st = fresh()
seq(st, [1, 2, 3])
p, v = run(st, 13, max_catch_up=5)
check(p.kind == "HOLD" and v == 102 and "10" in p.message, "jump beyond max_catch_up holds: %r" % p.message)
p, v = run(st, 14, max_catch_up=5)
check(p.kind == "STEP" and v == 103, "after a hold, sequential frames step from the held state")
check(13 not in st.cached_frames(), "holds are not cached")
st = fresh()
seq(st, [1, 2, 3])
p, v = run(st, 5, max_catch_up=1)
check(p.kind == "HOLD", "max_catch_up 1: jump of 2 holds")
p, v = run(fresh(), 1, max_catch_up=0)
check(p.kind == "RESET", "max_catch_up 0 does not affect reset")

# --- reset and resimulate / hold backward ----------------------------------------------------
st = fresh(cache_frames=3)
seq(st, range(1, 21), cache_frames=3)
check(st.cached_frames() == [18, 19, 20], "frame cache keeps the last 3 frames: %s" % st.cached_frames())
p, v = run(st, 10, cache_frames=3)
check(p.kind == "RESIM" and p.steps == 9 and v == 109, "uncached backward frame resimulates from start (109)")
p, v = run(st, 11, cache_frames=3)
check(p.kind == "STEP" and v == 110, "continues sequentially after a resimulation")
st = fresh(cache_frames=3)
seq(st, range(1, 21), cache_frames=3)
p, v = run(st, 10, max_catch_up=5, cache_frames=3)
check(p.kind == "HOLD" and v == 119 and "10" in p.message, "backward too far holds: %r" % p.message)
st = fresh(cache_frames=0)
seq(st, range(1, 6), cache_frames=0)
check(st.cached_frames() == [], "cache_frames 0 disables the cache")
p, v = run(st, 3, cache_frames=0)
check(p.kind == "RESIM" and v == 102, "no cache: scrub back resimulates")

# --- copies --------------------------------------------------------------------------------
st = fresh()
P.copies = 0
seq(st, [1, 2, 3])
check(P.copies == 3, "one cache copy per committed frame (%d)" % P.copies)
p = st.advance(2)
cached = st.cache[2][0]
out = p.run(lambda: None, lambda s: s)
check(out is not cached and out.v == cached.v, "restore hands out a copy, the cache stays pristine")

# --- stream keys -----------------------------------------------------------------------------
fresh()
S.clear()
a = S.get_stream("n", "RENDER", (8, 8), False)
check(S.get_stream("n", "RENDER", (8, 8), False) is a, "same key, same stream")
others = [S.get_stream("n", "BACKDROP", (8, 8), False), S.get_stream("n", "RENDER", (9, 8), False),
          S.get_stream("n", "RENDER", (8, 8), True), S.get_stream("m", "RENDER", (8, 8), False)]
check(all(o is not a for o in others) and len({id(o) for o in others}) == 4,
      "kind, size, device and uid separate streams")
seq(a, [1, 2, 3])
check(run(others[0], 4)[0].kind == "RESET", "another stream does not see the first one's state")
seq(others[0], [1, 2])
check(run(a, 4)[0].kind == "STEP", "interleaved streams are independent")
S.clear("n")
check(S.stream_count() == 1 and S.get_stream("n", "RENDER", (8, 8), False) is not a,
      "clear(uid) drops that node's streams only")
S.clear()
check(S.stream_count() == 0, "clear() drops everything")
a = S.get_stream("n", "RENDER", (8, 8), False)
seq(a, [1, 2, 3])
a.reset()
check(run(a, 4)[0].kind == "RESET", "stream.reset() restarts")

# --- memory ------------------------------------------------------------------------------------
st = fresh(stream_bytes=1000)
seq(st, range(1, 30))
check(st.nbytes <= 1000 and len(st.cache) < 29, "per-stream byte budget evicts (%d bytes, %d frames)"
      % (st.nbytes, len(st.cache)))
check(max(st.cached_frames()) == 29, "newest frame survives eviction")

S.clear()
S.CONFIG.update(cache_frames=32, stream_bytes=1 << 40, global_bytes=1500, ttl=0)
a = S.get_stream("a", "RENDER", (8, 8), False)
b = S.get_stream("b", "RENDER", (8, 8), False)
seq(a, range(1, 6))
seq(b, range(1, 6))
check(S.total_bytes() <= 1500, "global budget holds (%d)" % S.total_bytes())
run(a, 5)
seq(b, range(6, 9))
check(S.total_bytes() <= 1500 and 8 in b.cached_frames(), "global LRU keeps the busy stream's frames")
check(len(a.cache) < 5, "global LRU evicted the idle stream's frames first (%s)" % a.cached_frames())
S.configure(global_bytes=10)
check(S.total_bytes() <= 400, "shrinking the budget evicts (%d bytes)" % S.total_bytes())

now = [0.0]
S._clock[0] = lambda: now[0]
S.clear()
S.CONFIG.update(global_bytes=1 << 40, ttl=100.0)
a = S.get_stream("a", "RENDER", (8, 8), False)
a.used = now[0]
now[0] = 50.0
S.get_stream("b", "RENDER", (8, 8), False)
now[0] = 120.0
S.get_stream("c", "RENDER", (8, 8), False)
check(S.stream_count() == 2 and S.streams_for("a") == [], "unused streams expire after the ttl")

print("=" * 60)
if FAILURES:
    raise AssertionError("%d failure(s):\n  %s" % (len(FAILURES), "\n  ".join(FAILURES)))
print("ALL PASSED")
