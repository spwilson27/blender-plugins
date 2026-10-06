# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Reaction-Diffusion (Gray-Scott): two fields U, V advance a few iterations per frame.

    U' = U + dt (Du lap(U) - U V^2 + F (1 - U))
    V' = V + dt (Dv lap(V) + U V^2 - (F + k) V)              (both clipped to [0, 1])

``lap`` is the 9-point Laplacian (0.2 orthogonal, 0.05 diagonal, -1 centre; see ``lib/np_rd.py``).
Feed F / Kill k come from the sockets (the Preset enum writes them), optionally multiplied per
pixel by the ``Feed Map`` / ``Kill Map`` inputs.

The simulation runs at the output size divided by ``Scale`` (the output is upsampled bilinearly),
seeded from the ``Seed`` image (V = 1, U = 0.5 where its luminance is above 0.5) or, when that is
not linked, from a hashed noise of small blobs. Outputs: V, U and ``Color`` (V through a smoothstep
between two colours).

State: (U, V) at simulation size, numpy float32 on the CPU, one RG32F texture on the GPU. The first
frame, or any frame at or before the scene start, seeds the fields and runs ``Pre-roll`` frames
worth of iterations (the stateless fallback for a still render). See ``lib/state.py`` for repeats,
scrubbing and jumps.
"""

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, IntProperty

from ..lib import gpu as lab_gpu, np_rd, np_sampling
from ..lib.glsl import rd as glsl_rd
from ..lib.node import In, LabNode, Out, StatefulNode

MENU = "Simulate"

F32 = np.float32
MAX_ITERATIONS = 1000

_PRESET_ITEMS = [
    ('CORAL', "Coral", "Branching coral"),
    ('MITOSIS', "Mitosis", "Spots that split"),
    ('WORMS', "Worms", "Wandering worm-like strands"),
    ('SPOTS', "Spots", "Stable round spots"),
    ('MAZE', "Maze", "Labyrinth stripes"),
    ('BUBBLES', "Bubbles", "Rings and bubbles"),
    ('CUSTOM', "Custom", "Keep the Feed / Kill values as they are"),
]
_EDGE_ITEMS = [
    ('WRAP', "Wrap", "The field tiles"),
    ('CLAMP', "Clamp", "Closed borders (no flux through the edge)"),
]


def _preset_update(self, context):
    fk = np_rd.PRESETS.get(self.preset)
    if fk is not None and "Feed" in self.inputs and "Kill" in self.inputs:
        self.inputs["Feed"].default_value = fk[0]
        self.inputs["Kill"].default_value = fk[1]


class CompositorNodeLabReactionDiffusion(StatefulNode, LabNode, bpy.types.CompositorNode):
    '''Gray-Scott reaction-diffusion: patterns that grow frame by frame (spots, coral, mazes)'''
    bl_idname = "CompositorNodeLabReactionDiffusion"
    bl_label = "Reaction-Diffusion"

    SOCKETS = [
        In("Seed", "COLOR", (0.0, 0.0, 0.0, 1.0), hide_value=True),
        In("Feed", "FLOAT", 0.0545),
        In("Kill", "FLOAT", 0.062),
        In("Du", "FLOAT", 1.0),
        In("Dv", "FLOAT", 0.5),
        In("dt", "FLOAT", 1.0),
        In("Iterations per Frame", "INT", 20),
        In("Feed Map", "FLOAT", 1.0),
        In("Kill Map", "FLOAT", 1.0),
        Out("V", "FLOAT"),
        Out("U", "FLOAT"),
        Out("Color", "COLOR"),
    ]
    PROPS = ["preset", "scale", "edge_mode", None, "seed", "seed_density", "seed_noise", "preroll",
             None, "color_a", "color_b", "map_low", "map_high"]

    preset: EnumProperty(name="Preset", items=_PRESET_ITEMS, default='CORAL',
                         update=_preset_update,
                         description="Sets the Feed and Kill sockets (edit them to go custom)")
    scale: IntProperty(
        name="Scale", default=2, min=1, max=16,
        description="Simulation resolution divisor: the fields are simulated at the output size "
                    "divided by this and upsampled bilinearly (changing it restarts the state)")
    edge_mode: EnumProperty(name="Edges", items=_EDGE_ITEMS, default='WRAP')
    seed: IntProperty(name="Seed", default=1, min=0, max=2 ** 31 - 1,
                      description="Random seed of the noise seeding and of the initial V noise")
    seed_density: FloatProperty(
        name="Density", default=0.12, min=0.0, max=1.0,
        description="Without a Seed image: chance that each 8x8 simulation-pixel cell holds a blob")
    seed_noise: FloatProperty(
        name="Noise", default=0.02, min=0.0, max=1.0,
        description="Amplitude of the white noise added to V outside the seeded area")
    preroll: IntProperty(
        name="Pre-roll", default=0, min=0, max=2000,
        description="Frames worth of iterations (Iterations per Frame each) run when the state is "
                    "(re)started, so a still render shows a developed pattern")
    color_a: FloatVectorProperty(name="Color A", size=4, subtype='COLOR', min=0.0, soft_max=1.0,
                                 default=(0.02, 0.03, 0.09, 1.0),
                                 description="Colour where V is at or below Low")
    color_b: FloatVectorProperty(name="Color B", size=4, subtype='COLOR', min=0.0, soft_max=1.0,
                                 default=(1.0, 0.62, 0.2, 1.0),
                                 description="Colour where V is at or above High")
    map_low: FloatProperty(name="Low", default=0.05, min=0.0, max=1.0,
                           description="V mapped to Color A")
    map_high: FloatProperty(name="High", default=0.35, min=0.0, max=1.0,
                            description="V mapped to Color B")

    def draw_buttons(self, context, layout):
        self.draw_props(layout)
        self.draw_state_buttons(layout)

    # -- parameters ----------------------------------------------------------
    def _params(self, inputs, ctx):
        def fl(name, default, lo, hi):
            v = self.in_float(inputs, name, default)
            return F32(min(max(v, lo), hi)) if v == v else F32(default)

        w, h = ctx.size
        sw, sh = np_rd.sim_size(w, h, self.scale)
        lo, hi = float(self.map_low), float(self.map_high)
        return {
            "w": w, "h": h, "sw": sw, "sh": sh,
            "feed": fl("Feed", 0.0545, 0.0, 1.0), "kill": fl("Kill", 0.062, 0.0, 1.0),
            "du": fl("Du", 1.0, 0.0, 5.0), "dv": fl("Dv", 0.5, 0.0, 5.0),
            "dt": fl("dt", 1.0, 0.0, 5.0),
            "n": min(max(self.in_int(inputs, "Iterations per Frame", 20), 0), MAX_ITERATIONS),
            "edge": self.edge_mode,
            "seed": int(self.seed), "density": F32(self.seed_density),
            "noise": F32(self.seed_noise),
            "a": tuple(F32(c) for c in self.color_a), "b": tuple(F32(c) for c in self.color_b),
            "low": F32(lo), "inv": F32(1.0 / max(hi - lo, 1e-6)),
        }

    # -- CPU -----------------------------------------------------------------
    def _cpu_map(self, inputs, name, p, ctx):
        """None (unlinked: value scalar returned separately), else the map at simulation size."""
        if not self.in_is_image(inputs, name):
            return None, F32(self.in_float(inputs, name, 1.0))
        arr = self.in_image_array(inputs, name, ctx.shape, 1)
        arr = np.nan_to_num(np.asarray(arr, F32), nan=1.0, posinf=4.0, neginf=0.0)
        return np_rd.resample(arr, p["sw"], p["sh"], "CLAMP")[..., 0], None

    def cpu(self, inputs, outputs, ctx):
        outs = {k: self.out_array(outputs, k) for k in ("V", "U", "Color")}
        if all(o is None for o in outs.values()):
            return
        p = self._params(inputs, ctx)
        sw, sh = p["sw"], p["sh"]

        fmap, fval = self._cpu_map(inputs, "Feed Map", p, ctx)
        kmap, kval = self._cpu_map(inputs, "Kill Map", p, ctx)
        feed = p["feed"] * fval if fmap is None else (p["feed"] * fmap).astype(F32)
        kill = p["kill"] * kval if kmap is None else (p["kill"] * kmap).astype(F32)

        linked = self.in_is_image(inputs, "Seed")

        def init():
            if linked:
                img = np.asarray(self.in_image_array(inputs, "Seed", ctx.shape, 3), F32)
                img = np_rd.resample(img, sw, sh, "CLAMP")
                mask = np_rd.luminance(img) > F32(0.5)
            else:
                mask = np_rd.blob_mask(sw, sh, p["seed"], p["density"])
            u, v = np_rd.init_state(mask, p["noise"], p["seed"])
            n = p["n"] * self.preroll
            if n:
                u, v = np_rd.iterate(u, v, n, feed, kill, p["du"], p["dv"], p["dt"], p["edge"])
            return u, v

        def step(s):
            return np_rd.iterate(s[0], s[1], p["n"], feed, kill, p["du"], p["dv"], p["dt"],
                                 p["edge"])

        u, v = self.advance(ctx, signature=self.scale).run(init, step)
        uv = np.stack([u, v], axis=-1)
        if (sw, sh) != (p["w"], p["h"]):
            xs, ys = np_rd.resample_coords(p["w"], p["h"], sw, sh)
            uv = np_sampling.sample_bilinear(uv, xs, ys,
                                             'REPEAT' if p["edge"] == 'WRAP' else 'CLAMP')
        if outs["V"] is not None:
            outs["V"][..., 0] = uv[..., 1]
        if outs["U"] is not None:
            outs["U"][..., 0] = uv[..., 0]
        if outs["Color"] is not None:
            outs["Color"][...] = np_rd.colorize(uv[..., 1], p["a"], p["b"], p["low"], p["inv"])

    # -- GPU -----------------------------------------------------------------
    @staticmethod
    def _new_state(p):
        import gpu
        return gpu.types.GPUTexture((p["sw"], p["sh"]), format='RG32F')

    def _gpu_init(self, inputs, p):
        seed = inputs.get("Seed")
        has_seed = lab_gpu.is_texture(seed)
        tex = self._new_state(p)
        sratio = (F32(seed.width / p["sw"]), F32(seed.height / p["sh"])) if has_seed else (1.0, 1.0)
        lab_gpu.kernel(
            glsl_rd.INIT_BODY, {"State": tex},
            {"Seed": ("color", seed if has_seed else (0.0, 0.0, 0.0, 1.0))},
            uniforms={"d_has_seed": ("int", int(has_seed)),
                      "d_sratio": ("vec2", tuple(float(v) for v in sratio)),
                      "d_seed": ("int", p["seed"]), "d_density": ("float", float(p["density"])),
                      "d_noise": ("float", float(p["noise"]))},
            libs=("hash",), sampling=True)
        return tex

    def _gpu_iterate(self, state, n, p, maps):
        """n iterations from ``state`` into a fresh texture (``state`` is not modified)."""
        if n <= 0:
            from ..lib import state as lab_state
            return lab_state.gpu_copy(state)
        edge = 1 if p["edge"] == 'WRAP' else 0
        cur = state
        for i in range(n):
            dst = self._new_state(p) if i == n - 1 else lab_gpu.scratch(
                p["sw"], p["sh"], "rd_ping" if i % 2 == 0 else "rd_pong", "RG32F")
            lab_gpu.kernel(
                glsl_rd.STEP_BODY, {"State": dst},
                {"Prev": ("color", cur), "FMap": ("color", maps["f"]),
                 "KMap": ("color", maps["k"])},
                uniforms={"d_edge": ("int", edge), "d_feed": ("float", float(maps["feed"])),
                          "d_kill": ("float", float(maps["kill"])),
                          "d_du": ("float", float(p["du"])), "d_dv": ("float", float(p["dv"])),
                          "d_dt": ("float", float(p["dt"])),
                          "d_has_f": ("int", int(maps["has_f"])),
                          "d_has_k": ("int", int(maps["has_k"])),
                          "d_fratio": ("vec2", maps["fratio"]),
                          "d_kratio": ("vec2", maps["kratio"])},
                sampling=True)
            cur = dst
        return cur

    def gpu(self, inputs, outputs, ctx):
        dst = {k: self.out_texture(outputs, k) for k in ("V", "U", "Color")}
        if all(t is None for t in dst.values()):
            return
        p = self._params(inputs, ctx)

        maps = {"has_f": False, "has_k": False, "fratio": (1.0, 1.0), "kratio": (1.0, 1.0), "feed": p["feed"],
                "kill": p["kill"], "f": 1.0, "k": 1.0}
        for key, name, par, has, rat in (("f", "Feed Map", "feed", "has_f", "fratio"),
                                         ("k", "Kill Map", "kill", "has_k", "kratio")):
            m = inputs.get(name)
            if lab_gpu.is_texture(m):
                maps[has] = True
                maps[key] = m
                maps[rat] = (float(F32(m.width / p["sw"])), float(F32(m.height / p["sh"])))
            else:
                maps[par] = p[par] * F32(self.in_float(inputs, name, 1.0))

        def init():
            s = self._gpu_init(inputs, p)
            n = p["n"] * self.preroll
            return self._gpu_iterate(s, n, p, maps) if n else s

        final = self.advance(ctx, signature=self.scale).run(init, lambda s: self._gpu_iterate(s, p["n"], p, maps))
        up = (F32(p["sw"] / p["w"]), F32(p["sh"] / p["h"]))
        lab_gpu.kernel(
            glsl_rd.output_body(dst["V"] is not None, dst["U"] is not None,
                                dst["Color"] is not None),
            {"V": dst["V"], "U": dst["U"], "Color": dst["Color"]},
            {"State": ("color", final)},
            uniforms={"d_up": ("vec2", tuple(float(v) for v in up)),
                      "d_edge": ("int", 1 if p["edge"] == 'WRAP' else 0),
                      "d_a": ("vec4", tuple(float(c) for c in p["a"])),
                      "d_b": ("vec4", tuple(float(c) for c in p["b"])),
                      "d_low": ("float", float(p["low"])), "d_inv": ("float", float(p["inv"]))},
            sampling=True)


NODE_CLASSES = [CompositorNodeLabReactionDiffusion]
