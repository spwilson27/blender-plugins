# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Cellular Automata: Life-like and Generations rules on a grid of cells, advanced once per frame.

Each cell has a state ``0 .. C-1`` (0 dead, 1 alive, 2 .. C-1 dying / refractory, the Generations
extension of B/S rules) and an age. The rule is a B/S string (``B3/S23``, Generations:
``B2/S345/4``) or a preset. One evaluation step is: OR the *Inject* mask into the alive cells, then
run *Generations per Frame* generations. The grid has ``ceil(size / Cell Size)`` cells and covers
the image; neighbourhood Moore (8) or von Neumann (4); edges wrap or are dead. See ``lib/np_ca.py``
for the exact semantics (shared by the numpy and GLSL implementations, integer logic: the CPU and
GPU are bit-identical).

State: ``(state uint8, age uint16)`` arrays on the CPU, one RG32F texture (r state, g age) on the
GPU. The first frame (and any frame at or before the scene start) seeds the grid: from the *Seed*
image (alive where luminance > Threshold, sampled at the cell centres) or, with *Seed* unlinked,
random cells of the given Density (hash noise of the cell and Random Seed), optionally after
*Pre-roll* generations (the stateless fallback: a still render shows an evolved pattern).
Outputs: *Cells* (alive 1, dying shades, dead 0), *Age* (age / Age Range, clamped) and *Color*
(alive cells: age through Color A / B / C; dying cells: Color C, fading to the background with the
square of their shade, so the fade reads evenly after the display transform).
"""

import bpy
import numpy as np
from bpy.props import EnumProperty, FloatVectorProperty, IntProperty, StringProperty

from ..lib import gpu as lab_gpu, np_ca
from ..lib.node import In, LabNode, Out, StatefulNode

MENU = "Simulate"

F32 = np.float32
LUMA = (0.2126, 0.7152, 0.0722)

_PRESET_ITEMS = [(i, "%s  (%s)" % (label, rule) if rule else label, desc)
                 for i, label, rule, desc in np_ca.PRESETS]
_NEIGHBOURHOOD = [('MOORE', "Moore", "The 8 surrounding cells"),
                  ('VON_NEUMANN', "von Neumann", "The 4 edge-adjacent cells")]
_EDGES = [('WRAP', "Wrap", "The grid is a torus"), ('DEAD', "Dead", "Outside the grid is dead")]

_GEN_BODY = """
    out_State = vec4(lab_ca_step(texel, res, d_birth, d_survive, d_states, d_moore, d_wrap), 0.0, 0.0);
"""

_INIT_BODY = """
    float alive;
    if (d_random != 0) {
      alive = (lab_rand(texel, uint(d_seed)) < d_density) ? 1.0 : 0.0;
    }
    else {
      vec4 c = in_Seed(texel * d_cell + ivec2(d_cell / 2));
      alive = (dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)) > d_threshold) ? 1.0 : 0.0;
    }
    out_State = vec4(alive, 0.0, 0.0, 0.0);
"""

_INJECT_BODY = """
    vec4 c = in_Prev(texel);
    vec4 j = in_Inject(texel * d_cell + ivec2(d_cell / 2));
    bool paint = dot(j.rgb, vec3(0.2126, 0.7152, 0.0722)) > 0.5;
    if (paint && int(c.r + 0.5) != 1) {
      out_State = vec4(1.0, 0.0, 0.0, 0.0);
    }
    else {
      out_State = vec4(c.r, c.g, 0.0, 0.0);
    }
"""

_VIEW_HEAD = """
    vec4 c = in_State(texel / d_cell);
    int s = int(c.r + 0.5);
    float shade = (s >= 1) ? float(d_states - s) * d_inv_states : 0.0;
    float t = min(c.g * d_inv_age, 1.0);
"""
_VIEW_COLOR = """
    float lo = min(t * 2.0, 1.0);
    float hi = max(t * 2.0 - 1.0, 0.0);
    vec3 grad = d_ca + (d_cb - d_ca) * lo + (d_cc - d_cb) * hi;
    if (s >= 2) {
      grad = d_cc;
    }
    out_Color = vec4(d_bg + (grad - d_bg) * (shade * shade), 1.0);
"""


def _luma(v):
    return F32(v[0]) * F32(LUMA[0]) + F32(v[1]) * F32(LUMA[1]) + F32(v[2]) * F32(LUMA[2])


def _update_preset(self, context):
    if self.preset != 'CUSTOM':
        self.rule = np_ca.PRESET_RULES[self.preset]


class CompositorNodeLabCellularAutomata(StatefulNode, LabNode, bpy.types.CompositorNode):
    '''Cellular automata (Life, Generations, B/S rules) stepped once per frame, with age output'''
    bl_idname = "CompositorNodeLabCellularAutomata"
    bl_label = "Cellular Automata"

    SOCKETS = [
        In("Seed", "COLOR", (0.0, 0.0, 0.0, 1.0)),
        In("Inject", "FLOAT", 0.0, min=0.0, max=1.0),
        In("Threshold", "FACTOR", 0.5),
        In("Density", "FACTOR", 0.35, clamp=True),
        In("Random Seed", "INT", 0, min=0, max=1000),
        In("Generations per Frame", "INT", 1, min=0, max=16, clamp=(0, 1024)),
        In("Cell Size", "INT", 2, min=1, max=32, clamp=(1, 256)),
        Out("Cells", "FLOAT"),
        Out("Age", "FLOAT"),
        Out("Color", "COLOR"),
    ]
    PROPS = ["preset", "rule", "neighbourhood", "edges", "preroll", "age_range", None,
             "color_a", "color_b", "color_c", "background"]

    preset: EnumProperty(name="Rule", items=_PRESET_ITEMS, default='LIFE', update=_update_preset)
    rule: StringProperty(
        name="B/S", default="B3/S23",
        description="Birth / Survival neighbour counts, with an optional third part for the "
                    "number of states (Generations): B3/S23, B2/S345/4. Used by the Custom preset")
    neighbourhood: EnumProperty(name="Neighbourhood", items=_NEIGHBOURHOOD, default='MOORE')
    edges: EnumProperty(name="Edges", items=_EDGES, default='WRAP')
    preroll: IntProperty(
        name="Pre-roll", default=0, min=0, soft_max=256, max=4096,
        description="Generations run on the seed when the state is (re)started, so a still "
                    "render shows an evolved pattern (0: the first frame is the seed)")
    age_range: IntProperty(
        name="Age Range", default=20, min=1, soft_max=1000, max=65535,
        description="Age (generations since birth) that maps to the end of the Age output and "
                    "of the colour gradient")
    color_a: FloatVectorProperty(name="Color A", subtype='COLOR', size=3, min=0.0, soft_max=1.0,
                                 default=(1.0, 0.95, 0.65), description="Newborn cells")
    color_b: FloatVectorProperty(name="Color B", subtype='COLOR', size=3, min=0.0, soft_max=1.0,
                                 default=(0.95, 0.28, 0.08), description="Middle-aged cells")
    color_c: FloatVectorProperty(name="Color C", subtype='COLOR', size=3, min=0.0, soft_max=1.0,
                                 default=(0.10, 0.12, 0.60), description="Old cells")
    background: FloatVectorProperty(name="Background", subtype='COLOR', size=3, min=0.0,
                                    soft_max=1.0, default=(0.004, 0.004, 0.012),
                                    description="Dead cells (dying cells fade to it)")

    def draw_buttons(self, context, layout):
        layout.prop(self, "preset", text="")
        if self.preset == 'CUSTOM':
            row = layout.row()
            try:
                np_ca.parse_rule(self.rule)
            except ValueError:
                row.alert = True
            row.prop(self, "rule", text="")
        else:
            layout.label(text=np_ca.PRESET_RULES[self.preset])
        for name in ("neighbourhood", "edges", "preroll", "age_range"):
            layout.prop(self, name)
        col = layout.column(align=True)
        for name in ("color_a", "color_b", "color_c", "background"):
            col.prop(self, name)
        self.draw_state_buttons(layout)

    # -- parameters ----------------------------------------------------------
    def rule_text(self):
        return np_ca.preset_rule(self.preset, self.rule)

    def _params(self, inputs, ctx):
        birth, survive, states = np_ca.parse_rule(self.rule_text())   # ValueError -> node message
        bmask, smask = np_ca.rule_masks(birth, survive)
        cell = min(max(self.in_int(inputs, "Cell Size", 2), 1), 256)
        return {
            "birth": birth, "survive": survive, "states": states,
            "bmask": bmask, "smask": smask,
            "moore": self.neighbourhood == 'MOORE', "wrap": self.edges == 'WRAP',
            "gens": min(max(self.in_int(inputs, "Generations per Frame", 1), 0), 1024),
            "cell": cell, "grid": np_ca.grid_size(ctx.size, cell),
            "random": not self.in_is_image(inputs, "Seed"),
            "threshold": F32(self.in_float(inputs, "Threshold", 0.5)),
            "density": F32(self.in_float(inputs, "Density", 0.35)),
            "seed": int(self.in_int(inputs, "Random Seed", 0)),
            "inv_states": F32(1.0 / (states - 1)),
            "inv_age": F32(1.0 / max(self.age_range, 1)),
            "colors": [tuple(float(v) for v in getattr(self, n))
                       for n in ("color_a", "color_b", "color_c", "background")],
        }

    def _plan(self, ctx, grid):
        """The plan; the state restarts when the grid size changes (Cell Size, output size)."""
        return self.advance(ctx, signature=tuple(grid))

    # -- CPU -----------------------------------------------------------------
    def _cpu_mask(self, inputs, ctx, p):
        """Inject mask on the grid (bool), or None."""
        v = inputs.get("Inject")
        if v is None:
            return None
        if not self.in_is_image(inputs, "Inject"):
            return None if not _luma(self.in_color(inputs, "Inject", (0, 0, 0, 1))) > F32(0.5) \
                else np.ones(p["grid"][::-1], bool)
        img = self.in_image_array(inputs, "Inject", ctx.shape, 4)
        return np_ca.luminance(np_ca.sample_centres(img, p["grid"], p["cell"])) > F32(0.5)

    def _frame_cpu(self, st, mask, p, gens):
        s, age = st
        if mask is not None:
            s, age = np_ca.inject(s, age, mask)
        for _ in range(gens):
            s, age = np_ca.step(s, age, p["birth"], p["survive"], p["states"], p["moore"],
                                p["wrap"])
        return s, age

    def cpu(self, inputs, outputs, ctx):
        outs = {n: self.out_array(outputs, n) for n in ("Cells", "Age", "Color")}
        if all(o is None for o in outs.values()):
            return
        p = self._params(inputs, ctx)
        mask = self._cpu_mask(inputs, ctx, p)

        def init():
            if p["random"]:
                alive = np_ca.random_cells(p["grid"], p["seed"], p["density"])
            else:
                img = self.in_image_array(inputs, "Seed", ctx.shape, 4)
                alive = np_ca.luminance(np_ca.sample_centres(img, p["grid"], p["cell"])) \
                    > p["threshold"]
            st = (alive.astype(np.uint8), np.zeros(alive.shape, np.uint16))
            if mask is not None:
                st = np_ca.inject(st[0], st[1], mask)
            return self._frame_cpu(st, None, p, self.preroll)

        plan = self._plan(ctx, p["grid"])
        s, age = plan.run(init, lambda st: self._frame_cpu(st, mask, p, p["gens"]))
        self._view_cpu(outs, s, age, p, ctx)

    def _view_cpu(self, outs, s, age, p, ctx):
        h, w = ctx.shape
        c = p["cell"]

        def up(a):
            a = np.repeat(np.repeat(a, c, axis=0), c, axis=1) if c > 1 else a
            return a[:h, :w]

        shade = np_ca.shade(s, p["states"])
        t = np.minimum(age.astype(F32) * p["inv_age"], F32(1.0))
        if outs["Cells"] is not None:
            outs["Cells"][..., 0] = up(shade)
        if outs["Age"] is not None:
            outs["Age"][..., 0] = up(t)
        if outs["Color"] is not None:
            a, b, cc, bg = (np.asarray(x, F32) for x in p["colors"])
            grad = np_ca.gradient(t, a, b, cc)
            grad = np.where((s >= 2)[..., None], cc, grad).astype(F32)
            col = (bg + (grad - bg) * (shade * shade)[..., None]).astype(F32)
            out = outs["Color"]
            out[..., :3] = np.stack([up(col[..., i]) for i in range(3)], axis=-1)
            out[..., 3] = 1.0

    # -- GPU -----------------------------------------------------------------
    @staticmethod
    def _new_tex(grid):
        import gpu
        return gpu.types.GPUTexture(grid, format='RG32F')

    def _passes(self, src, passes, grid):
        """Run ``passes`` (functions (src, dst)) ping-ponging through scratch textures; the last
        writes a fresh texture, which is returned."""
        for i, fn in enumerate(passes):
            last = i == len(passes) - 1
            dst = self._new_tex(grid) if last else lab_gpu.scratch(grid[0], grid[1],
                                                                    "ca_%d" % (i % 2), "RG32F")
            fn(src, dst)
            src = dst
        return src

    def _gen_pass(self, p):
        uniforms = {"d_birth": ("int", p["bmask"]), "d_survive": ("int", p["smask"]),
                    "d_states": ("int", p["states"]), "d_moore": ("int", int(p["moore"])),
                    "d_wrap": ("int", int(p["wrap"]))}

        def run(src, dst):
            lab_gpu.kernel(_GEN_BODY, {"State": dst}, {"Prev": ("color", src)},
                           uniforms=uniforms, libs=("ca",))
        return run

    def _inject_pass(self, inject, p):
        def run(src, dst):
            lab_gpu.kernel(_INJECT_BODY, {"State": dst},
                           {"Prev": ("color", src), "Inject": ("color", inject)},
                           uniforms={"d_cell": ("int", p["cell"])})
        return run

    def _inject_value(self, inputs):
        """The Inject input for the GPU (texture / single value), or None if it paints nothing."""
        if self.in_is_image(inputs, "Inject"):
            return inputs["Inject"]
        v = self.in_color(inputs, "Inject", (0.0, 0.0, 0.0, 1.0))
        return v if _luma(v) > F32(0.5) else None

    def gpu(self, inputs, outputs, ctx):
        outs = {n: self.out_texture(outputs, n) for n in ("Cells", "Age", "Color")}
        if all(o is None for o in outs.values()):
            return
        p = self._params(inputs, ctx)
        grid = p["grid"]
        inject = self._inject_value(inputs)
        gen = self._gen_pass(p)
        inj = None if inject is None else self._inject_pass(inject, p)
        seed = self.in_texture_or_value(inputs, "Seed", (0.0, 0.0, 0.0, 1.0))

        def init():
            s = self._new_tex(grid) if not (inj or self.preroll) else lab_gpu.scratch(
                grid[0], grid[1], "ca_init", "RG32F")
            lab_gpu.kernel(_INIT_BODY, {"State": s}, {"Seed": ("color", seed)},
                           uniforms={"d_random": ("int", int(p["random"])),
                                     "d_seed": ("int", p["seed"]),
                                     "d_density": ("float", float(p["density"])),
                                     "d_threshold": ("float", float(p["threshold"])),
                                     "d_cell": ("int", p["cell"])},
                           libs=("hash",))
            passes = ([inj] if inj else []) + [gen] * self.preroll
            return self._passes(s, passes, grid) if passes else s

        def step(st):
            passes = ([inj] if inj else []) + [gen] * p["gens"]
            if not passes:
                passes = [lambda src, dst: lab_gpu.pointwise(
                    "    out_State = in_Src(texel);\n", {"State": dst},
                    inputs={"Src": ("color", src)})]
            return self._passes(st, passes, grid)

        plan = self._plan(ctx, grid)
        final = plan.run(init, step)
        body = _VIEW_HEAD
        if outs["Cells"] is not None:
            body += "    out_Cells = vec4(shade, 0.0, 0.0, 1.0);\n"
        if outs["Age"] is not None:
            body += "    out_Age = vec4(t, 0.0, 0.0, 1.0);\n"
        if outs["Color"] is not None:
            body += _VIEW_COLOR
        a, b, cc, bg = p["colors"]
        lab_gpu.kernel(
            body, outs, {"State": ("color", final)},
            uniforms={"d_cell": ("int", p["cell"]), "d_states": ("int", p["states"]),
                      "d_inv_states": ("float", float(p["inv_states"])),
                      "d_inv_age": ("float", float(p["inv_age"])),
                      "d_ca": ("vec3", a), "d_cb": ("vec3", b), "d_cc": ("vec3", cc),
                      "d_bg": ("vec3", bg)})


NODE_CLASSES = [CompositorNodeLabCellularAutomata]
