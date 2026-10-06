# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
GUI test (F4): the evaluation kind a Python node sees in the real UI.

    Blender --factory-startup --python test_eval_kind_gui.py

Records (kind, is_animation_playing) seen by a node while the node editor backdrop, the rendered
3D viewport and a render with a UI evaluate it. Exits with 0 when all checks pass, 1 otherwise.
"""
import os
import sys
import traceback

import bpy
import numpy as np

SEEN = []


class CompositorNodeKindProbe(bpy.types.CompositorNode):
    bl_idname = "CompositorNodeKindProbe"
    bl_label = "Kind Probe"

    def init(self, context):
        self.inputs.new('NodeSocketFloat', "Value")
        self.outputs.new('NodeSocketColor', "Image")

    def evaluate_cpu(self, inputs, outputs, context):
        SEEN.append((context.kind, context.is_animation_playing, context.frame_start,
                     context.frame_end))
        np.asarray(outputs["Image"])[...] = (0.2, 0.4, 0.6, 1.0)


bpy.utils.register_class(CompositorNodeKindProbe)

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 64, 48
scene.render.resolution_percentage = 100
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.compositor_device = 'CPU'
scene.frame_start, scene.frame_end = 3, 40
scene.frame_current = 5

tree = bpy.data.node_groups.new("Compositing", 'CompositorNodeTree')
tree.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
scene.compositing_node_group = tree
probe = tree.nodes.new(CompositorNodeKindProbe.bl_idname)
n_out = tree.nodes.new('NodeGroupOutput')
n_view = tree.nodes.new('CompositorNodeViewer')
tree.links.new(probe.outputs[0], n_out.inputs[0])
tree.links.new(probe.outputs[0], n_view.inputs[0])

# Animate an input so that every frame change re-evaluates the compositor (during playback too).
for _frame, _value in ((1, 0.0), (40, 1.0)):
    probe.inputs["Value"].default_value = _value
    probe.inputs["Value"].keyframe_insert("default_value", frame=_frame)

results = []
TIMEOUT = 100  # ticks of 0.1s


def record(desc, ok, detail=""):
    results.append((desc, ok, detail))


def finish():
    failed = [r for r in results if not r[1]]
    print("=" * 60)
    for desc, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}: {desc} ({detail})")
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    sys.stdout.flush()
    os._exit(1 if failed or not results else 0)


def find_area(kind):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == kind:
                return window, area
    return None, None


def setup_node_editor():
    # Keep the 3D view for the viewport phases.
    area = max((a for a in bpy.context.screen.areas if a.type != 'VIEW_3D'),
               key=lambda a: a.width * a.height)
    area.ui_type = 'CompositorNodeTree'
    area.spaces.active.node_tree = tree
    area.spaces.active.show_backdrop = True
    return area


def setup_viewport():
    window, area = find_area('VIEW_3D')
    if area is None:
        return None
    space = area.spaces.active
    space.shading.type = 'RENDERED'
    if hasattr(space.shading, "use_compositor"):
        space.shading.use_compositor = 'ALWAYS'
    return area


def kinds(since):
    return {k for (k, _p, _s, _e) in SEEN[since:]}


# Phases, run from a timer. Each phase: start() -> optional, check() -> True when done.
phase = {"i": 0, "t": 0, "mark": 0, "started": False}


def start_backdrop():
    setup_node_editor()
    phase["mark"] = len(SEEN)


def check_backdrop():
    if "BACKDROP" in kinds(phase["mark"]):
        recs = [r for r in SEEN[phase["mark"]:] if r[0] == "BACKDROP"]
        record("BACKDROP kind in node editor", True)
        record("BACKDROP not playing when idle", all(not r[1] for r in recs), str(recs[-1]))
        record("BACKDROP frame range", all(r[2:] == (3, 40) for r in recs), str(recs[-1]))
        return True
    return None


def start_backdrop_play():
    phase["mark"] = len(SEEN)
    scene.frame_set(5)
    window, area = find_area('NODE_EDITOR')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.animation_play()


def check_backdrop_play():
    recs = [r for r in SEEN[phase["mark"]:] if r[0] == "BACKDROP"]
    if any(r[1] for r in recs):
        record("BACKDROP is_animation_playing True during playback", True)
        stop_playback()
        return True
    return None


def stop_playback():
    if bpy.context.screen.is_animation_playing:
        window, area = find_area('NODE_EDITOR')
        if area is None:
            window, area = bpy.context.window_manager.windows[0], bpy.context.screen.areas[0]
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.animation_cancel(restore_frame=False)


def start_viewport():
    area = setup_viewport()
    phase["mark"] = len(SEEN)
    phase["viewport"] = area is not None
    scene.frame_set(5)


def check_viewport():
    if not phase["viewport"]:
        record("VIEWPORT area available", False, "no 3D view")
        return True
    if "VIEWPORT" in kinds(phase["mark"]):
        recs = [r for r in SEEN[phase["mark"]:] if r[0] == "VIEWPORT"]
        record("VIEWPORT kind in rendered 3D view", True)
        record("VIEWPORT not playing when idle", not recs[-1][1], str(recs[-1]))
        record("VIEWPORT frame range", recs[-1][2:] == (3, 40), str(recs[-1]))
        return True
    return None


def start_viewport_play():
    phase["mark"] = len(SEEN)
    window, area = find_area('VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.animation_play()


def check_viewport_play():
    recs = [r for r in SEEN[phase["mark"]:] if r[0] == "VIEWPORT"]
    if any(r[1] for r in recs):
        record("VIEWPORT is_animation_playing True during playback", True)
        stop_playback()
        return True
    return None


def start_render():
    phase["mark"] = len(SEEN)
    phase["render_done"] = False
    # Plain (blocking) render from the UI: the render pipeline compositor.
    scene.render.filepath = os.path.join(os.environ.get("TMPDIR", "/tmp"), "eval_kind_gui_")
    bpy.ops.render.render('INVOKE_DEFAULT')


def check_render():
    if "RENDER" in kinds(phase["mark"]):
        recs = [r for r in SEEN[phase["mark"]:] if r[0] == "RENDER"]
        record("RENDER kind for render.render('INVOKE_DEFAULT')", True)
        record("RENDER not playing", all(not r[1] for r in recs), str(recs[-1]))
        return True
    return None


def start_render_blocking():
    phase["mark"] = len(SEEN)
    bpy.ops.render.render()


def check_render_blocking():
    ok = "RENDER" in kinds(phase["mark"])
    record("RENDER kind for plain render.render()", ok, str(SEEN[phase["mark"]:][-1:]))
    return True


PHASES = [
    ("backdrop", start_backdrop, check_backdrop),
    ("backdrop_play", start_backdrop_play, check_backdrop_play),
    ("viewport", start_viewport, check_viewport),
    ("viewport_play", start_viewport_play, check_viewport_play),
    ("render", start_render, check_render),
    ("render_blocking", start_render_blocking, check_render_blocking),
]


def tick():
    try:
        if phase["i"] >= len(PHASES):
            finish()
            return None
        name, start, check = PHASES[phase["i"]]
        if not phase["started"]:
            stop_playback()
            start()
            phase["started"] = True
            phase["t"] = 0
            return 0.2
        phase["t"] += 1
        if check() is True:
            phase["i"] += 1
            phase["started"] = False
            return 0.5
        if phase["t"] > TIMEOUT:
            record(f"phase {name}", False, f"timed out; seen={SEEN[phase['mark']:][-3:]}")
            phase["i"] += 1
            phase["started"] = False
        return 0.1
    except Exception:
        traceback.print_exc()
        record("test harness", False, "exception, see traceback")
        finish()


bpy.app.timers.register(tick, first_interval=1.5)
