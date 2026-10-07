# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Compositor Lab",
    "author": "Sean Wilson",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "Compositor > Add > Lab",
    "description": "Experimental compositor nodes evaluated in Python (CPU numpy / GPU compute)",
    "category": "Compositing",
}

import importlib
import pkgutil
import sys
import traceback

import bpy

# Reinstalling or re-enabling the add-on in a running Blender reloads this package but not its
# submodules, which would mix new modules with old ones still cached (e.g. a new node calling a
# helper the old lib lacks). Drop them so everything is imported fresh from disk.
for _name in [n for n in sys.modules if n.startswith(__name__ + ".")]:
    del sys.modules[_name]

SECTIONS = ("Filter", "Generate", "Simulate", "Utility")

# Registered node classes by section, and per-module load errors (for diagnostics / tests).
REGISTERED = {s: [] for s in SECTIONS}
ERRORS = {}

_menu_class = None
_menu_targets = []


def _log(msg):
    print("compositor_lab: " + msg)


def _discover():
    """Import nodes/*.py; yield (module_name, module). A broken module is logged and skipped."""
    from . import nodes

    for info in sorted(pkgutil.iter_modules(nodes.__path__), key=lambda i: i.name):
        name = info.name
        try:
            mod = importlib.import_module("%s.nodes.%s" % (__name__, name))
            if not hasattr(mod, "NODE_CLASSES"):
                raise AttributeError("module has no NODE_CLASSES")
            yield name, mod
        except Exception:
            ERRORS[name] = traceback.format_exc()
            _log("failed to import node module %r:\n%s" % (name, ERRORS[name]))


# ---------------------------------------------------------------------------
# Add menu: Add > Lab > Filter / Generate / Simulate / Utility
# ---------------------------------------------------------------------------

def _make_menu_class():
    try:
        from bl_ui import node_add_menu
        base = node_add_menu.AddNodeMenu
    except Exception:
        base = bpy.types.Menu

    class NODE_MT_compositor_lab(base):
        bl_idname = "NODE_MT_compositor_lab"
        bl_label = "Lab"

        def draw(self, context):
            layout = self.layout
            first = True
            for section in SECTIONS:
                classes = REGISTERED[section]
                if not classes:
                    continue
                if not first:
                    layout.separator()
                first = False
                layout.label(text=section)
                for cls in classes:
                    _add_item(self, layout, cls)

    return NODE_MT_compositor_lab


def _add_item(menu, layout, cls):
    try:
        menu.node_operator(layout, cls.bl_idname)  # AddNodeMenu (Blender 5.x)
    except AttributeError:
        props = layout.operator("node.add_node", text=cls.bl_label)
        props.type = cls.bl_idname
        if hasattr(props, "use_transform"):
            props.use_transform = True


def _draw_lab_submenu(self, context):
    self.layout.separator()
    self.layout.menu("NODE_MT_compositor_lab")


def _register_menu():
    global _menu_class
    _menu_class = _make_menu_class()
    bpy.utils.register_class(_menu_class)
    for name in ("NODE_MT_compositor_node_add_all", "NODE_MT_add"):
        menu = getattr(bpy.types, name, None)
        if menu is None:
            continue
        try:
            menu.append(_draw_lab_submenu)
            _menu_targets.append(menu)
            return
        except Exception as ex:
            _log("could not extend %s: %s" % (name, ex))


def _unregister_menu():
    global _menu_class
    for menu in _menu_targets:
        try:
            menu.remove(_draw_lab_submenu)
        except Exception:
            pass
    _menu_targets.clear()
    if _menu_class is not None:
        try:
            bpy.utils.unregister_class(_menu_class)
        except Exception:
            pass
        _menu_class = None


# ---------------------------------------------------------------------------
# Socket ranges: nodes saved with older ranges pick up the current SOCKETS spec on load.
# ---------------------------------------------------------------------------

def refresh_socket_ranges(*_args):
    """Re-apply the slider ranges of the SOCKETS specs to every Lab node in every node tree."""
    from .lib.node import LabNode, apply_ranges

    try:
        trees = list(bpy.data.node_groups)
    except AttributeError:      # bpy.data is restricted while an add-on is being enabled
        return
    for scene in bpy.data.scenes:
        tree = getattr(scene, "node_tree", None)
        if tree is not None and tree not in trees:
            trees.append(tree)
    for tree in trees:
        for node in tree.nodes:
            if isinstance(node, LabNode):
                try:
                    apply_ranges(node)
                except Exception:
                    _log("could not update socket ranges of %r:\n%s"
                         % (node.name, traceback.format_exc()))


def _load_post_ranges(*_args):
    refresh_socket_ranges()


_load_post_ranges = bpy.app.handlers.persistent(_load_post_ranges)


def _register_range_handler():
    handlers = bpy.app.handlers.load_post
    if not any(getattr(h, "__name__", "") == "_load_post_ranges"
               and getattr(h, "__module__", "") == __name__ for h in handlers):
        handlers.append(_load_post_ranges)


def _unregister_range_handler():
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_load_post_ranges" and \
                getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.load_post.remove(h)


# ---------------------------------------------------------------------------

def register():
    ERRORS.clear()
    for section in REGISTERED.values():
        section.clear()
    for name, mod in _discover():
        section = getattr(mod, "MENU", "Utility")
        if section not in REGISTERED:
            section = "Utility"
        for cls in mod.NODE_CLASSES:
            try:
                bpy.utils.register_class(cls)
                REGISTERED[section].append(cls)
            except Exception:
                ERRORS["%s.%s" % (name, cls.__name__)] = traceback.format_exc()
                _log("failed to register %s:\n%s" % (cls.__name__, traceback.format_exc()))
    try:
        from .lib import state as lab_state
        lab_state.register()
    except Exception:
        ERRORS["lib.state"] = traceback.format_exc()
        _log("state registration failed:\n" + traceback.format_exc())
    try:
        _register_menu()
    except Exception:
        _log("menu registration failed:\n" + traceback.format_exc())
    _register_range_handler()
    try:
        refresh_socket_ranges()
    except Exception:
        _log("socket range update failed:\n" + traceback.format_exc())


def unregister():
    _unregister_range_handler()
    _unregister_menu()
    for section in REGISTERED.values():
        for cls in reversed(section):
            try:
                bpy.utils.unregister_class(cls)
            except Exception:
                _log("failed to unregister %s" % cls.__name__)
        section.clear()
    try:
        from .lib import state as lab_state
        lab_state.unregister()
    except Exception:
        pass
    try:
        from .lib import gpu as lab_gpu
        lab_gpu.clear_cache()
    except Exception:
        pass


if __name__ == "__main__":
    register()
