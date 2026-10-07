# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Reloading the package (what reinstalling / re-enabling the add-on in a running Blender does)
must re-import every submodule from disk, not keep stale ones cached in sys.modules.

Run: Blender -b --factory-startup --python-exit-code 1 --python test_reload.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
                                "addons"))
import compositor_lab  # noqa: E402

compositor_lab.register()
stale_node = sys.modules["compositor_lab.lib.node"]
stale_node.STALE_MARKER = True
stale_cls = compositor_lab.REGISTERED["Simulate"][0]

compositor_lab.unregister()
importlib.reload(compositor_lab)
compositor_lab.register()

fresh_node = sys.modules["compositor_lab.lib.node"]
assert fresh_node is not stale_node, "lib.node was not re-imported"
assert not hasattr(fresh_node, "STALE_MARKER"), "stale lib.node still in use"
for section, classes in compositor_lab.REGISTERED.items():
    for cls in classes:
        assert cls.__module__ in sys.modules, cls
        assert sys.modules[cls.__module__] is not None
assert compositor_lab.REGISTERED["Simulate"][0] is not stale_cls, "node classes not reloaded"
total = sum(len(c) for c in compositor_lab.REGISTERED.values())
assert total >= 25, total
compositor_lab.unregister()
print("PASS: reload re-imports all %d node modules and the lib fresh" % total)
