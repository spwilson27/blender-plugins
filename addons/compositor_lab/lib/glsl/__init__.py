# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""GLSL snippets as Python strings. ``resolve("noise", "blend")`` returns the concatenated
source of those modules and their dependencies, each once, dependencies first."""

import importlib

NAMES = ("hash", "color", "noise", "blend", "exact")


def _module(name):
    return importlib.import_module(__name__ + "." + name)


def resolve(*names):
    seen = []

    def visit(n):
        if n in seen:
            return
        mod = _module(n)
        for d in mod.DEPS:
            visit(d)
        seen.append(n)

    for n in names:
        visit(n)
    return "\n".join(_module(n).SOURCE for n in seen)
