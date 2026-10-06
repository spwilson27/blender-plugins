#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Build dist/compositor_lab.zip from addons/compositor_lab (stdlib only).

The zip has a single top-level folder ``compositor_lab/``, which Blender's
Preferences > Add-ons > Install from Disk accepts.

Usage: tools/build_zip.py [OUTPUT.zip]
"""
import os
import sys
import zipfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
PACKAGE = os.path.join(ROOT, "addons", "compositor_lab")


def build_zip(output=None):
    output = output or os.path.join(ROOT, "dist", "compositor_lab.zip")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(PACKAGE):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith((".pyc", ".pyo")) or name == ".DS_Store":
                    continue
                full = os.path.join(dirpath, name)
                arc = os.path.join("compositor_lab", os.path.relpath(full, PACKAGE))
                zf.write(full, arc)
    return output


if __name__ == "__main__":
    print(build_zip(sys.argv[1] if len(sys.argv) > 1 else None))
