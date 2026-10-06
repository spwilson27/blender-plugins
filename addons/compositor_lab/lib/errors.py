# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Error log for node evaluation.

Blender catches an exception raised inside ``evaluate_cpu`` / ``evaluate_gpu``, shows it as the
node's info message and writes default outputs, so a broken shader does not fail anything by
itself. ``LabNode`` and the shader builders therefore ``record`` every such error here before
re-raising. Tests (``tests/lab/harness.py``) call ``take()`` after each render and fail on
anything left. Thread-safe (the compositor evaluates nodes on worker threads).
"""

import threading

_lock = threading.Lock()
_log = []


class LabShaderError(RuntimeError):
    """A compute shader failed to compile or link (the message names the shader)."""


def record(message):
    with _lock:
        _log.append(str(message))


def take():
    """Return and clear the recorded error messages."""
    with _lock:
        out = list(_log)
        _log.clear()
    return out


def clear():
    with _lock:
        _log.clear()
