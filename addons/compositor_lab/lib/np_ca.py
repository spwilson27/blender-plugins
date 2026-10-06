# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Cellular automata (Life-like and Generations rules), numpy side. GLSL twin: ``glsl/ca.py``.

A grid cell has a state ``s`` in ``0 .. C-1`` (``uint8``) and an age (``uint16``, saturating):

* ``s == 0``  dead: becomes alive (state 1) if the number of alive neighbours is in B;
* ``s == 1``  alive: stays alive if that number is in S, otherwise starts dying (state 2), or
  dies (state 0) when ``C == 2``;
* ``s >= 2``  dying (refractory): ``s + 1`` each generation, back to 0 after ``C - 1``.

Only state-1 cells count as neighbours. Age is the number of generations since the cell was
born (0 at birth, +1 per generation while ``s != 0``, 0 while dead). Everything is integer logic,
so the CPU and GPU agree exactly.
"""

import re

import numpy as np

F32 = np.float32
AGE_MAX = 65535

PRESETS = [
    # (id, label, rule, description)
    ('LIFE', "Life", "B3/S23", "Conway's Game of Life"),
    ('HIGHLIFE', "HighLife", "B36/S23", "Life with a replicator"),
    ('SEEDS', "Seeds", "B2/S", "Every cell dies after one generation"),
    ('DAY_NIGHT', "Day & Night", "B3678/S34678", "Symmetric under swapping alive and dead"),
    ('BRIANS_BRAIN', "Brian's Brain", "B2/S/3", "Three states: firing, refractory, ready"),
    ('STAR_WARS', "Star Wars", "B2/S345/4", "Generations rule with gliders"),
    ('MAZE', "Maze", "B3/S12345", "Grows labyrinths"),
    ('CUSTOM', "Custom", "", "Use the rule string"),
]
PRESET_RULES = {p[0]: p[2] for p in PRESETS}

_RULE_RE = re.compile(r"^\s*([BbSs])\s*([0-8]*)\s*/\s*([BbSs])\s*([0-8]*)\s*(?:/\s*[CcGg]?\s*(\d+))?\s*$")


def parse_rule(text):
    """``"B3/S23"``, ``"B2/S/3"`` (Generations: third part = number of states C) or ``"B36/S23/C4"``
    -> ``(birth, survive, C)`` with sets of neighbour counts 0..8. The two halves may come in either
    order, case and spaces do not matter, ``B`` / ``S`` may be empty. Raises ``ValueError``."""
    m = _RULE_RE.match(str(text))
    if m is None:
        raise ValueError("Invalid rule %r: expected B<digits>/S<digits>[/<states>], e.g. B3/S23 "
                         "or B2/S345/4" % (text,))
    k1, d1, k2, d2, c = m.groups()
    if k1.upper() == k2.upper():
        raise ValueError("Invalid rule %r: need one B and one S part" % (text,))
    parts = {k1.upper(): d1, k2.upper(): d2}
    birth, survive = (set(int(ch) for ch in parts[k]) for k in ("B", "S"))
    states = 2 if c is None else int(c)
    if not 2 <= states <= 256:
        raise ValueError("Invalid rule %r: the number of states must be 2..256 (got %d)"
                         % (text, states))
    return birth, survive, states


def rule_masks(birth, survive):
    """Bit masks (bit n set: n neighbours) for the GPU."""
    return sum(1 << n for n in birth), sum(1 << n for n in survive)


def preset_rule(preset, custom):
    return custom if preset == 'CUSTOM' else PRESET_RULES[preset]


def grid_size(size, cell):
    """Cells of a ``cell`` pixel grid covering ``size`` (w, h): ceil division."""
    c = max(int(cell), 1)
    return (-(-int(size[0]) // c), -(-int(size[1]) // c))


def alive_neighbours(alive, moore=True, wrap=True):
    """Number of alive (uint8 0/1) neighbours of every cell."""
    if wrap:
        p = np.pad(alive, 1, mode="wrap")
    else:
        p = np.pad(alive, 1, mode="constant")
    h, w = alive.shape
    n = p[0:h, 1:w + 1] + p[2:h + 2, 1:w + 1] + p[1:h + 1, 0:w] + p[1:h + 1, 2:w + 2]
    if moore:
        n = n + p[0:h, 0:w] + p[0:h, 2:w + 2] + p[2:h + 2, 0:w] + p[2:h + 2, 2:w + 2]
    return n


def step(s, age, birth, survive, states, moore=True, wrap=True):
    """One generation. ``s`` uint8 (h, w), ``age`` uint16 (h, w); returns new arrays."""
    n = alive_neighbours((s == 1).astype(np.uint8), moore, wrap)
    b = np.zeros(9, bool)
    sv = np.zeros(9, bool)
    b[list(birth)] = True
    sv[list(survive)] = True
    fade = np.uint8(2 if states > 2 else 0)
    ns = np.where(s == 0, b[n].astype(np.uint8),
                  np.where(s == 1, np.where(sv[n], np.uint8(1), fade),
                           np.where(s + 1 >= states, 0, s + 1).astype(np.uint8))).astype(np.uint8)
    nage = np.where(ns == 0, 0,
                    np.where(s == 0, 0, np.minimum(age.astype(np.uint32) + 1, AGE_MAX)))
    return ns, nage.astype(np.uint16)


def inject(s, age, mask):
    """OR ``mask`` (bool) into the alive cells: cells not already alive become alive (age 0)."""
    new = mask & (s != 1)
    return (np.where(new, 1, s).astype(np.uint8), np.where(new, 0, age).astype(np.uint16))


def luminance(rgba):
    """Rec. 709 luminance of the rgb channels, float32 (the GPU twin uses a dot product)."""
    return (rgba[..., 0] * F32(0.2126) + rgba[..., 1] * F32(0.7152)
            + rgba[..., 2] * F32(0.0722)).astype(F32)


def sample_centres(img, grid, cell):
    """Pixels at the cell centres (clamped to the image): ``img`` (H, W, ...) -> (gh, gw, ...)."""
    gw, gh = grid
    ys = np.minimum(np.arange(gh) * cell + cell // 2, img.shape[0] - 1)
    xs = np.minimum(np.arange(gw) * cell + cell // 2, img.shape[1] - 1)
    return img[ys[:, None], xs[None, :]]


def random_cells(grid, seed, density):
    """Alive where the hash noise of the cell (white noise, ``lab_rand``) < density."""
    from . import np_noise

    gw, gh = grid
    iy, ix = np.mgrid[0:gh, 0:gw].astype(np.int32)
    return np_noise.rand(ix, iy, seed) < F32(density)


def gradient(t, a, b, c):
    """Three-stop gradient: a at t = 0, b at 0.5, c at 1. ``t`` (h, w) float32, colours (3,)."""
    a, b, c = (np.asarray(x, F32) for x in (a, b, c))
    lo = np.minimum(t * F32(2.0), F32(1.0))[..., None]
    hi = np.maximum(t * F32(2.0) - F32(1.0), F32(0.0))[..., None]
    return (a + (b - a) * lo + (c - b) * hi).astype(F32)


def shade(s, states):
    """Brightness of a cell: 1 for alive, fading to 1 / (C - 1) for the last dying state, 0 dead."""
    inv = F32(1.0 / (states - 1))
    return np.where(s >= 1, (states - s.astype(np.int32)).astype(F32) * inv, F32(0.0)).astype(F32)
