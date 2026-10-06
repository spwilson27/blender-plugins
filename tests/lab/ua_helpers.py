# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Helpers for the Utility A node tests (Expression, Image Statistics, Auto Levels, Palette
Extract): reading single-value outputs through the compositor, and test images."""

import numpy as np

import harness as H


def single(idname, device, size, out, props=None, inputs=None, images=None):
    """Render the node with the given (single-value) output linked to the group output and return
    the value as a float64 array: the pixel at (0, 0) of the rendered image. Also checks that the
    image is uniform (a single value fills the frame)."""
    img = H.render_node(idname, device, size, props=props, inputs=inputs, images=images,
                        out_socket=out)
    v = img[0, 0].astype(np.float64)
    if not np.allclose(img, img[0, 0], atol=0.0, rtol=0.0, equal_nan=True):
        H.report(False, "%s [%s]: single value output is not uniform" % (idname, out))
    return v


def flat_colour_image(size, colours, seed=0, alpha=1.0):
    """(h, w, 4) image made of vertical bands / random blocks of exactly the given colours.
    Returns (img, counts) where counts[i] is the number of pixels of colour i."""
    w, h = size
    rng = np.random.default_rng(seed)
    bw = max(w // 16, 1)
    cols = (np.arange(w) // bw) % len(colours)
    # shuffle the band order deterministically so the layout is not a simple ramp
    perm = rng.permutation(len(colours))
    cols = perm[cols]
    img = np.zeros((h, w, 4), np.float32)
    pal = np.asarray(colours, np.float32)
    img[..., :3] = pal[cols][None, :, :] * alpha
    img[..., 3] = alpha
    counts = np.bincount(np.broadcast_to(cols[None, :], (h, w)).ravel(), minlength=len(colours))
    return img, counts
