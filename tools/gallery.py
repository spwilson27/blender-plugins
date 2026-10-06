# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Render every Compositor Lab node (plus Pixel Sort) with its default settings into a gallery.

Run inside Blender (needs a build with Python compositor nodes and a GPU)::

    Blender -b --factory-startup --python tools/gallery.py -- [--out DIR] [--device GPU|CPU]

Nodes with image inputs get a shared, colourful test image (a second one for further inputs such
as Blend Modes B or the Displace map); the generators (Noise, Voronoi, Pattern, Flow Field) are
rendered standalone. Everything is 480x270. Output (default ``dist/``):

* ``gallery/<Node Name>.png``: one PNG per node (and ``source.png``, the test image),
* ``gallery.png``: a contact sheet with each tile labelled with the node name (drawn with a tiny
  built-in bitmap font, so it needs nothing but numpy).

It reuses the test harness (``tests/lab/harness.py``) for building the node tree and rendering.
"""

import os
import re
import sys

import bpy
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "tests", "lab"))
sys.path.insert(0, os.path.join(ROOT, "addons"))

import harness as H  # noqa: E402

SIZE = (480, 270)
GENERATORS = {"CompositorNodeLabNoise", "CompositorNodeLabVoronoi", "CompositorNodeLabPattern",
              "CompositorNodeLabFlowField"}
IMAGE_SOCKETS = ("Image", "A", "B", "Map", "Mask")           # inputs that get the test image
OUT_SOCKET = {"CompositorNodeLabImageStatistics": "Mean",     # single-value output
              "CompositorNodeLabPaletteExtract": "Quantized"}
COLUMNS = 5
LABEL_H = 22
GAP = 6

# ---------------------------------------------------------------------------
# Test image
# ---------------------------------------------------------------------------


def hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.astype(int) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def source_image(w, h):
    """Colourful scene-linear RGBA (h, w, 4), row 0 = bottom: hue sweep, shapes, stripes, text-like
    detail, so every kind of filter has something to show."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / w, y / h
    hue = (u + 0.25 * np.sin(v * 5.0 + u * 3.0)) % 1.0
    rgb = hsv_to_rgb(hue, 0.55 + 0.4 * v, 0.35 + 0.65 * (1 - 0.5 * v))
    cx, cy = 0.5 * w, 0.52 * h
    r = np.hypot(x - cx, y - cy)
    rgb = np.where((r < 0.30 * h)[..., None], hsv_to_rgb((hue + 0.5) % 1.0, 0.25, 0.95), rgb)
    ring = (np.abs(r - 0.38 * h) < 3.0)[..., None]
    rgb = np.where(ring, 0.05, rgb)
    stripes = ((x + y * 0.5) // 14) % 2 == 0
    box = (u > 0.05) & (u < 0.28) & (v > 0.08) & (v < 0.45)
    rgb = np.where((box & stripes)[..., None], np.array([0.9, 0.85, 0.1]), rgb)
    rgb = np.where((box & ~stripes)[..., None], np.array([0.1, 0.1, 0.5]), rgb)
    checker = (((x // 12) + (y // 12)) % 2 == 0)
    box2 = (u > 0.72) & (u < 0.95) & (v > 0.55) & (v < 0.92)
    rgb = np.where((box2 & checker)[..., None], 0.97, np.where(box2[..., None], 0.03, rgb))
    line = (np.abs(y - (0.15 * h + 0.2 * np.sin(x / 25.0) * h * 0.1)) < 1.5)[..., None]
    rgb = np.where(line, 1.0, rgb)
    out = np.empty((h, w, 4), np.float32)
    out[..., :3] = np.clip(rgb, 0.0, 1.0) ** 2.2        # treat the above as display values
    out[..., 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def srgb8(arr):
    """Scene-linear (h, w, 4) -> sRGB-encoded float RGBA in 0..1 (alpha dropped)."""
    c = np.clip(np.nan_to_num(arr[..., :3]), 0.0, 1.0)
    enc = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(enc * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)


def save_png(path, rgb):
    """``rgb``: (h, w, 3) display values in 0..1, row 0 = bottom."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new("gallery_tmp", w, h, alpha=False)
    try:
        img.colorspace_settings.name = 'Non-Color'
        px = np.ones((h, w, 4), np.float32)
        px[..., :3] = rgb
        img.pixels.foreach_set(px.ravel())
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
    finally:
        bpy.data.images.remove(img)


# 5x7 bitmap font (rows top to bottom, '#' = set). Lower case is drawn as upper case.
_FONT_SRC = """
A 01110 10001 10001 11111 10001 10001 10001
B 11110 10001 10001 11110 10001 10001 11110
C 01110 10001 10000 10000 10000 10001 01110
D 11110 10001 10001 10001 10001 10001 11110
E 11111 10000 10000 11110 10000 10000 11111
F 11111 10000 10000 11110 10000 10000 10000
G 01110 10001 10000 10111 10001 10001 01111
H 10001 10001 10001 11111 10001 10001 10001
I 01110 00100 00100 00100 00100 00100 01110
J 00111 00010 00010 00010 00010 10010 01100
K 10001 10010 10100 11000 10100 10010 10001
L 10000 10000 10000 10000 10000 10000 11111
M 10001 11011 10101 10101 10001 10001 10001
N 10001 11001 10101 10011 10001 10001 10001
O 01110 10001 10001 10001 10001 10001 01110
P 11110 10001 10001 11110 10000 10000 10000
Q 01110 10001 10001 10001 10101 10010 01101
R 11110 10001 10001 11110 10100 10010 10001
S 01111 10000 10000 01110 00001 00001 11110
T 11111 00100 00100 00100 00100 00100 00100
U 10001 10001 10001 10001 10001 10001 01110
V 10001 10001 10001 10001 10001 01010 00100
W 10001 10001 10001 10101 10101 11011 10001
X 10001 10001 01010 00100 01010 10001 10001
Y 10001 10001 01010 00100 00100 00100 00100
Z 11111 00001 00010 00100 01000 10000 11111
0 01110 10001 10011 10101 11001 10001 01110
1 00100 01100 00100 00100 00100 00100 01110
2 01110 10001 00001 00010 00100 01000 11111
3 11110 00001 00001 01110 00001 00001 11110
4 00010 00110 01010 10010 11111 00010 00010
5 11111 10000 11110 00001 00001 10001 01110
6 00110 01000 10000 11110 10001 10001 01110
7 11111 00001 00010 00100 01000 01000 01000
8 01110 10001 10001 01110 10001 10001 01110
9 01110 10001 10001 01111 00001 00010 01100
/ 00001 00001 00010 00100 01000 10000 10000
+ 00000 00100 00100 11111 00100 00100 00000
- 00000 00000 00000 11111 00000 00000 00000
( 00010 00100 01000 01000 01000 00100 00010
) 01000 00100 00010 00010 00010 00100 01000
. 00000 00000 00000 00000 00000 01100 01100
: 00000 01100 01100 00000 01100 01100 00000
"""
FONT = {}
for _line in _FONT_SRC.strip().splitlines():
    _ch, *_rows = _line.split()
    FONT[_ch] = np.array([[c == "1" for c in row] for row in _rows], bool)
FONT[" "] = np.zeros((7, 5), bool)


def draw_text(canvas, text, x, y, scale=2, color=(1.0, 1.0, 1.0)):
    """Draw ``text`` into ``canvas`` ((h, w, 3), row 0 = top) with its top-left at (x, y)."""
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        big = np.kron(glyph, np.ones((scale, scale), bool))
        gh, gw = big.shape
        if x + gw > canvas.shape[1] or y + gh > canvas.shape[0]:
            break
        region = canvas[y:y + gh, x:x + gw]
        region[big] = color
        x += gw + scale


def contact_sheet(tiles):
    """tiles: [(label, rgb (h, w, 3) bottom-up)] -> (H, W, 3) bottom-up sheet."""
    tw, th = SIZE
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    cell_h = th + LABEL_H
    width = COLUMNS * tw + (COLUMNS + 1) * GAP
    height = rows * cell_h + (rows + 1) * GAP
    sheet = np.full((height, width, 3), 0.08, np.float32)       # top-down while drawing
    for i, (label, rgb) in enumerate(tiles):
        cx = GAP + (i % COLUMNS) * (tw + GAP)
        cy = GAP + (i // COLUMNS) * (cell_h + GAP)
        sheet[cy:cy + LABEL_H, cx:cx + tw] = 0.16
        draw_text(sheet, label, cx + 6, cy + 4)
        sheet[cy + LABEL_H:cy + cell_h, cx:cx + tw] = rgb[::-1]
    return sheet[::-1].copy()


# ---------------------------------------------------------------------------

def filename(label):
    return re.sub(r"[^A-Za-z0-9+]+", "_", label).strip("_")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_dir = os.path.join(ROOT, "dist")
    device = "GPU"
    if "--out" in argv:
        out_dir = os.path.abspath(argv[argv.index("--out") + 1])
    if "--device" in argv:
        device = argv[argv.index("--device") + 1].upper()
    os.makedirs(os.path.join(out_dir, "gallery"), exist_ok=True)

    compositor_lab = H.setup(detect=False)
    import pixel_sort_node
    pixel_sort_node.register()
    H.FEATURES["F1"] = True          # generators have the render-sized domain (needed by the build)

    w, h = SIZE
    src = source_image(w, h)
    src2 = np.ascontiguousarray(src[::-1, :, [1, 2, 0, 3]])      # flipped, channels rolled
    tiles = [("Source", srgb8(src))]
    save_png(os.path.join(out_dir, "gallery", "source.png"), tiles[0][1])

    entries = [(c.bl_label, c.bl_idname, c)
               for section in compositor_lab.SECTIONS for c in compositor_lab.REGISTERED[section]]
    entries.sort(key=lambda e: e[0].lower())
    entries.append(("Pixel Sort", "CompositorNodePixelSort", None))
    failures = []
    for label, idname, cls in entries:
        in_names = [s.name for s in getattr(cls, "SOCKETS", ()) if hasattr(s, "default")]
        images = {}
        if idname not in GENERATORS:
            for n in (in_names or ["Image"]):
                if n in IMAGE_SOCKETS:
                    images[n] = src if not images else src2
        out_socket = OUT_SOCKET.get(idname)
        try:
            if idname in GENERATORS:
                res = H.render_generator(idname, device, SIZE, out_socket=out_socket,
                                         allow_errors=True)
            else:
                res = H.render_node(idname, device, SIZE, images=images, out_socket=out_socket,
                                    allow_errors=True)
            errors = list(H.LAST_ERRORS)
        except Exception as ex:
            res, errors = np.zeros((h, w, 4), np.float32), [repr(ex)]
        rgb = srgb8(res)
        if errors:
            failures.append((label, errors))
            rgb = rgb * 0.5
            rgb[..., 0] = np.maximum(rgb[..., 0], 0.4)
        tiles.append((label + (" (ERROR)" if errors else ""), rgb))
        save_png(os.path.join(out_dir, "gallery", filename(label) + ".png"), rgb)
        print("rendered %s%s" % (label, "  ERRORS: %s" % errors if errors else ""))
        sys.stdout.flush()

    sheet = contact_sheet(tiles)
    sheet_path = os.path.join(out_dir, "gallery.png")
    save_png(sheet_path, sheet)
    print("wrote %s (%d tiles) and %s" % (sheet_path, len(tiles), os.path.join(out_dir, "gallery")))
    if failures:
        raise SystemExit("gallery: %d node(s) failed: %s" % (len(failures), [f[0] for f in failures]))


main()
