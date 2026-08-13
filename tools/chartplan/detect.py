#!/usr/bin/env python3
"""Detect the geometry of a TradingView mobile screenshot.

Finds the things that are measurable from pixels alone: horizontal price
gridlines, the pane separators, the right-hand price gutter, where the candle
data stops, and the vertical gridlines that mark the time axis.

What it deliberately does NOT do is read the numbers on the price axis --
that needs OCR and goes wrong in ways that are hard to spot. The caller
supplies two numbers instead (the price on the top gridline and the step
between gridlines), which is enough to turn any y pixel into a price.

    python3 detect.py screenshot.png -o geometry.json

Assumes a portrait phone screenshot with the price gutter occupying less
than the right-hand fifth of the frame -- true of the TradingView iOS/Android
chart tab. Verify the printed summary against the image before trusting it.
"""

import argparse
import json

import numpy as np
from PIL import Image

GRID_LO, GRID_HI = 28, 95  # TradingView's gridline grey, on the dark theme


def _group(values, gap=3):
    """Collapse runs of adjacent coordinates into their centres."""
    values = list(values)
    if not values:
        return []
    out, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(cur)
            cur = [v]
    out.append(cur)
    return [int(round(float(np.mean(g)))) for g in out]


def _grid_mask(img):
    """Faint neutral grey: gridlines, pane separators, axis rules."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (np.abs(r - g) < 14) & (np.abs(g - b) < 14) & (r > GRID_LO) & (r < GRID_HI)


def _near(img, colour, tol=30):
    """Pixels within tol of an exact RGB triple, per channel."""
    return (
        (np.abs(img[..., 0] - colour[0]) < tol)
        & (np.abs(img[..., 1] - colour[1]) < tol)
        & (np.abs(img[..., 2] - colour[2]) < tol)
    )


def detect(path):
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    h, w, _ = img.shape

    # Sample well inside the frame: past the left edge, short of the gutter.
    inner = slice(40, int(w * 0.80))
    grid = _grid_mask(img)
    row_frac = grid[:, inner].mean(1)

    # A separator runs edge to edge; a gridline is dotted and comes out ~0.2.
    separators = _group(np.flatnonzero(row_frac > 0.80), gap=4)
    gridlines_all = _group(np.flatnonzero((row_frac > 0.15) & (row_frac <= 0.80)), gap=3)

    # The price pane is everything above the first separator.
    pane_bottom = separators[0] if separators else int(h * 0.60)
    pane_top = int(h * 0.08)
    gridlines = [y for y in gridlines_all if pane_top < y < pane_bottom]

    # The gutter begins where the gridlines stop running.
    gutter = int(w * 0.85)
    if gridlines:
        ends = []
        for y in gridlines:
            row = np.flatnonzero(grid[y, :])
            row = row[row > int(w * 0.5)]
            if len(row):
                ends.append(int(row.max()))
        if ends:
            gutter = int(np.median(ends)) + 4

    # Candle ink: TradingView's up-teal and down-red, matched tightly so the
    # current-price tag and the news/earnings badges mostly fall out.
    pane = img[pane_top:pane_bottom, :gutter, :]
    up = _near(pane, (38, 166, 154))
    down = _near(pane, (239, 83, 80))
    ink = (up | down).sum(0)
    cols = [x for x in range(gutter) if ink[x] > 6]
    # Badges pinned under the bars (earnings, news, splits) share the down-red
    # hue closely enough to survive that filter and sit to the right of the
    # last bar, so this is a hint, not a measurement -- read the last bar off
    # the ruler image before anchoring a projection to it.
    data_left, data_right = (cols[0], cols[-1]) if cols else (0, gutter)

    # Vertical gridlines mark the time axis (one per month on a daily chart).
    col_frac = grid[pane_top:pane_bottom, :gutter].mean(0)
    vthresh = max(0.10, float(col_frac.max()) * 0.60)
    time_x = _group(np.flatnonzero(col_frac > vthresh), gap=4)

    return {
        "image": {"width": w, "height": h},
        "separators": separators,
        "gutter_x": gutter,
        "price_pane": {"top": pane_top, "bottom": pane_bottom},
        "gridlines_y": gridlines,
        "gridline_spacing": (
            round(float(np.median(np.diff(gridlines))), 3) if len(gridlines) > 1 else None
        ),
        "data_x_hint": {"left": data_left, "right": data_right},
        "time_gridline_x": time_x,
        "time_gridline_spacing": (
            round(float(np.median(np.diff(time_x))), 2) if len(time_x) > 1 else None
        ),
    }


def ruler(path, geo, out):
    """Overlay a labelled coordinate grid so the geometry can be eyeballed.

    Read this image before writing a plan: it is how you confirm which price
    sits on the top gridline and where the last bar actually is.
    """
    from PIL import ImageDraw

    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size

    for x in range(0, w, 50):
        d.line([(x, 0), (x, h)], fill=(90, 90, 0), width=1)
    for x in range(0, w, 100):
        d.line([(x, 0), (x, h)], fill=(255, 255, 0), width=1)
        d.text((x + 3, 4), str(x), fill=(255, 255, 0))

    for y in geo["gridlines_y"]:
        d.line([(0, y), (w, y)], fill=(0, 255, 255), width=1)
        d.text((6, y + 3), f"y={y}", fill=(0, 255, 255))
    for y in geo["separators"]:
        d.line([(0, y), (w, y)], fill=(255, 0, 255), width=2)
    for x in geo["time_gridline_x"]:
        d.line([(x, 0), (x, h)], fill=(255, 140, 0), width=2)

    hint = geo["data_x_hint"]["right"]
    d.line([(hint, 0), (hint, h)], fill=(255, 255, 255), width=2)
    d.text((hint + 6, 60), f"last-bar hint x={hint}", fill=(255, 255, 255))

    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("screenshot")
    ap.add_argument("-o", "--out", default="geometry.json")
    ap.add_argument("--ruler", help="write a coordinate-grid overlay here")
    args = ap.parse_args()

    geo = detect(args.screenshot)
    geo["source"] = args.screenshot
    with open(args.out, "w") as fh:
        json.dump(geo, fh, indent=2)

    print(f"image        {geo['image']['width']}x{geo['image']['height']}")
    print(f"separators   {geo['separators']}")
    print(f"gutter x     {geo['gutter_x']}")
    print(f"price pane   {geo['price_pane']['top']} -> {geo['price_pane']['bottom']}")
    print(f"gridlines y  {geo['gridlines_y']}")
    print(f"             spacing {geo['gridline_spacing']} px per step")
    print(f"data x hint  {geo['data_x_hint']['left']} -> {geo['data_x_hint']['right']}")
    print(f"time grid x  {geo['time_gridline_x']} (spacing {geo['time_gridline_spacing']})")
    print(f"\nwrote {args.out}")

    if args.ruler:
        ruler(args.screenshot, geo, args.ruler)
        print(f"wrote {args.ruler}  <- read this to confirm prices and last bar")


if __name__ == "__main__":
    main()
