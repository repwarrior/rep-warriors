#!/usr/bin/env python3
"""Draw a TradingView-style chart from OHLC data.

For instruments where there is no screenshot to annotate. The output matches
the layout render.py expects -- same 1290px portrait frame, a price pane over a
MACD pane, gridlines on round numbers, price gutter on the right -- and the
geometry is written out directly rather than detected, so it is exact.

    python3 synth.py ohlc.csv -o chart.png -g geometry.json \\
        --symbol "BTC / USD" --subtitle Bitcoin

The CSV wants `timestamp,open,high,low,close` with a header, oldest or newest
first (it gets sorted).
"""

import argparse
import csv
import json
from datetime import date

from PIL import Image, ImageDraw, ImageFont

W, H = 1290, 2400
HEADER_H = 175
PRICE_BOT = 1596
MACD_TOP, MACD_BOT = 1610, 2300
AXIS_Y = 2340
PLOT_L, PLOT_R = 28, 1045
GUTTER = 1053

BG = (0, 0, 0)
GRID = (46, 50, 58)
SEP = (60, 65, 74)
UP = (38, 166, 154)
DOWN = (239, 83, 80)
TEXT = (190, 196, 204)
DIM = (120, 127, 136)
MACD_LINE = (41, 98, 255)
SIG_LINE = (255, 109, 0)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size)


def nice_step(span, target=9):
    """A round gridline step that yields roughly `target` lines."""
    raw = span / target
    mag = 10 ** len(str(int(raw)).lstrip("-")) / 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= mag * m:
            return mag * m
    return mag * 10


def ema(xs, n):
    k = 2 / (n + 1)
    out = [xs[0]]
    for v in xs[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def load(path, start=None):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if start and r["timestamp"] < start:
                continue
            rows.append((
                r["timestamp"], float(r["open"]), float(r["high"]),
                float(r["low"]), float(r["close"]),
            ))
    rows.sort()
    return rows


def draw(rows, out, geo_out, symbol, subtitle, decimals, future_bars):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    closes = [r[4] for r in rows]

    pad = (max(highs) - min(lows)) * 0.06
    step = nice_step(max(highs) - min(lows) + 2 * pad)
    top = (int((max(highs) + pad) / step) + 1) * step
    bot = int((min(lows) - pad) / step) * step

    pane_top, pane_bot = HEADER_H + 40, PRICE_BOT - 20
    ppu = (pane_bot - pane_top) / (top - bot)

    def py(price):
        return pane_top + (top - price) * ppu

    # Bars are laid out with room on the right for the projected path.
    n = len(rows)
    slot = (PLOT_R - PLOT_L) / (n + future_bars)
    body = max(3, int(slot * 0.62))

    def bx(i):
        return PLOT_L + slot * (i + 0.5)

    # --- gridlines and the price gutter ------------------------------------
    gridlines = []
    p = top
    while p >= bot - 1e-9:
        y = int(round(py(p)))
        if pane_top - 2 <= y <= pane_bot + 2:
            gridlines.append(y)
            for x in range(PLOT_L, GUTTER, 6):
                d.point([(x, y)], fill=GRID)
            d.text((GUTTER + 14, y - 15), f"{p:,.{decimals}f}", font=font(27), fill=TEXT)
        p -= step

    # --- month gridlines and labels ----------------------------------------
    time_x = []
    seen = set()
    for i, r in enumerate(rows):
        y_, m_, _ = (int(v) for v in r[0].split("-"))
        if (y_, m_) in seen:
            continue
        seen.add((y_, m_))
        if i == 0:
            continue
        x = int(bx(i))
        time_x.append(x)
        for yy in range(pane_top, MACD_BOT, 6):
            d.point([(x, yy)], fill=GRID)
        d.text((x - 26, AXIS_Y), MONTHS[m_ - 1], font=font(30), fill=TEXT)

    # --- candles ------------------------------------------------------------
    for i, (_, o, hi, lo, c) in enumerate(rows):
        colour = UP if c >= o else DOWN
        x = bx(i)
        d.line([(x, py(hi)), (x, py(lo))], fill=colour, width=3)
        y1, y2 = py(max(o, c)), py(min(o, c))
        d.rectangle([x - body / 2, y1, x + body / 2, max(y2, y1 + 2)], fill=colour)

    # --- last price line and tag -------------------------------------------
    last = closes[-1]
    ly = int(py(last))
    for x in range(PLOT_L, GUTTER, 10):
        d.line([(x, ly), (x + 4, ly)], fill=DOWN if closes[-1] < closes[-2] else UP)
    tagc = DOWN if closes[-1] < closes[-2] else UP
    d.rectangle([GUTTER + 6, ly - 26, GUTTER + 232, ly + 26], fill=tagc)
    d.text((GUTTER + 18, ly - 18), f"{last:,.{decimals}f}", font=font(29, True), fill=(255, 255, 255))

    # --- header -------------------------------------------------------------
    d.text((30, 46), symbol, font=font(44, True), fill=(235, 238, 242))
    if subtitle:
        wsym = d.textlength(symbol, font=font(44, True))
        d.text((30 + wsym + 20, 58), subtitle, font=font(32), fill=DIM)
    chg = last - closes[-2]
    pct = chg / closes[-2] * 100
    col = UP if chg >= 0 else DOWN
    d.text(
        (30, 110),
        f"{last:,.{decimals}f}  {chg:+,.{decimals}f} ({pct:+.2f}%)",
        font=font(38, True), fill=col,
    )

    # --- MACD pane ----------------------------------------------------------
    d.line([(0, PRICE_BOT), (W, PRICE_BOT)], fill=SEP, width=2)
    macd = [a - b for a, b in zip(ema(closes, 12), ema(closes, 26))]
    sig = ema(macd, 9)
    hist = [a - b for a, b in zip(macd, sig)]
    span = max(max(map(abs, macd)), max(map(abs, sig)), max(map(abs, hist))) * 1.15
    mid = (MACD_TOP + MACD_BOT) / 2
    scale = (MACD_BOT - MACD_TOP) / 2 / span

    def my(v):
        return mid - v * scale

    for x in range(PLOT_L, GUTTER, 8):
        d.point([(x, int(mid))], fill=SEP)
    for i, v in enumerate(hist):
        rising = i == 0 or v >= hist[i - 1]
        colour = UP if v >= 0 else DOWN
        if not rising:
            colour = tuple(int(ch * 0.55) for ch in colour)
        x = bx(i)
        ya, yb = sorted((my(v), my(0)))
        d.rectangle([x - body / 2, ya, x + body / 2, max(yb, ya + 2)], fill=colour)
    for series, colour in ((macd, MACD_LINE), (sig, SIG_LINE)):
        d.line([(bx(i), my(v)) for i, v in enumerate(series)], fill=colour, width=4)
    d.line([(0, MACD_BOT + 30), (W, MACD_BOT + 30)], fill=SEP, width=2)
    d.text((30, MACD_TOP + 10), "MACD 12 26 9", font=font(26), fill=DIM)

    im.save(out)

    geo = {
        "image": {"width": W, "height": H},
        "separators": [PRICE_BOT, MACD_BOT + 30],
        "gutter_x": GUTTER,
        "price_pane": {"top": pane_top, "bottom": pane_bot},
        "gridlines_y": gridlines,
        "gridline_spacing": (
            round((gridlines[-1] - gridlines[0]) / (len(gridlines) - 1), 3)
            if len(gridlines) > 1 else None
        ),
        "data_x_hint": {"left": int(bx(0)), "right": int(bx(n - 1))},
        "time_gridline_x": time_x,
        "time_gridline_spacing": None,
        "source": out,
        "synth": {
            "top_gridline_price": top,
            "step": step,
            "last_bar_x": int(bx(n - 1)),
            "bar_slot_px": round(slot, 3),
            "bars": n,
            "first": rows[0][0],
            "last": rows[-1][0],
            "last_close": last,
        },
    }
    with open(geo_out, "w") as fh:
        json.dump(geo, fh, indent=2)
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="chart.png")
    ap.add_argument("-g", "--geometry", default="geometry.json")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--start", help="drop rows before this timestamp")
    ap.add_argument("--decimals", type=int, default=0)
    ap.add_argument("--future-bars", type=int, default=38,
                    help="empty slots kept on the right for the projected path")
    args = ap.parse_args()

    rows = load(args.csv, args.start)
    geo = draw(rows, args.out, args.geometry, args.symbol, args.subtitle,
               args.decimals, args.future_bars)
    s = geo["synth"]
    print(f"wrote {args.out} ({s['bars']} bars, {s['first']} -> {s['last']})")
    print(f"wrote {args.geometry}")
    print(f"calibration: top_gridline_price={s['top_gridline_price']:,.0f} "
          f"step={s['step']:,.0f}")
    print(f"last bar x={s['last_bar_x']}  bar slot={s['bar_slot_px']}px")


if __name__ == "__main__":
    main()
