# rep-warriors

Turns a plain TradingView screenshot into an annotated trade-plan image.

The chart in the output is the user's own screenshot — levels, zones, the
expected-path arrow and the summary panel are drawn on top of it, so what comes
back matches what is on their phone.

```
                screenshot.png
                      |
        detect.py  ---+---> geometry.json  (gridlines, panes, gutter, time axis)
                      |     ruler.png      (read this to calibrate)
                      |
                 plan.json                 (levels in prices, written by hand)
                      |
        render.py  ---+---> plan.png       (HTML + SVG overlay, shot in Chromium)
```

## Usage

```bash
python3 tools/chartplan/detect.py examples/gold-2026-08-13.png \
  -o plans/2026-08-13-gold-1d.geometry.json --ruler /tmp/ruler.png

python3 tools/chartplan/render.py plans/2026-08-13-gold-1d.json \
  -o out/gold-2026-08-13-plan.png
```

`.claude/skills/trade-plan-chart/` documents the full workflow, including how
the levels are derived from real price data rather than eyeballed.

## Why the calibration is manual

`detect.py` finds gridlines, pane separators, the price gutter and the time
axis from pixels alone — all reliable. It deliberately does not read the price
axis: OCR on axis labels fails quietly, and a misread digit puts every level on
the chart in the wrong place. Instead the plan supplies two numbers, the price
on the top gridline and the step between gridlines, which is enough to map any
y pixel to a price exactly.

The same caution applies to `data_x_hint`: news and earnings badges pinned under
the bars share TradingView's candle red, so the detected right edge can overshoot
the last bar. Confirm it against the ruler image.

## Requirements

Python with `pillow` and `numpy`, Node with `playwright`, and a Chromium build
(`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers` in this environment).

## Not advice

Everything here is technical analysis read off a chart. It is not financial
advice, and nothing in it accounts for the risk the reader is actually taking.
