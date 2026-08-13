---
name: trade-plan-chart
description: Turn a raw TradingView screenshot into an annotated trade-plan image - support/resistance zones, targets, pullback zone, expected-path arrow, momentum callout and a summary panel, drawn on top of the user's own chart. Use whenever the user sends a chart screenshot (stocks, crypto, FX, metals, indices) and wants levels, a prediction, a trade plan, or "the annotated version".
---

# Trade plan from a chart screenshot

The user sends a plain TradingView screenshot. They get back the same chart with
the plan drawn on it. Do not redraw the candles from scratch — annotate *their*
image, so what they see matches what is on their phone.

## The split that makes this reliable

The script measures geometry from pixels. You read the numbers and do the
analysis. Never try to OCR the price axis in code — it fails quietly and every
level ends up subtly wrong.

## Steps

### 1. Find the uploaded file

Screenshots land in `/root/.claude/uploads/<session-id>/`. Newest first:

```bash
ls -t /root/.claude/uploads/*/ | head
```

Copy it into `examples/` if it is worth keeping as a regression case.

### 2. Detect geometry, then look at the ruler

```bash
python3 tools/chartplan/detect.py <shot.png> \
  -o plans/<date>-<symbol>-<tf>.geometry.json --ruler /tmp/ruler.png
```

Then **Read `/tmp/ruler.png`**. This is not optional — it is where you pick up
the two things the detector cannot know:

- **Which price sits on the top detected gridline**, and the step between
  gridlines. The ruler prints `y=` on each one; match them to the axis labels.
- **Where the last bar actually is.** `data_x_hint` overshoots whenever an
  earnings/news badge is pinned under the bars, as in the gold example. Read
  the real x off the ruler; the projected path is anchored to it.

Sanity-check the calibration against the current-price line before going on:
`price = top_gridline_price - (y - y0) / px_per_unit` should reproduce the
number in the red tag to within a point or two. If it does not, you have the
wrong gridline price — fix it now, not after rendering.

### 3. Get real data for the instrument

Do not eyeball the levels. Pull the actual series and compute them:

- metals — `GOLD_SILVER_HISTORY` (daily closes, no OHLC)
- stocks/ETFs — `TIME_SERIES_DAILY`
- crypto — `DIGITAL_CURRENCY_DAILY` / `CRYPTO_INTRADAY`
- FX — `FX_DAILY` (note: `XAU` is rejected here, use the metals endpoint)

Large responses get spilled to a file; query it with `jq` or Python rather than
reading the whole thing.

Then compute, from the data:

- **Time-at-price histogram** (50-unit bins over the visible window). Heavy bins
  are shelves — real support and resistance. Thin bins are air pockets worth
  calling out, price travels through them fast.
- **Swing pivots** (local max/min over a ±3 bar window) for the structure.
- **The base**: the consolidation range before the current leg. Its height
  projected from the breakout is the measured move — check whether it has
  already been met, that often explains a stall.
- **Fib retracements** of the active leg for the pullback zone.
- **MACD** (12/26/9 on closes). Report the histogram's last few values, not just
  the sign: a histogram rolling over while still positive is a pause, and saying
  so is more useful than "momentum bullish".

### 4. Write the plan

Copy `plans/2026-08-13-gold-1d.json` and edit. Every level is a *price*; the
renderer converts to pixels. Positions (`label_y`, `text_x`, path `x`) are in
source-image pixels — take them off the ruler.

Zone kinds are `resistance`, `support`, `pullback`. Tag kinds are `resistance`,
`support`, `neutral`. Callouts take `text` (`\n` splits lines) and an optional
leader line via `from_x/from_y/to_x/to_y`.

The path is a polyline of `{x, price}` from the last bar into the empty space on
the right. Make it say something: where the pullback lands, where the first
target is, where it stalls.

### 5. Render, then look at it

```bash
python3 tools/chartplan/render.py plans/<name>.json -o out/<name>.png
```

**Read the PNG.** Labels collide with candles, with the axis, and with the path
on almost every first pass. Move them and re-render until nothing overlaps.
Empty regions are cheap to find on the ruler image. Budget two or three passes.

### 6. Send it

`SendUserFile` with `display: "render"` so it opens inline on their phone.

## Writing the analysis

Say what the chart shows and what would falsify it. Ground every claim in a
number you computed. State the invalidation level explicitly — a plan without
one is a guess.

Keep the disclaimer in the footer honest and small: these are levels read off a
chart, not advice, and the user is the one taking the risk.
