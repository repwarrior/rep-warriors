#!/usr/bin/env python3
"""Draw a trade plan over a TradingView screenshot.

Takes a plan (JSON) that names levels in *price* terms and a geometry file
from detect.py, builds an HTML page with the screenshot underneath and an SVG
overlay on top, then screenshots it with headless Chromium.

    python3 render.py plan.json -o out/plan.png

Prices are converted to pixels with a two-number calibration the plan supplies
(`top_gridline_price` and `step`), so nothing here depends on reading the
price axis by OCR.
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import tempfile

# TradingView-ish palette, so the annotations sit naturally on the screenshot.
RESISTANCE = "#f2434a"
SUPPORT = "#2ea86a"
PATH = "#2f6bff"
ZONE = "#3d7dff"
NEUTRAL = "#e6e8ea"
ACCENT = "#f5c542"
BG = "#0c0e12"


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


class Scale:
    """Maps prices to y pixels in the source screenshot's coordinate space."""

    def __init__(self, gridlines, top_price, step):
        self.y0 = gridlines[0]
        self.top_price = top_price
        spacing = (gridlines[-1] - gridlines[0]) / (len(gridlines) - 1)
        self.px_per_unit = spacing / abs(step)

    def y(self, price):
        return self.y0 + (self.top_price - price) * self.px_per_unit

    def price(self, y):
        return self.top_price - (y - self.y0) / self.px_per_unit


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def fmt(v, dp):
    return f"{v:,.{dp}f}"


def build_svg(plan, geo, scale, crop):
    """Everything drawn in source-pixel coordinates, clipped to the crop box."""
    x0, y0, x1, y1 = crop
    W, H = x1 - x0, y1 - y0
    dp = plan.get("decimals", 0)
    label_x = plan.get("label_right_x", geo["gutter_x"] - 190)
    plot_left = plan.get("plot_left_x", 20)
    plot_right = plan.get("plot_right_x", geo["gutter_x"] - 8)
    tag_x = plan.get("tag_x", geo["gutter_x"] - 175)

    o = [
        f'<svg class="ov" viewBox="{x0} {y0} {W} {H}" width="{W}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        '<defs>',
        f'<marker id="ah" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" '
        f'markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{PATH}"/></marker>',
        f'<marker id="ahs" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ACCENT}"/></marker>',
        '</defs>',
    ]

    # --- price zones (a shaded band between two prices) ---------------------
    for z in plan.get("zones", []):
        colour = {"resistance": RESISTANCE, "support": SUPPORT,
                  "pullback": ZONE, "note": ACCENT}[z["kind"]]
        ya, yb = scale.y(z["hi"]), scale.y(z["lo"])
        zx0 = z.get("x0", plot_left)
        zx1 = z.get("x1", plot_right)
        o.append(
            f'<rect x="{zx0}" y="{ya:.1f}" width="{zx1 - zx0}" height="{yb - ya:.1f}" '
            f'fill="{colour}" fill-opacity="{z.get("fill", 0.10)}" stroke="{colour}" '
            f'stroke-width="2.5" stroke-dasharray="{z.get("dash", "12 8")}" rx="4"/>'
        )
        if z.get("label"):
            ty = z.get("label_y", ya - 16)
            anchor = z.get("anchor", "end")
            tx = z.get("label_x", label_x if anchor == "end" else zx0 + 12)
            o.append(
                f'<text x="{tx}" y="{ty:.1f}" text-anchor="{anchor}" class="zl" '
                f'fill="{colour}">{esc(z["label"])}</text>'
            )
            o.append(
                f'<text x="{tx}" y="{ty + 34:.1f}" text-anchor="{anchor}" class="zv" '
                f'fill="{colour}">{fmt(z["lo"], dp)} - {fmt(z["hi"], dp)}</text>'
            )

    # --- single levels ------------------------------------------------------
    for lv in plan.get("levels", []):
        colour = {"resistance": RESISTANCE, "support": SUPPORT, "neutral": NEUTRAL}[lv["kind"]]
        y = scale.y(lv["price"])
        o.append(
            f'<line x1="{lv.get("x0", plot_left)}" y1="{y:.1f}" '
            f'x2="{lv.get("x1", plot_right)}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="2.5" stroke-dasharray="12 8"/>'
        )
        if lv.get("label"):
            o.append(
                f'<text x="{lv.get("label_x", label_x)}" y="{y - 14:.1f}" text-anchor="end" '
                f'class="zl" fill="{colour}">{esc(lv["label"])}</text>'
            )

    # --- price tags in the right-hand gutter --------------------------------
    for t in plan.get("tags", []):
        colour = {"resistance": RESISTANCE, "support": SUPPORT, "neutral": NEUTRAL}[t["kind"]]
        y = scale.y(t["price"])
        o.append(
            f'<rect x="{tag_x}" y="{y - 26:.1f}" width="168" height="52" rx="7" '
            f'fill="{BG}" fill-opacity="0.92" stroke="{colour}" stroke-width="2.5"/>'
        )
        o.append(
            f'<text x="{tag_x + 84}" y="{y + 12:.1f}" text-anchor="middle" class="tag" '
            f'fill="{colour}">{fmt(t["price"], dp)}</text>'
        )

    # --- the projected path -------------------------------------------------
    p = plan.get("path")
    if p:
        pts = " ".join(f'{pt["x"]},{scale.y(pt["price"]):.1f}' for pt in p["points"])
        o.append(
            f'<polyline points="{pts}" fill="none" stroke="{PATH}" stroke-width="9" '
            f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ah)" '
            f'opacity="0.95"/>'
        )

    # --- callouts: a text label with a leader line to a point ---------------
    for c in plan.get("callouts", []):
        colour = c.get("colour", ACCENT)
        tx, ty = c["text_x"], c["text_y"]
        lines = c["text"].split("\n")
        anchor = c.get("anchor", "middle")
        for i, line in enumerate(lines):
            o.append(
                f'<text x="{tx}" y="{ty + i * 34}" text-anchor="{anchor}" class="co" '
                f'fill="{colour}">{esc(line)}</text>'
            )
        if "to_x" in c:
            o.append(
                f'<line x1="{c["from_x"]}" y1="{c["from_y"]}" x2="{c["to_x"]}" '
                f'y2="{c["to_y"]}" stroke="{colour}" stroke-width="5" '
                f'marker-end="url(#ahs)" stroke-linecap="round"/>'
            )

    o.append("</svg>")
    return "\n".join(o)


def build_html(plan, geo, scale):
    src = plan.get("source", geo.get("source"))
    crop = plan.get("crop") or [0, 0, geo["image"]["width"], geo["image"]["height"]]
    x0, y0, x1, y1 = crop
    W, H = x1 - x0, y1 - y0
    dp = plan.get("decimals", 0)
    m = plan.get("meta", {})
    s = plan.get("summary", {})

    # Only key what the plan actually draws, so a chart with no forecast does
    # not advertise one.
    kinds = {z["kind"] for z in plan.get("zones", [])} | {
        lv["kind"] for lv in plan.get("levels", [])}
    legend = []
    if plan.get("path"):
        legend.append((f'<span class="sw" style="background:{PATH}"></span>', "Expected path"))
    if "support" in kinds:
        legend.append((f'<span class="sw dash" style="border-color:{SUPPORT}"></span>', "Support"))
    if "resistance" in kinds:
        legend.append((f'<span class="sw dash" style="border-color:{RESISTANCE}"></span>', "Resistance"))
    if "pullback" in kinds:
        legend.append((f'<span class="sw box" style="border-color:{ZONE}"></span>',
                       plan.get("zone_legend", "Pullback zone")))
    if "note" in kinds:
        legend.append((f'<span class="sw box" style="border-color:{ACCENT}"></span>',
                       plan.get("note_legend", "Thin tape")))
    for extra in plan.get("legend_extra", []):
        legend.append((f'<span class="sw" style="background:{ACCENT}"></span>', extra))
    legend_html = "".join(
        f'<div class="lg">{sw}<span>{esc(txt)}</span></div>' for sw, txt in legend
    )
    key_html = f'<div id="key">{legend_html}</div>' if len(legend) >= 2 else ""

    bullets = "".join(f"<li>{esc(b)}</li>" for b in s.get("points", []))
    bias = s.get("bias", "")
    bias_colour = SUPPORT if "BULL" in bias.upper() else (
        RESISTANCE if "BEAR" in bias.upper() else NEUTRAL
    )

    b = plan.get("badge", {})
    badge = (
        f'<div id="badge"><b>{esc(b.get("title", "TRADE PLAN"))}</b>'
        f'<span>{esc(b.get("sub", "read off this chart"))}</span></div>'
    )

    invalid = ""
    if s.get("invalidation"):
        invalid = (
            f'<div class="inv">Invalidation &middot; '
            f'{esc(s["invalidation"])}</div>'
        )

    return f"""<!doctype html>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; font-family: "DejaVu Sans", system-ui, sans-serif; }}
  #sheet {{ width: {W}px; background: {BG}; }}
  #chart {{ position: relative; width: {W}px; height: {H}px; overflow: hidden; }}
  #chart img {{ position: absolute; left: {-x0}px; top: {-y0}px;
                width: {geo["image"]["width"]}px; height: {geo["image"]["height"]}px; }}
  .ov {{ position: absolute; left: 0; top: 0; }}
  .zl {{ font-size: 27px; font-weight: 700; letter-spacing: .3px; }}
  .zv {{ font-size: 29px; font-weight: 700; }}
  .co {{ font-size: 27px; font-weight: 700; }}
  .tag {{ font-size: 28px; font-weight: 700; }}
  #badge {{ position: absolute; left: 50%; top: 14px; transform: translateX(-50%);
            border: 2.5px solid {NEUTRAL}; border-radius: 10px; padding: 10px 22px;
            text-align: center; background: rgba(12,14,18,.88); }}
  #badge b {{ color: {NEUTRAL}; font-size: 30px; letter-spacing: 1.5px; }}
  #badge span {{ display: block; color: #9aa3ad; font-size: 20px; margin-top: 3px; }}
  #foot {{ display: flex; gap: 18px; padding: 22px 26px 26px; }}
  #sum {{ flex: 1 1 66%; border: 2.5px solid {ACCENT}; border-radius: 12px; padding: 20px 24px; }}
  #sum h2 {{ color: {ACCENT}; font-size: 27px; letter-spacing: 2px; margin-bottom: 12px; }}
  #sum li {{ color: #dfe3e8; font-size: 25px; line-height: 1.5; margin: 0 0 8px 22px; }}
  .biasline {{ margin-top: 14px; font-size: 27px; font-weight: 700; color: {bias_colour}; }}
  .inv {{ margin-top: 8px; font-size: 23px; color: #9aa3ad; }}
  #key {{ flex: 0 0 30%; border: 2.5px solid #2a2f3a; border-radius: 12px; padding: 20px; }}
  .lg {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
         color: #cfd4da; font-size: 23px; }}
  .sw {{ width: 34px; height: 6px; border-radius: 3px; flex: 0 0 34px; }}
  .sw.dash {{ height: 0; border-top: 4px dashed; background: none; }}
  .sw.box {{ height: 22px; border: 2.5px dashed; background: rgba(61,125,255,.16);
             border-radius: 4px; }}
  #meta {{ padding: 0 26px 22px; color: #767e88; font-size: 20px;
           display: flex; justify-content: space-between; }}
</style>
<div id="sheet">
  <div id="chart">
    <img src="{data_uri(src)}">
    {build_svg(plan, geo, scale, crop)}
    {badge}
  </div>
  <div id="foot">
    <div id="sum">
      <h2>SUMMARY</h2>
      <ul>{bullets}</ul>
      <div class="biasline">{esc(bias)}</div>
      {invalid}
    </div>
    {key_html}
  </div>
  <div id="meta">
    <span>{esc(m.get("symbol", ""))} &middot; {esc(m.get("timeframe", ""))}
      &middot; last {fmt(m["price"], dp) if m.get("price") is not None else ""}</span>
    <span>{esc(m.get("asof", ""))} &middot; levels are analysis, not advice</span>
  </div>
</div>
"""


SHOT_JS = """
const { chromium } = require('playwright');
(async () => {
  const [html, out, width] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: parseInt(width), height: 1200 } });
  await p.goto('file://' + html, { waitUntil: 'load' });
  const el = await p.$('#sheet');
  await el.screenshot({ path: out });
  await b.close();
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("-o", "--out", default="out/plan.png")
    ap.add_argument("--geometry", help="override the geometry file named in the plan")
    args = ap.parse_args()

    with open(args.plan) as fh:
        plan = json.load(fh)
    geo_path = args.geometry or plan["geometry"]
    with open(geo_path) as fh:
        geo = json.load(fh)

    cal = plan["calibration"]
    scale = Scale(geo["gridlines_y"], cal["top_gridline_price"], cal["step"])
    html = build_html(plan, geo, scale)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "sheet.html")
        jp = os.path.join(td, "shot.js")
        with open(hp, "w") as fh:
            fh.write(html)
        with open(jp, "w") as fh:
            fh.write(SHOT_JS)
        crop = plan.get("crop") or [0, 0, geo["image"]["width"], geo["image"]["height"]]
        width = crop[2] - crop[0]
        env = dict(os.environ, NODE_PATH="/opt/node22/lib/node_modules")
        subprocess.run(
            ["node", jp, hp, os.path.abspath(args.out), str(width)],
            check=True, env=env,
        )

    lo = scale.price(crop[3])
    hi = scale.price(crop[1])
    print(f"wrote {args.out}")
    print(f"crop spans roughly {lo:,.0f} to {hi:,.0f} on the price axis")


if __name__ == "__main__":
    main()
