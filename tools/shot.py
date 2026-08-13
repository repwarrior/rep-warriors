#!/usr/bin/env python3
"""Screenshot a local HTML file to PNG with headless Chromium.

    python3 shot.py page.html -o out.png --width 1290 --selector "#sheet"

Used for the annotation overlays and for standalone graphics that need the same
typography and colour handling as the rest of the pipeline.
"""

import argparse
import os
import subprocess
import tempfile

SHOT_JS = """
const { chromium } = require('playwright');
(async () => {
  const [html, out, width, selector] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: parseInt(width), height: 1200 } });
  await p.goto('file://' + html, { waitUntil: 'load' });
  const el = selector ? await p.$(selector) : null;
  if (el) await el.screenshot({ path: out });
  else await p.screenshot({ path: out, fullPage: true });
  await b.close();
})();
"""


def shoot(html_path, out, width=1290, selector=None):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        jp = os.path.join(td, "shot.js")
        with open(jp, "w") as fh:
            fh.write(SHOT_JS)
        cmd = ["node", jp, os.path.abspath(html_path), os.path.abspath(out), str(width)]
        if selector:
            cmd.append(selector)
        subprocess.run(cmd, check=True,
                       env=dict(os.environ, NODE_PATH="/opt/node22/lib/node_modules"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("-o", "--out", default="out/page.png")
    ap.add_argument("--width", type=int, default=1290)
    ap.add_argument("--selector", help="screenshot just this element")
    args = ap.parse_args()
    shoot(args.html, args.out, args.width, args.selector)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
