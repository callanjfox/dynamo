#!/usr/bin/env python3
"""Build the customer performance diagnosis page as one self-contained HTML file.

    python3 build_diag.py --scenarios busy=<run> ...      # writes diag/data/*.json
    python3 diag/build_static.py diagnosis.html

The page needs only a browser (it loads d3 from cdnjs). The data is embedded, so the file can be
emailed, opened locally or hosted anywhere. diag/index.html and diag/app.js are the same sources a
dashboard host would load; this file supplies the small `dash` object they expect.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

TOKENS = """
:root {
  color-scheme: light dark;
  --color-bg: #ffffff; --color-panel: #f6f6f4; --color-fg: #1f2328; --color-fg-muted: #5a606a;
  --color-border-line: #dcdad4; --color-bad: #c33;
  --cds-chart-grid: #e8e6e0; --cds-chart-axis: #8a8f98; --cds-chart-reference: #8a8f98; --cds-chart-status-critical: #c33;
  --cds-text-secondary: #5a606a; --cds-surface-popover: #ffffff; --cds-border: #dcdad4;
  --cds-shadow-popover: 0 4px 16px rgba(0, 0, 0, .18);
  --cds-radius: 8px; --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 12px; --cds-gap-lg: 20px; --cds-gap-xl: 32px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 16px;
  --cds-font-size-caption: 12.5px; --cds-font-size-body: 14.5px; --cds-font-size-heading: 18px; --cds-font-size-title: 26px;
  --cds-font-weight-medium: 600;
  --font-anthropic-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; --font-anthropic-serif: Georgia, "Times New Roman", serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --color-bg: #16171a; --color-panel: #1d1f23; --color-fg: #e8e6e1; --color-fg-muted: #a3a7ae; --color-border-line: #33363c; --color-bad: #ff7b72;
    --cds-chart-grid: #2c2f35; --cds-chart-axis: #7d828b; --cds-chart-reference: #7d828b; --cds-chart-status-critical: #ff7b72;
    --cds-text-secondary: #a3a7ae; --cds-surface-popover: #26282d; --cds-border: #33363c;
  }
}
:root[data-theme="dark"] {
  --color-bg: #16171a; --color-panel: #1d1f23; --color-fg: #e8e6e1; --color-fg-muted: #a3a7ae; --color-border-line: #33363c; --color-bad: #ff7b72;
  --cds-chart-grid: #2c2f35; --cds-chart-axis: #7d828b; --cds-chart-reference: #7d828b; --cds-chart-status-critical: #ff7b72;
  --cds-text-secondary: #a3a7ae; --cds-surface-popover: #26282d; --cds-border: #33363c;
}
body { margin: 0; background: var(--color-bg); color: var(--color-fg); font: var(--cds-font-size-body)/1.55 var(--font-anthropic-sans);
  padding: 24px max(16px, env(safe-area-inset-right, 0px)) 56px max(16px, env(safe-area-inset-left, 0px)); }
"""

COLORS = ["#2f6fed", "#e8743b", "#2a9d6a", "#9b5de5", "#c9a227", "#d1495b"]


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "diagnosis.html")
    data = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "data").glob("*.json"))}
    for need in ("timeline", "run_meta"):
        if need not in data:
            sys.exit(f"missing diag/data/{need}.json: run build_diag.py first")
    shim = "window.dash={colors:%s,data:(id)=>({status:DATA[id]?'ok':'empty',data:DATA[id]}),onData:(f)=>{window.addEventListener('load',()=>f());}};" % json.dumps(COLORS)
    page = "".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Customer performance diagnosis</title><style>", TOKENS, "</style></head><body>",
        (HERE / "index.html").read_text(),
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>',
        "<script>const DATA=", json.dumps(data, separators=(",", ":")), ";", shim, "</script>",
        "<script>", (HERE / "app.js").read_text(), "</script></body></html>",
    ])
    out.write_text(page)
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB, {len(data)} datasets")


if __name__ == "__main__":
    main()
