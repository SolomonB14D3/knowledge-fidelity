#!/usr/bin/env python3
"""
NoetherSolve Results Dashboard
───────────────────────────────
Reads results/candidates.tsv and generates a self-contained HTML report
at results/dashboard.html with embedded charts.

Usage:
    python dashboard.py                  # write results/dashboard.html
    python dashboard.py --open           # write + open in default browser
    python dashboard.py --out my.html    # custom output path
    python dashboard.py --png            # also save individual PNG charts
"""

import argparse
import base64
import io
import os
import sys
import csv
import textwrap
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib/numpy not found — charts will be text-only.\n"
          "  pip install matplotlib numpy", file=sys.stderr)

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_TSV = os.path.join(HERE, "results", "candidates.tsv")

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "DUAL-PASS":              "#2ecc71",   # green
    "QUADRANT3→FLIPPED":      "#27ae60",   # dark green (also a win)
    "ORACLE-FAIL+CHECKER-PASS": "#e67e22", # orange  (open gap)
    "CHECKER-FAIL":           "#e74c3c",   # red
    "ORACLE-FAIL":            "#c0392b",   # dark red
    "pending":                "#95a5a6",   # grey
}
QUADRANT_LABEL = {
    "DUAL-PASS":              "Q1 Dual-Pass",
    "QUADRANT3→FLIPPED":      "Q3→Flipped",
    "ORACLE-FAIL+CHECKER-PASS": "Q3 Open Gap",
    "CHECKER-FAIL":           "Q4 Checker-Fail",
    "ORACLE-FAIL":            "Oracle-Fail",
    "pending":                "Pending",
}

# ─────────────────────────────────────────────────────────────────────────────
# TSV parser
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    """Parse candidates.tsv into a list of row dicts."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Parse margin: may be "−29.959→+3.987" or "+4.50" or "pending"
            raw_margin = row.get("margin_mean", "").strip()
            margin_start = None
            margin_end   = None
            if "→" in raw_margin:
                parts = raw_margin.split("→")
                try:
                    margin_start = float(parts[0].replace("−", "-").replace("–", "-"))
                    margin_end   = float(parts[1].replace("−", "-").replace("–", "-"))
                except ValueError:
                    pass
            elif raw_margin not in ("pending", "baseline", ""):
                try:
                    margin_start = float(raw_margin.replace("−", "-").replace("–", "-"))
                    margin_end   = margin_start
                except ValueError:
                    pass

            # Parse frac_var from classification column
            frac_var = None
            classification = row.get("classification", "")
            for token in classification.split(";"):
                token = token.strip()
                if "frac_var=" in token:
                    try:
                        frac_var = float(token.split("frac_var=")[1].split()[0].rstrip(","))
                    except (ValueError, IndexError):
                        pass

            rows.append({
                "timestamp":      row.get("timestamp", ""),
                "hypothesis":     row.get("hypothesis", ""),
                "verdict":        row.get("verdict", "pending").strip(),
                "classification": classification,
                "margin_raw":     raw_margin,
                "margin_start":   margin_start,
                "margin_end":     margin_end,
                "frac_var":       frac_var,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators (return PNG bytes encoded as base64 string)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def chart_quadrant_donut(rows: list[dict]) -> str:
    """Donut chart of verdict distribution."""
    from collections import Counter
    counts = Counter(r["verdict"] for r in rows)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLOURS.get(l, "#bdc3c7") for l in labels]
    display_labels = [QUADRANT_LABEL.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    wedges, _ = ax.pie(
        values, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2),
    )
    # Centre label
    total = sum(values)
    ax.text(0, 0, f"{total}\ncandidates", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")

    ax.legend(
        wedges, [f"{l}  ({v})" for l, v in zip(display_labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        frameon=False, fontsize=9, labelcolor="white",
        ncol=2,
    )
    ax.set_title("Verdict Distribution", color="white", fontsize=13, pad=12)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_margin_bar(rows: list[dict]) -> str:
    """Horizontal bar chart — baseline margin (and final margin) per candidate."""
    # Only rows with numeric margin_start
    valid = [r for r in rows if r["margin_start"] is not None]
    if not valid:
        return ""

    labels   = [textwrap.shorten(r["hypothesis"], 55) for r in valid]
    m_start  = [r["margin_start"] for r in valid]
    m_end    = [r["margin_end"]   if r["margin_end"] is not None else r["margin_start"]
                for r in valid]
    verdicts = [r["verdict"] for r in valid]
    colors   = [COLOURS.get(v, "#95a5a6") for v in verdicts]

    n = len(valid)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * n + 1.5)), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    y = np.arange(n)
    bar_h = 0.35

    # Baseline bars (translucent)
    ax.barh(y + bar_h/2, m_start, height=bar_h,
            color=[c + "66" for c in colors], label="baseline")

    # Final bars (solid, shown only when different from baseline)
    has_repair = any(r["margin_end"] != r["margin_start"] for r in valid
                     if r["margin_end"] is not None)
    if has_repair:
        ax.barh(y - bar_h/2, m_end, height=bar_h,
                color=colors, label="after adapter")

    ax.axvline(0, color="#586069", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#c9d1d9")
    ax.set_xlabel("Oracle margin  (log P(truth) − log P(best distractor))",
                  color="#8b949e", fontsize=9)
    ax.set_title("Oracle Margins per Candidate", color="white", fontsize=13)
    fig.tight_layout()
    if has_repair:
        # Figure-level legend anchored above the axes, below the figure top edge —
        # uses figure coordinates so it can never overlap axes content or labels.
        baseline_patch = mpatches.Patch(color="#2ecc7166", label="baseline")
        adapted_patch  = mpatches.Patch(color="#2ecc71",   label="after adapter")
        fig.legend(handles=[baseline_patch, adapted_patch],
                   loc="upper right", bbox_to_anchor=(0.98, 0.98),
                   bbox_transform=fig.transFigure,
                   frameon=False, fontsize=8, labelcolor="white", ncol=1)
    return _fig_to_b64(fig)


def chart_frac_var_scatter(rows: list[dict]) -> str:
    """frac_var vs oracle margin scatter — the 'quality space'."""
    valid = [r for r in rows
             if r["frac_var"] is not None and r["margin_end"] is not None]
    if not valid:
        return ""

    fv      = np.array([r["frac_var"]   for r in valid])
    mg      = np.array([r["margin_end"] for r in valid])
    colors  = [COLOURS.get(r["verdict"], "#95a5a6") for r in valid]
    labels  = [textwrap.shorten(r["hypothesis"], 40) for r in valid]

    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    sc = ax.scatter(fv, mg, c=colors, s=80, zorder=3, edgecolors="#30363d", linewidths=0.5)

    # Annotate each point
    for i, (x, y_val, lbl) in enumerate(zip(fv, mg, labels)):
        ax.annotate(lbl, (x, y_val), textcoords="offset points",
                    xytext=(5, 2), fontsize=6.5, color="#8b949e")

    # Target zone shading: frac_var < 5e-3 AND margin > 0
    ax.axhline(0,    color="#586069", linewidth=1, linestyle="--")
    ax.axvline(5e-3, color="#586069", linewidth=1, linestyle="--")
    ax.fill_betweenx([-999, 0], 0, 5e-3, alpha=0.06, color="red")
    ax.fill_betweenx([0, 999],  0, 5e-3, alpha=0.10, color="green",
                     label="target zone (checker PASS + oracle PASS)")

    ax.set_xscale("log")
    ax.set_xlabel("frac_var  σ/|mean|  (lower = better conserved)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Oracle margin (positive = model knows it)", color="#8b949e", fontsize=9)
    ax.set_title("Quality Space: frac_var vs Oracle Margin", color="white", fontsize=13)

    # Legend patches
    patches = [mpatches.Patch(color=c, label=QUADRANT_LABEL.get(v, v))
               for v, c in COLOURS.items() if v in {r["verdict"] for r in valid}]
    patches.append(mpatches.Patch(color="green", alpha=0.3, label="target zone"))
    ax.legend(handles=patches, loc="upper right", frameon=False,
              fontsize=7.5, labelcolor="white")

    ax.set_xlim(max(1e-8, fv.min() * 0.1), max(fv) * 10)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_discovery_timeline(rows: list[dict]) -> str:
    """Timeline of discoveries (DUAL-PASS + FLIPPED)."""
    wins = [r for r in rows if r["verdict"] in ("DUAL-PASS", "QUADRANT3→FLIPPED")]
    if not wins:
        return ""

    fig, ax = plt.subplots(figsize=(8, max(2, 0.6 * len(wins) + 1.2)), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    for i, r in enumerate(wins):
        c = COLOURS.get(r["verdict"], "#2ecc71")
        ax.scatter(i, r["margin_end"] or 0, color=c, s=120, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(textwrap.shorten(r["hypothesis"], 50),
                    (i, r["margin_end"] or 0),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color="#c9d1d9", va="center")

    ax.axhline(0, color="#586069", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels([r["timestamp"] for r in wins], rotation=30, ha="right",
                       fontsize=8, color="#8b949e")
    ax.set_ylabel("Final oracle margin", color="#8b949e", fontsize=9)
    ax.set_title("Discovery Timeline (DUAL-PASS + FLIPPED)", color="white", fontsize=13)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NoetherSolve Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    padding: 24px;
  }}
  h1 {{ color: #e6edf3; font-size: 1.6em; margin-bottom: 4px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 32px; font-size: 0.9em; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 24px;
    margin-bottom: 32px;
  }}
  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
  }}
  .card h2 {{ color: #e6edf3; font-size: 1em; margin-bottom: 12px; }}
  .card img {{ width: 100%; height: auto; border-radius: 4px; }}
  .wide {{ grid-column: 1 / -1; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
  }}
  th {{
    background: #21262d;
    color: #8b949e;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #30363d;
  }}
  td {{
    padding: 7px 10px;
    border-bottom: 1px solid #21262d;
    vertical-align: top;
    max-width: 340px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    color: #0d1117;
  }}
  .pos {{ color: #2ecc71; }}
  .neg {{ color: #e74c3c; }}
  .footer {{
    color: #484f58;
    font-size: 0.78em;
    margin-top: 16px;
    text-align: center;
  }}
  .stat-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }}
  .stat {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 20px;
    min-width: 120px;
  }}
  .stat .val {{ font-size: 1.9em; font-weight: 700; color: #e6edf3; }}
  .stat .lbl {{ color: #8b949e; font-size: 0.82em; margin-top: 2px; }}
</style>
</head>
<body>

<h1>🔬 NoetherSolve Dashboard</h1>
<p class="subtitle">Find what LLMs don't know about what the universe conserves. Then fix it. &nbsp;·&nbsp; Generated {generated_at}</p>

<div class="stat-row">
  <div class="stat"><div class="val">{n_total}</div><div class="lbl">Candidates tested</div></div>
  <div class="stat"><div class="val" style="color:#2ecc71">{n_wins}</div><div class="lbl">Discoveries (PASS or FLIPPED)</div></div>
  <div class="stat"><div class="val" style="color:#e67e22">{n_open}</div><div class="lbl">Open gaps (checker PASS)</div></div>
  <div class="stat"><div class="val" style="color:#e74c3c">{n_fail}</div><div class="lbl">Checker failures</div></div>
  <div class="stat"><div class="val">{n_domains}</div><div class="lbl">Domains active</div></div>
</div>

<div class="grid">
  {chart_donut}
  {chart_scatter}
  <div class="card wide">
    <h2>Oracle Margins per Candidate</h2>
    {chart_margin}
  </div>
  {chart_timeline}
</div>

<div class="card">
  <h2>All Candidates</h2>
  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>Hypothesis</th>
        <th>frac_var</th>
        <th>Margin</th>
        <th>Verdict</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</div>

<p class="footer">NoetherSolve · Built on the STEM Truth Oracle (Paper 9, DOI: 10.5281/zenodo.19005729) and Snap-On Communication Modules (Paper 8, DOI: 10.5281/zenodo.18902616)</p>
</body>
</html>
"""


VERDICT_COLOURS_CSS = {
    "DUAL-PASS":                "#2ecc71",
    "QUADRANT3→FLIPPED":        "#27ae60",
    "ORACLE-FAIL+CHECKER-PASS": "#e67e22",
    "CHECKER-FAIL":             "#e74c3c",
    "ORACLE-FAIL":              "#c0392b",
    "pending":                  "#586069",
}


def _img_card(title: str, b64: str, wide: bool = False) -> str:
    if not b64:
        return ""
    cls = "card wide" if wide else "card"
    return f'<div class="{cls}"><h2>{title}</h2><img src="data:image/png;base64,{b64}" alt="{title}"></div>'


def _verdict_badge(verdict: str) -> str:
    colour = VERDICT_COLOURS_CSS.get(verdict, "#586069")
    label  = QUADRANT_LABEL.get(verdict, verdict)
    return f'<span class="badge" style="background:{colour}">{label}</span>'


def _margin_cell(row: dict) -> str:
    raw = row["margin_raw"]
    if not raw or raw in ("pending", "baseline", ""):
        return f'<td style="color:#586069">{raw or "—"}</td>'
    if row["margin_end"] is not None and row["margin_end"] > 0:
        cls = "pos"
    elif row["margin_end"] is not None:
        cls = "neg"
    else:
        cls = ""
    return f'<td class="{cls}">{raw}</td>'


def build_table_rows(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        fv_str = f"{r['frac_var']:.2e}" if r["frac_var"] is not None else "—"
        notes  = r["classification"]
        # Truncate notes for display
        if len(notes) > 90:
            notes = notes[:87] + "…"
        parts.append(
            f"<tr>"
            f"<td>{r['timestamp']}</td>"
            f'<td style="max-width:260px;white-space:normal">{r["hypothesis"]}</td>'
            f"<td>{fv_str}</td>"
            f"{_margin_cell(r)}"
            f"<td>{_verdict_badge(r['verdict'])}</td>"
            f'<td style="color:#8b949e;white-space:normal;max-width:280px">{notes}</td>'
            f"</tr>"
        )
    return "\n      ".join(parts)


def build_stats(rows: list[dict]) -> dict:
    n_total  = len(rows)
    n_wins   = sum(1 for r in rows if r["verdict"] in ("DUAL-PASS", "QUADRANT3→FLIPPED"))
    n_open   = sum(1 for r in rows if r["verdict"] == "ORACLE-FAIL+CHECKER-PASS")
    n_fail   = sum(1 for r in rows if r["verdict"] == "CHECKER-FAIL")
    # Infer domains from hypothesis text
    domains  = set()
    for r in rows:
        h = r["hypothesis"].lower()
        if "vortex" in h or "vp" in h.split()[0]:
            domains.add("point-vortex")
        elif "figure" in h or "3-body" in h or "r12" in h or "choreograph" in h:
            domains.add("figure-8 3-body")
        elif "kinetic" in h or "pilot" in h:
            domains.add("kinetic-energy pilot")
        else:
            domains.add("unknown")
    return dict(n_total=n_total, n_wins=n_wins, n_open=n_open,
                n_fail=n_fail, n_domains=len(domains))


def generate_dashboard(out_path: str, png_dir: str | None = None) -> None:
    rows = load_candidates(CANDIDATES_TSV)
    if not rows:
        print(f"  No candidates found in {CANDIDATES_TSV}", file=sys.stderr)
        return

    stats = build_stats(rows)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if HAS_MPL:
        b64_donut    = chart_quadrant_donut(rows)
        b64_margin   = chart_margin_bar(rows)
        b64_scatter  = chart_frac_var_scatter(rows)
        b64_timeline = chart_discovery_timeline(rows)

        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
            for name, b64 in [("quadrant_donut.png", b64_donut),
                               ("margin_bar.png",     b64_margin),
                               ("frac_var_scatter.png", b64_scatter),
                               ("discovery_timeline.png", b64_timeline)]:
                if b64:
                    with open(os.path.join(png_dir, name), "wb") as f:
                        f.write(base64.b64decode(b64))
            print(f"  PNGs saved to: {png_dir}")

        chart_donut    = _img_card("Verdict Distribution", b64_donut)
        chart_scatter  = _img_card("Quality Space", b64_scatter)
        chart_margin   = f'<img src="data:image/png;base64,{b64_margin}" alt="margins" style="width:100%;height:auto;border-radius:4px">' if b64_margin else "<p style='color:#586069'>No margin data</p>"
        chart_timeline = _img_card("Discovery Timeline", b64_timeline)
    else:
        chart_donut    = '<div class="card"><p style="color:#586069">Install matplotlib for charts.</p></div>'
        chart_scatter  = ""
        chart_margin   = "<p style='color:#586069'>Install matplotlib for charts.</p>"
        chart_timeline = ""

    html = HTML_TEMPLATE.format(
        generated_at  = generated_at,
        chart_donut   = chart_donut,
        chart_scatter = chart_scatter,
        chart_margin  = chart_margin,
        chart_timeline = chart_timeline,
        table_rows    = build_table_rows(rows),
        **stats,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard written: {out_path}  ({len(rows)} candidates)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate NoetherSolve results dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out",  default=os.path.join(HERE, "results", "dashboard.html"),
                        help="Output HTML path")
    parser.add_argument("--open", action="store_true", help="Open dashboard in browser after writing")
    parser.add_argument("--png",  action="store_true", help="Also save individual PNG charts")
    args = parser.parse_args()

    png_dir = os.path.join(HERE, "results", "charts") if args.png else None
    generate_dashboard(args.out, png_dir)

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(args.out)}")
        print(f"  Opened in browser.")


if __name__ == "__main__":
    main()
