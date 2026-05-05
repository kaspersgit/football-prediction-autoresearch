import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Football Prediction Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; background: #f9f9f9; color: #222; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 2px solid #283593; padding-bottom: 6px; }}
  .metrics {{ display: flex; gap: 24px; margin: 24px 0; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 20px 32px; box-shadow: 0 2px 8px rgba(0,0,0,.1); text-align: center; flex: 1; }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; color: {roi_color}; }}
  .metric-card:nth-child(1) .value {{ color: #1565c0; }}
  .metric-card:nth-child(3) .value {{ color: #2e7d32; }}
  .metric-card .label {{ color: #777; font-size: 0.9em; margin-top: 4px; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 16px 0; }}
  .explanation {{ background: #e8eaf6; border-left: 4px solid #3949ab; padding: 12px 18px; border-radius: 4px; margin: 12px 0; }}
  /* ── filter bar ─────────────────────────────────────────────────────────── */
  .filter-bar {{
    background: white; border-radius: 8px; padding: 14px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 20px 0;
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
  }}
  .filter-group {{ display: flex; flex-direction: column; gap: 4px; min-width: 140px; }}
  .filter-group label {{ font-size: 0.82em; color: #555; font-weight: 600; }}
  .filter-group input[type=range] {{ width: 100%; accent-color: #3949ab; }}
  .outcome-toggles {{ display: flex; gap: 6px; }}
  .outcome-btn {{
    padding: 4px 14px; border-radius: 20px; border: 2px solid #3949ab;
    background: white; color: #3949ab; cursor: pointer; font-size: 0.85em; font-weight: 600;
    transition: background 0.15s, color 0.15s;
  }}
  .outcome-btn.active {{ background: #3949ab; color: white; }}
  .filter-count {{
    margin-left: auto; background: #e8eaf6; border-radius: 20px;
    padding: 6px 16px; font-size: 0.9em; color: #3949ab; font-weight: 600;
    white-space: nowrap;
  }}
  /* ── canvas cards ───────────────────────────────────────────────────────── */
  .top-bets-card {{
    background: white; border-radius: 8px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 20px 0;
  }}
  .top-bets-card h3 {{ margin: 0 0 14px 0; color: #283593; font-size: 1.05em; }}
  canvas {{ display: block; width: 100%; }}
</style>
</head>
<body>
<h1>Football Prediction Evaluation Report</h1>
<h2>Summary Metrics</h2>
<div class="metrics">
  <div class="metric-card"><div class="value">{accuracy:.1%}</div><div class="label">Accuracy</div></div>
  <div class="metric-card"><div class="value" style="color:{roi_color}">{roi:+.2f}%</div><div class="label">ROI (Return on Investment)</div></div>
  <div class="metric-card"><div class="value">{stability:.3f}</div><div class="label">Profit Stability (Sharpe-like)</div></div>
</div>
<div class="explanation">
  <b>ROI</b>: total profit / total staked × 100. Positive means profit over the test period.<br>
  <b>Stability</b>: mean profit per bet / std(profit per bet). Higher = more consistent returns. Above 0.05 is good.
</div>

<!-- ── filter bar ─────────────────────────────────────────────────────────── -->
<div class="filter-bar">
  <div class="filter-group">
    <label>Min edge <span id="lbl-min-edge">0%</span></label>
    <input type="range" id="filter-min-edge" min="-10" max="15" step="1" value="0"
           oninput="document.getElementById('lbl-min-edge').textContent=this.value+'%';applyFilters()">
  </div>
  <div class="filter-group">
    <label>Max edge <span id="lbl-max-edge">15%</span></label>
    <input type="range" id="filter-max-edge" min="0" max="20" step="1" value="15"
           oninput="document.getElementById('lbl-max-edge').textContent=this.value+'%';applyFilters()">
  </div>
  <div class="filter-group">
    <label>Outcome</label>
    <div class="outcome-toggles">
      <button class="outcome-btn active" id="btn-All" onclick="setOutcome('All')">All</button>
      <button class="outcome-btn" id="btn-H" onclick="setOutcome('H')">H</button>
      <button class="outcome-btn" id="btn-D" onclick="setOutcome('D')">D</button>
      <button class="outcome-btn" id="btn-A" onclick="setOutcome('A')">A</button>
    </div>
  </div>
  <div class="filter-group">
    <label>Min odds <span id="lbl-min-odds">1.0</span></label>
    <input type="range" id="filter-min-odds" min="1.0" max="3.0" step="0.1" value="1.0"
           oninput="document.getElementById('lbl-min-odds').textContent=parseFloat(this.value).toFixed(1);applyFilters()">
  </div>
  <div class="filter-group">
    <label>Max odds <span id="lbl-max-odds">4.0</span></label>
    <input type="range" id="filter-max-odds" min="1.5" max="10.0" step="0.5" value="4.0"
           oninput="document.getElementById('lbl-max-odds').textContent=parseFloat(this.value).toFixed(1);applyFilters()">
  </div>
  <div class="filter-count">Showing <span id="visible-count">…</span> bets</div>
</div>

<h2>Probability Calibration</h2>
<div class="top-bets-card" id="calibration-container">
  <h3>Probability Calibration (filtered)</h3>
  <canvas id="calibration-canvas" height="300"></canvas>
</div>

<h2>ROI by Edge Bucket</h2>
<div class="top-bets-card" id="edge-bucket-container">
  <h3>ROI by Edge Bucket</h3>
  <canvas id="edge-bucket-canvas" height="220"></canvas>
</div>

<h2>Profit Distribution (sorted by profit)</h2>
<img src="data:image/png;base64,{bar_chart_b64}" alt="Profit distribution bar chart">
<h2>Cumulative Profit Over Time</h2>
<img src="data:image/png;base64,{cumulative_chart_b64}" alt="Cumulative profit chart">
<h2>Bet Details</h2>
<p>Total bets: {n_bets} &nbsp;|&nbsp; Correct: {n_correct} &nbsp;|&nbsp; Wrong: {n_wrong}</p>

<script>
// ── filter state ──────────────────────────────────────────────────────────────
var filteredBets = [];
var _activeOutcome = 'All';

function setOutcome(outcome) {{
  _activeOutcome = outcome;
  ['All','H','D','A'].forEach(function(o) {{
    var btn = document.getElementById('btn-' + o);
    if (btn) btn.classList.toggle('active', o === outcome);
  }});
  applyFilters();
}}

function applyFilters() {{
  var minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  var maxEdge = parseFloat(document.getElementById('filter-max-edge').value) / 100;
  var minOdds = parseFloat(document.getElementById('filter-min-odds').value);
  var maxOdds = parseFloat(document.getElementById('filter-max-odds').value);

  if (maxEdge < minEdge) maxEdge = minEdge;
  if (maxOdds < minOdds) maxOdds = minOdds;

  filteredBets = ALL_BETS.filter(function(b) {{
    if (b.edge < minEdge || b.edge > maxEdge) return false;
    if (b.odds < minOdds || b.odds > maxOdds) return false;
    if (_activeOutcome !== 'All' && b.outcome !== _activeOutcome) return false;
    return true;
  }});

  var countEl = document.getElementById('visible-count');
  if (countEl) countEl.textContent = filteredBets.length;

  if (typeof rebuildCalibration === 'function') rebuildCalibration(filteredBets);
  if (typeof rebuildEdgeBuckets === 'function') rebuildEdgeBuckets(filteredBets);
}}

// ── calibration chart (canvas) ────────────────────────────────────────────────
function rebuildCalibration(bets) {{
  var canvas = document.getElementById('calibration-canvas');
  if (!canvas) return;
  var container = document.getElementById('calibration-container');
  var W = Math.max((container ? container.clientWidth : 0) - 48, 360);
  var H = 300;
  var dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  var pl = 52, pr = 24, pt = 16, pb = 60;
  var cw = W - pl - pr, ch = H - pt - pb;

  function sx(v) {{ return pl + v * cw; }}
  function sy(v) {{ return pt + (1 - v) * ch; }}

  // Perfect calibration diagonal
  ctx.strokeStyle = '#bbb';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(0), sy(0));
  ctx.lineTo(sx(1), sy(1));
  ctx.stroke();
  ctx.setLineDash([]);

  // Axes
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pl, pt); ctx.lineTo(pl, pt + ch); ctx.lineTo(pl + cw, pt + ch);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#888';
  ctx.font = '11px Arial,sans-serif';
  ctx.textAlign = 'center';
  for (var t = 0; t <= 10; t++) {{
    var frac = t / 10;
    var xp = sx(frac);
    ctx.fillText((frac * 100).toFixed(0) + '%', xp, pt + ch + 16);
    ctx.fillStyle = '#eee';
    ctx.fillRect(xp - 0.5, pt, 1, ch);
    ctx.fillStyle = '#888';
  }}
  ctx.textAlign = 'right';
  for (var r = 0; r <= 5; r++) {{
    var rv = r / 5;
    var yp = sy(rv);
    ctx.fillText((rv * 100).toFixed(0) + '%', pl - 6, yp + 4);
    ctx.fillStyle = '#eee';
    ctx.fillRect(pl, yp - 0.5, cw, 1);
    ctx.fillStyle = '#888';
  }}
  ctx.textAlign = 'center';
  ctx.fillStyle = '#555';
  ctx.font = '12px Arial,sans-serif';
  ctx.fillText('Mean predicted probability', pl + cw / 2, H - 8);
  ctx.save();
  ctx.translate(14, pt + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Actual win rate', 0, 0);
  ctx.restore();

  var OUTCOMES = [
    {{key: 'H', color: '#1565c0', label: 'Home (H)'}},
    {{key: 'D', color: '#757575', label: 'Draw (D)'}},
    {{key: 'A', color: '#e65100', label: 'Away (A)'}}
  ];
  var N_BINS = 8;

  OUTCOMES.forEach(function(oc) {{
    var subset = bets.filter(function(b) {{ return b.outcome === oc.key; }});
    if (subset.length < 5) return;

    // Build bins
    var bins = [];
    for (var bi = 0; bi < N_BINS; bi++) bins.push([]);
    subset.forEach(function(b) {{
      var idx = Math.min(Math.floor(b.model_prob * N_BINS), N_BINS - 1);
      bins[idx].push(b);
    }});

    var points = [];
    bins.forEach(function(bin, bi) {{
      if (bin.length < 5) return;
      var meanProb = bin.reduce(function(s, b) {{ return s + b.model_prob; }}, 0) / bin.length;
      var winRate = bin.filter(function(b) {{ return b.y_true === b.outcome; }}).length / bin.length;
      points.push({{x: meanProb, y: winRate}});
    }});

    if (points.length < 2) return;

    ctx.strokeStyle = oc.color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    points.forEach(function(p, i) {{
      var xp = sx(p.x), yp = sy(p.y);
      if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
    }});
    ctx.stroke();

    // Dots
    points.forEach(function(p) {{
      ctx.beginPath();
      ctx.arc(sx(p.x), sy(p.y), 5, 0, 2 * Math.PI);
      ctx.fillStyle = oc.color;
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }});
  }});

  // Legend
  var legendX = pl + 8, legendY = pt + ch + 36;
  var legendItems = [
    {{color: '#bbb', label: 'Perfect calibration', dashed: true}},
    {{color: '#1565c0', label: 'Home (H)'}},
    {{color: '#757575', label: 'Draw (D)'}},
    {{color: '#e65100', label: 'Away (A)'}}
  ];
  var lx = legendX;
  legendItems.forEach(function(item) {{
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    if (item.dashed) ctx.setLineDash([5, 3]); else ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(lx, legendY); ctx.lineTo(lx + 22, legendY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#444';
    ctx.font = '11px Arial,sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(item.label, lx + 26, legendY + 4);
    lx += ctx.measureText(item.label).width + 48;
  }});
}}

// ── ROI by edge bucket (canvas) ───────────────────────────────────────────────
function rebuildEdgeBuckets(bets) {{
  var canvas = document.getElementById('edge-bucket-canvas');
  if (!canvas) return;
  var container = document.getElementById('edge-bucket-container');
  var W = Math.max((container ? container.clientWidth : 0) - 48, 360);
  var H = 220;
  var dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  var BUCKETS = [
    {{label: '<0%',  test: function(e) {{ return e < 0; }}}},
    {{label: '0–3%', test: function(e) {{ return e >= 0 && e < 0.03; }}}},
    {{label: '3–6%', test: function(e) {{ return e >= 0.03 && e < 0.06; }}}},
    {{label: '6–9%', test: function(e) {{ return e >= 0.06 && e < 0.09; }}}},
    {{label: '>9%',  test: function(e) {{ return e >= 0.09; }}}}
  ];

  var bucketData = BUCKETS.map(function(bk) {{
    var subset = bets.filter(function(b) {{ return bk.test(b.edge); }});
    var n = subset.length;
    var roi = 0;
    if (n > 0) {{
      var totalReturn = subset.reduce(function(s, b) {{
        return s + (b.y_true === b.outcome ? (b.odds - 1) : -1);
      }}, 0);
      roi = (totalReturn / n) * 100;
    }}
    return {{label: bk.label, roi: roi, n: n}};
  }});

  var pl = 80, pr = 32, pt = 16, pb = 36;
  var cw = W - pl - pr, ch = H - pt - pb;
  var nBuckets = bucketData.length;
  var barH = Math.floor(ch / nBuckets * 0.72);
  var barGap = (ch - barH * nBuckets) / (nBuckets + 1);

  var maxAbsRoi = Math.max(20, Math.max.apply(null, bucketData.map(function(d) {{ return Math.abs(d.roi); }})));
  maxAbsRoi = Math.ceil(maxAbsRoi / 5) * 5;

  function sx(roi) {{ return pl + cw / 2 + (roi / maxAbsRoi) * (cw / 2); }}
  var x0 = sx(0);

  // Vertical grid lines
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  [-maxAbsRoi, -maxAbsRoi/2, 0, maxAbsRoi/2, maxAbsRoi].forEach(function(v) {{
    var xp = sx(v);
    ctx.beginPath(); ctx.moveTo(xp, pt); ctx.lineTo(xp, pt + ch); ctx.stroke();
  }});

  // Zero line
  ctx.strokeStyle = '#bbb';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x0, pt); ctx.lineTo(x0, pt + ch); ctx.stroke();

  // Axis x labels
  ctx.fillStyle = '#888';
  ctx.font = '11px Arial,sans-serif';
  ctx.textAlign = 'center';
  [-maxAbsRoi, -maxAbsRoi/2, 0, maxAbsRoi/2, maxAbsRoi].forEach(function(v) {{
    ctx.fillText(v.toFixed(0) + '%', sx(v), pt + ch + 16);
  }});
  ctx.fillStyle = '#555';
  ctx.fillText('ROI %', pl + cw / 2, H - 4);

  // Bars
  bucketData.forEach(function(d, i) {{
    var barY = pt + barGap + i * (barH + barGap);
    var barW = Math.abs(sx(d.roi) - x0);
    var barX = d.roi >= 0 ? x0 : x0 - barW;
    var color = d.roi >= 0 ? '#43a047' : '#e53935';

    ctx.fillStyle = color;
    ctx.fillRect(barX, barY, barW, barH);

    // Bucket label on left
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial,sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(d.label, pl - 8, barY + barH / 2 + 4);

    // n-bets label on bar
    if (d.n > 0) {{
      var labelX = d.roi >= 0 ? x0 + barW + 4 : x0 - barW - 4;
      var align = d.roi >= 0 ? 'left' : 'right';
      ctx.fillStyle = '#555';
      ctx.font = '11px Arial,sans-serif';
      ctx.textAlign = align;
      ctx.fillText('n=' + d.n, labelX, barY + barH / 2 + 4);
    }}
  }});
}}

// ── boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {{
  applyFilters();
}});
</script>
</body>
</html>"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _bar_chart(profits: pd.Series) -> str:
    sorted_profits = profits.sort_values().values
    colors = ["#e53935" if p < 0 else "#43a047" for p in sorted_profits]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(sorted_profits)), sorted_profits, color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Bet (sorted by profit)")
    ax.set_ylabel("Profit (units)")
    ax.set_title("Per-Bet Profit Distribution (sorted)")
    green_patch = mpatches.Patch(color="#43a047", label="Win")
    red_patch = mpatches.Patch(color="#e53935", label="Loss")
    ax.legend(handles=[green_patch, red_patch])
    return _fig_to_b64(fig)


def _cumulative_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["cumulative_profit"], color="#1565c0", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(
        df["Date"], df["cumulative_profit"], 0,
        where=df["cumulative_profit"] >= 0, alpha=0.15, color="#43a047"
    )
    ax.fill_between(
        df["Date"], df["cumulative_profit"], 0,
        where=df["cumulative_profit"] < 0, alpha=0.15, color="#e53935"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit Over Time")
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _all_predictions_script(all_predictions: "pd.DataFrame | None") -> str:
    if all_predictions is None or all_predictions.empty:
        return "<script>var ALL_BETS=[];</script>"
    needed = ["date", "league", "home", "away", "outcome", "model_prob",
              "implied_prob", "odds", "y_true", "edge"]
    cols = [c for c in needed if c in all_predictions.columns]
    df = all_predictions[cols].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    records = df.to_dict("records")
    return f"<script>\nvar ALL_BETS={json.dumps(records, separators=(',', ':'))};\n</script>"


def generate_report(
    results_df: pd.DataFrame,
    accuracy: float,
    roi: float,
    stability: float,
    output_path: Path,
    all_predictions: "pd.DataFrame | None" = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bar_b64 = _bar_chart(results_df["profit"])
    cum_b64 = _cumulative_chart(results_df)

    n_bets = len(results_df)
    n_correct = (results_df["y_true"] == results_df["y_pred"]).sum()
    n_wrong = n_bets - n_correct
    roi_color = "#2e7d32" if roi >= 0 else "#c62828"

    html = _HTML_TEMPLATE.format(
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        roi_color=roi_color,
        bar_chart_b64=bar_b64,
        cumulative_chart_b64=cum_b64,
        n_bets=n_bets,
        n_correct=n_correct,
        n_wrong=n_wrong,
    )
    all_bets_script = _all_predictions_script(all_predictions)
    html = html.replace("</head>", f"{all_bets_script}\n</head>", 1)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {output_path}")
