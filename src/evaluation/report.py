import json
from pathlib import Path

import pandas as pd

from src.config import LEAGUE_NAMES, SUPPORTED_LEAGUES

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
<p style="color:#777;font-size:.88em;margin:4px 0 12px">Accuracy is over all test matches. Bets, ROI, and t-stat react to the filters below.</p>
<div class="metrics">
  <div class="metric-card"><div class="value">{accuracy:.1%}</div><div class="label">Accuracy (all matches)</div></div>
  <div class="metric-card"><div class="value" id="summary-bets">{n_bets}</div><div class="label">Bets Placed</div></div>
  <div class="metric-card"><div class="value" id="summary-roi" style="color:{roi_color}">{roi:+.2f}%</div><div class="label">ROI (stake-weighted)</div></div>
  <div class="metric-card"><div class="value" id="summary-tstat">{tstat:+.2f}</div><div class="label">t-stat (≥2 = significant)</div></div>
</div>
<div class="explanation">
  <b>ROI</b>: total profit / total staked × 100. Uses flat staking: 1 unit per bet.<br>
  <b>t-stat</b>: stability × √N. Above ±2 is statistically significant at 5% level.
</div>

<!-- ── filter bar ─────────────────────────────────────────────────────────── -->
<div class="filter-bar">
  <div class="filter-group">
    <label>Min edge <span id="lbl-min-edge">0%</span></label>
    <input type="range" id="filter-min-edge" min="0" max="15" step="1" value="0"
           oninput="document.getElementById('lbl-min-edge').textContent=this.value+'%';applyFilters()">
  </div>
  <div class="filter-group">
    <label>Max edge <span id="lbl-max-edge">25%</span></label>
    <input type="range" id="filter-max-edge" min="0" max="25" step="1" value="25"
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
    <label>Max odds <span id="lbl-max-odds">10.0</span></label>
    <input type="range" id="filter-max-odds" min="1.5" max="10.0" step="0.5" value="10.0"
           oninput="document.getElementById('lbl-max-odds').textContent=parseFloat(this.value).toFixed(1);applyFilters()">
  </div>
  <div class="filter-group">
    <label>Leagues</label>
    <div class="outcome-toggles">
      {league_buttons}
    </div>
    <div class="outcome-toggles" style="margin-top:6px">
      <button class="outcome-btn" onclick="setLeaguePreset('all')">All markets</button>
      <button class="outcome-btn" onclick="setLeaguePreset('production')">Production markets</button>
    </div>
  </div>
  <div class="filter-count">Showing <span id="visible-count">…</span> bets</div>
</div>

<h2>Probability Calibration</h2>
<div class="top-bets-card" id="calibration-container">
  <h3>Probability Calibration (filtered)</h3>
  <canvas id="calibration-canvas" height="300"></canvas>
</div>

<h2>ROI by Edge Bucket</h2>
<div class="explanation">
  ROI here is <b>stake-weighted</b> (profit / stake) from placed bets in the currently selected leagues. Higher edge buckets tend to be rarer; small sample sizes make per-bucket noise high.
</div>
<div class="top-bets-card" id="edge-bucket-container">
  <h3>ROI by Edge Bucket (stake-weighted)</h3>
  <canvas id="edge-bucket-canvas" height="220"></canvas>
</div>

<h2>Outcome &amp; League Breakdown</h2>
<div style="display:flex;gap:20px;flex-wrap:wrap;margin:20px 0" id="breakdown-flex">
  <div class="top-bets-card" style="flex:1;min-width:320px">
    <h3>By Outcome</h3>
    <div id="outcome-table-container"></div>
  </div>
  <div class="top-bets-card" style="flex:1;min-width:320px">
    <h3>By League</h3>
    <div id="league-table-container"></div>
  </div>
</div>

<h2>Profit Distribution (sorted by bet profit)</h2>
<div class="top-bets-card" id="profit-bar-container">
  <h3>Per-Bet Profit (flat staking)</h3>
  <canvas id="profit-bar-canvas" height="220"></canvas>
</div>

<h2>Cumulative Profit Over Time</h2>
<div class="top-bets-card" id="cumulative-chart-container">
  <h3>Cumulative Profit (flat staking)</h3>
  <canvas id="cumulative-chart-canvas" height="260"></canvas>
</div>

<h2>Bet Details</h2>
<p>Total bets: {n_bets} &nbsp;|&nbsp; Correct: {n_correct} &nbsp;|&nbsp; Wrong: {n_wrong}</p>

<script>
// ── filter state ──────────────────────────────────────────────────────────────
var filteredBets = [];
var _activeOutcome = 'All';
var _activeLeagues={active_leagues_json};
var _productionLeagues={production_leagues_json};

function setOutcome(outcome) {{
  _activeOutcome = outcome;
  ['All','H','D','A'].forEach(function(o) {{
    var btn = document.getElementById('btn-' + o);
    if (btn) btn.classList.toggle('active', o === outcome);
  }});
  applyFilters();
}}

function toggleLeague(league) {{
  _activeLeagues[league] = !_activeLeagues[league];
  var btn = document.getElementById('btn-lg-' + league);
  if (btn) btn.classList.toggle('active', !!_activeLeagues[league]);
  applyFilters();
}}

function setLeaguePreset(preset) {{
  Object.keys(_activeLeagues).forEach(function(league) {{
    _activeLeagues[league] = preset === 'all' || _productionLeagues.indexOf(league) !== -1;
    var btn = document.getElementById('btn-lg-' + league);
    if (btn) btn.classList.toggle('active', !!_activeLeagues[league]);
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

  // BACKTEST_BETS = placed bets (stake/profit already computed, filtered by backtest config)
  filteredBets = BACKTEST_BETS.filter(function(b) {{
    if (b.edge < minEdge || b.edge > maxEdge) return false;
    if (b.odds < minOdds || b.odds > maxOdds) return false;
    if (_activeOutcome !== 'All' && b.outcome !== _activeOutcome) return false;
    if (b.league && !_activeLeagues[b.league]) return false;
    return true;
  }});

  // ALL_BETS = all predictions (used only for calibration chart)
  var filteredAllBets = ALL_BETS.filter(function(b) {{
    if (b.edge < minEdge || b.edge > maxEdge) return false;
    if (b.odds < minOdds || b.odds > maxOdds) return false;
    if (_activeOutcome !== 'All' && b.outcome !== _activeOutcome) return false;
    if (b.league && !_activeLeagues[b.league]) return false;
    return true;
  }});

  var countEl = document.getElementById('visible-count');
  if (countEl) countEl.textContent = filteredBets.length;

  updateSummaryCards(filteredBets);
  if (typeof rebuildCalibration === 'function') rebuildCalibration(filteredAllBets);
  if (typeof rebuildEdgeBuckets === 'function') rebuildEdgeBuckets(filteredBets);
  if (typeof rebuildOutcomeTable === 'function') rebuildOutcomeTable(filteredBets);
  if (typeof rebuildLeagueTable === 'function') rebuildLeagueTable(filteredBets);
  if (typeof rebuildProfitBar === 'function') rebuildProfitBar(filteredBets);
  if (typeof rebuildCumulativeChart === 'function') rebuildCumulativeChart(filteredBets);
}}

function updateSummaryCards(bets) {{
  var n = bets.length;
  var roi = NaN, stab = 0, tstat = NaN;
  if (n > 0) {{
    var totalProfit = bets.reduce(function(s,b){{return s+b.profit;}},0);
    var totalStake  = bets.reduce(function(s,b){{return s+b.stake;}},0);
    roi = totalStake > 0 ? (totalProfit / totalStake) * 100 : NaN;
    var profits = bets.map(function(b){{return b.profit;}});
    var mean = totalProfit / n;
    var variance = profits.reduce(function(s,v){{return s+(v-mean)*(v-mean);}},0) / Math.max(n-1,1);
    var std = Math.sqrt(variance);
    stab = std > 0 ? mean / std : 0;
    tstat = stab * Math.sqrt(n);
  }}
  var betsEl = document.getElementById('summary-bets');
  if (betsEl) betsEl.textContent = n;
  var roiEl = document.getElementById('summary-roi');
  if (roiEl) {{
    roiEl.textContent = isNaN(roi) ? 'n/a' : (roi >= 0 ? '+' : '') + roi.toFixed(2) + '%';
    roiEl.style.color = (!isNaN(roi) && roi >= 0) ? '#2e7d32' : '#c62828';
  }}
  var tstatEl = document.getElementById('summary-tstat');
  if (tstatEl) {{
    tstatEl.textContent = isNaN(tstat) ? 'n/a' : (tstat >= 0 ? '+' : '') + tstat.toFixed(2);
    tstatEl.style.color = (!isNaN(tstat) && Math.abs(tstat) >= 2) ? '#2e7d32' : '#c62828';
  }}
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
      var totalProfit = subset.reduce(function(s, b) {{ return s + b.profit; }}, 0);
      var totalStake  = subset.reduce(function(s, b) {{ return s + b.stake;  }}, 0);
      roi = totalStake > 0 ? (totalProfit / totalStake) * 100 : 0;
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

// ── breakdown table helpers ───────────────────────────────────────────────────
function _breakdownStats(bets) {{
  var n = bets.length;
  if (n < 2) return null;
  var wins = bets.filter(function(b) {{ return b.y_true === b.outcome; }});
  var win_rate = wins.length / n;
  var avg_edge = bets.reduce(function(s, b) {{ return s + b.edge; }}, 0) / n;
  // Stake-weighted ROI: profit.sum() / stake.sum() * 100
  var totalProfit = bets.reduce(function(s, b) {{ return s + b.profit; }}, 0);
  var totalStake  = bets.reduce(function(s, b) {{ return s + b.stake; }}, 0);
  var roi = totalStake > 0 ? (totalProfit / totalStake) * 100 : 0;
  // t-stat: mean profit per bet / (std / sqrt(n))
  var profits = bets.map(function(b) {{ return b.profit; }});
  var mean_profit = totalProfit / n;
  var variance = profits.reduce(function(s, v) {{ return s + (v - mean_profit) * (v - mean_profit); }}, 0) / (n - 1);
  var std_dev = Math.sqrt(variance);
  var t_stat = std_dev > 0 ? mean_profit / (std_dev / Math.sqrt(n)) : 0;
  return {{n: n, win_rate: win_rate, avg_edge: avg_edge, roi: roi, t_stat: t_stat}};
}}

function _breakdownRow(label, bets) {{
  var s = _breakdownStats(bets);
  if (!s) {{
    return '<tr><td>' + label + '</td><td>' + bets.length + '</td>' +
           '<td>—</td><td>—</td>' +
           '<td style="color:#aaa">—</td>' +
           '<td style="color:#aaa">—</td></tr>';
  }}
  var roiColor = s.roi > 0 ? '#2e7d32' : '#c62828';
  var tColor = Math.abs(s.t_stat) > 2 ? '#2e7d32' : '#c62828';
  return '<tr>' +
    '<td>' + label + '</td>' +
    '<td>' + s.n + '</td>' +
    '<td>' + (s.win_rate * 100).toFixed(1) + '%</td>' +
    '<td>' + (s.avg_edge * 100).toFixed(1) + '%</td>' +
    '<td style="color:' + roiColor + ';font-weight:600">' + (s.roi >= 0 ? '+' : '') + s.roi.toFixed(1) + '%</td>' +
    '<td style="color:' + tColor + ';font-weight:600">' + s.t_stat.toFixed(2) + '</td>' +
    '</tr>';
}}

function _breakdownTableHtml(rows) {{
  return '<table style="width:100%;border-collapse:collapse;font-size:.88em">' +
    '<thead><tr style="border-bottom:2px solid #eee;color:#888;font-size:.8em;text-transform:uppercase;letter-spacing:.4px">' +
    '<th style="text-align:left;padding:6px 8px">Label</th>' +
    '<th style="text-align:center;padding:6px 8px">Bets</th>' +
    '<th style="text-align:center;padding:6px 8px">Win%</th>' +
    '<th style="text-align:center;padding:6px 8px">Avg edge</th>' +
    '<th style="text-align:center;padding:6px 8px">ROI</th>' +
    '<th style="text-align:center;padding:6px 8px">t-stat</th>' +
    '</tr></thead><tbody>' +
    rows.join('') +
    '</tbody></table>';
}}

// ── outcome breakdown table ───────────────────────────────────────────────────
function rebuildOutcomeTable(bets) {{
  var container = document.getElementById('outcome-table-container');
  if (!container) return;
  var OUTCOMES = [
    {{key: 'H', label: 'Home'}},
    {{key: 'D', label: 'Draw'}},
    {{key: 'A', label: 'Away'}}
  ];
  var rows = OUTCOMES.map(function(oc) {{
    var subset = bets.filter(function(b) {{ return b.outcome === oc.key; }});
    return _breakdownRow(oc.label, subset);
  }});
  container.innerHTML = _breakdownTableHtml(rows);
}}

// ── league breakdown table ────────────────────────────────────────────────────
function rebuildLeagueTable(bets) {{
  var container = document.getElementById('league-table-container');
  if (!container) return;
  var LG_NAMES = {league_names_json};
  var LG_ORDER = {league_order_json};
  var rows = [];
  LG_ORDER.forEach(function(code) {{
    var subset = bets.filter(function(b) {{ return b.league === code; }});
    if (subset.length === 0) return;
    rows.push(_breakdownRow(LG_NAMES[code] || code, subset));
  }});
  if (rows.length === 0) {{
    container.innerHTML = '<p style="color:#aaa;padding:8px">No bets match current filters</p>';
    return;
  }}
  container.innerHTML = _breakdownTableHtml(rows);
}}

// ── profit bar chart (canvas) ─────────────────────────────────────────────────
function rebuildProfitBar(bets) {{
  var canvas = document.getElementById('profit-bar-canvas');
  if (!canvas) return;
  var container = document.getElementById('profit-bar-container');
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

  if (bets.length === 0) {{
    ctx.fillStyle = '#aaa';
    ctx.font = '13px Arial,sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No bets match current filters', W / 2, H / 2);
    return;
  }}

  var pl = 48, pr = 16, pt = 12, pb = 36;
  var cw = W - pl - pr, ch = H - pt - pb;

  // Sort bets by actual profit
  var sorted = bets.slice().sort(function(a, b) {{ return a.profit - b.profit; }});
  var returns = sorted.map(function(b) {{ return b.profit; }});

  var minR = Math.min.apply(null, returns);
  var maxR = Math.max.apply(null, returns);
  var yRange = maxR - minR || 1;
  var yMin = minR - yRange * 0.06;
  var yMax = maxR + yRange * 0.06;
  if (yMax === yMin) {{ yMin -= 0.5; yMax += 0.5; }}
  var yr = yMax - yMin;

  function sy(v) {{ return pt + (1 - (v - yMin) / yr) * ch; }}
  var y0 = sy(0);

  // Zero line
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pl, y0); ctx.lineTo(pl + cw, y0); ctx.stroke();

  // Y axis grid + labels
  ctx.fillStyle = '#888'; ctx.font = '10px Arial,sans-serif'; ctx.textAlign = 'right';
  var tickVals = [];
  var rawStep = yr / 5;
  var mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  var niceStep = Math.ceil(rawStep / mag) * mag;
  var firstTick = Math.ceil(yMin / niceStep) * niceStep;
  for (var tv = firstTick; tv <= yMax + niceStep * 0.01; tv += niceStep) {{
    tickVals.push(parseFloat(tv.toFixed(8)));
  }}
  tickVals.forEach(function(v) {{
    var yp = sy(v);
    if (yp < pt - 4 || yp > pt + ch + 4) return;
    ctx.fillStyle = '#888'; ctx.fillText(v.toFixed(1), pl - 4, yp + 4);
    ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(pl, yp); ctx.lineTo(pl + cw, yp); ctx.stroke();
  }});

  // Y axis label
  ctx.save(); ctx.translate(12, pt + ch / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#555'; ctx.font = '11px Arial,sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Profit (units)', 0, 0); ctx.restore();

  // Bars — one thin bar per bet, sorted
  var n = sorted.length;
  var barW = Math.max(1, Math.floor(cw / n * 0.9));
  if (barW < 1) barW = 1;
  var step = cw / n;
  returns.forEach(function(r, i) {{
    var bx = pl + (i + 0.5) * step - barW / 2;
    var barTop = sy(Math.max(r, 0));
    var barBot = sy(Math.min(r, 0));
    var barH = Math.max(1, barBot - barTop);
    ctx.fillStyle = r >= 0 ? '#43a047' : '#e53935';
    ctx.fillRect(bx, barTop, barW, barH);
  }});

  // X axis label
  ctx.fillStyle = '#555'; ctx.font = '11px Arial,sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Bet (sorted by unit return)', pl + cw / 2, H - 4);
}}

// ── cumulative profit chart (canvas) ─────────────────────────────────────────
function rebuildCumulativeChart(bets) {{
  var canvas = document.getElementById('cumulative-chart-canvas');
  if (!canvas) return;
  var container = document.getElementById('cumulative-chart-container');
  var W = Math.max((container ? container.clientWidth : 0) - 48, 360);
  var H = 260;
  var dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  if (bets.length === 0) {{
    ctx.fillStyle = '#aaa'; ctx.font = '13px Arial,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('No bets match current filters', W / 2, H / 2);
    return;
  }}

  // Sort by date, then aggregate: one data point per day (cumulative after that day)
  var sorted = bets.slice().sort(function(a, b) {{
    return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
  }});

  // Build series: aggregate actual profit per day, then cumulative
  var byDate = {{}};
  sorted.forEach(function(b) {{
    byDate[b.date] = (byDate[b.date] || 0) + b.profit;
  }});
  var dates = Object.keys(byDate).sort();
  var series = [{{date: dates[0], y: 0}}];
  var cum = 0;
  dates.forEach(function(d) {{
    cum += byDate[d];
    series.push({{date: d, y: cum}});
  }});

  var allY = series.map(function(s) {{ return s.y; }});
  var rawMin = Math.min.apply(null, allY);
  var rawMax = Math.max.apply(null, allY);
  var spread = Math.max(Math.abs(rawMax), Math.abs(rawMin)) || 1;
  var yMin = Math.min(0, rawMin) - spread * 0.08;
  var yMax = Math.max(0, rawMax) + spread * 0.08;
  if (yMax === yMin) {{ yMin -= 1; yMax += 1; }}
  var yRange = yMax - yMin;

  // x: timestamp-based
  var minTs = new Date(series[0].date).getTime();
  var maxTs = new Date(series[series.length - 1].date).getTime();
  var tsRange = maxTs - minTs || 1;

  var pl = 52, pr = 24, pt = 16, pb = 50;
  var cw = W - pl - pr, ch = H - pt - pb;

  function sxD(dateStr) {{ return pl + (new Date(dateStr).getTime() - minTs) / tsRange * cw; }}
  function sy(v) {{ return pt + (1 - (v - yMin) / yRange) * ch; }}
  var y0 = sy(0);

  // Green fill above zero
  ctx.fillStyle = 'rgba(67,160,71,0.14)';
  ctx.beginPath(); ctx.moveTo(sxD(series[0].date), y0);
  series.forEach(function(s) {{ ctx.lineTo(sxD(s.date), Math.min(sy(s.y), y0)); }});
  ctx.lineTo(sxD(series[series.length-1].date), y0); ctx.closePath(); ctx.fill();

  // Red fill below zero
  ctx.fillStyle = 'rgba(229,57,53,0.14)';
  ctx.beginPath(); ctx.moveTo(sxD(series[0].date), y0);
  series.forEach(function(s) {{ ctx.lineTo(sxD(s.date), Math.max(sy(s.y), y0)); }});
  ctx.lineTo(sxD(series[series.length-1].date), y0); ctx.closePath(); ctx.fill();

  // Breakeven dashed line
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1.2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(pl, y0); ctx.lineTo(pl + cw, y0); ctx.stroke();
  ctx.setLineDash([]);

  // Line
  ctx.strokeStyle = '#1565c0'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.beginPath();
  series.forEach(function(s, i) {{
    var xp = sxD(s.date), yp = sy(s.y);
    if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
  }});
  ctx.stroke();

  // Y axis ticks
  ctx.fillStyle = '#999'; ctx.font = '11px Arial,sans-serif'; ctx.textAlign = 'right';
  var tickStep = Math.pow(10, Math.floor(Math.log10(yRange / 4)));
  if (yRange / tickStep < 3) tickStep /= 2;
  if (yRange / tickStep > 8) tickStep *= 2;
  var firstTick2 = Math.ceil(yMin / tickStep) * tickStep;
  for (var t = firstTick2; t <= yMax + tickStep * 0.01; t += tickStep) {{
    var yp = sy(t);
    if (yp < pt - 5 || yp > pt + ch + 5) continue;
    ctx.fillStyle = '#999'; ctx.fillText(t.toFixed(0), pl - 5, yp + 4);
    if (Math.abs(t) > 0.001) {{
      ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(pl, yp); ctx.lineTo(pl + cw, yp); ctx.stroke();
    }}
  }}

  // X axis labels (monthly)
  var mnNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var labelDates = []; var lastMon = '';
  series.forEach(function(s) {{
    var ym = s.date.slice(0, 7);
    if (ym !== lastMon) {{ lastMon = ym; labelDates.push(s.date); }}
  }});
  var lstep = Math.ceil(labelDates.length / 10);
  ctx.fillStyle = '#999'; ctx.font = '11px Arial,sans-serif'; ctx.textAlign = 'center';
  labelDates.forEach(function(d, j) {{
    if (j % lstep !== 0) return;
    var p = d.split('-');
    ctx.fillText(mnNames[parseInt(p[1]) - 1] + " '" + p[0].slice(2), sxD(d), pt + ch + 22);
  }});

  // Y axis label
  ctx.save(); ctx.translate(12, pt + ch / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#555'; ctx.font = '11px Arial,sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Cumulative units', 0, 0); ctx.restore();

  // Axes border
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pl, pt); ctx.lineTo(pl, pt + ch); ctx.lineTo(pl + cw, pt + ch); ctx.stroke();

  // Final value annotation
  var fin = series[series.length - 1].y;
  ctx.font = 'bold 11px Arial,sans-serif'; ctx.textAlign = 'right';
  ctx.fillStyle = fin >= 0 ? '#1565c0' : '#c62828';
  ctx.fillText((fin >= 0 ? '+' : '') + fin.toFixed(1) + 'u', pl + cw, sy(fin) - 6);
}}

// ── boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {{
  applyFilters();
}});
</script>
</body>
</html>"""



def _backtest_bets_script(results_df: pd.DataFrame) -> str:
    """Generate BACKTEST_BETS JS array from placed bets (includes stake, profit, edge, league)."""
    if results_df is None or results_df.empty:
        return "<script>var BACKTEST_BETS=[];</script>"
    df = results_df.copy()
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "y_pred": "outcome"})
    if "edge" not in df.columns and "model_prob" in df.columns and "implied_prob" in df.columns:
        df["edge"] = df["model_prob"] - df["implied_prob"]
    needed = ["date", "league", "home", "away", "outcome", "y_true",
              "odds", "model_prob", "implied_prob", "edge", "stake", "profit"]
    cols = [c for c in needed if c in df.columns]
    records = df[cols].to_dict("records")
    return f"<script>\nvar BACKTEST_BETS={json.dumps(records, separators=(',', ':'))};\n</script>"


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


def _observed_leagues(
    results_df: pd.DataFrame,
    all_predictions: "pd.DataFrame | None",
) -> list[str]:
    observed: set[str] = set()
    for frame in (results_df, all_predictions):
        if frame is not None and "league" in frame.columns:
            observed.update(str(value) for value in frame["league"].dropna().unique())
    canonical = [league for league in SUPPORTED_LEAGUES if league in observed]
    return canonical + sorted(observed - set(canonical))


def generate_report(
    results_df: pd.DataFrame,
    accuracy: float,
    roi: float,
    stability: float,
    output_path: Path,
    all_predictions: "pd.DataFrame | None" = None,
    production_leagues: "set[str] | frozenset[str] | None" = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_bets = len(results_df)
    n_correct = (results_df["y_true"] == results_df["y_pred"]).sum()
    n_wrong = n_bets - n_correct
    roi_color = "#2e7d32" if roi >= 0 else "#c62828"
    tstat = stability * (n_bets ** 0.5)
    observed_leagues = _observed_leagues(results_df, all_predictions)
    league_buttons = "\n      ".join(
        f'<button class="outcome-btn active" id="btn-lg-{league}" '
        f'onclick="toggleLeague(\'{league}\')">{league} · {LEAGUE_NAMES.get(league, league)}</button>'
        for league in observed_leagues
    )
    active_leagues_json = json.dumps(
        {league: True for league in observed_leagues}, separators=(",", ":")
    )
    production_order = [
        league for league in observed_leagues if league in (production_leagues or set())
    ]
    production_leagues_json = json.dumps(production_order, separators=(",", ":"))
    league_names_json = json.dumps(
        {league: LEAGUE_NAMES.get(league, league) for league in observed_leagues},
        separators=(",", ":"),
    )
    league_order_json = json.dumps(observed_leagues, separators=(",", ":"))

    html = _HTML_TEMPLATE.format(
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        roi_color=roi_color,
        n_bets=n_bets,
        n_correct=n_correct,
        n_wrong=n_wrong,
        tstat=tstat,
        league_buttons=league_buttons,
        active_leagues_json=active_leagues_json,
        production_leagues_json=production_leagues_json,
        league_names_json=league_names_json,
        league_order_json=league_order_json,
    )
    backtest_script = _backtest_bets_script(results_df)
    all_bets_script = _all_predictions_script(all_predictions)
    html = html.replace("</head>", f"{backtest_script}\n{all_bets_script}\n</head>", 1)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {output_path}")
