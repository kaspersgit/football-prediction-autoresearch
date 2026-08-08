import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


_LEAGUE_NAMES = {
    # short codes (historical data / backtest CSV)
    "E0": "England", "D1": "Germany", "SP1": "Spain",
    "I1": "Italy", "F1": "France", "N1": "Netherlands", "P1": "Portugal",
    # full-name keys from load_fixtures() _LEAGUE_MAP
    "england": "England", "germany": "Germany", "spain": "Spain",
    "italy": "Italy", "france": "France", "netherlands": "Netherlands", "portugal": "Portugal",
}

# Canonical order for league columns in the monthly table
_LEAGUE_ORDER = ["england", "germany", "spain", "italy", "france", "netherlands", "portugal"]
_LEAGUE_CODES = {"england": "E0", "germany": "D1", "spain": "SP1",
                 "italy": "I1", "france": "F1", "netherlands": "N1", "portugal": "P1"}

_OUTCOME_LABEL = {"H": "Home", "D": "Draw", "A": "Away"}
_OUTCOME_COLOR = {"H": "#1565c0", "D": "#616161", "A": "#e65100"}


# ── probability bar ──────────────────────────────────────────────────────────

def _prob_bar(h: float, d: float, a: float) -> str:
    hp, dp, ap = int(h * 100), int(d * 100), int(a * 100)
    return (
        f'<div class="prob-bar" title="Home {hp}% / Draw {dp}% / Away {ap}%">'
        f'<div style="width:{hp}%;background:#1565c0"></div>'
        f'<div style="width:{dp}%;background:#9e9e9e"></div>'
        f'<div style="width:{ap}%;background:#e65100"></div>'
        f'</div>'
        f'<div class="prob-labels">'
        f'<span style="color:#1565c0">{hp}%</span>'
        f'<span style="color:#9e9e9e">{dp}%</span>'
        f'<span style="color:#e65100">{ap}%</span>'
        f'</div>'
    )


# ── odds cell ────────────────────────────────────────────────────────────────

def _odds_cell(b365: float, cmax: float, bk: str, outcome: str, edge: float | None) -> str:
    color = _OUTCOME_COLOR[outcome]
    has_max = not math.isnan(cmax) and bk
    max_better = has_max and cmax > b365 + 0.005

    b365_html = (
        f'<span class="b365-val value-price" style="color:{color}">{b365:.2f}</span>'
        if edge is not None else
        f'<span class="b365-val">{b365:.2f}</span>'
    )
    edge_html = (
        f'<span class="edge-chip" style="background:{color}">+{edge:.1%}</span>'
        if edge is not None else ""
    )
    max_html = (
        f'<span class="max-odds">↑ {cmax:.2f} <em>{bk}</em></span>'
        if max_better else ""
    )
    css_extra = f' value-odds" style="border-color:{color}' if edge is not None else ""
    return (
        f'<td class="odds-cell{css_extra}">'
        f'{b365_html}{edge_html}{max_html}'
        f'</td>'
    )


# ── top value bets table ─────────────────────────────────────────────────────

def _empty_fixtures_notice_html() -> str:
    """Banner shown when live inference produced no production fixtures."""
    return (
        '<div class="empty-fixtures-notice">'
        "No upcoming production fixtures available. "
        "The page will refresh when the next scheduled prediction run finds matches."
        "</div>"
    )


def _top_bets_html(all_bets: list[dict]) -> str:
    if not all_bets:
        return '<tr><td colspan="8"><p>No value bets found at this threshold.</p></td></tr>'
    rows = []
    for i, b in enumerate(all_bets, 1):
        color = _OUTCOME_COLOR[b["outcome"]]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
        cmax = b.get("max_odds", float("nan"))
        bk = b.get("max_bk", "")
        has_max = not math.isnan(cmax) and bk
        max_cell = (
            f'<td class="odds-val">{cmax:.2f} <span class="bk-tag">{bk}</span></td>'
            if has_max else '<td class="odds-val">—</td>'
        )
        rows.append(
            f'<tr class="bet-row" data-edge="{b["edge"]:.4f}" data-odds="{b["b365_odds"]:.2f}" data-model-prob="{b["model_prob"]:.4f}">'
            f'<td class="rank">{medal}</td>'
            f'<td>{b["date"]}</td>'
            f'<td><span class="league-tag">{_LEAGUE_NAMES.get(b["league"], b["league"])}</span></td>'
            f'<td><strong>{b["home"]}</strong> vs {b["away"]}</td>'
            f'<td><span class="bet-outcome" style="color:{color};border-color:{color}">'
            f'{_OUTCOME_LABEL[b["outcome"]]}</span></td>'
            f'<td><span class="edge-val" style="color:{color}">+{b["edge"]:.1%}</span></td>'
            f'<td class="odds-val">{b["b365_odds"]:.2f}</td>'
            f'{max_cell}'
            f'</tr>'
        )
    return "\n".join(rows)


# ── per-league fixture section ───────────────────────────────────────────────

def _league_section_html(league: str, rows: list[dict]) -> str:
    league_name = _LEAGUE_NAMES.get(league, league)
    n_value = sum(1 for r in rows if r["ValueBets"])
    fixture_rows = []
    for r in rows:
        date_str = r["Date"].strftime("%a %b %d")
        has_value = bool(r["ValueBets"])
        row_class = "value-row" if has_value else ""
        edge_map = {o: e for o, e in r["ValueBets"]}

        odds_cells = ""
        for outcome in ["H", "D", "A"]:
            b365 = r[f"B365{outcome}"]
            cmax = r.get(f"CustomMax{outcome}", float("nan"))
            bk = r.get(f"CustomMaxBk{outcome}", "")
            odds_cells += _odds_cell(b365, cmax, bk, outcome, edge_map.get(outcome))

        fixture_rows.append(
            f'<tr class="{row_class}">'
            f'<td class="date-cell">{date_str}</td>'
            f'<td class="team-cell"><strong>{r["HomeTeam"]}</strong></td>'
            f'<td class="vs-cell">vs</td>'
            f'<td class="team-cell">{r["AwayTeam"]}</td>'
            f'<td class="prob-cell">{_prob_bar(r["ModelH"], r["ModelD"], r["ModelA"])}</td>'
            f'{odds_cells}'
            f'</tr>'
        )

    badge = (
        f'<span class="value-count">{n_value} value bet{"s" if n_value != 1 else ""}</span>'
        if n_value else ""
    )
    return f"""
<div class="league-section">
  <div class="league-header">
    <span class="league-name">{league_name}</span>
    {badge}
  </div>
  <table class="fixture-table">
    <thead>
      <tr>
        <th>Date</th><th colspan="3">Fixture</th>
        <th>Model probs (H/D/A)</th>
        <th>B365 H</th><th>B365 D</th><th>B365 A</th>
      </tr>
      <tr class="subheader">
        <th colspan="5"></th>
        <th colspan="3" style="color:#888;font-size:.7em;font-weight:500">↑ best available odds + bookmaker</th>
      </tr>
    </thead>
    <tbody>
      {"".join(fixture_rows)}
    </tbody>
  </table>
</div>"""


# ── historical monthly × league performance ──────────────────────────────────

def _monthly_league_table_html(bets: pd.DataFrame | None) -> str:
    """Container for month × league ROI table — populated by JS from BACKTEST_BETS."""
    if bets is None or bets.empty:
        return ""
    return """
<div class="top-bets-card" style="margin-bottom:24px">
  <div class="card-header">
    Historical Performance by Month
    <span id="perf-table-subtitle" style="font-size:.8em;font-weight:400;color:#888;margin-left:8px">backtest · flat staking</span>
  </div>
  <div style="overflow-x:auto;padding:4px 0" id="perf-table-container"></div>
</div>"""


# ── backtest data embed ───────────────────────────────────────────────────────

def _backtest_data_script(historical_bets: pd.DataFrame | None) -> str:
    if historical_bets is None or historical_bets.empty:
        return "<script>var BACKTEST_BETS=[];</script>"
    needed = ["Date", "league", "stake", "profit", "odds", "model_prob", "implied_prob", "y_true", "y_pred"]
    cols = [c for c in needed if c in historical_bets.columns]
    df = historical_bets[cols].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"Date": "date"})
    records = df.to_dict("records")
    return f"<script>\nvar BACKTEST_BETS={json.dumps(records, separators=(',', ':'))};\n</script>"


# ── profit curve ─────────────────────────────────────────────────────────────

def _profit_curve_html(historical_bets: pd.DataFrame | None) -> str:
    if historical_bets is None or historical_bets.empty:
        return ""
    return """
<div class="top-bets-card" style="margin-bottom:24px">
  <div class="card-header">
    Backtest Profit Curve
    <span id="profit-curve-subtitle" style="font-size:.8em;font-weight:400;color:#888;margin-left:8px">last 2 seasons walk-forward</span>
  </div>
  <div style="padding:16px 24px" id="profit-curve-container">
    <canvas id="profit-curve-canvas" style="width:100%;max-width:860px;display:block;margin:0 auto"></canvas>
  </div>
</div>"""


def _forecast_card_html(historical_bets: pd.DataFrame | None) -> str:
    if historical_bets is None or historical_bets.empty:
        return ""
    return """
<div class="top-bets-card" style="margin-bottom:24px" id="forecast-card">
  <div class="card-header">
    Matchweek Forecast
    <span id="forecast-subtitle" style="font-size:.8em;font-weight:400;color:#888;margin-left:8px">predicted ROI · 95% CI</span>
  </div>
  <div id="forecast-card-container" style="padding:16px 24px;min-height:60px"></div>
</div>"""


# ── filter bar ───────────────────────────────────────────────────────────────

def _filter_bar_html(default_threshold: float, has_historical: bool = False) -> str:
    thr_pct = int(round(default_threshold * 100))

    if has_historical:
        forecast_card_js = """
// ── matchweek forecast card ───────────────────────────────────────────────────
function rebuildForecastCard(backtestFiltered, nUpcoming) {
  var container = document.getElementById('forecast-card-container');
  if (!container) return;
  var sub = document.getElementById('forecast-subtitle');

  if (!BACKTEST_BETS.length || backtestFiltered.length < 30) {
    container.innerHTML = '<p style="text-align:center;color:#aaa;padding:8px 0">Insufficient historical data for this filter combination</p>';
    return;
  }

  // Expected ROI from backtest (calibrated point estimate)
  var returns = backtestFiltered.map(function(b) { return b.profit / b.stake; });
  var nHist = returns.length;
  var mean = returns.reduce(function(a, b) { return a + b; }, 0) / nHist;

  if (nUpcoming < 5) {
    var sign = mean >= 0 ? '+' : '';
    container.innerHTML =
      '<div style="padding:8px 0;color:#555;font-size:.9em">Predicted ROI: <strong style="color:' +
      (mean >= 0 ? '#2e7d32' : '#c62828') + '">' + sign + (mean * 100).toFixed(1) + '%</strong>' +
      ' &nbsp;·&nbsp; <span style="color:#f57c00">⚠ CI unreliable — fewer than 5 bets</span></div>';
    if (sub) sub.textContent = 'predicted ROI · ' + nHist + ' hist. bets';
    return;
  }

  // Per-bet Bernoulli variance using each upcoming bet's actual odds
  var upcomingRows = document.querySelectorAll('.bet-row');
  var totalVar = 0, count = 0;
  upcomingRows.forEach(function(row) {
    if (row.style.display === 'none') return;
    var odds = parseFloat(row.dataset.odds);
    var mp = parseFloat(row.dataset.modelProb);
    if (isNaN(odds) || isNaN(mp)) return;
    totalVar += mp * (1 - mp) * odds * odds;
    count++;
  });
  if (count === 0) {
    container.innerHTML = '<p style="text-align:center;color:#aaa;padding:8px 0">Insufficient historical data for this filter combination</p>';
    return;
  }

  var se = Math.sqrt(totalVar) / count;
  var lower = mean - 1.96 * se;
  var upper = mean + 1.96 * se;
  var lower1s = mean - se;
  var upper1s = mean + se;

  function fmt(v) { return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; }
  var roiColor = mean >= 0 ? '#2e7d32' : '#c62828';

  // Map value → % position within [lower, upper]
  var span = upper - lower || 0.01;
  function toPos(v) { return Math.max(0, Math.min(100, (v - lower) / span * 100)); }
  var meanPos = toPos(mean);
  var l1sPos = toPos(lower1s);
  var u1sPos = toPos(upper1s);

  if (sub) sub.textContent = 'predicted ROI · ' + nHist + ' hist. bets · ' + count + ' upcoming bets';

  container.innerHTML =
    '<div style="display:flex;gap:32px;align-items:center;flex-wrap:wrap">' +
    '<div style="min-width:120px">' +
      '<div style="font-size:2em;font-weight:700;color:' + roiColor + '">' + fmt(mean) + '</div>' +
      '<div style="font-size:.8em;color:#888;margin-top:2px">predicted ROI</div>' +
      '<div style="font-size:.85em;color:#555;margin-top:6px">' + count + ' bet' + (count !== 1 ? 's' : '') + ' this week</div>' +
    '</div>' +
    '<div style="flex:1;min-width:220px;padding:8px 0">' +
      '<div style="position:relative;height:20px;margin:0 8px">' +
        '<div style="position:absolute;top:9px;left:0;right:0;height:2px;background:#ddd;border-radius:1px"></div>' +
        '<div style="position:absolute;top:6px;height:8px;background:' + roiColor + ';opacity:.22;border-radius:3px;left:' + l1sPos.toFixed(1) + '%;width:' + (u1sPos - l1sPos).toFixed(1) + '%"></div>' +
        '<div style="position:absolute;top:2px;width:3px;height:16px;background:' + roiColor + ';border-radius:2px;left:calc(' + meanPos.toFixed(1) + '% - 1.5px)"></div>' +
      '</div>' +
      '<div style="display:flex;justify-content:space-between;font-size:.75em;color:#999;margin-top:4px;padding:0 8px">' +
        '<span>' + fmt(lower) + '</span>' +
        '<span style="color:' + roiColor + ';font-weight:600">' + fmt(mean) + '</span>' +
        '<span>' + fmt(upper) + '</span>' +
      '</div>' +
      '<div style="text-align:center;font-size:.72em;color:#bbb;margin-top:2px">← 95% confidence interval →</div>' +
    '</div>' +
    '</div>';
}
"""
        # Leading spaces must match the indentation of rebuildProfitCurve above
        forecast_card_call = "    rebuildForecastCard(filtered, shown);"
    else:
        forecast_card_js = ""
        forecast_card_call = ""

    return f"""
<div class="filter-bar">
  <div class="filter-group">
    <label>Min edge <span id="lbl-threshold">3%</span></label>
    <input type="range" id="filter-threshold" min="0" max="15" step="1" value="3"
           oninput="document.getElementById('lbl-threshold').textContent=this.value+'%';applyFilters()">
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
  <div class="filter-group filter-count">
    Showing <span id="visible-count">…</span> bets
  </div>
</div>
<script>
// ── backtest filtering helpers ────────────────────────────────────────────────
function filterBacktestBets(minEdge, minOdds, maxOdds) {{
  return BACKTEST_BETS.filter(function(b) {{
    var edge = b.model_prob - b.implied_prob;
    return edge >= minEdge && b.odds >= minOdds && b.odds <= maxOdds;
  }});
}}

// ── profit curve (canvas) ─────────────────────────────────────────────────────
function rebuildProfitCurve(bets) {{
  var canvas = document.getElementById('profit-curve-canvas');
  if (!canvas || !BACKTEST_BETS.length) return;
  var container = document.getElementById('profit-curve-container');
  var W = Math.max((container ? container.clientWidth : 0) - 32, 400);
  var H = 270;
  var dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  var sorted = bets.slice().sort(function(a, b) {{
    return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
  }});

  if (sorted.length === 0) {{
    ctx.fillStyle = '#aaa';
    ctx.font = '13px -apple-system,sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No bets match current filters', W / 2, H / 2);
    return;
  }}

  // Aggregate bets by date so each day contributes exactly one point per series.
  var byDate = {{}};
  sorted.forEach(function(b) {{
    if (!byDate[b.date]) byDate[b.date] = [];
    byDate[b.date].push(b.profit / b.stake);
  }});
  var dates = Object.keys(byDate).sort();

  // Series 1: flat 1-unit staking — pure model signal, no bankroll assumption.
  // Y = 100 + cumulative flat profit (baseline 100 = zero profit).
  var cum = 0;
  var series1 = [{{date: dates[0], y: 100}}];
  dates.forEach(function(d) {{
    byDate[d].forEach(function(r) {{ cum += r; }});
    series1.push({{date: d, y: 100 + cum}});
  }});

  // Series 2: 0.5%-per-bet fixed-fractional compounding.
  // Each bet within a day compounds the bankroll individually before the next.
  var FRAC = 0.005;
  var bankroll = 100;
  var series2 = [{{date: dates[0], y: 100}}];
  dates.forEach(function(d) {{
    byDate[d].forEach(function(r) {{ bankroll *= (1 + FRAC * r); }});
    series2.push({{date: d, y: bankroll}});
  }});

  // Combined date range (timestamp-based x axis)
  var allPts = series1.concat(series2);
  var minTs = Math.min.apply(null, allPts.map(function(s) {{ return new Date(s.date).getTime(); }}));
  var maxTs = Math.max.apply(null, allPts.map(function(s) {{ return new Date(s.date).getTime(); }}));
  var tsRange = maxTs - minTs || 1;

  // Combined y range, anchored at 100
  var allY = allPts.map(function(s) {{ return s.y; }});
  var rawMin = Math.min.apply(null, allY), rawMax = Math.max.apply(null, allY);
  var spread = Math.abs(rawMax - rawMin) || 1;
  var yMin = Math.min(100, rawMin) - spread * 0.08;
  var yMax = Math.max(100, rawMax) + spread * 0.08;
  if (yMax === yMin) {{ yMin -= 1; yMax += 1; }}
  var yRange = yMax - yMin;

  var pl = 56, pr = 24, pt = 16, pb = 50;
  var cw = W - pl - pr, ch = H - pt - pb;

  function sxD(dateStr) {{ return pl + (new Date(dateStr).getTime() - minTs) / tsRange * cw; }}
  function sy(y) {{ return pt + (1 - (y - yMin) / yRange) * ch; }}
  var y0 = sy(100);

  // Green/red fill follows series 2 (the bankroll simulation — the one with % meaning)
  ctx.fillStyle = 'rgba(67,160,71,0.12)';
  ctx.beginPath(); ctx.moveTo(sxD(series2[0].date), y0);
  series2.forEach(function(s) {{ ctx.lineTo(sxD(s.date), Math.min(sy(s.y), y0)); }});
  ctx.lineTo(sxD(series2[series2.length-1].date), y0); ctx.closePath(); ctx.fill();

  ctx.fillStyle = 'rgba(229,57,53,0.12)';
  ctx.beginPath(); ctx.moveTo(sxD(series2[0].date), y0);
  series2.forEach(function(s) {{ ctx.lineTo(sxD(s.date), Math.max(sy(s.y), y0)); }});
  ctx.lineTo(sxD(series2[series2.length-1].date), y0); ctx.closePath(); ctx.fill();

  // Breakeven line at 100
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pl, y0); ctx.lineTo(pl + cw, y0); ctx.stroke();

  // Series 2: 0.5%/bet compounding (orange, dashed) — drawn first so blue sits on top
  ctx.strokeStyle = '#e65100'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  series2.forEach(function(s, i) {{
    var x = sxD(s.date), yy = sy(s.y);
    if (i === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
  }});
  ctx.stroke();
  ctx.setLineDash([]);

  // Series 1: flat 1-unit (dark blue, solid)
  ctx.strokeStyle = '#1a237e'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.beginPath();
  series1.forEach(function(s, i) {{
    var x = sxD(s.date), yy = sy(s.y);
    if (i === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
  }});
  ctx.stroke();

  // Y axis ticks
  ctx.fillStyle = '#999'; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'right';
  var tickStep = Math.pow(10, Math.floor(Math.log10(yRange / 4)));
  if (yRange / tickStep < 3) tickStep /= 2;
  if (yRange / tickStep > 8) tickStep *= 2;
  var firstTick = Math.ceil(yMin / tickStep) * tickStep;
  for (var t = firstTick; t <= yMax + tickStep * 0.01; t += tickStep) {{
    var yp = sy(t);
    if (yp < pt - 5 || yp > pt + ch + 5) continue;
    ctx.fillStyle = '#999'; ctx.fillText(Math.round(t), pl - 5, yp + 4);
    if (Math.abs(t - 100) > 0.01) {{
      ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(pl, yp); ctx.lineTo(pl + cw, yp); ctx.stroke();
    }}
  }}

  // X axis labels (monthly, max ~10)
  var mnNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var labelDates = []; var lastMon = '';
  series1.forEach(function(s) {{
    var ym = s.date.slice(0, 7);
    if (ym !== lastMon) {{ lastMon = ym; labelDates.push(s.date); }}
  }});
  var lstep = Math.ceil(labelDates.length / 10);
  ctx.fillStyle = '#999'; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
  labelDates.forEach(function(d, j) {{
    if (j % lstep !== 0) return;
    var p = d.split('-');
    ctx.fillText(mnNames[parseInt(p[1]) - 1] + " '" + p[0].slice(2), sxD(d), pt + ch + 22);
  }});

  // Axes border
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pl, pt); ctx.lineTo(pl, pt + ch); ctx.lineTo(pl + cw, pt + ch); ctx.stroke();

  // Final value annotations
  var fin1 = series1[series1.length - 1].y;
  var fin2 = series2[series2.length - 1].y;
  ctx.font = 'bold 11px -apple-system,sans-serif'; ctx.textAlign = 'right';
  ctx.fillStyle = fin1 >= 100 ? '#1a237e' : '#c62828';
  ctx.fillText(fin1.toFixed(0) + 'u', pl + cw, sy(fin1) + (fin1 >= fin2 ? -8 : 14));
  ctx.fillStyle = fin2 >= 100 ? '#bf360c' : '#c62828';
  ctx.fillText(fin2.toFixed(0) + '%', pl + cw, sy(fin2) + (fin2 > fin1 ? -8 : 14));

  // Legend
  var lx = pl, ly = pt + ch + 38;
  ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
  ctx.strokeStyle = '#1a237e'; ctx.lineWidth = 2; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 20, ly); ctx.stroke();
  var flatReturn = fin1 - 100;
  ctx.fillStyle = '#444';
  ctx.fillText('Flat 1-unit · model signal (' + (flatReturn >= 0 ? '+' : '') + flatReturn.toFixed(0) + ' units)', lx + 24, ly + 4);
  var lx2 = lx + 220;
  ctx.strokeStyle = '#e65100'; ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(lx2, ly); ctx.lineTo(lx2 + 20, ly); ctx.stroke();
  ctx.setLineDash([]);
  var pctReturn = fin2 - 100;
  ctx.fillStyle = '#444';
  ctx.fillText('0.5%/bet compounding · bankroll (' + (pctReturn >= 0 ? '+' : '') + pctReturn.toFixed(0) + '%)', lx2 + 24, ly + 4);
}}

// ── monthly performance table ─────────────────────────────────────────────────
function rebuildPerformanceTable(bets) {{
  var container = document.getElementById('perf-table-container');
  if (!container || !BACKTEST_BETS.length) return;

  if (bets.length === 0) {{
    container.innerHTML = '<p style="text-align:center;color:#aaa;padding:24px">No bets match current filters</p>';
    return;
  }}

  var LG_NAMES = {{'E0':'England','D1':'Germany','SP1':'Spain','I1':'Italy','F1':'France','N1':'Netherlands','P1':'Portugal'}};
  var LG_ORDER = ['E0','D1','SP1','I1','F1','N1','P1'];
  var mnNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  function fmtMon(m) {{ var p=m.split('-'); return mnNames[parseInt(p[1])-1]+' '+p[0].slice(2); }}
  function getMonth(d) {{ return d.slice(0, 7); }}

  var months = [], leagues = [], byMonLg = {{}};
  bets.forEach(function(b) {{
    var m = getMonth(b.date), lg = b.league;
    if (!byMonLg[m]) byMonLg[m] = {{}};
    if (!byMonLg[m][lg]) byMonLg[m][lg] = [];
    byMonLg[m][lg].push(b);
    if (months.indexOf(m) < 0) months.push(m);
    if (leagues.indexOf(lg) < 0) leagues.push(lg);
  }});
  months.sort();
  var activeLgs = LG_ORDER.filter(function(lg) {{ return leagues.indexOf(lg) >= 0; }});

  function cellHtml(arr, extra) {{
    if (!arr || !arr.length) return '<td class="perf-empty' + (extra?' '+extra:'') + '">—</td>';
    var n=arr.length, flatProfit=0, wins=0;
    arr.forEach(function(b) {{ flatProfit += b.profit/b.stake; if(b.y_true===b.y_pred) wins++; }});
    var roi = n > 0 ? flatProfit/n*100 : 0;
    var cls = (roi > 0 ? 'perf-pos' : 'perf-neg') + (extra?' '+extra:'');
    return '<td class="'+cls+'" title="'+wins+'W/'+(n-wins)+'L">'+n+'b&nbsp;<strong>'+(roi>=0?'+':'')+roi.toFixed(0)+'%</strong></td>';
  }}

  var hdr = activeLgs.map(function(lg) {{ return '<th>'+LG_NAMES[lg].slice(0,3)+'</th>'; }}).join('');
  var body = months.map(function(m) {{
    var mBets = bets.filter(function(b) {{ return getMonth(b.date)===m; }});
    var cells = activeLgs.map(function(lg) {{ return cellHtml(byMonLg[m] && byMonLg[m][lg]); }}).join('');
    return '<tr><td class="month-label">'+fmtMon(m)+'</td>'+cells+cellHtml(mBets,'perf-total')+'</tr>';
  }}).join('');

  var foot = activeLgs.map(function(lg) {{
    var lb = bets.filter(function(b) {{ return b.league===lg; }});
    var n=lb.length, fp=0;
    lb.forEach(function(b) {{ fp += b.profit/b.stake; }});
    var roi = n>0 ? fp/n*100 : 0;
    return '<td class="'+(roi>0?'perf-pos':'perf-neg')+' perf-total"><strong>'+(roi>=0?'+':'')+roi.toFixed(0)+'%</strong><br><small>'+n+'b</small></td>';
  }}).join('');
  var flatProfitAll=0;
  bets.forEach(function(b) {{ flatProfitAll += b.profit/b.stake; }});
  var roiAll = bets.length>0 ? flatProfitAll/bets.length*100 : 0;
  foot += '<td class="'+(roiAll>0?'perf-pos':'perf-neg')+' perf-total"><strong>'+(roiAll>=0?'+':'')+roiAll.toFixed(0)+'%</strong><br><small>'+bets.length+'b</small></td>';

  container.innerHTML =
    '<table class="perf-table"><thead><tr><th>Month</th>'+hdr+'<th>Total</th></tr></thead>' +
    '<tbody>'+body+'</tbody>' +
    '<tfoot><tr><td class="month-label"><strong>All</strong></td>'+foot+'</tr></tfoot></table>';
}}

{forecast_card_js}
// ── main filter function ───────────────────────────────────────────────────────
function applyFilters() {{
  var minEdge = parseFloat(document.getElementById('filter-threshold').value) / 100;
  var minOdds = parseFloat(document.getElementById('filter-min-odds').value);
  var maxOdds = parseFloat(document.getElementById('filter-max-odds').value);

  // Filter top value bets table
  var rows = document.querySelectorAll('.bet-row');
  var shown = 0;
  rows.forEach(function(row) {{
    var edge = parseFloat(row.dataset.edge);
    var odds = parseFloat(row.dataset.odds);
    var visible = (edge >= minEdge) && (odds >= minOdds) && (odds <= maxOdds);
    row.style.display = visible ? '' : 'none';
    if (visible) shown++;
  }});
  document.getElementById('visible-count').textContent = shown;
  var badge = document.getElementById('value-bets-badge');
  if (badge) badge.textContent = shown;

  // Re-number visible rows
  var rank = 1;
  rows.forEach(function(row) {{
    if (row.style.display !== 'none') {{
      var medals = {{1:'🥇',2:'🥈',3:'🥉'}};
      row.querySelector('.rank').textContent = medals[rank] || ('#'+rank);
      rank++;
    }}
  }});

  // Rebuild backtest sections from embedded data
  if (BACKTEST_BETS.length) {{
    var filtered = filterBacktestBets(minEdge, minOdds, maxOdds);
    rebuildProfitCurve(filtered);
    rebuildPerformanceTable(filtered);
{forecast_card_call}
    var sub = document.getElementById('perf-table-subtitle');
    if (sub) sub.textContent = 'backtest · flat staking · ' + filtered.length + ' bets';
    var psub = document.getElementById('profit-curve-subtitle');
    if (psub) psub.textContent = 'last 2 seasons walk-forward · ' + filtered.length + ' bets';
  }}
}}
window.addEventListener('DOMContentLoaded', applyFilters);
</script>"""


# ── main HTML generator ──────────────────────────────────────────────────────

def generate_predictions_html(
    pred_rows: list[dict],
    threshold: float,
    fetched_at: datetime,
    profit_curve_path: Path | None = None,
    historical_bets: pd.DataFrame | None = None,
) -> str:
    all_bets = []
    for r in pred_rows:
        for outcome, edge in r["ValueBets"]:
            cmax = r.get(f"CustomMax{outcome}", float("nan"))
            bk = r.get(f"CustomMaxBk{outcome}", "")
            all_bets.append({
                "date": r["Date"].strftime("%a %b %d"),
                "league": r["League"],
                "home": r["HomeTeam"],
                "away": r["AwayTeam"],
                "outcome": outcome,
                "edge": edge,
                "b365_odds": r[f"B365{outcome}"],
                "max_odds": cmax if not math.isnan(cmax) else float("nan"),
                "max_bk": bk,
                "model_prob": r[f"Model{outcome}"],
            })
    all_bets.sort(key=lambda x: x["edge"], reverse=True)

    by_league: dict[str, list] = {}
    for r in pred_rows:
        by_league.setdefault(r["League"], []).append(r)

    league_html = "\n".join(
        _league_section_html(lg, by_league[lg])
        for lg in _LEAGUE_ORDER
        if lg in by_league
    )

    fetch_str = fetched_at.strftime("%d %b %Y, %H:%M")
    total_fixtures = len(pred_rows)
    total_value = len(all_bets)
    empty_notice_html = _empty_fixtures_notice_html() if total_fixtures == 0 else ""
    profit_html = _profit_curve_html(historical_bets)
    monthly_html = _monthly_league_table_html(historical_bets)
    forecast_html = _forecast_card_html(historical_bets)
    filter_html = _filter_bar_html(threshold, has_historical=historical_bets is not None and not historical_bets.empty)
    backtest_script = _backtest_data_script(historical_bets)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictions — {fetched_at.strftime('%d %b %Y')}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    background: #f0f2f5; color: #1a1a2e;
    margin: 0; padding: 24px 16px 48px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}

  .page-header {{
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    color: white; border-radius: 12px;
    padding: 28px 32px 24px; margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(26,35,126,.25);
  }}
  .page-header h1 {{ margin: 0 0 6px; font-size: 1.7em; letter-spacing: -.3px; }}
  .page-header .meta {{ opacity: .8; font-size: .9em; margin-bottom: 14px; }}
  .warning-banner {{
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
    border-radius: 6px; padding: 8px 14px; font-size: .85em; display: inline-block;
  }}
  .stats-row {{ display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap; }}
  .stat-pill {{
    background: rgba(255,255,255,.18); border-radius: 20px;
    padding: 6px 16px; font-size: .9em; font-weight: 600;
  }}
  .empty-fixtures-notice {{
    background: #fff8e1; border: 1px solid #ffe082; color: #6d4c00;
    border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;
    font-size: .95em;
  }}

  /* Filter bar */
  .filter-bar {{
    background: white; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    padding: 16px 24px; margin-bottom: 20px;
    display: flex; gap: 32px; align-items: center; flex-wrap: wrap;
  }}
  .filter-group {{ display: flex; flex-direction: column; gap: 4px; }}
  .filter-group label {{ font-size: .8em; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; }}
  .filter-group input[type=range] {{ width: 140px; accent-color: #1a237e; }}
  .filter-count {{ font-size: .9em; color: #1a237e; font-weight: 700; margin-left: auto; }}

  /* Cards */
  .top-bets-card {{
    background: white; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08); margin-bottom: 24px; overflow: hidden;
  }}
  .card-header {{
    padding: 16px 24px; border-bottom: 1px solid #eee;
    font-weight: 700; font-size: 1.05em; color: #1a237e;
    display: flex; align-items: center; gap: 10px;
  }}
  .card-header .count-badge {{
    background: #1a237e; color: white; border-radius: 12px; padding: 2px 10px; font-size: .8em;
  }}
  .top-bets-card table {{ width: 100%; border-collapse: collapse; font-size: .9em; }}
  .top-bets-card th {{
    background: #f5f6fa; color: #666; font-weight: 600;
    text-transform: uppercase; font-size: .75em; letter-spacing: .5px;
    padding: 10px 16px; text-align: left;
  }}
  .top-bets-card td {{ padding: 12px 16px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
  .top-bets-card tr:last-child td {{ border-bottom: none; }}
  .top-bets-card tr:hover td {{ background: #fafbff; }}
  .rank {{ font-size: 1.1em; text-align: center !important; width: 40px; }}
  .league-tag {{ background: #e8eaf6; color: #3949ab; border-radius: 4px; padding: 2px 8px; font-size: .8em; font-weight: 600; }}
  .bet-outcome {{ border: 1.5px solid; border-radius: 4px; padding: 2px 8px; font-size: .82em; font-weight: 700; }}
  .edge-val {{ font-weight: 700; font-size: 1em; }}
  .odds-val {{ font-weight: 600; color: #333; }}
  .bk-tag {{ font-size:.78em; color:#666; font-weight:400; }}

  /* League sections */
  .league-section {{
    background: white; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08); margin-bottom: 20px; overflow: hidden;
  }}
  .league-header {{
    padding: 14px 24px; border-bottom: 1px solid #eee;
    display: flex; align-items: center; gap: 12px;
  }}
  .league-name {{ font-weight: 700; font-size: 1.05em; color: #1a237e; }}
  .value-count {{
    background: #e8f5e9; color: #2e7d32; border-radius: 12px;
    padding: 2px 10px; font-size: .8em; font-weight: 600;
  }}
  .fixture-table {{ width: 100%; border-collapse: collapse; font-size: .88em; }}
  .fixture-table th {{
    background: #f5f6fa; color: #888; font-weight: 600;
    text-transform: uppercase; font-size: .72em; letter-spacing: .4px;
    padding: 8px 12px; text-align: left;
  }}
  .fixture-table .subheader th {{ background: #fafafa; padding: 2px 12px; border-bottom: 1px solid #eee; }}
  .fixture-table td {{ padding: 10px 12px; border-bottom: 1px solid #f5f5f5; vertical-align: middle; }}
  .fixture-table tr:last-child td {{ border-bottom: none; }}
  .fixture-table tr:hover td {{ background: #fafbff; }}
  .value-row td {{ background: #f1f8e9 !important; }}
  .value-row:hover td {{ background: #e8f5e9 !important; }}
  .date-cell {{ color: #666; white-space: nowrap; width: 100px; }}
  .team-cell {{ width: 180px; }}
  .vs-cell {{ color: #bbb; text-align: center; width: 28px; padding: 0 !important; }}
  .prob-cell {{ width: 200px; }}
  .prob-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; margin-bottom: 4px; }}
  .prob-bar div {{ border-radius: 4px; }}
  .prob-labels {{ display: flex; justify-content: space-between; font-size: .75em; font-weight: 600; }}
  .odds-cell {{ text-align: center; width: 82px; font-weight: 600; color: #444; vertical-align: middle; }}
  .value-odds {{ border-left: 3px solid; }}
  .b365-val {{ display: block; font-size: .92em; }}
  .value-price {{ font-weight: 700; }}
  .edge-chip {{
    display: inline-block; color: white; border-radius: 3px;
    padding: 1px 5px; font-size: .72em; font-weight: 700; margin-top: 2px;
  }}
  .max-odds {{ display: block; font-size: .75em; color: #2e7d32; font-weight: 500; margin-top: 3px; }}
  .max-odds em {{ font-style: normal; color: #555; }}

  /* Monthly performance table */
  .perf-table {{ border-collapse: collapse; font-size: .82em; min-width: 600px; }}
  .perf-table th {{
    background: #f5f6fa; color: #888; font-weight: 600;
    text-transform: uppercase; font-size: .72em; letter-spacing: .4px;
    padding: 8px 10px; text-align: center; white-space: nowrap;
  }}
  .perf-table td {{ padding: 6px 10px; border-bottom: 1px solid #f5f5f5; text-align: center; white-space: nowrap; }}
  .perf-table tfoot td {{ border-top: 2px solid #e0e0e0; padding: 8px 10px; }}
  .month-label {{ text-align: left !important; color: #555; font-weight: 600; width: 60px; }}
  .perf-pos {{ background: #f1f8e9; color: #2e7d32; }}
  .perf-neg {{ background: #ffeaea; color: #c62828; }}
  .perf-empty {{ color: #bbb; }}
  .perf-total {{ font-weight: 700; }}

  .page-footer {{ text-align: center; color: #aaa; font-size: .82em; margin-top: 32px; }}
  .back-link {{ display: inline-block; margin-bottom: 16px; color: #1565c0; }}
</style>
{backtest_script}
</head>
<body>
<div class="container">

  <a class="back-link" href="index.html">&larr; Back to index</a>

  <div class="page-header">
    <h1>⚽ Match Week Predictions</h1>
    <div class="meta">Odds fetched {fetch_str} &nbsp;·&nbsp; Edge threshold {threshold:+.0%}</div>
    <div class="warning-banner">⚠ Verify odds are still current before placing any bet</div>
    <div class="stats-row">
      <div class="stat-pill">{total_fixtures} fixtures</div>
      <div class="stat-pill">{total_value} value bets</div>
    </div>
  </div>

  {filter_html}

  {empty_notice_html}

  {forecast_html}

  <div class="top-bets-card">
    <div class="card-header">
      Top Value Bets
      <span class="count-badge" id="value-bets-badge">{len(all_bets)}</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Date</th><th>League</th><th>Fixture</th>
          <th>Bet</th><th>Edge</th><th>B365 Odds</th><th>Best Odds</th>
        </tr>
      </thead>
      <tbody>
        {_top_bets_html(all_bets)}
      </tbody>
    </table>
  </div>

  {profit_html}

  {monthly_html}

  {league_html}

  <div class="page-footer">
    Generated {fetch_str} · Football Prediction Model
  </div>
</div>
</body>
</html>"""


def save_predictions_html(
    pred_rows: list[dict],
    threshold: float,
    fetched_at: datetime,
    output_path: Path,
    profit_curve_path: Path | None = None,
    historical_bets: pd.DataFrame | None = None,
) -> Path:
    html = generate_predictions_html(
        pred_rows, threshold, fetched_at,
        profit_curve_path=profit_curve_path,
        historical_bets=historical_bets,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
