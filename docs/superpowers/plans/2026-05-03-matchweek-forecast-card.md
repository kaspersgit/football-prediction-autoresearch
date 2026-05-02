# Matchweek Forecast Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live "Matchweek Forecast" card to the predictions report showing predicted ROI and a 95% CI for the current set of upcoming value bets, updating as filter sliders change.

**Architecture:** All computation is client-side JS — Python only renders an empty card shell. Expected ROI is the backtest mean profit/stake for the filtered historical cohort. CI width uses per-bet Bernoulli variance from each upcoming bet's actual odds (so low-odds filters produce tighter CIs). The card sits between the filter bar and the top-bets table and hooks into the existing `applyFilters()` pipeline.

**Tech Stack:** Python (HTML generation), vanilla JavaScript (stats + DOM rendering), pytest

---

## File map

| File | Change |
|------|--------|
| `src/evaluation/predictions_report.py` | Add `model_prob` to `all_bets`; add `data-model-prob` to `.bet-row`; add `_forecast_card_html()`; wire card into page template; add `rebuildForecastCard` JS + call in `applyFilters` |
| `tests/test_predictions_report.py` | New file — unit tests for Python-side changes |

---

## Task 1: Add `model_prob` to `all_bets` and `data-model-prob` to bet rows

**Files:**
- Modify: `src/evaluation/predictions_report.py` (function `generate_predictions_html` ~line 533, function `_top_bets_html` ~line 89)
- Test: `tests/test_predictions_report.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_predictions_report.py`:

```python
import math
from datetime import datetime

import pandas as pd
import pytest

from src.evaluation.predictions_report import (
    _forecast_card_html,
    _top_bets_html,
    generate_predictions_html,
)


def _pred_rows():
    return [
        {
            "Date": datetime(2026, 5, 10),
            "League": "england",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "ModelH": 0.55,
            "ModelD": 0.25,
            "ModelA": 0.20,
            "B365H": 1.90,
            "B365D": 3.50,
            "B365A": 4.50,
            "CustomMaxH": float("nan"),
            "CustomMaxD": float("nan"),
            "CustomMaxA": float("nan"),
            "CustomMaxBkH": "",
            "CustomMaxBkD": "",
            "CustomMaxBkA": "",
            "ValueBets": [("H", 0.023)],
        }
    ]


def _historical_bets():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=50),
        "league": ["E0"] * 50,
        "stake": [1.0] * 50,
        "profit": ([0.9, -1.0, 0.9, -1.0, 0.9]) * 10,
        "odds": [1.90] * 50,
        "model_prob": [0.55] * 50,
        "implied_prob": [0.526] * 50,
        "y_true": ["H", "A", "H", "A", "H"] * 10,
        "y_pred": ["H", "H", "H", "H", "H"] * 10,
    })


def test_all_bets_includes_model_prob():
    rows = _pred_rows()
    html = generate_predictions_html(
        rows, threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=_historical_bets(),
    )
    # The bet row must carry data-model-prob
    assert 'data-model-prob="0.5500"' in html


def test_top_bets_html_includes_model_prob_attr():
    bets = [
        {
            "date": "Sat May 10",
            "league": "england",
            "home": "Arsenal",
            "away": "Chelsea",
            "outcome": "H",
            "edge": 0.023,
            "b365_odds": 1.90,
            "max_odds": float("nan"),
            "max_bk": "",
            "model_prob": 0.55,
        }
    ]
    html = _top_bets_html(bets)
    assert 'data-model-prob="0.5500"' in html
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  uv run pytest tests/test_predictions_report.py -v 2>&1 | tail -20
```

Expected: FAIL — `TypeError` because `model_prob` key missing from `all_bets` dict.

- [ ] **Step 3: Add `model_prob` to `all_bets` in `generate_predictions_html`**

In `src/evaluation/predictions_report.py`, find the `all_bets.append({...})` block (~line 536) and add one line:

```python
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
```

- [ ] **Step 4: Add `data-model-prob` to `.bet-row` in `_top_bets_html`**

In `_top_bets_html` (~line 89), change the opening `<tr>` line:

Old:
```python
        f'<tr class="bet-row" data-edge="{b["edge"]:.4f}" data-odds="{b["b365_odds"]:.2f}">'
```

New:
```python
        f'<tr class="bet-row" data-edge="{b["edge"]:.4f}" data-odds="{b["b365_odds"]:.2f}" data-model-prob="{b["model_prob"]:.4f}">'
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  uv run pytest tests/test_predictions_report.py -v 2>&1 | tail -20
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  git add src/evaluation/predictions_report.py tests/test_predictions_report.py && \
  git commit -m "feat: add model_prob to all_bets and data-model-prob to bet rows"
```

---

## Task 2: Add `_forecast_card_html` Python function and insert into page

**Files:**
- Modify: `src/evaluation/predictions_report.py`
- Test: `tests/test_predictions_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_predictions_report.py`:

```python
def test_forecast_card_html_empty_when_no_historical():
    assert _forecast_card_html(None) == ""
    assert _forecast_card_html(pd.DataFrame()) == ""


def test_forecast_card_html_contains_container_id():
    html = _forecast_card_html(_historical_bets())
    assert "forecast-card-container" in html


def test_generate_predictions_html_includes_forecast_card():
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=_historical_bets(),
    )
    assert "forecast-card-container" in html


def test_generate_predictions_html_no_forecast_card_without_historical():
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=None,
    )
    assert "forecast-card-container" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  uv run pytest tests/test_predictions_report.py -v 2>&1 | tail -20
```

Expected: FAIL — `ImportError` because `_forecast_card_html` not yet defined.

- [ ] **Step 3: Add `_forecast_card_html` function**

Add this function to `src/evaluation/predictions_report.py` after `_profit_curve_html` (~line 208):

```python
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
```

- [ ] **Step 4: Wire the card into `generate_predictions_html`**

In `generate_predictions_html`, add `forecast_html` alongside the other HTML fragments (~line 563):

```python
    profit_html = _profit_curve_html(historical_bets)
    monthly_html = _monthly_league_table_html(historical_bets)
    forecast_html = _forecast_card_html(historical_bets)
    filter_html = _filter_bar_html(threshold)
    backtest_script = _backtest_data_script(historical_bets)
```

Then insert `{forecast_html}` into the return template, between `{filter_html}` and `<div class="top-bets-card">`:

```python
  {filter_html}

  {forecast_html}

  <div class="top-bets-card">
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  uv run pytest tests/test_predictions_report.py -v 2>&1 | tail -20
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  git add src/evaluation/predictions_report.py tests/test_predictions_report.py && \
  git commit -m "feat: add forecast card HTML shell to predictions report"
```

---

## Task 3: Add `rebuildForecastCard` JS and wire into `applyFilters`

**Files:**
- Modify: `src/evaluation/predictions_report.py` (JS inside `_filter_bar_html`)

No unit tests for JS (inline template string, no JS test runner in this project). Verified by visual inspection.

- [ ] **Step 1: Add `rebuildForecastCard` JS function**

In `_filter_bar_html` in `src/evaluation/predictions_report.py`, add the following JS block **before** the `// ── main filter function` comment (~line 477). The entire function to insert:

```javascript
// ── matchweek forecast card ───────────────────────────────────────────────────
function rebuildForecastCard(backtestFiltered, nUpcoming) {{
  var container = document.getElementById('forecast-card-container');
  if (!container) return;
  var sub = document.getElementById('forecast-subtitle');

  if (!BACKTEST_BETS.length || backtestFiltered.length < 30) {{
    container.innerHTML = '<p style="text-align:center;color:#aaa;padding:8px 0">Insufficient historical data for this filter combination</p>';
    return;
  }}

  // Expected ROI from backtest (calibrated point estimate)
  var returns = backtestFiltered.map(function(b) {{ return b.profit / b.stake; }});
  var nHist = returns.length;
  var mean = returns.reduce(function(a, b) {{ return a + b; }}, 0) / nHist;

  if (nUpcoming < 5) {{
    var sign = mean >= 0 ? '+' : '';
    container.innerHTML =
      '<div style="padding:8px 0;color:#555;font-size:.9em">Predicted ROI: <strong style="color:' +
      (mean >= 0 ? '#2e7d32' : '#c62828') + '">' + sign + (mean * 100).toFixed(1) + '%</strong>' +
      ' &nbsp;·&nbsp; <span style="color:#f57c00">⚠ CI unreliable — fewer than 5 bets</span></div>';
    if (sub) sub.textContent = 'predicted ROI · ' + nHist + ' hist. bets';
    return;
  }}

  // Per-bet Bernoulli variance using each upcoming bet's actual odds
  var upcomingRows = document.querySelectorAll('.bet-row');
  var totalVar = 0, count = 0;
  upcomingRows.forEach(function(row) {{
    if (row.style.display === 'none') return;
    var odds = parseFloat(row.dataset.odds);
    var mp = parseFloat(row.dataset.modelProb);
    if (isNaN(odds) || isNaN(mp)) return;
    totalVar += mp * (1 - mp) * odds * odds;
    count++;
  }});
  if (count === 0) return;

  var se = Math.sqrt(totalVar) / count;
  var lower = mean - 1.96 * se;
  var upper = mean + 1.96 * se;
  var lower1s = mean - se;
  var upper1s = mean + se;

  function fmt(v) {{ return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; }}
  var roiColor = mean >= 0 ? '#2e7d32' : '#c62828';

  // Map value → % position within [lower, upper]
  var span = upper - lower || 0.01;
  function toPos(v) {{ return Math.max(0, Math.min(100, (v - lower) / span * 100)); }}
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
}}

```

- [ ] **Step 2: Call `rebuildForecastCard` inside `applyFilters`**

In `applyFilters`, inside the `if (BACKTEST_BETS.length)` block, add the call after `rebuildPerformanceTable`:

```javascript
  if (BACKTEST_BETS.length) {{
    var filtered = filterBacktestBets(minEdge, minOdds, maxOdds);
    rebuildProfitCurve(filtered);
    rebuildPerformanceTable(filtered);
    rebuildForecastCard(filtered, shown);
    var sub = document.getElementById('perf-table-subtitle');
    if (sub) sub.textContent = 'backtest · 20/odds staking · ' + filtered.length + ' bets';
    var psub = document.getElementById('profit-curve-subtitle');
    if (psub) psub.textContent = 'last 2 seasons walk-forward · ' + filtered.length + ' bets';
  }}
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  uv run pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 4: Visual verification**

Generate a predictions report (or use the most recent one in `reports/`) and open it in a browser. Verify:
- Forecast card appears between filter bar and top-bets table
- Big ROI number shown in green/red
- CI range bar renders with thin outer line and shaded ±1σ band
- Dragging the min/max odds sliders updates the card live
- When filtering to low odds (max odds ~1.8), CI narrows compared to wide odds range
- When fewer than 5 bets visible, warning text appears instead of range bar

- [ ] **Step 5: Commit**

```bash
cd /home/tappe/projects/football_pred_autoresearch && \
  git add src/evaluation/predictions_report.py && \
  git commit -m "feat: matchweek forecast card with predicted ROI and 95% CI"
```
