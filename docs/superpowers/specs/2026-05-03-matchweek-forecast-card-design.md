# Matchweek Forecast Card

**Date:** 2026-05-03  
**Status:** Approved

## Goal

Add a "Matchweek Forecast" card to the predictions report that shows the predicted ROI for the current set of upcoming value bets, together with a 95% confidence interval. The card updates live as filter sliders change.

## Placement

Between the filter bar and the top-bets table — the first piece of analysis the user sees after adjusting filters.

## Statistics

### Point estimate (expected ROI)

Derived from `BACKTEST_BETS` filtered to the current slider settings (same `filterBacktestBets(minEdge, minOdds, maxOdds)` call used elsewhere). The backtest mean profit/stake is the calibrated expected ROI — conservative and empirically grounded rather than using raw model edge.

```
mean_return = mean(b.profit / b.stake  for b in filtered_backtest)
```

### Confidence interval

**Hybrid approach**: expected return from backtest (calibrated), variance from the specific odds of each upcoming bet (Bernoulli formula).

Per-bet variance for upcoming bet i:
```
p_i     = 1/O_i + edge_i          // model prob, from data-odds and data-edge on .bet-row
var_i   = p_i * (1 - p_i) * O_i²  // Bernoulli outcome variance
```

Total variance over N upcoming bets (unit stake each):
```
total_var = sum(var_i  for i in upcoming_bets)
se        = sqrt(total_var) / N_upcoming
lower_95  = mean_return - 1.96 * se
upper_95  = mean_return + 1.96 * se
```

This means the CI width automatically narrows when filtering to low-odds bets (lower O² per bet), correctly reflecting reduced outcome variance.

## Card Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Matchweek Forecast                    based on 847 hist. bets  │
├──────────────────┬──────────────────────────────────────────────┤
│  +4.2%           │  ──────────[━━━━━━━━━━━━━━━]────────────    │
│  predicted ROI   │  −8%              +4.2%             +17%     │
│  5 bets          │  ←────────── 95% CI ──────────────→         │
└──────────────────┴──────────────────────────────────────────────┘
```

- **Left panel**: large predicted ROI (green if positive, red if negative), count of upcoming bets
- **Right panel**: horizontal range bar
  - Thin line: full 95% CI span
  - Thick segment: ±1σ inner band
  - Tick/dot: point estimate
  - Axis labels at lower, mean, upper
- **Subtitle**: "based on N historical bets at current filters"

## Edge Cases

| Condition | Behaviour |
|-----------|-----------|
| N_upcoming < 5 | Show warning "CI unreliable — fewer than 5 bets" instead of range bar |
| N_backtest < 30 | Show "Insufficient historical data for this filter combination" |
| No backtest data at all | Card hidden (same pattern as profit curve / monthly table) |

## Implementation

### Python (predictions_report.py)

Add `_forecast_card_html(historical_bets)` — returns a card shell with `id="forecast-card-container"` when `historical_bets` is not empty, empty string otherwise. No computation happens in Python; all stats computed in JS.

Add `data-model-prob` attribute to each `.bet-row` in `_top_bets_html` so JS can read p_i directly (avoids recomputing from edge + implied prob):

```python
data-model-prob="{b['model_prob']:.4f}"
```

`model_prob` is already available in the `all_bets` dict (sourced from `ModelH/D/A` columns).

### JavaScript (inline in filter_bar_html)

Add `rebuildForecastCard(backtestFiltered, upcomingRows)` function:

1. Guard: return early if no backtest data or card container missing
2. Compute `mean_return` from `backtestFiltered`
3. Read `data-odds` and `data-model-prob` from each visible `.bet-row`
4. Compute per-bet variance, sum to `total_var`, derive `se`, `lower`, `upper`
5. Render HTML into `#forecast-card-container`

Call `rebuildForecastCard` at the end of `applyFilters()`, passing `filtered` (already computed) and the count of visible rows.

### Data dependency

`model_prob` must be added to the `all_bets` dict in `generate_predictions_html`. The source field is `r["Model{outcome}"]` (e.g. `r["ModelH"]` for a Home value bet), already present on every pred_row.

## Files changed

- `src/evaluation/predictions_report.py` — add `_forecast_card_html`, add `data-model-prob` to bet rows, add `model_prob` to `all_bets`, call `rebuildForecastCard` in `applyFilters`, insert card into page HTML
