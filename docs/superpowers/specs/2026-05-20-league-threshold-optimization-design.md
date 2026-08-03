# League-Specific Threshold Optimization

**Date:** 2026-05-20
**Status:** Approved

## Goal

Replace the single global edge threshold with a per-league threshold dict, optimized on prior out-of-sample walk-forward results. The global LGBM model, feature pipeline, and calibration are unchanged.

## Architecture

The only change is bet filtering. After predictions are made, instead of applying one global threshold, a `dict[league → threshold]` is applied per-league.

The backtest pipeline gains a **calibration phase**: before evaluating season T, per-league thresholds are derived from all accumulated OOS results from prior test seasons. Evaluation never feeds back into calibration.

### Season-by-season flow (TEST_SEASONS=4)

| Season | Calibration input | Thresholds applied |
|--------|------------------|--------------------|
| S1 | none | global default (0.0) |
| S2 | S1 results | per-league, calibrated on S1 |
| S3 | S1+S2 results | per-league, calibrated on S1+S2 |
| S4 | S1+S2+S3 results | per-league, calibrated on S1+S2+S3 |

Final reported metrics cover all 4 seasons with a clean causal chain (no lookahead).

## Threshold Selection

**Grid:** `[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]`

**Objective per league:** maximize `ROI × √bets` (stability metric), subject to minimum **20 bets** over the calibration window.

**Fallback:** if no threshold clears 20 bets in the calibration window, use the global default (0.0).

**Tie-breaking:** prefer the lower threshold (more conservative, more bets).

**Output:** a frozen `dict[league → float]` computed once before each evaluation season, logged with that season's results.

## Implementation

### New: `src/evaluation/threshold_selector.py`

```
select_league_thresholds(
    season_results: list[dict],   # accumulated OOS results from prior seasons
    leagues: list[str],
    grid: list[float]
) -> dict[str, float]
```

Pure function. Takes accumulated prior-season OOS results, sweeps the grid per league, returns the threshold dict.

### Modified: `src/model/train.py`

- `TEST_SEASONS` default changes from 2 → 4.
- The season loop calls `select_league_thresholds` on all prior test-season results before evaluating the current season.
- The per-league threshold dict is passed to the evaluation/reporting step.

### Modified: `src/evaluation/report.py` (or equivalent)

- Per-league section shows the chosen threshold alongside bet count and ROI, so calibration choices are auditable.

### New: `models/league_thresholds.json`

After each full backtest run, the final per-league threshold dict (from the last calibration step) is saved here. Updated automatically; committed by the `[skip ci]` chore job.

### Modified: `predict.sh` and live prediction path

- `predict.sh` no longer injects `--threshold 0.04`.
- Live prediction loads `models/league_thresholds.json` and applies per-league thresholds.

## What Does Not Change

- LGBM hyperparameters (`_LGBM_CFG`)
- Calibration setup (`CalibratedClassifierCV`, `cv=10`, isotonic)
- Feature pipeline (`features.py`)
- Evaluation metrics (ROI, stability, t-stat)
- League set (E0, N1, P1, G1, D1, SP1, I1, F1)

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Calibration window too thin (S1 only = ~150–300 bets/league) | min-bets floor of 20; fallback to 0.0 |
| Overfitting threshold to noise | Only one scalar per league tuned; ±8.6% per-league CI is the reference noise floor |
| Live prediction drift | `league_thresholds.json` is regenerated on every full backtest run |
