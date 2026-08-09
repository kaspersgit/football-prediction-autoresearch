# Current autoresearch state

## Current best

The latest recorded result is `EXP-20260804-001`, re-run on the current all-market/production-allowlist evaluation split. It measured on 2026-08-04. The earlier `EXP-20260519-S101` figures (ROI +9.65%) predate that split and are no longer comparable.

Production portfolio (E0, N1, P1, G1 — the metric that determines keep/revert decisions):

| Metric | `threshold=0.0` |
|---|---:|
| Bets | 2,101 |
| ROI | +1.03% |
| Per-league | England +0.72%, Netherlands +4.55%, Portugal −6.59%, Greece +1.09% |

All-market diagnostic (11 leagues, no max-edge/overround cap — informational, not a decision metric):

| Metric | `threshold=0.0` |
|---|---:|
| Accuracy | 0.518 |
| ROI | −4.35% |
| Stability | −0.0290 |
| t-statistic | −2.76 |
| Bets | 9,096 / 9,906 (91.8%) |

Production ROI is positive but well below screening significance and far weaker than the stale historical number. Portugal is the weakest production league (−6.59%) and a candidate for re-evaluation. (Per-league row corrected 2026-08-10 — the previously recorded values here were the all-market, unfiltered per-league table, not the actual production-portfolio split; see `EXP-20260810-001`'s note.)

### Pinnacle-confirmation filter — validated in backtest, not yet live

`EXP-20260810-001` re-verified the historical Pinnacle-confirmation filter (skip a bet unless Pinnacle's historical fair probability exceeds B365's fair probability by more than a 0.015 margin, using `PSCH/PSCD/PSCA`) against the current per-league model. Result: production portfolio 745 bets, ROI +15.46% (vs. 2,101 bets / +1.03% with the filter off), all 4 production leagues improved (+7.4pp to +20.1pp each), stability 0.0069 → 0.1029, t-stat 0.32 → 2.81 (crosses significance). This decisively clears `EVALUATION.md`'s keep/revert bar.

The filter code is landed (`pinnacle_confirmation_margin` param on `compute_value_betting_results` and `_build_prediction_rows`; `--pinnacle-filter` CLI flag for `main.py`'s backtest) but **defaults to off everywhere, including the live Predict workflow** (`_run_predict()` does not pass it to `_build_prediction_rows`). Live Pinnacle odds (`PSH/PSD/PSA`) are fetched and attached to fixtures (see "Live Pinnacle odds" below), so the live path is ready — flipping the filter on for real betting is a deliberate, separate decision pending explicit sign-off, not something this backtest result triggers automatically.

## Verified configuration

- Training: one LightGBM model with isotonic calibration per league and test season (`--per-league`), using four walk-forward test seasons.
- Model: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=0.05`.
- Calibration: isotonic, `cv=10`, `ensemble=False`.
- Features: 30 features covering EWM form, Elo, Elo momentum, market probabilities and overround, league identity, head-to-head, draw rate, market bias, match balance, and attack/defence ratings.
- Betting edge: model probability minus vig-normalized B365 fair probability.
- Filters: maximum odds `5.0`, maximum edge `0.20`, and maximum overround `0.07`.
- Evaluation leagues: England, Germany, Spain, Italy, France, Netherlands, Portugal, Greece, Scotland, Belgium, and Turkey. The research headline and report default include every observed supported league at the fixed CLI threshold.
- Production leagues: England (`E0`), Netherlands (`N1`), Portugal (`P1`), and Greece (`G1`). Only these leagues may appear in live predictions.
- Thresholds: backtesting calibrates one threshold per supported league from prior test seasons and writes `models/league_thresholds.json`; the production simulation and live prediction use those thresholds only for production leagues.
- Pinnacle: live odds fetched via The Odds API for production leagues and attached to fixtures (`src/data/pinnacle_odds.py`); the confirmation filter itself is validated in backtest (`EXP-20260810-001`) but stays off in live predictions pending explicit sign-off (see "Pinnacle-confirmation filter" above).
- Staking: flat one unit per backtest bet.

Executable betting defaults live in `src/config.py`. Model and feature parameters live in `src/model/train.py` and `src/model/features.py`.

## Canonical commands

```bash
# Primary research comparison
uv run python main.py --per-league --threshold 0.0

# Normal backtest CLI defaults (global model, threshold 0.03)
uv run python main.py

# Per-league comparison with the Pinnacle-confirmation filter on (validated, not yet live)
uv run python main.py --per-league --threshold 0.0 --pinnacle-filter

# Production prediction shortcut (saved league thresholds, CLI default 0.03 as fallback)
./predict.sh

# Tests
uv run pytest tests/ -v
```

## Active hypotheses

These ideas are not recorded as completed experiments in the consolidated ledger:

1. Add referee tendency features if stable historical coverage can be obtained.
2. Test opening-to-closing market movement without introducing inference-only features.
3. Test an ensemble only if its component model adds independent out-of-sample signal.
4. Evaluate an xG-surplus feature after acquiring consistent historical coverage for all target leagues.
5. Fix `main.py:_run_compare_vig`'s per-league breakdown, which crashes with `KeyError: 'league'` (merges on a column not present in `results["odds_test"]`). Found in `EXP-20260804-002`.
6. Decide whether to wire `pinnacle_confirmation_margin=DEFAULT_PINNACLE_CONFIRMATION_MARGIN` into `main.py:_run_predict()`'s live `_build_prediction_rows` call — validated in backtest (`EXP-20260810-001`, production ROI +1.03% → +15.46%), pending explicit user sign-off before it affects real betting.

Cleared this iteration: item 1 (all-market baseline re-run) done in `EXP-20260804-001`; item 6 (fair vs raw edge baseline) tested and reverted in `EXP-20260804-002`; Pinnacle-confirmation filter re-verified and kept at the backtest level in `EXP-20260810-001`.

## File responsibilities

- `GUIDE.md`: iteration procedure.
- `EVALUATION.md`: stable evaluation and decision rules.
- `current.md`: this current snapshot and active queue.
- `experiments.md`: append-only historical record.
