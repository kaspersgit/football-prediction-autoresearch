# Current autoresearch state

## Current best

The latest recorded result is `EXP-20260804-001`, re-run on the current all-market/production-allowlist evaluation split. It measured on 2026-08-04. The earlier `EXP-20260519-S101` figures (ROI +9.65%) predate that split and are no longer comparable.

Production portfolio (E0, N1, P1, G1 — the metric that determines keep/revert decisions):

| Metric | `threshold=0.0` |
|---|---:|
| Bets | 2,101 |
| ROI | +1.03% |
| Per-league | England +1.80%, Netherlands +7.97%, Portugal +0.21%, Greece −5.81% |

All-market diagnostic (11 leagues, no max-edge/overround cap — informational, not a decision metric):

| Metric | `threshold=0.0` |
|---|---:|
| Accuracy | 0.518 |
| ROI | −4.35% |
| Stability | −0.0290 |
| t-statistic | −2.76 |
| Bets | 9,096 / 9,906 (91.8%) |

Production ROI is positive but well below screening significance and far weaker than the stale historical number. Greece is the weakest production league (−5.81%) and a candidate for re-evaluation.

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
- Pinnacle: not used because its odds are unavailable for live fixtures.
- Staking: flat one unit per backtest bet.

Executable betting defaults live in `src/config.py`. Model and feature parameters live in `src/model/train.py` and `src/model/features.py`.

## Canonical commands

```bash
# Primary research comparison
uv run python main.py --per-league --threshold 0.0

# Normal backtest CLI defaults (global model, threshold 0.03)
uv run python main.py

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

Cleared this iteration: item 1 (all-market baseline re-run) done in `EXP-20260804-001`; item 6 (fair vs raw edge baseline) tested and reverted in `EXP-20260804-002`.

## File responsibilities

- `GUIDE.md`: iteration procedure.
- `EVALUATION.md`: stable evaluation and decision rules.
- `current.md`: this current snapshot and active queue.
- `experiments.md`: append-only historical record.
