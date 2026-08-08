# Current autoresearch state

## Current best

The latest recorded production-portfolio result is iteration `EXP-20260519-S101`. It was measured on 2026-05-19 and predates the all-market evaluation split, so it should not be compared directly with new all-market headline metrics.

| Metric | `threshold=0.0` |
|---|---:|
| Accuracy | 0.359 |
| ROI | +9.65% |
| Stability | 0.0615 |
| t-statistic | +2.56 |
| Bets | 1,728 / 5,879 (29.4%) |

The result is above the approximate `t > 2` screening threshold, but it remains a historical backtest and may include selection effects from repeated experimentation.

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

1. Re-run the all-market baseline and record the new aggregate and per-league metrics.
2. Add referee tendency features if stable historical coverage can be obtained.
3. Test opening-to-closing market movement without introducing inference-only features.
4. Test an ensemble only if its component model adds independent out-of-sample signal.
5. Evaluate an xG-surplus feature after acquiring consistent historical coverage for all target leagues.
6. Compare fair-probability edge with raw implied-probability edge on a separately held-out period.

## File responsibilities

- `GUIDE.md`: iteration procedure.
- `EVALUATION.md`: stable evaluation and decision rules.
- `current.md`: this current snapshot and active queue.
- `experiments.md`: append-only historical record.
