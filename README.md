# Football Prediction Autoresearch

ML-driven value betting system for European football. Trains walk-forward models per league, identifies edges over B365 market odds, and outputs HTML evaluation and prediction reports.

**Leagues:** England (E0), Germany (D1), Spain (SP1), Italy (I1), France (F1), Netherlands (N1), Portugal (P1)

## Setup

```bash
uv sync
```

Data is downloaded automatically from [football-data.co.uk](https://www.football-data.co.uk) on first run.

## Usage

### Backtest (last 2 seasons, default)

```bash
uv run python main.py
uv run python main.py --per-league          # one model per league (recommended)
uv run python main.py --threshold 0.05      # stricter edge filter (default: 0.03)
uv run python main.py --max-odds 5.0        # higher odds cutoff (default: 4.0)
uv run python main.py --update              # re-download latest results, then backtest
```

Opens `reports/evaluation_report.html` with ROI breakdown, calibration chart, and profit curve.

### Predict upcoming fixtures

```bash
./predict.sh                     # edge >= 0.04 (production default)
./predict.sh --threshold 0.0     # all value bets (exploration)
uv run python main.py --predict  # same as predict.sh, threshold 0.03
```

Outputs a timestamped HTML report to `reports/predictions_<timestamp>.html`.

## Staking

Inverse-odds staking: `stake = max(3, 20 / B365_odds)`. Edge is measured against the vig-stripped B365 fair price. The Pinnacle line is used as an additional filter when available.

## Model

XGBoost/LightGBM ensemble, trained walk-forward (one model per league per test season). Features include rolling xG, form, odds-implied probabilities, and head-to-head history.

## Project layout

```
main.py                  # pipeline entrypoint
predict.sh               # prediction shortcut with production threshold
src/
  data/                  # download + loader
  model/                 # features + walk-forward training
  evaluation/            # metrics, HTML reports
reports/                 # generated outputs (gitignored)
docs/                    # experiment logs and evaluation standards
```
