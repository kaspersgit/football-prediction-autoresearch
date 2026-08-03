# Football Prediction Autoresearch

Football value-betting research pipeline for European leagues. It trains walk-forward models, compares their probabilities with B365 market odds, and produces HTML backtest and upcoming-fixture reports.

Evaluation covers all 11 supported leagues. Production predictions are limited to England (`E0`), Netherlands (`N1`), Portugal (`P1`), and Greece (`G1`) through an explicit allowlist.

## Setup

```bash
uv sync --extra dev
```

Historical match data is downloaded from [football-data.co.uk](https://www.football-data.co.uk) on first use.

## Backtesting

Use the explicit research configuration when comparing experiments:

```bash
uv run python main.py --per-league --threshold 0.0
```

Other supported commands:

```bash
uv run python main.py                         # global model, threshold 0.03
uv run python main.py --per-league            # one model per league, threshold 0.03
uv run python main.py --threshold 0.05        # override the minimum model edge
uv run python main.py --max-odds 4.0          # override the default maximum of 5.0
uv run python main.py --update                # refresh current-season results first
uv run python main.py --monthly --per-league  # experimental monthly retraining
```

Research evaluation uses flat one-unit stakes and includes every supported league. The common filters are:

- maximum B365 odds: `5.0`;
- maximum model edge: `0.20`;
- maximum B365 overround: `0.07`.

The HTML report opens with all observed leagues enabled and includes a **Production markets** preset. The run writes:

- `reports/evaluation_report.html` and `reports/evaluation_bets.csv` for all-market research evaluation;
- `reports/backtest_bets.csv` and `reports/profit_curve.png` for the production portfolio simulation; and
- `models/league_thresholds.json` with calibrated thresholds for every supported league.

## Upcoming predictions

```bash
./predict.sh                     # production shortcut, using saved league thresholds
./predict.sh --threshold 0.0     # fallback for leagues without a saved threshold
uv run python main.py --predict  # direct CLI, threshold 0.03
```

Prediction reports use saved per-league thresholds when `models/league_thresholds.json` exists. They otherwise use the CLI threshold. Live inference only emits fixtures in `PRODUCTION_LEAGUES`; supported leagues outside that allowlist remain trained and evaluated. The odds, edge, and overround filters match the production simulation, and the report displays an inverse-odds stake suggestion of `max(3, 20 / B365_odds)`; this does not change the flat-stake evaluation metric.

The run writes timestamped CSV and HTML reports under `reports/`.

## Model

The recommended mode trains one LightGBM classifier per league and test season, followed by isotonic probability calibration (`cv=10`, `ensemble=False`). Features cover recent form, Elo and Elo momentum, B365-implied fair probabilities, market overround and bias, league identity, head-to-head results, draw tendency, match balance, and attack/defence ratings.

Pinnacle odds and xG are not part of the current model. The betting edge is measured against vig-normalized B365 implied probabilities.

## Research documentation

- [`autoresearch/GUIDE.md`](autoresearch/GUIDE.md): iteration procedure.
- [`autoresearch/EVALUATION.md`](autoresearch/EVALUATION.md): evaluation and keep/revert rules.
- [`autoresearch/current.md`](autoresearch/current.md): latest recorded configuration, metrics, and active hypotheses.
- [`autoresearch/experiments.md`](autoresearch/experiments.md): append-only experiment history.

## Project layout

```text
main.py                  # pipeline entry point
predict.sh               # prediction shortcut with saved league thresholds
src/config.py            # shared betting defaults
src/data/                # downloads and loading
src/model/               # feature generation and walk-forward training
src/evaluation/          # betting metrics and HTML reports
autoresearch/            # procedure, policy, current state, and history
docs/index.html          # GitHub Pages landing page
reports/                 # generated outputs
tests/                   # automated tests
```
