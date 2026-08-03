# Evaluation and production market separation

## Stakeholder summary

Evaluation will cover every supported league, including leagues that are not currently
approved for production betting. Production inference will remain restricted to an
explicit allowlist. This lets us detect when a model change makes an excluded league
profitable without exposing that league in live predictions first.

## Goal

Separate two decisions that are currently mixed:

1. **Is the model profitable in a league?** Evaluation answers this for every supported
   league using the same walk-forward predictions and research betting policy.
2. **May production emit a bet in this league?** Production answers this using an
   explicit allowlist and saved per-league thresholds.

The repository currently supports only the 1X2 market. In this design, “market” means a
league within that market.

## League configuration

`src/config.py` will become the canonical source for two sets:

- `SUPPORTED_LEAGUES`: E0, D1, SP1, I1, F1, N1, P1, G1, SC0, B1, and T1.
- `PRODUCTION_LEAGUES`: E0, N1, P1, and G1.

The downloader, fixture loader, per-league training, walk-forward evaluation, report,
and prediction path will use these constants. `EXCLUDED_BETTING_LEAGUES` will be derived
from the difference for compatibility where a skip set is still useful.

All supported leagues remain available for training and evaluation. Changing the
production allowlist will therefore not change the evaluation universe.

## Evaluation data flow

The season-level walk-forward split remains unchanged: the latest four seasons are test
seasons, and each test season is predicted using only earlier seasons.

For every supported league with data:

1. Train and calibrate its model on earlier seasons.
2. Produce out-of-sample H/D/A probabilities for the test season.
3. Evaluate value bets using the CLI research threshold (normally `0.0`) and the common
   odds, edge, season-game, and overround limits.
4. Include the league in the combined metrics and per-league report breakdown.

The main evaluation metrics will be calculated from this all-market dataset. A league
will not be removed because its historical ROI is negative.

Per-league production thresholds will still be calibrated from prior out-of-sample
seasons. Thresholds will be calculated and saved for every supported league, but live
inference will consult them only for leagues in `PRODUCTION_LEAGUES`.

## Production data flow

Live inference will:

1. Train models for all supported leagues with available data.
2. Build features for all supported upcoming fixtures.
3. Retain only fixtures whose league is in `PRODUCTION_LEAGUES`.
4. Apply the saved league threshold and the production odds, edge, and overround limits.
5. Emit only the resulting production bets.

This keeps excluded leagues warm and measurable while preventing them from appearing as
actionable bets.

## Reports and artifacts

`reports/evaluation_report.html` will default to all observed supported leagues. League
buttons will be generated from the report data instead of hardcoded. The report will
also provide an explicit “Production markets” preset for viewing the current production
portfolio without changing the default all-market view.

The initial summary cards and the interactive summary will use the same all-market bet
dataset, removing the current mismatch between server-rendered production metrics and
embedded all-league bets.

Artifacts will have separate responsibilities:

- `reports/evaluation_bets.csv`: all supported leagues under the research policy.
- `reports/backtest_bets.csv`: production-portfolio simulation used by the live
  prediction report's historical performance section.
- `models/league_thresholds.json`: thresholds for all supported leagues.

## Failure handling

A supported league without sufficient training or test data will be skipped for that
walk-forward period and reported through the existing training output. It must not
receive zero-filled probabilities that look like real predictions.

An upcoming fixture without a trained league model will not be emitted as a production
prediction.

## Tests

The implementation will add or update tests that verify:

- the production allowlist is a subset of the supported universe;
- downloader and training league lists use the canonical configuration;
- all-market evaluation does not apply the production allowlist;
- production fixture filtering uses only `PRODUCTION_LEAGUES`;
- report league controls are generated dynamically, default to all markets, and retain
  the production-only preset;
- the complete existing test suite remains green.

## Scope boundaries

This change does not add betting types beyond 1X2, change the model algorithm, promote
new production leagues, or select a new production threshold. It only separates market
measurement from market activation.
