# Live Pinnacle odds via The Odds API — design

## Goal

Give live predictions access to real Pinnacle odds so the Pinnacle-confirmation filter — the strongest feature ever found in this project's backtests — can be evaluated and, if it still holds up, restored for production betting.

## Background

The filter (only bet when `pinnacle_fair[outcome] > b365_fair[outcome] + margin`) drove nearly all of the model's historical backtest ROI (`EXP-20260513-S057` through `S062`). It was fully removed in `EXP-20260517-S072` after discovering `fixtures.csv` from football-data.co.uk never carries live Pinnacle odds (0/112 rows populated) — the filter had been silently skipped at every live prediction all along, so the backtest number never reflected achievable live performance. `autoresearch/current.md` still records the gap: "Pinnacle: not used because its odds are unavailable for live fixtures."

The user has an API key for The Odds API (`THEODDS_API` in `.env`), confirmed working, which serves live pre-match Pinnacle odds per league at ~1 credit per call.

## Scope

Production leagues only: `E0`, `N1`, `P1`, `G1` (matches `PRODUCTION_LEAGUES` in `src/config.py`). These are the only leagues that place live bets, so this keeps API usage and team-matching work bounded. At ~4 calls per Predict run × 2 runs/week, usage stays near 35 credits/month against a 500/month free-tier quota.

## Components

### `src/data/pinnacle_odds.py` (new)

`fetch_pinnacle_odds(leagues: set[str]) -> pd.DataFrame` — for each requested league:

1. Look up its Odds API sport key from a module-level `_LEAGUE_TO_SPORT_KEY` dict (`E0` → `soccer_epl`, `N1` → `soccer_netherlands_eredivisie`, `P1` → `soccer_portugal_primeira_liga`, `G1` → `soccer_greece_super_league`).
2. Call `GET /v4/sports/{sport_key}/odds/?apiKey=...&regions=eu&markets=h2h&bookmakers=pinnacle&oddsFormat=decimal`.
3. Normalize `home_team`/`away_team` through the alias table (below); rows for teams not found in the alias table are dropped, not guessed.
4. Return a DataFrame with columns `league, HomeTeam, AwayTeam, PSH, PSD, PSA`.

Failure handling, in order of precedence:

- `THEODDS_API` not set in the environment → return an empty DataFrame immediately, log once. No behavior change for anyone without the key.
- Per-league request error (timeout, non-200, rate limit) → catch, log, skip that league, continue with the rest.
- Any exception must never propagate out of this function — a failure here must degrade to "no Pinnacle data," never break the Predict workflow. This mirrors the empty-fixtures handling already established in `_save_empty_predictions_report`.

### `src/data/team_aliases.py` (new)

`ODDS_API_TEAM_ALIASES: dict[str, dict[str, str]]`, keyed by league code, mapping an Odds API team name to the matching football-data.co.uk name, e.g.:

```python
ODDS_API_TEAM_ALIASES = {
    "N1": {
        "FC Utrecht": "Utrecht",
        "FC Twente Enschede": "Twente",
        "FC Zwolle": "Zwolle",
        ...
    },
    "P1": {...},
    "G1": {...},
    "E0": {...},  # likely empty or near-empty — names mostly already match
}
```

Built once by diffing each league's current Odds API team list against its football-data CSV team list; teams whose names already match exactly need no entry. Ambiguous or diacritic-heavy cases (Portuguese, Greek clubs) are resolved by hand, not fuzzy-matched — a wrong match would silently misprice a bet.

### Integration point: `main.py:_run_predict()`

Immediately after `fixtures_df = load_fixtures()`, call `attach_pinnacle_odds(fixtures_df)`, which left-merges `fetch_pinnacle_odds(PRODUCTION_LEAGUES)` onto `fixtures_df` on `(league, HomeTeam, AwayTeam)`, overwriting the `PSH`/`PSD`/`PSA` placeholder `NaN` columns that `load_fixtures()` already reserves via `_FIXTURE_PINNACLE_COLS`. No downstream schema change: `src/model/features.py` already passes `PSH`/`PSD`/`PSA` through as pass-through metadata on `fixture_features`.

### Restoring the Pinnacle-confirmation filter

Two call sites, both currently containing no trace of the old filter (fully deleted in `EXP-20260517-S072`):

- `src/evaluation/metrics.py:compute_value_betting_results` — re-add the Pinnacle-confirmation check as an opt-in parameter (default off), using historical `PSCH`/`PSCD`/`PSCA` already present in the training CSVs. Skip the check (never veto) when Pinnacle columns are null for a row, same as the original implementation.
- `main.py:_build_prediction_rows` — same check, using the new live `PSH`/`PSD`/`PSA` columns. Only enabled once the backtest re-verification below passes.

Margin constant: start from the last-confirmed value (`0.015`, from `EXP-20260513-S062`), added to `src/config.py`.

## Evaluation plan (per `autoresearch/EVALUATION.md`)

The model has changed substantially since the filter was last validated (Elo, Dixon-Coles ratings, production/all-market split, and per-league thresholds didn't exist in May). The old backtest numbers do not transfer automatically. Sequence:

1. Land `metrics.py`'s restored filter behind a default-off parameter — no behavior change yet.
2. Run it as a new dated `EXP-*` iteration against the current model and current evaluation split (`uv run python main.py --per-league --threshold 0.0`), with the filter on vs off, using historical `PSCH`/`PSCD`/`PSCA` — no API calls needed for this step.
3. Apply the keep/revert rules from `EVALUATION.md`. Only if ROI and stability hold up does the live path (`main.py`) get wired to use it.
4. Record the outcome in `autoresearch/experiments.md` and update `autoresearch/current.md`.

## Testing

- `tests/test_pinnacle_odds.py`: mock `requests` calls; cover missing API key, per-league request failure, successful parse + alias resolution, and unmatched-team skip.
- `tests/test_metrics.py`: extend with cases for the restored Pinnacle-confirmation parameter — filter active with agreement, filter active with disagreement (veto), and null-Pinnacle passthrough.

## Manual step (not automated)

`THEODDS_API` must be added as a GitHub Actions repository secret before `predict.yml` can use it in CI. This is a deliberate manual step, not scripted as part of this change.

## Out of scope

- Non-production leagues (`D1`, `SP1`, `I1`, `F1`, `SC0`, `B1`, `T1`) are not fetched live. They remain backtest-only, using historical Pinnacle columns as today.
- Replacing B365 as the primary reference odds — this change only restores the confirmation filter, it does not change the fair-odds baseline.
- Historical backfill of live Odds API data — backtests continue to use football-data.co.uk's historical `PSCH`/`PSCD`/`PSCA`, which already cover the full training history.
