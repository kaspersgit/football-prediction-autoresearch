# Current autoresearch state

## Current best

`EXP-20260810-002` (2026-08-10) is the latest state: the Pinnacle-confirmation filter is now **live** and `PRODUCTION_LEAGUES` was re-chosen from a production-methodology screen (real per-league calibrated thresholds + max-edge/overround caps + the filter, using `PSH/PSD/PSA` — the realistic opening-odds proxy for live fetching) across all 11 supported leagues.

New production set: **England (E0), Netherlands (N1), Greece (G1), France (F1)**. Portugal (P1) was dropped.

Per-league result from the screen (opening-odds proxy, real thresholds/caps):

| League | Bets | ROI | Decision |
|---|---:|---:|---|
| Netherlands | 190 | +16.27% | keep |
| England | 44 | +17.93% | keep |
| Greece | 33 | +3.36% | keep |
| France | 34 | −0.21% | **added** — flat under this test, but the user's judgment is that live Predict runs close to kickoff will trend nearer Pinnacle's closing line than this worst-case opening-odds proxy |
| Portugal | 36 | −0.53% | **dropped** — flat/slightly negative under both the closing-odds and opening-odds proxies |
| Belgium, Scotland | 20–28 | +75–78% | **not added** — huge swings on tiny samples, read as noise per `EVALUATION.md`'s own small-sample guidance |
| Germany, Italy, Spain, Turkey | 29–206 | −7% to −19% | not added — clear, well-sampled negatives |

(League-selection numbers above are from the opening-odds screen used to decide the allowlist; the reference *performance* number for this league set has since moved to the closing-odds/veto-on-missing result below — see "Pinnacle-confirmation filter.")

All-market diagnostic (11 leagues, no max-edge/overround cap, filter off — informational, not a decision metric):

| Metric | `threshold=0.0` |
|---|---:|
| Accuracy | 0.518 |
| ROI | −4.35% |
| Stability | −0.0290 |
| t-statistic | −2.76 |
| Bets | 9,096 / 9,906 (91.8%) |

### Pinnacle-confirmation filter — now live (updated 2026-08-10, `EXP-20260810-004`)

**Current reference performance** (production portfolio, `E0/N1/G1/F1`, closing odds `PSCH/PSCD/PSCA`, veto-on-missing-data):

| League | Bets | ROI |
|---|---:|---:|
| England | 175 | +8.16% |
| France | 106 | +24.97% |
| Greece | 87 | +9.95% |
| Netherlands | 171 | +41.79% |
| **Total** | **539** | **+22.42%** |

Stability 0.1475, t-stat **+3.42** (crosses significance). Per-season: 2023/24 +24.58% (328 bets), 2024/25 +19.40% (168 bets), 2025/26 +17.81% (43 bets, truncated — see archive gap below). **All 3 seasons profitable with consistent magnitude** — the strongest season-breadth result of any Pinnacle-filter variant tested.

**History of how this number was reached:** `EXP-20260810-001` first validated the filter on closing odds with the original "skip check if Pinnacle data missing" null-handling (745 bets, +15.46%, t-stat 2.81). `EXP-20260810-002` re-tested with `PSH/PSD/PSA` (opening odds — the only kind a live snapshot can ever produce) under the same null-handling, weaker but still positive (303 bets, +13.11%, t-stat 1.42, below significance). Per explicit user direction, `EXP-20260810-004` then **inverted the null-handling**: missing Pinnacle data now **vetoes** the bet instead of letting it through unfiltered — tightening "only bet when Pinnacle actually confirms" to mean what it says.

**Archive gap discovered while testing this:** football-data.co.uk's historical Pinnacle-odds coverage (both opening and closing columns) drops to a flat **0% for all four production leagues from mid-January 2026 onward**, and hasn't recovered as of this run (2026-08-10). This is why the 2025/26 season contributes only 43 bets instead of a full season's worth. **This does not affect live betting** — live Pinnacle odds come from The Odds API independently, fetched fresh at prediction time, unrelated to football-data.co.uk's archive. It only limits how much of the current season can be used as backtest evidence. Given this, and since closing vs. opening odds show the identical coverage gap (so switching doesn't dodge the problem), the user chose closing odds as the reference methodology anyway — explicitly optimistic (a live snapshot is never a true closing line), with the live-vs-closing comparison below still the open question.

`pinnacle_confirmation_margin=DEFAULT_PINNACLE_CONFIRMATION_MARGIN` (0.015) remains wired into all three live call sites in `main.py:_run_predict()` (`_build_prediction_rows`, `_print_predictions`, `_save_predictions_csv`); the null-handling inversion applies automatically since it lives in the shared filter logic. `reports/backtest_bets.csv` reflects this run.

**Follow-ups queued** (see Active hypotheses): (1) once enough live Predict runs have accumulated, check how close live-fetched Pinnacle odds actually land to closing-line behavior — this run's headline number is explicitly optimistic pending that check; (2) keep an eye on whether football-data.co.uk's Pinnacle coverage resumes for future seasons (the mid-January 2026 cutoff may or may not be permanent).

## Verified configuration

- Training: one LightGBM model with isotonic calibration per league and test season (`--per-league`), using four walk-forward test seasons.
- Model: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=0.05`.
- Calibration: isotonic, `cv=10`, `ensemble=False`.
- Features: 30 features covering EWM form, Elo, Elo momentum, market probabilities and overround, league identity, head-to-head, draw rate, market bias, match balance, and attack/defence ratings.
- Betting edge: model probability minus vig-normalized B365 fair probability.
- Filters: maximum odds `5.0`, maximum edge `0.20`, and maximum overround `0.07`.
- Diagnostics: `compute_season_breadth` (per-season profitability breadth, printed by `main.py`'s primary comparison as "SEASON BREADTH" — flags a change that only looks good pooled because one strong season masks several weak ones; require ≥3/4 profitable seasons) supplements the pooled ROI/stability/t-stat metrics as of 2026-08-10.
- Evaluation leagues: England, Germany, Spain, Italy, France, Netherlands, Portugal, Greece, Scotland, Belgium, and Turkey. The research headline and report default include every observed supported league at the fixed CLI threshold.
- Production leagues: England (`E0`), Netherlands (`N1`), Greece (`G1`), and France (`F1`). Only these leagues may appear in live predictions. Re-chosen 2026-08-10 (`EXP-20260810-002`) — Portugal was dropped, France added; see "Current best" above.
- Thresholds: backtesting calibrates one threshold per supported league from prior test seasons and writes `models/league_thresholds.json`; the production simulation and live prediction use those thresholds only for production leagues.
- Pinnacle: live odds fetched via The Odds API for production leagues and attached to fixtures (`src/data/pinnacle_odds.py`, with date-aware matching against `commence_time` to avoid attaching the wrong matchweek). The confirmation filter is validated (`EXP-20260810-001`/`-002`) and **live** as of 2026-08-10 (see "Pinnacle-confirmation filter" above).
- Staking: flat one unit per backtest bet.

Executable betting defaults live in `src/config.py`. Model and feature parameters live in `src/model/train.py` and `src/model/features.py`.

## Canonical commands

```bash
# Primary research comparison
uv run python main.py --per-league --threshold 0.0

# Normal backtest CLI defaults (global model, threshold 0.03)
uv run python main.py

# Per-league comparison with the Pinnacle-confirmation filter on (closing odds, historical validation)
uv run python main.py --per-league --threshold 0.0 --pinnacle-filter

# Same, but using PSH/PSD/PSA (opening odds) -- the realistic proxy for live fetching
uv run python main.py --per-league --threshold 0.0 --pinnacle-filter-opening

# Screen every supported league under the real production methodology (diagnostic only)
uv run python main.py --per-league --threshold 0.0 --pinnacle-filter-opening --all-leagues-production

# Production prediction shortcut (saved league thresholds, CLI default 0.03 as fallback)
./predict.sh

# Tests
uv run pytest tests/ -v
```

## Active hypotheses

These ideas are not recorded as completed experiments in the consolidated ledger:

1. Add referee tendency features if stable historical coverage can be obtained.
2. Test an ensemble only if its component model adds independent out-of-sample signal.
3. Evaluate an xG-surplus feature after acquiring consistent historical coverage for all target leagues.
4. Fix `main.py:_run_compare_vig`'s per-league breakdown, which crashes with `KeyError: 'league'` (merges on a column not present in `results["odds_test"]`). Found in `EXP-20260804-002`.
5. Once enough live Predict runs have accumulated, check how close live-fetched Pinnacle odds (`PSH/PSD/PSA` via `attach_pinnacle_odds`) actually land to historical closing-line behavior in practice — the reference performance number (`EXP-20260810-004`) is explicitly optimistic (uses closing odds `PSCH/PSCD/PSCA`, which a live snapshot can never truly be) pending this check. If live odds don't track closing-line behavior well, reconsider France's inclusion in `PRODUCTION_LEAGUES` (added on a flat opening-odds result, per user judgment about live timing).
6. Keep an eye on whether football-data.co.uk's Pinnacle-odds coverage (both `PSH/PSD/PSA` and `PSCH/PSCD/PSCA`) resumes for future seasons — it dropped to 0% for all four production leagues from mid-January 2026 onward and hadn't recovered as of `EXP-20260810-004` (2026-08-10). Does not affect live betting (The Odds API is independent), but limits backtest evidence for the current season until/unless it resumes.
7. Populate the `ODDS_API_TEAM_ALIASES` tables for the 6 leagues never yet fetched live (`D1`, `SP1`, `I1`, `SC0`, `B1`, `T1`) if `PRODUCTION_LEAGUES` widens again — repeat the live-diff process used for E0/N1/P1/G1/F1.
8. The all-market season-breadth diagnostic already fails (0/3 profitable) at the current verified baseline, independent of any feature — this is a known, pre-existing property of the diagnostic (see `EXP-20260810-003`'s correction), not a new problem. Worth keeping in mind so a future iteration doesn't mistake "season breadth still fails" for evidence against that iteration's own change, the way `EXP-20260810-003` initially did.

Cleared this iteration: item 1 (all-market baseline re-run) done in `EXP-20260804-001`; item 6 (fair vs raw edge baseline) tested and reverted in `EXP-20260804-002`; Pinnacle-confirmation filter re-verified and kept at the backtest level in `EXP-20260810-001`, then re-validated against the realistic opening-odds proxy and made live in `EXP-20260810-002`; opening-to-closing market movement tested (as a live-computable, lagged team-level feature) and reverted in `EXP-20260810-003`; Pinnacle filter tightened to veto on missing data and reference methodology switched to closing odds in `EXP-20260810-004`.

## File responsibilities

- `GUIDE.md`: iteration procedure.
- `EVALUATION.md`: stable evaluation and decision rules.
- `current.md`: this current snapshot and active queue.
- `experiments.md`: append-only historical record.
