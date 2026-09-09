# Current autoresearch state

## Current best

`EXP-20260810-002` (2026-08-10) is the latest state: the Pinnacle-confirmation filter is now **live** and `PRODUCTION_LEAGUES` was re-chosen from a production-methodology screen (real per-league calibrated thresholds + max-edge/overround caps + the filter, using `PSH/PSD/PSA` — the realistic opening-odds proxy for live fetching) across all 11 supported leagues.

New production set: **England (E0), Netherlands (N1), Greece (G1), France (F1)**. Portugal (P1) was dropped.

**Update 2026-08-28 (`EXP-20260828-001`):** Portugal (P1) was re-added to `PRODUCTION_LEAGUES`, per explicit user direction, after an all-leagues-production re-screen (same calibrated-threshold methodology as this section's table) showed it flat-to-slightly-positive (+3.01% ROI, 135 bets) — the weakest of the resulting five kept leagues, but not the clearly-negative case the other six non-production leagues showed. Current production set is now **E0, N1, G1, F1, P1**; see `experiments.md` for the full screen results and the live team-alias verification.

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

**Updated 2026-08-23 (`EXP-20260823-002`) — threshold grid capped at 0.05, same Pinnacle-filter/closing-odds methodology:**

| League | Bets | ROI |
|---|---:|---:|
| England | 174 | +9.94% |
| France | 182 | +16.53% |
| Greece | 142 | +16.57% |
| Netherlands | 173 | +35.52% |
| **Total** | **671** | **+19.73%** |

Stability 0.1287, t-stat **+3.33** (still crosses significance). Bet count up 35.6% vs. the 539-bet table above, on the same league set and filter — the table above is now historical context for how the Pinnacle filter itself was validated, not the current reference number. See `EXP-20260823-002` for why this was kept despite lower per-bet ROI/stability than the pre-cap grid (deliberate volume-for-stability trade-off, on explicit user direction).

**Updated 2026-08-28 (`EXP-20260828-001`) — Portugal (P1) added, same threshold grid/Pinnacle-filter/closing-odds methodology:**

| League | Bets | ROI |
|---|---:|---:|
| England | 166 | +5.90% |
| France | 139 | +18.17% |
| Greece | 145 | +9.89% |
| Netherlands | 163 | +38.75% |
| Portugal | 135 | +3.01% |
| **Total** | **748** | **+15.59%** |

Total ROI down from +19.73% to +15.59% (Portugal's weaker per-bet ROI dilutes the blend, as expected — this was a volume/coverage decision on explicit user direction, not a metric-improvement one; see `EXP-20260828-001` in `experiments.md`). Also confirmed via a live Odds API diff that all 5 production leagues' team-name alias tables are current — the only unmapped names anywhere (E0, F1, G1, P1) are newly promoted clubs with no historical rows to alias against, not real mismatches; N1 had none at all.

## Verified configuration

- Training: one LightGBM model with isotonic calibration per league and test season (`--per-league`), using three walk-forward test seasons (`TEST_SEASONS = 3`; only seasons with ≥1,000 rows count as a completed test season — see `EXP-20260810-005` — so an in-progress season, e.g. a handful of 2026/27 fixtures, can never silently displace a full season from the window).
- Model: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=0.05`.
- Calibration: isotonic, `cv=10`, `ensemble=False`.
- Features: 34 features covering EWM form, Elo, Elo momentum, market probabilities and overround, league identity (one-hot per `SUPPORTED_LEAGUES` entry except the `I1` reference — `EXP-20260828-002`), head-to-head, draw rate, market bias, match balance, and attack/defence ratings. The league-identity one-hot columns are constant (zero-variance, unused by tree splits) within every per-league model, i.e. under both the live prediction pipeline and the standard `--per-league` research methodology — they only matter for the dormant global (single shared model) mode.
- Betting edge: model probability minus vig-normalized B365 fair probability.
- Filters: maximum odds `5.0`, maximum edge `0.20`, and maximum overround `0.07`.
- Diagnostics: `compute_season_breadth` (per-season profitability breadth, printed by `main.py`'s primary comparison as "SEASON BREADTH" — flags a change that only looks good pooled because one strong season masks several weak ones; require ≥3/4 profitable seasons) supplements the pooled ROI/stability/t-stat metrics as of 2026-08-10.
- Evaluation leagues: England, Germany, Spain, Italy, France, Netherlands, Portugal, Greece, Scotland, Belgium, and Turkey. The research headline and report default include every observed supported league at the fixed CLI threshold.
- Production leagues: England (`E0`), Netherlands (`N1`), Greece (`G1`), France (`F1`), and Portugal (`P1`). Only these leagues may appear in live predictions. Re-chosen 2026-08-10 (`EXP-20260810-002`) — Portugal was dropped, France added; Portugal re-added 2026-08-28 (`EXP-20260828-001`); see "Current best" above.
- Thresholds: backtesting calibrates one threshold per supported league from prior test seasons over `_THRESHOLD_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]` (capped at 0.05, down from 0.10 — `EXP-20260823-002`, 2026-08-23) and writes `models/league_thresholds.json`; the production simulation and live prediction use those thresholds only for production leagues.
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

# Contrarian "hot-hand fallacy" replication (diagnostic only, no training) -- EXP-20260909-001
uv run python main.py --hot-hand-diagnostic

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
7. Populate the `ODDS_API_TEAM_ALIASES` tables for `D1`, `SP1`, `I1`, `SC0`, `B1`, `T1` only if `PRODUCTION_LEAGUES` widens to include one of them — a live diff was run for all six on 2026-08-28 (`EXP-20260828-001`) purely as part of the all-leagues-production financial screen and found real gaps (13 unmapped in D1, 13 in SP1, 4 in I1, 2 in SC0, 11 in B1, 11 in T1), but none of those six leagues were added, so the tables were deliberately left unpopulated — re-diff live (names go stale season to season) before actually promoting any of them, don't reuse this run's list unchanged.
8. The all-market season-breadth diagnostic already fails (0/3 profitable) at the current verified baseline, independent of any feature — this is a known, pre-existing property of the diagnostic (see `EXP-20260810-003`'s correction), not a new problem. Worth keeping in mind so a future iteration doesn't mistake "season breadth still fails" for evidence against that iteration's own change, the way `EXP-20260810-003` initially did.
9. ~~Contrarian "hot hand fallacy" momentum signal.~~ **DONE — REJECTED at the diagnostic stage, `EXP-20260909-001` (2026-09-09).** Step 1 (standalone diagnostic, `--hot-hand-diagnostic`) run on all 11 leagues / 2013–2026 across six rating constructions (window 6 & 5 × vig-stripped trailing / raw `1/odds` trailing / cumulative-season). Backing the colder team loses 3.3–4.0% ROI everywhere (t ≈ −4 to −5 vs zero, ~the vig); colder beats hotter by only ~1–2pp with paired t ≈ 0.2–1.3 (never significant; article claimed p=0.002); the only significant single-league result (Belgium, paired t −2.27) contradicts the hypothesis. Did not replicate → per the plan's step 4, stopped before any feature work. The diagnostic tool is kept.
10. **Pinnacle-confirmation margin — next iteration (item 9 above is now concluded).** A 2026-08-29 diagnostic sweep (throwaway-branch CI runs, not yet a formal `EXP-...` entry — no code changed) re-measured this against the current baseline (5-league production portfolio, `EXP-20260828-001`/`-002`): margin ∈ {none, 0.000, 0.005, 0.010, 0.015 (current default), 0.020, 0.025, 0.030} → {3003 bets/−2.80%/t=−1.03, 1444/+9.17%/t=+2.28, 1198/+11.31%/t=+2.58, 947/+10.23%/t=+2.14, 748/+15.59%/t=+2.92, 590/+17.18%/t=+2.91, 451/+19.79%/t=+3.03, 362/+20.42%/t=+2.88}. Margin=0.005 passes the new volume-for-ROI trade rule numerically (+60.2% bets, t-stat 2.58 ≥ 2.5, ROI decline 4.28pp ≤ 5pp) — but the per-league breakdown fails `EVALUATION.md` rule 3 outright: only England improves (166→268 bets, +5.90%→+16.55%); France, Greece, Netherlands, and Portugal all worsen, with Portugal flipping to outright negative (+3.01%→−2.51%). **Not adopted** on this evidence — the aggregate gain is one league's good luck masking broad-based deterioration elsewhere, exactly the failure mode the volume-for-ROI rule's rules-2–4 carve-out was written to catch. Formalize this as a real `EXP-...` entry (with a decision) — item 9 concluded in `EXP-20260909-001`, so this is now the next iteration — reusing these numbers rather than re-running the sweep from scratch unless the production baseline has moved meaningfully by then. Historical context: two other volume levers were tested previously — `max_overround` raised to 0.09 (reverted, `EXP-20260823-001`) and the threshold grid capped at 0.05 (kept, `EXP-20260823-002`, +35.6% bets on explicit user direction).

11. **COD season-luck quantile — the "done properly" overreaction test** (source: Wheatcroft, *Profiting from overreaction in soccer betting odds*, JQAS 2020, [PDF](https://researchonline.lse.ac.uk/id/eprint/115490/1/10.1515_jqas_2019_0009.pdf); read 2026-09-09). This is **not** a re-run of item 9 — construction and mechanism differ. Claim: bettors attribute a team's *season-to-date* over/under-performance vs its own market-implied schedule to genuine quality change, so teams that have banked fewer points than the odds implied get systematically generous prices going forward.
   - **The COD statistic:** for a team at match `N` of a season, Monte-Carlo simulate each of its prior matches from that match's normalised odds-implied `(win/draw/loss)` probabilities (`m ≈ 512` sims), sum simulated points per sim → a distribution of season-to-date point totals. COD = the quantile of the team's *actual* points total within that distribution. Low COD ⇒ under-performed its own odds (unlucky / market over-adjusted). Computed over all prior matches this season (`r = N−1`); require ≥ 6 prior matches; reset each season. Uses best available odds across books (our `CustomMax` → `B365` fallback).
   - **Their result:** backing the lowest-COD teams (home *and* away) is significantly profitable — full 90% bootstrap band above zero, net of a positive overround — across 20 leagues **including the Eredivisie, Primeira Liga and Belgian First Division A** (our `N1`, `P1`, `B1`). Effect is stronger than, and not explained by, the favourite-longshot bias. Greece not in their set.
   - **How it differs from item 9 (`EXP-20260909-001`):** season-cumulative not trailing-6; a *points quantile benchmarked against a simulation of the odds themselves* (auto-adjusts for schedule strength) rather than a mean probability residual; within-season reset captures "this season's narrative" mispricing specifically. Item 9's rejection does not pre-empt this — but note item 9 did show the broad "overreaction" family not paying on our leagues against sharp-ish prices, so treat a null result here as expected-ish, not surprising.
   - **Plan (diagnostic-first, exactly like item 9):**
     1. `main.py --cod-diagnostic` in `src/evaluation/` (standalone, no training). Compute COD per team per match over all 11 leagues / all seasons; bucket bets by COD decile; report ROI of backing low-COD teams vs the market, per league and per season, with `colder`-style vs-zero and paired (low vs high COD) t-stats. Best-price settlement (`CustomMax`→`B365`), flat stakes. Vig-strip odds-implied probs the multiplicative way (matches `_fair_probs` in `hot_hand.py`).
     2. Only if the low-COD ROI edge replicates with a plausible t-stat *and* holds in ≥ 2 of `N1/P1/B1` (the leagues the paper covers): add `home_cod` / `away_cod` (or `cod_diff`) to `FEATURE_COLS` and run the full per-league walk-forward on CI, same keep bar as any feature per `EVALUATION.md`.
     3. Correlation check before trusting it as new info: against `home_market_bias`/`away_market_bias` (a rolling actual-minus-market residual — conceptually adjacent), `market_overround`, and the Elo-momentum features. Precedent: `EXP-20260426-D055`.
     4. If step 1 doesn't replicate, stop — do not build a feature on the paper's numbers alone.
12. **Sharp-vs-soft line *structure*, beyond the static Pinnacle-confirmation filter.** The current filter is binary ("does Pinnacle fair prob beat B365 fair prob by `DEFAULT_PINNACLE_CONFIRMATION_MARGIN`"). Literature (e.g. [statsbet.org — reading sharp money movement](https://statsbet.org/blog/dropping-odds-explained); closing-line-value research generally) says the *magnitude* and *direction* of the sharp/soft divergence carry more than a threshold crossing. We already fetch B365 + Pinnacle live and have historical `PS*`/`PSC*`, so these are cheap filter/staking refinements, not new data:
   - **(a) Size-graded confirmation:** scale conviction / stake with the Pinnacle-minus-B365 fair-prob gap instead of pass/fail. Test as a staking rule first (ROI% is stake-invariant under flat, so measure via a Kelly-style or banded stake and report bankroll-growth + drawdown, not just ROI).
   - **(b) Dispersion filter:** require B365 to be an outlier vs Pinnacle *and* vs the `CustomMax` multi-book consensus — i.e. only bet when the soft price we're taking is loose relative to *both* sharp references, not just Pinnacle.
   - **(c) Pinnacle drift:** did Pinnacle move toward our side between open and close (`PSH → PSCH`)? Historical only for backtest; live we only get one snapshot, but the daily vig tracker (`data/diagnostics/pinnacle_vig_history.csv`) is starting to accumulate intra-week movement.
   - **Not** a re-run of `EXP-20260810-003` (which added *historical team-level* open→close movement as a *model feature* and failed) — this is using *this match's* live sharp-line structure as a *bet filter*, the same mechanism that already works for us.
   - **Plan:** extend the existing `--pinnacle-filter` backtest path with each variant behind its own flag; compare on the 5-league production portfolio (closing-odds methodology, per `EXP-20260828-001`) against the current binary filter as baseline. Keep bar: `EVALUATION.md` including the volume-for-ROI trade rule. Main risk: heavy overlap with the existing confirmation filter — a variant only earns its place if it beats the binary filter, not just beats no-filter.
13. **Lower-priority leads from the 2026-09-09 literature scan** (not yet worked; recorded so they aren't lost):
    - **Non-transitive triads** (van Ours, *Non-transitive patterns in sports match outcomes*, Empirical Economics 2025, [link](https://link.springer.com/article/10.1007/s00181-025-02838-6)): A>B>C>A dominance loops persist across 25 EPL seasons and bookmakers don't price them (they prioritise odds *consistency*). Testable on our data with no external inputs — build a directed dominance graph from recent results/H2H, find intransitive triads, check whether the "should-lose-by-transitivity but historically wins" side is underpriced. Partial overlap with `h2h_home_win_rate`.
    - **Asian-Handicap-implied 1X2 as the edge baseline** (from *Forecasting soccer matches with betting odds: a tale of two markets*, Int. J. Forecasting 2025, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0169207024000670)): AH markets show no favourite-longshot bias and concentrate sharp money; a 1X2 fair prob derived from AH + totals lines may be a cleaner `baseline_prob` than vig-stripped B365 1X2. Infrastructure change (better baseline that every downstream edge calc inherits), not a feature. football-data.co.uk carries AH/totals columns in many files — check coverage across our leagues/seasons first.
    - **Asymmetric rest disadvantage:** raw "days rest" was reverted twice (`S023`, `D088`) as a *symmetric* feature; the untested angle is the *asymmetry* in the situations where it bites — a short-rest heavy favourite (odds < ~1.6) vs a rested underdog. Narrow diagnostic, our data only.
    - **xG mean reversion** (Wilkens, *Can simple models predict football*, 2026, [link](https://journals.sagepub.com/doi/10.1177/22150218261416681); ~10% avg-odds / ~15% best-odds ROI on Bundesliga): strongest single academic lead but blocked on data — Understat/FBref xG covers the big-5 leagues, not `N1/P1/G1`. Already tracked as item 3; this is the supporting evidence for why it's worth doing *if* the production set ever changes or as an evaluation-only signal.

Cleared this iteration: item 1 (all-market baseline re-run) done in `EXP-20260804-001`; item 6 (fair vs raw edge baseline) tested and reverted in `EXP-20260804-002`; Pinnacle-confirmation filter re-verified and kept at the backtest level in `EXP-20260810-001`, then re-validated against the realistic opening-odds proxy and made live in `EXP-20260810-002`; opening-to-closing market movement tested (as a live-computable, lagged team-level feature) and reverted in `EXP-20260810-003`; Pinnacle filter tightened to veto on missing data and reference methodology switched to closing odds in `EXP-20260810-004`; walk-forward test-season selection fixed to exclude in-progress seasons, `TEST_SEASONS` set to 3 explicitly in `EXP-20260810-005` (also incidentally fixed the all-market season-breadth diagnostic, now 3/3 PASS instead of the pre-existing 0/3 noted in item 8 below — that item is now historical, from before this fix); item 7's live team-alias diff was run for all six non-production leagues in `EXP-20260828-001`, and Portugal (P1) was added back to `PRODUCTION_LEAGUES` from that same screen — item 7 itself stays open for the other six since none of them were promoted; item 9 (contrarian hot-hand fallacy) replicated as a standalone diagnostic in `EXP-20260909-001` and was rejected — it did not reproduce on our leagues, so no feature was built (the `--hot-hand-diagnostic` tool is kept).

## File responsibilities

- `GUIDE.md`: iteration procedure.
- `EVALUATION.md`: stable evaluation and decision rules.
- `current.md`: this current snapshot and active queue.
- `experiments.md`: append-only historical record.
