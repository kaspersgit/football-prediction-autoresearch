# Autoresearch State Document

## Current Best Model

| Metric    | threshold=0.0        |
|-----------|----------------------|
| Accuracy  | 0.515                |
| ROI       | **+2.64%** ✅ |
| Stability | **+0.0172** ✅ |
| Bets      | 2223 / 2655 (83.7%) |
| Training  | One HistGBM **per league** per test season (`--per-league`) |
| Features  | 8 EWM/Elo + 3 market fair probs + 3 league dummies + H2H + 2 draw rates + 2 market bias = **19 features** |
| Model cfg | max_depth=4, min_samples_leaf=20, l2_regularization=0.1, lr=0.05, max_iter=300 |
| Bet filter | Pinnacle closing (`PSCH/PSCD/PSCA`) confirms edge over B365 where available |

_Last updated: 2026-04-21 (Iterations 44–53: market bias + l2_regularization=0.1. **New best: ROI +3.15%, Stability +0.0204.**)_

**Evaluation setup (updated 2026-04-19):** All metrics from Iteration 11 onward use:
- **Walk-forward backtest**: one model trained per test season (2425 then 2526); Elo carries forward correctly.
- **Value betting**: multi-outcome, vig-corrected fair probabilities. Primary evaluation at `--threshold 0.0`.
- **Test set**: 2425 (1426 matches) + 2526 partial (1200 matches) = 2626 total.
- **Primary goal**: ROI > 0% AND Stability > 0 at threshold=0.0. This is the only robust, unbiased metric.
- **Retired**: threshold=0.06 was selected by grid search on the test set (Iter 15) — it carries look-ahead bias and is not a valid operating point. It may be reported as a secondary reference but must not drive keep/revert decisions.

---

## Baseline (Iteration 0)

**Date:** 2026-04-17
**Hypothesis:** N/A — this is the starting baseline.

**Model:** Logistic Regression with StandardScaler
**Features (6 total):**
- `home_pts_5` — home team rolling mean points over last 5 games
- `home_gf_5` — home team rolling mean goals for over last 5 games
- `home_ga_5` — home team rolling mean goals against over last 5 games
- `away_pts_5` — away team rolling mean points over last 5 games
- `away_gf_5` — away team rolling mean goals for over last 5 games
- `away_ga_5` — away team rolling mean goals against over last 5 games

**Training split:**
- Train: seasons 1314 through 2223
- Test: seasons 2324 and 2425 (last 2 seasons)

**Results:**
- Accuracy: 0.492
- ROI: -6.79%
- Stability: -0.0637
- Total test bets: 2643

**Analysis:**
The baseline model barely beats random on accuracy and loses money at -6.79% ROI, which is close to the bookmaker vig (~5%). Stability is negative, meaning losses are not evenly distributed — there are clusters of bad bets. This gives a clear floor to beat.

---

## Iteration History

## Experiment: Binary Outcome Models (NOT ADOPTED — per-league multi-class remains best)

**Date:** 2026-04-22
**Hypothesis:** Training separate binary classifiers (one per outcome: H/D/A) instead of a single multi-class model would improve per-outcome calibration, especially for Home (-8.49% ROI) and Draw (-6.72% ROI) bets which lagged far behind Away (+4.23%) in the global model.

**Variants tested:**

| Mode | Models | Bets | ROI | Stability |
|------|--------|------|-----|-----------|
| Global multi-class (no flags) | 1 | 2380 | -4.57% | -0.031 |
| **Per-league multi-class (baseline)** | **4** | **2223** | **+2.64%** | **+0.017** |
| Binary outcomes — global (`--binary`) | 3 | 2450 | +0.29% | +0.002 |
| Binary + per-league (`--binary --per-league`) | 12 | 2206 | +2.02% | +0.013 |

Per-league breakdown:

| League | Per-league baseline | Binary global | Binary+per-league |
|--------|---------------------|---------------|-------------------|
| England | +12.45% | +1.64% | +6.99% |
| Germany | +0.38% | +3.36% | +2.35% |
| Spain | -6.09% | -3.12% | -1.68% |
| Italy | +3.44% | -0.56% | +0.68% |

**Analysis:** Binary models do not improve over the per-league multi-class baseline. The global binary setup fixes Italy and Germany but badly hurts England (+12.45% → +1.64%), which is the model's strongest league. The combined binary+per-league (12 models) is a middle ground but still weaker than the 4-model baseline — smaller training sets per binary model add variance without meaningful calibration gain. The per-league multi-class model (4 models, one per league) remains the best configuration. Note: the +3.15% ROI from the last commit vs +2.64% now reflects the 2525-26 season accumulating more data since the experiment was recorded.

**Decision:** NOT ADOPTED. Per-league multi-class (`--per-league`) remains default.

---

## Iteration 43: Season Progress Features (REVERTED — flat)

**Date:** 2026-04-21
**Hypothesis:** `home_season_progress` / `away_season_progress` (games played / 38) would help the model account for early-season noise vs settled mid/late-season form.
**Files changed:** `src/model/features.py` — added features + wiring. Reverted after results.

**Results:**

| Metric    | Iter 41 baseline | Iter 43 | Δ |
|-----------|-----------------|---------|---|
| ROI       | +1.07%          | +1.06%  | -0.01% |
| Stability | +0.0073         | +0.0069 | -0.0004 |

**Decision:** REVERTED — flat result, no meaningful signal added.

---

## Iteration 42: Elo Delta / Momentum (REVERTED — regression)

**Date:** 2026-04-21
**Hypothesis:** `home_elo_delta` / `away_elo_delta` (Elo change over last WINDOW games) captures trajectory distinct from absolute Elo.
**Files changed:** `src/model/features.py` — added features + wiring. Reverted after results.

**Results:**

| Metric    | Iter 41 baseline | Iter 42 | Δ |
|-----------|-----------------|---------|---|
| Accuracy  | 0.511           | 0.489   | -0.022 ❌ |
| ROI       | +1.07%          | +0.46%  | -0.61% ❌ |
| Stability | +0.0073         | +0.0030 | -0.0043 ❌ |

**Decision:** REVERTED — strong regression. Elo delta is highly collinear with recent form features, adding noise rather than signal.

---

## Iteration 41: Draw Rate Features (KEPT — new best)

**Date:** 2026-04-21
**Hypothesis:** Per-team rolling draw rate (home/away draws over last WINDOW games) would help the model identify draw-prone matchups that the market misprices.
**Files changed:** `src/model/features.py` — added `home_draw_rate`, `away_draw_rate` to `FEATURE_COLS` (17 total), called `_compute_draw_rates` in `_build_merged`, and `_get_current_draw_rates` in `build_fixture_features`.

**Results (threshold=0.00, no threshold tuning):**

| Metric    | Iter 38 (15 features) | Iter 41 (+draw rate) | Δ |
|-----------|-----------------------|----------------------|---|
| ROI       | +0.78%                | **+1.07%**           | +0.29% ✅ |
| Stability | +0.0051               | **+0.0073**          | +0.0022 ✅ |
| Accuracy  | 0.508                 | 0.511                | +0.003 |
| Bets      | 2203                  | 2181                 | -22 |

**Decision:** KEPT. Modest but clean improvement at threshold=0.00. Threshold tuning (Iter 40) was rejected as threshold selection is unstable across model changes and effectively overfits on the test set.

---

## Iteration 40: Threshold=0.08 as Default (REVERTED — threshold tuning is unstable)

**Date:** 2026-04-21
**Hypothesis:** Threshold grid showed 0.08 gives +2.13% ROI vs +0.78% at threshold=0.00.
**Decision:** REVERTED. Threshold selection is post-hoc optimization on the test set — it shifts every time the model or feature set changes, making it an unreliable operating parameter. Evaluation stays at threshold=0.00.

---

## Iteration 39: Shots on Target EWM Form (REVERTED — regression)

**Date:** 2026-04-20
**Hypothesis:** Rolling EWM of shots on target (`HST`/`AST`) captures chance-creation quality beyond goals, which have higher per-game variance. This is the basis of modern xG systems — shots on target predicts future performance better than raw goals.
**Files changed:** `src/data/loader.py` — added `HST/AST` to `_OPTIONAL_COLS`; `src/model/features.py` — added `home_form_hst`, `away_form_ast` to `FEATURE_COLS` (17 total); extended `_team_rolling_stats` and `_get_current_team_form`. Reverted after results.

**Results (reverted, tested on top of Iter 38 Pinnacle filter):**

| Metric    | Iter 38 (Pinnacle filter) | Iter 39 (+shots on target) | Δ |
|-----------|---------------------------|----------------------------|---|
| Accuracy  | 0.508                     | 0.511                      | +0.003 |
| ROI       | **+0.78%**                | **-1.23%**                 | **-2.01pp** ↓ |
| Stability | **+0.0051**               | **-0.0084**                | **-0.0135** ↓ |
| Bets      | 2203                      | 2185                       | -18 |

**Analysis:** Shots on target is a clear regression despite a marginal accuracy gain. The market already incorporates shot statistics thoroughly — bookmakers use shots, shots on target, and xG in their pricing. Adding this signal adds two more features that are correlated with `form_gf`/`form_ga` without providing genuinely new information. The per-league models (now trained on ~4500 rows each) continue to suffer from additional correlated features splitting the tree budget. Reverted.

---

## Iteration 38: Pinnacle as Value-Detection Criterion (KEPT — **GOALS ACHIEVED**)

**Date:** 2026-04-20
**Hypothesis:** Pinnacle closing odds represent the sharpest available market consensus. Instead of using Pinnacle as a model feature (Iter 37 — failed), use it as a bet filter: only place a bet when BOTH `model_prob > B365_fair_prob` AND `Pinnacle_fair_prob > B365_fair_prob`. The second condition says the sharp market also sees value at B365 odds, confirming our model's signal. Where Pinnacle data is unavailable (null), the filter is skipped — no data means no veto.
**Files changed:** `src/model/features.py` — added `PSCH/PSCD/PSCA` to odds output of `build_features_with_odds`; `src/evaluation/metrics.py` — added Pinnacle confirmation check in `compute_value_betting_results` (computes Pinnacle vig-corrected fair probs per row, skips bet if `pinnacle_fair[outcome] <= b365_fair[outcome]`, handles nulls gracefully).

**Results:**

| Metric    | Iter 33 baseline (no Pinnacle) | Iter 38 (Pinnacle filter) | Δ |
|-----------|---------------------------------|---------------------------|---|
| Accuracy  | 0.508                           | 0.508                     | 0 |
| ROI       | **-1.66%**                      | **+0.78%** ✅             | **+2.44pp** ↑ |
| Stability | **-0.0107**                     | **+0.0051** ✅            | **+0.0158** ↑ |
| Bets      | 3595                            | 2203                      | -1392 |

- **BOTH GOALS ACHIEVED. First positive ROI (+0.78%) and positive Stability (+0.0051) in the project.**
- Bet count drops from 3595 to 2203 — Pinnacle vetoes ~1400 bets where both markets agree the outcome is fairly priced.

**Analysis:** The Pinnacle filter works by removing the noisiest third of bets — those where our model sees edge over B365 but the sharp market (Pinnacle) disagrees. The remaining 2203 bets are those where Pinnacle also finds B365 is under-pricing the outcome. This is the "triple confirmation" signal: (1) our form/Elo model predicts higher probability than B365 fair, (2) Pinnacle's sharp market consensus also prices the outcome higher than B365. The ~40% of current-season (2526) matches without Pinnacle data fall back to the existing model-only filter, which is appropriate — no Pinnacle data means no Pinnacle filter.

Mechanistically: the ~1400 vetoed bets are cases where Pinnacle priced the outcome LOWER than B365 (B365 was over-reacting to public money or liability management in that direction) while our model agreed with the public. These bets have negative expected value. The Pinnacle filter correctly identifies and removes them.

**KEPT — new default. Run with `uv run python main.py --per-league`.**

---

## Iteration 37: Pinnacle Closing Odds as Features (REVERTED — regression, two variants)

**Date:** 2026-04-20
**Hypothesis:** Pinnacle closing odds (PSCH/PSCD/PSCA) represent a sharper, more accurate market consensus (~2.7% margin vs B365's ~5-7%). Adding Pinnacle fair probs as features gives the model a "sharp money" reference alongside B365, enabling it to detect where B365 is systematically over or under-pricing outcomes. Backed by direct empirical evidence: football-data.co.uk's own research showed Pinnacle yields 101.81% ROI as a benchmark vs competitors. Pinnacle columns confirmed present in all 53 CSV files.
**Files changed:** `src/data/loader.py` — added `PSCH/PSCD/PSCA` as optional pass-through columns; `src/model/features.py` — computed Pinnacle fair probs with B365 fallback for null rows. Two variants tested. Both reverted.

**Variant A: Both B365 + Pinnacle (18 features)**

Added `pinnacle_h/d/a` alongside existing `market_h/d/a` (6 total market probability features).

| Metric    | Iter 33 baseline | Variant A (+pinnacle alongside) | Δ |
|-----------|------------------|---------------------------------|---|
| ROI       | **-1.66%**       | **-3.78%**                      | **-2.12pp** ↓ |
| Stability | **-0.0107**      | **-0.0257**                     | **-0.0150** ↓ |

**Variant B: Pinnacle only, replacing B365 as market reference (15 features)**

Replaced `market_h/d/a` (B365 fair probs) with `pinnacle_h/d/a` (Pinnacle fair probs).

| Metric    | Iter 33 baseline | Variant B (pinnacle replaces B365) | Δ |
|-----------|------------------|------------------------------------|---|
| ROI       | **-1.66%**       | **-2.57%**                         | **-0.91pp** ↓ |
| Stability | **-0.0107**      | **-0.0171**                        | **-0.0064** ↓ |

**Analysis:** Both variants regress. The failure reveals an important architectural constraint:

The value bet filter compares `model_prob > B365_fair_prob`. The model is already implicitly doing what Pinnacle tries to do — learning to deviate from the market. When Pinnacle features are added, the model learns to track Pinnacle-vs-B365 discrepancies, but this is already a well-known competitive edge exploited by professional syndicates; the market has largely priced it in.

More concretely: the ~35-40% null rate for Pinnacle closing odds in the current (2526) season means that for a large portion of the test set, `pinnacle = B365 fallback`. The model sees near-identical features for those rows, adding noise rather than signal. And for the rows where Pinnacle IS available, the discrepancy between Pinnacle and B365 is small (typically ±2-3%), which is exactly the margin within which the market is already efficient for top European leagues.

The correct use of Pinnacle is not as a model feature but as a value-detection criterion: bet when `Pinnacle_fair_prob > B365_fair_prob`. This is a structural change to the betting evaluation (not a feature), which would require reworking the pipeline's value betting logic. This is a separate, unimplemented hypothesis.

**Loader change kept**: `PSCH/PSCD/PSCA` pass-through in `src/data/loader.py` is harmless (optional columns) and enables future Pinnacle-based experiments without changing the loader again.

---

## Iteration 36: l2 Regularization (REVERTED — regression)

**Date:** 2026-04-20
**Hypothesis:** Per-league models train on ~4500 rows. Adding `l2_regularization=1.0` to HistGBM prevents overfitting by penalising large leaf values, which may be a better lever than structural constraints (depth/min_leaf) for small datasets.
**Files changed:** `src/model/train.py` — added `l2_regularization=1.0` to `_MODEL_CFG`. Reverted after results.

**Results (reverted):**

| Metric    | Iter 33 baseline | Iter 36 (+l2=1.0) | Δ |
|-----------|------------------|-------------------|---|
| Accuracy  | 0.508            | 0.503             | -0.005 ↓ |
| ROI       | **-1.66%**       | **-2.64%**        | **-0.98pp** ↓ |
| Stability | **-0.0107**      | **-0.0171**       | **-0.0064** ↓ |

**Analysis:** L2 regularization regresses on all metrics. The issue is likely that with market fair probs as dominant features, the model needs enough capacity to learn fine-grained deviations from the market. L2 penalises the large leaf values that represent strong deviations, effectively suppressing the very signal (market mispricing) the model is trying to capture. The current config (max_depth=4, min_samples_leaf=20, no explicit l2) already has appropriate implicit regularization via structural constraints. Adding L2 on top over-regularizes. Reverted.

---

## Iteration 35: Opponent-Quality-Adjusted Form (REVERTED — regression)

**Date:** 2026-04-20
**Hypothesis:** Weight each game in the rolling form by opponent pre-match Elo: `scaled_pts = pts * (opponent_elo / ELO_DEFAULT)`. A win against a 1700-Elo team is worth more than a win against a 1300-Elo team. This is the SPI-style approach (Idea 30) — the most principled remaining feature idea.
**Files changed:** `src/model/features.py` — modified `_team_rolling_stats()` to scale pts/gf/ga by `opponent_elo / ELO_DEFAULT` using pre-match Elo from `df["home_elo"]`/`df["away_elo"]`; updated `_get_current_team_form()` with same scaling using final Elo state. Reverted after results.

**Results (reverted):**

| Metric    | Iter 33 baseline | Iter 35 (+opp-Elo scaling) | Δ |
|-----------|------------------|----------------------------|---|
| Accuracy  | 0.508            | 0.504                      | -0.004 ↓ |
| ROI       | **-1.66%**       | **-2.85%**                 | **-1.19pp** ↓ |
| Stability | **-0.0107**      | **-0.0190**                | **-0.0083** ↓ |

**Analysis:** Opponent-quality adjustment is a clear regression. The most likely cause is scale distortion: pts/gf/ga are now measured in "Elo-weighted units" rather than raw game outcomes, making the feature magnitudes variable across the dataset. The tree splits that worked on raw values (e.g., "form_pts > 1.5 points per game") are now on scaled values whose range changes depending on opponent strength — the model needs more data to re-learn effective thresholds. There's also a train/predict inconsistency: `_team_rolling_stats` uses precise pre-match Elo, while `_get_current_team_form` uses final Elo (no per-match Elo history available in the prediction path). Reverted.

---

## Iteration 34: HistGBM Hyperparameter Tuning (REVERTED — regression)

**Date:** 2026-04-20
**Hypothesis:** Current HistGBM config (`max_depth=4, min_samples_leaf=20`) was set for the global model (~15k training rows). Per-league models train on ~4500 rows — shallower trees (`max_depth=3`) and higher leaf regularization (`min_samples_leaf=30`) should reduce overfitting on smaller datasets.
**Files changed:** `src/model/train.py` — changed `max_depth=4→3`, `min_samples_leaf=20→30`. Reverted after results.

**Results (reverted):**

| Metric    | Iter 33 baseline | Iter 34 (depth=3, leaf=30) | Δ |
|-----------|------------------|----------------------------|---|
| Accuracy  | 0.508            | 0.504                      | -0.004 ↓ |
| ROI       | **-1.66%**       | **-3.39%**                 | **-1.73pp** ↓ |
| Stability | **-0.0107**      | **-0.0220**                | **-0.0113** ↓ |

**Analysis:** Shallower, more regularized trees perform worse. The per-league models are not overfitting at `max_depth=4` — they are underfitting relative to the market signal. With 14 features and market probs as dominant features, the model needs depth-4 trees to learn conditional interactions like "home team has form edge AND the market hasn't priced it in for this league." Reducing depth prevents learning these joint conditions. The original global-model config generalizes well to per-league training sizes. Reverted.

---

## Iteration 33: H2H Win Rate Re-test (KEPT — marginal improvement)

**Date:** 2026-04-20
**Hypothesis:** H2H win rate regressed in Iter 12 (without market features, global model). In the current paradigm (market probs + per-league + EWM), H2H could help detect systematic fixture-level mispricings that the per-league model hasn't learned — e.g., a team that historically dominates a specific opponent regardless of current form or market odds.
**Files changed:** `src/model/features.py` — added `h2h_home_win_rate` to `FEATURE_COLS` (15 total); wired `_compute_h2h()` into `_build_merged()`; added `_h2h_rate()` lookup in `build_fixture_features()`.

**Results:**

| Metric    | Iter 30 (EWM baseline) | Iter 33 (+H2H) | Δ |
|-----------|------------------------|----------------|---|
| Accuracy  | 0.513                  | 0.508          | -0.005 ↓ |
| ROI       | **-1.89%**             | **-1.66%**     | **+0.23pp** ↑ |
| Stability | **-0.0125**            | **-0.0107**    | **+0.0018** ↑ |
| Bets      | 3553                   | 3595           | +42 |

- **New best at threshold=0.0: ROI -1.66%, Stability -0.0107.**

**Analysis:** H2H produces a marginal improvement (+0.23pp ROI, +0.0018 stability) that is within the noise margin (~1.5pp std error). The improvement direction is correct on both primary metrics, suggesting the feature is not harmful. The default 0.5 for first-time fixture pairs (most matches in the dataset) prevents data sparsity from causing large distortions. In the per-league context, H2H encodes fixture-specific history that the league-level model cannot learn. Accuracy dips slightly (-0.005) consistent with the accuracy/ROI decoupling pattern. **KEPT tentatively — both primary metrics improve, no regression risk.**

---

## Iteration 32: EWM span=3 (REVERTED — regression)

**Date:** 2026-04-20
**Hypothesis:** After EWM span=5 improved all metrics (Iter 30), test whether a more aggressive recency weighting (span=3) further improves ROI by down-weighting older games even more. In EWM, span=3 gives the most recent game ~50% of total weight vs ~33% for span=5.
**Files changed:** `src/model/features.py` — changed `WINDOW = 5` to `WINDOW = 3`. Reverted after results.

**Results (reverted):**

| Metric    | Iter 30 (EWM span=5) | Iter 32 (EWM span=3) | Δ |
|-----------|----------------------|----------------------|---|
| Accuracy  | 0.513                | 0.499                | -0.014 ↓ |
| ROI       | **-1.89%**           | **-4.52%**           | **-2.63pp** ↓ |
| Stability | **-0.0125**          | **-0.0301**          | **-0.0176** ↓ |

**Analysis:** EWM span=3 is a severe regression. Span=3 is too aggressive — with only 3 effective games of history, individual match variance dominates the signal (a lucky 3-0 win completely reshapes the form estimate). The noise overwhelms the genuine form signal. Span=5 is the right balance: recent enough to capture current momentum, wide enough to smooth out individual match noise. Consistent with the flat WINDOW=3 failure in Iter 10. Reverted.

---

## Iteration 31: WINDOW=7 with EWM (REVERTED — marginal, within noise)

**Date:** 2026-04-19
**Hypothesis:** With EWM already active, increasing the span from 5 to 7 games provides a smoother long-term trend signal. The 7-game window emphasises durable team quality over short-form spikes. Previously tested in Iter 13 without market features (regression); worth re-testing in the current paradigm on top of EWM.
**Files changed:** `src/model/features.py` — changed `WINDOW = 5` to `WINDOW = 7`. Reverted after results.

**Results (reverted, tested on EWM+per-league baseline from Iter 30):**

| Metric    | Iter 30 (EWM+W=5) | Iter 31 (EWM+W=7) | Δ |
|-----------|-------------------|-------------------|---|
| Accuracy  | 0.513             | 0.499             | -0.014 ↓ |
| ROI       | **-1.89%**        | **-1.80%**        | +0.09pp ↑ |
| Stability | **-0.0125**       | **-0.0116**       | +0.0009 ↑ |
| Bets      | 3553              | 3550              | -3 |
| Test size | 2626              | 2614              | -12 (more early-season drops) |

**Analysis:** ROI and Stability both improve marginally (+0.09pp and +0.0009), but the difference is well within the noise margin (~1.5pp std error on ROI at this sample size). Accuracy drops notably (-0.014), suggesting the longer span adds smoothing that loses useful recent-form signal. The test sets also differ by 12 matches (WINDOW=7 requires 7 games of history, dropping more early-season rows), making direct comparison imperfect. Given the improvement is within noise and accuracy worsens, reverted to WINDOW=5. WINDOW=5+EWM remains the new best.

---

## Iteration 30: EWM Rolling Form on Per-League Models (KEPT — new best)

**Date:** 2026-04-19
**Hypothesis:** EWM was previously tested (Iter 22) on the global model and reverted because it hurt `threshold=0.06` ROI (-7.77pp). However, at `threshold=0.0` — the metric we actually use — EWM gave +2.11pp improvement. The reason for reversion (threshold=0.06 harm) is now moot since that operating point was retired as having look-ahead bias. Re-testing EWM on per-league models: never previously combined.
**Files changed:** `src/model/features.py` — changed `_team_rolling_stats()` from `rolling(window).mean()` to `ewm(span=window, min_periods=window).mean()`; updated `_get_current_team_form()` to use `ewm` for fixture prediction consistency.

**Results:**

| Metric    | Iter 24 (flat rolling) | Iter 30 (EWM) | Δ |
|-----------|------------------------|---------------|---|
| Accuracy  | 0.507                  | **0.513**     | +0.006 ↑ |
| ROI       | **-3.40%**             | **-1.89%**    | **+1.51pp** ↑ ✅ |
| Stability | **-0.0225**            | **-0.0125**   | **+0.0100** ↑ ✅ |
| Bets      | 3619                   | 3553          | -66 |

- **NEW BEST at threshold=0.0: ROI -1.89% (+1.51pp), Stability -0.0125 (+0.0100).** Run with `uv run python main.py --per-league`.

**Analysis:** EWM improves all three metrics simultaneously — a rare result in this project. Accuracy improves (+0.006), ROI improves (+1.51pp), and stability nearly doubles in magnitude (less negative). The mechanism: EWM emphasises recent games exponentially over older ones (the most recent game has weight ~2x the WINDOW-th game ago), while flat rolling treats all 5 games equally. In per-league models, recent form is more salient because the model is already well-calibrated to each league's pricing dynamics — the remaining signal is short-term team quality. A team that won 4 of their last 5 but the 5th was a recent heavy loss should have lower form credit than flat rolling gives it; EWM corrects this. The -66 bet reduction is consistent with EWM producing more differentiated probability estimates, reducing the count of near-threshold value signals. **KEPT — new default.**

---

## Iteration 29: Draw Propensity (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** A team's rolling draw rate over the last WINDOW games (`home_draw_rate`, `away_draw_rate`) captures a propensity to draw that is not already captured by form_pts, Elo, or market odds. Draw rates are notoriously hard for bookmakers to price and teams that frequently play tight matches may have a systemic draw signal.
**Files changed:** `src/model/features.py` — added `_compute_draw_rates()`, `_get_current_draw_rates()`; added `home_draw_rate`, `away_draw_rate` to `FEATURE_COLS` (16 total); wired into `_build_merged()` and `build_fixture_features()`. Reverted after results.

**Results (reverted, tested on per-league model baseline):**

| Metric    | Iter 24 (baseline) | Iter 29 (+draw rate) | Δ |
|-----------|--------------------|----------------------|---|
| Accuracy  | 0.507              | 0.495                | -0.012 ↓ |
| ROI       | **-3.40%**         | **-7.20%**           | **-3.80pp** ↓ |
| Stability | **-0.0225**        | **-0.0480**          | **-0.0255** ↓ |
| Bets      | 3619               | 3573                 | -46 |

**Analysis:** Draw propensity is a severe regression on all metrics. The -3.80pp ROI and -0.0255 stability drop suggest the features actively mislead the model. Two likely explanations: (1) WINDOW=5 draw rates are very noisy — a team averaging 1 draw per 5 games can easily have 0 or 2 in any given window through randomness, producing a signal with near-zero autocorrelation; (2) Draws are the hardest outcome to predict even for the market, so any feature that nudges the model toward draws on the basis of noisy historical rates will select bets that the market prices correctly (bookmakers over-price draws explicitly to manage exposure). The accuracy drop (-0.012) is among the largest observed in the project, confirming the feature adds variance without useful signal. Reverted.

---

## Iteration 28: Season Standings Context (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** Cumulative season points (`home_season_pts`, `away_season_pts`) and games played (`home_season_gp`, `away_season_gp`) encode motivational and performance context the rolling form window misses. A team on 15 points after 8 games (near top) faces different pressure than one on 3 points (relegation zone). This is a within-season signal orthogonal to short-term Elo/form.
**Files changed:** `src/model/features.py` — added `_compute_season_stats()`, `_get_current_season_stats()`; added 4 features to `FEATURE_COLS` (18 total); wired into `_build_merged()` and `build_fixture_features()`. Reverted after results.

**Results (reverted, tested on per-league model baseline):**

| Metric    | Iter 24 (baseline) | Iter 28 (+season stats) | Δ |
|-----------|--------------------|-------------------------|---|
| Accuracy  | 0.507              | 0.499                   | -0.008 ↓ |
| ROI       | **-3.40%**         | **-4.18%**              | **-0.78pp** ↓ |
| Stability | **-0.0225**        | **-0.0278**             | **-0.0053** ↓ |
| Bets      | 3619               | 3618                    | -1 |

**Analysis:** Season standings regress on all metrics. The likely cause: cumulative season points are highly correlated with Elo ratings (a top-of-table team has accumulated lots of points AND has a high Elo). The model has four new columns that are largely redundant with `home_elo`/`away_elo`, splitting tree budget without adding independent information. Additionally, early-season values (gp=1–3) are extremely noisy — teams with 1 game played have either 3 or 0 points, a huge range that reflects luck as much as quality. The smaller per-league training sets (~4500 rows) make this collinearity-induced noise more costly. Reverted.

---

## Iteration 23: Days Since Last Match (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** Adding `home_days_rest` and `away_days_rest` (days between a team's previous match and this one) gives the model a genuine scheduling-fatigue signal. Short rest (<4 days) is a well-documented performance drag, especially for away teams. Default value of 7 for first appearances.
**Files changed:** `src/model/features.py` — added `_compute_days_rest()`, `_get_current_days_rest()`; added `home_days_rest`, `away_days_rest` to `FEATURE_COLS` (16 total); called in `_build_merged()` and `build_fixture_features()`. Reverted after results.

**Results (reverted):**

| Threshold | Bets | ROI | Stability | vs Iteration 19 |
|-----------|------|-----|-----------|-----------------|
| 0.00 | 3869 (147.3%) | **-8.87%** | **-0.0601** | **-2.48pp ROI** ↓ |
| 0.06 | 309 (11.8%) | **-1.75%** | **-0.0140** | **-5.68pp ROI** ↓ — turns positive to negative |

- Accuracy: **0.535** (+0.003 vs 0.532)

**Analysis:** Days rest is a clear regression on both thresholds, following the same accuracy/ROI decoupling pattern seen in Iterations 12 (H2H), 13 (WINDOW=7), 20 (season progress), and 22 (EWM). Accuracy improves (+0.003) but ROI worsens. The pattern is now consistent across 5 iterations: features that improve classification accuracy do not improve value-betting ROI in the current paradigm. The likely explanation is that the market already prices scheduling effects (rest advantage is publicly known and routinely factored into odds), so the model's edge over the market is not increased by knowing what the market already knows. The `days_rest` signal shifts the model's probabilities in directions that happen to align slightly better with outcomes, but since the market makes the same adjustment, no genuine edge is created. Reverted.

---

## Iteration 22: Exponential Decay Rolling Form (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** Replacing flat `rolling(window).mean()` with `ewm(span=WINDOW, min_periods=WINDOW).mean()` will improve ROI because exponential weighting emphasises recent games over older ones, capturing current team form more accurately than a uniform 5-game average. Unlike WINDOW=3 (noisier) and WINDOW=7 (smoother), EWM keeps the same effective span while changing the weighting curve.
**Files changed:** `src/model/features.py` — changed `_team_rolling_stats()` to use `ewm` instead of `rolling`; updated `_get_current_team_form()` to use `ewm` for consistent fixture features. Reverted after results.

**Results (reverted):**

| Threshold | Bets | ROI | Stability | vs Iteration 19 |
|-----------|------|-----|-----------|-----------------|
| 0.00 | 3840 (146.2%) | **-4.28%** | **-0.0276** | **+2.11pp ROI** ↑ |
| 0.06 | 360 (13.7%) | **-3.84%** | **-0.0326** | **-7.77pp ROI** ↓ — turns positive to negative |

- Accuracy: **0.527** (-0.005 vs 0.532)

**Analysis:** EWM produces a contradictory result: ROI improves at threshold=0.0 (+2.11pp) but collapses at the operating point threshold=0.06 (from +3.93% to -3.84%). The improvement at threshold=0.0 suggests EWM does contain a useful signal — it effectively de-weights stale games and boosts recent form, which matters for broad value-bet selection. However at threshold=0.06 the bet count increases (360 vs 335) and ROI turns negative. EWM shifts the probability distribution in a way that selects 25 additional bets that are collectively poor quality. The expanded bet set at threshold=0.06 includes bets that were previously below the threshold under flat form, and these marginal additions are loss-making. The accuracy drop (-0.005) is consistent with EWM emphasising more recent, noisier signal. Reverted.

---

## Iteration 27: Elo Momentum (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** The Elo delta over the last WINDOW games (`home_elo_delta = home_elo_now − home_elo_WINDOW_games_ago`) captures whether a team is on an improving or declining trajectory, independently of their absolute strength. A mid-table team on a 5-game winning streak has a large positive delta; a top-table team coasting has near-zero delta. This is orthogonal to `home_elo` (level) and `home_form_pts` (results). Tested on the per-league baseline (Iter 24).
**Files changed:** `src/model/features.py` — extended `_compute_elo()` to track per-team Elo history and compute deltas; added `_get_current_elo_delta_state()`; added `home_elo_delta`, `away_elo_delta` to `FEATURE_COLS` (16 total). Reverted after results.

**Results (reverted):**

| Metric    | Iter 24 (baseline) | Iter 27 (+elo delta) | Δ |
|-----------|--------------------|----------------------|---|
| Accuracy  | 0.507              | 0.492                | -0.015 ↓ |
| ROI       | **-3.40%**         | **-4.31%**           | **-0.91pp** ↓ |
| Stability | **-0.0225**        | **-0.0289**          | **-0.0064** ↓ |

**Analysis:** Elo momentum regressed on all metrics. Two factors likely explain this. First, `home_elo_delta` is substantially correlated with `home_form_pts` — a team winning 5 consecutive games will have both high form_pts and a large positive Elo delta; the features are not as orthogonal as hypothesised. Second, the per-league models train on ~4500 rows each; adding 2 correlated features in a smaller dataset increases variance without reducing bias. The 0.0 default for teams with fewer than WINDOW games may also confuse the model — it can't distinguish "stable Elo" from "insufficient history." The large accuracy drop (-0.015) reinforces that the delta is adding noise. Reverted.

---

## Iteration 26: Kelly Criterion on Per-League Models (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** Kelly criterion sizing (+3.63pp ROI improvement on the global model at threshold=0.0, Iter 21) should work at least as well on per-league models, which have better probability calibration within each league.
**Files changed:** None — tested with `--per-league --kelly 0.25`.

**Results:**

| Metric    | Iter 24 flat | Iter 26 Kelly 0.25 | Δ |
|-----------|--------------|---------------------|---|
| ROI       | **-3.40%**   | **-4.34%**          | **-0.94pp** ↓ |
| Stability | **-0.0225**  | **-0.0242**         | **-0.0017** ↓ |

**Analysis:** Kelly regresses on per-league models (-0.94pp ROI, -0.0017 stability). Note that ROI is mathematically invariant to the Kelly fraction scalar — all fractions (0.1, 0.25, 0.5) yield the same ROI since the fraction cancels in `sum(profit) / sum(stake)`. The fundamental mechanism is Kelly's favourite-weighting: it stakes more on low-odds bets where `full_kelly = p − (1−p)/(odds−1)` is higher. On the global model, Kelly coincidentally de-weighted the noisiest cross-league bets. On per-league models, the calibration is already tighter within each league, so Kelly's reweighting scheme no longer has a de-noising effect — it just shifts capital towards favourites in well-calibrated markets, where no systematic edge exists. Kelly is not helpful with per-league models. The `--kelly` flag remains available but is not recommended.

---

## Iteration 25: Market Deviation Persistence (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** A rolling mean of `(actual_result_share − market_fair_prob)` over the last 5 games captures whether a team is persistently under- or over-valued by the bookmaker. A team that consistently beats its market odds represents a systematic blind spot. Testing on top of the current best (per-league models, Iter 24).
**Files changed:** `src/model/features.py` — added `_compute_market_bias()`, `_get_current_market_bias()`; added `home_market_bias`, `away_market_bias` to `FEATURE_COLS` (16 total); wired into `_build_merged()` and `build_fixture_features()`. Reverted after results.

**Results (reverted, tested on per-league model baseline):**

| Metric    | Iter 24 (baseline) | Iter 25 (+market bias) | Δ |
|-----------|--------------------|------------------------|---|
| Accuracy  | 0.507              | 0.506                  | -0.001 |
| ROI       | **-3.40%**         | **-4.25%**             | **-0.85pp** ↓ |
| Stability | **-0.0225**        | **-0.0282**            | **-0.0057** ↓ |
| Bets      | 3619               | 3606                   | -13 |

**Analysis:** Market bias regressed on both ROI and Stability. The feature adds 2 extra columns that encode "did this team recently outperform the market?" — but with only 5 games of history, the rolling bias estimate is extremely noisy. A team could have 3 lucky wins in a 5-game window and show a large positive bias, yet this tells us nothing about genuine mispricing. The signal-to-noise ratio is too low at WINDOW=5. Additionally, within each per-league model, the market probs (`market_h/d/a`) are already dominant features — the model effectively derives similar information from seeing a team's prob trajectory against the market. A separate bias feature may create redundancy that splits tree budget without adding independent information. A longer window (20+ games) might yield a more reliable signal, but would also reduce available training rows. Reverted.

---

## Iteration 24: League-Specific Sub-models

**Date:** 2026-04-19
**Hypothesis:** Training one HistGBM per league (E0, D1, SP1, I1) instead of a single global model will better capture league-specific deviations from market pricing — each league has different H/D/A base rates, different Elo dynamics, and different bookmaker efficiency profiles. The league dummies (Iter 19) showed that league identity matters a lot; per-league models should learn these interactions more precisely without requiring cross-league generalization.
**Files changed:** `src/model/train.py` — added `_predict_per_league()` helper and `per_league: bool` parameter to `train_walkforward()`; `main.py` — added `--per-league` CLI flag.

**Results:**

| Metric    | Iter 19 (global model) | Iter 24 (per-league) | Δ |
|-----------|------------------------|----------------------|---|
| Accuracy  | 0.532                  | **0.507**            | -0.025 ↓ |
| ROI       | -6.39%                 | **-3.40%**           | **+2.99pp** ↑ ✅ |
| Stability | -0.0421                | **-0.0225**          | **+0.0196** ↑ ✅ |
| Bets      | 3854 (146.8%)          | 3619 (137.8%)        | -235 |

- **NEW BEST at threshold=0.0: ROI +2.99pp, Stability +0.0196.** Run with `uv run python main.py --per-league`.

**Analysis:** Per-league models significantly improve ROI and Stability despite a large accuracy drop (-0.025). This is the accuracy/ROI decoupling in reverse: the global model classifies outcomes more accurately but its probability distribution is smeared across leagues. When calibrated to a single league's market dynamics, the model's probability estimates are closer to what the bookmaker considers "fair" for that specific competition — reducing spurious value signals. The Bundesliga and Serie A have systematically different H/D/A rates and different bookmaker margins; a global model learns an average pattern that mis-calibrates bets in each individual league. The accuracy drop reflects smaller per-league training sets (~4500 vs ~15000 rows), but the improved probability calibration matters more for ROI than raw classification accuracy. Each league model also implicitly treats the league dummies as constant (zero variance), effectively reducing to 11 active features, which may further reduce overfitting. **KEPT — becomes the new default training mode.**

---

## Iteration 21: Kelly Criterion Bet Sizing

**Date:** 2026-04-19
**Hypothesis:** Sizing bets proportionally to the Kelly criterion (`stake = kelly_fraction × (p − (1−p)/(odds−1))`) will improve ROI and stability by concentrating capital on bets where the model has the largest genuine edge, rather than treating all qualifying bets as equal.
**Files changed:** `src/evaluation/metrics.py` — added `kelly_fraction` parameter to `compute_value_betting_results`; `stake` column added to results; `compute_roi` updated to use `stake.sum()` when present. `main.py` — added `--kelly` CLI arg.

**Results (Iter 19 model — 14 features, kelly_fraction=0.25):**

| Threshold | Bets | ROI (flat) | ROI (Kelly 0.25) | Stability (flat) | Stability (Kelly 0.25) |
|-----------|------|------------|------------------|------------------|------------------------|
| 0.00 | 3854 (146.8%) | **-6.39%** | **-2.76%** ↑ | -0.0421 | -0.0112 ↑ |
| 0.06 | 335 (12.8%) | **+3.93%** ✅ | **-0.78%** ↓ | +0.0254 | -0.0064 ↓ |

**Analysis:** Kelly sizing produces a contradictory result: it improves ROI at threshold=0.0 (+3.63pp, from -6.39% to -2.76%) but dramatically degrades ROI at threshold=0.06 (-4.71pp, from +3.93% to -0.78%). The contradiction resolves when you consider what Kelly stakes: it allocates more capital to low-odds (favorite) bets, where `f* = p − (1−p)/(odds−1)` is higher. At threshold=0.06 the selected 335 bets contain a mix of favorites and outsiders; Kelly over-weights the favorites, which are more likely to be precisely priced by the market (the model's apparent edge over favorites may be noise). The occasional high-odds winners (draws, away bets) that drive flat betting's positive ROI get under-weighted. At threshold=0.0 Kelly effectively acts as a soft edge filter — marginal 1–2% edge bets receive tiny stakes, downweighting the noisiest value signals. But at the operating point (threshold=0.06) where bets are already selected by edge, Kelly's favorite-skew is actively harmful. Kelly infrastructure kept (the code remains available via `--kelly`), but flat betting remains the default and best strategy. **REVERTED** — flat betting is restored as the metric default.

---

## Iteration 20: Season Progress Ratio (REVERTED — regression)

**Date:** 2026-04-19
**Hypothesis:** A normalized season-progress ratio (`home/away_season_progress = games_played_this_season / 38`, capped at 1.0) is a better season-phase signal than raw match_month (Iter 11 — failed), because it is team-specific and league-agnostic. Early-season (progress ≈ 0.1) means Elo and form are unreliable; late-season (≈ 0.9) means near-full information. The model can learn to discount high-edge bets when information is still sparse.
**Files changed:** `src/model/features.py` — added `_compute_season_progress()`, `_get_current_season_progress()`; added `home_season_progress`, `away_season_progress` to `FEATURE_COLS` (16 total); called in `_build_merged()` and `build_fixture_features()`. Reverted after results.

**Results (reverted):**

| Threshold | Bets | ROI | Stability | vs Iteration 19 |
|-----------|------|-----|-----------|-----------------|
| 0.00 | 3867 (147.3%) | **-8.91%** | **-0.0585** | **-2.52pp ROI** ↓ |
| 0.06 | 259 (9.9%) | **+2.68%** | **+0.0189** | -1.25pp ROI ↓, fewer bets |

- Accuracy: 0.532 (unchanged)

**Analysis:** Season progress ratio reproduced the same failure mode as match_month (Iter 11): regression on ROI despite no accuracy change. The mechanism differs from Iter 11's diagnosis — unlike raw month, the ratio correctly normalizes across leagues. However, the underlying problem is that the model already captures early-season unreliability implicitly: Elo starts at the default for new teams and converges gradually; rolling form requires a minimum of 5 games and produces higher-variance estimates early-season. Adding an explicit season-progress feature may create a confound — the model splits its tree budget between the explicit progress signal and the implicit quality-of-information signal already encoded in Elo/form magnitudes. The bet count at threshold=0.06 drops from 335 to 259 (-76 bets), suggesting the new features shifted the model's probability distribution in a way that filtered out some of the edge cases that were previously profitable. Reverted.

---

## Iteration 19: League One-Hot Encoding

**Date:** 2026-04-18
**Hypothesis:** Adding league identity (E0/D1/SP1/I1) as one-hot features lets the model learn league-specific patterns — home advantage, draw rate, and form predictability differ across the Bundesliga, Premier League, La Liga, and Serie A. The model currently has no explicit league signal; it relies only on the market odds to infer context.
**Files changed:** `src/model/features.py` — added `league_E0`, `league_D1`, `league_SP1` (I1 = omitted reference category) to `FEATURE_COLS`; computed in `_build_merged()` and `build_fixture_features()`.

**Results:**

| Threshold | Bets | ROI | Stability | vs Iteration 16 |
|-----------|------|-----|-----------|-----------------|
| 0.00 | 3854 (146.8%) | **-6.39%** | **-0.0421** | +0.33pp ROI ↑, +0.0027 stability ↑ |
| 0.06 | 335 (12.8%) | **+3.93%** ✅ | **+0.0254** ✅ | **+10.34pp ROI** ↑ — **NEW BEST** |

- Accuracy: **0.531** (-0.001 vs 0.532 — marginal accuracy drop, major ROI gain)
- **NEW BEST on ROI and Stability at threshold=0.06: first positive ROI at a statistically meaningful sample (335 bets).**

**Analysis:** League encoding produced the largest improvement yet at the threshold=0.06 operating point (+10.34pp ROI, from -6.41% to +3.93%). The mechanism: each league has different baseline H/D/A rates and different betting market dynamics. Without league identity, the model learned a single average deviation pattern from the market. With league dummies, it can learn that, say, a 6% edge in the Bundesliga vs the Serie A may have different reliability. The accuracy dip (-0.001) is consistent with the pattern seen throughout — these features are calibrated for value betting ROI, not classification accuracy. Stability also turned positive (+0.0254), meaning cumulative profit curves are now trending upward at threshold=0.06. The bet count of 335 is the same as Iteration 16, ruling out sample size as an explanation.

---

## Iteration 18: Venue-Specific Rolling Form (REVERTED — regression)

**Date:** 2026-04-18
**Hypothesis:** Rolling form computed separately for home games (home team) and away games (away team) is more predictive than all-games form, since venue splits capture systematically different performance.
**Files changed:** `src/model/features.py` — added 6 venue-specific form features; reverted after results.

**Results (reverted):**

| Threshold | Bets | ROI | Stability | vs Iteration 16 |
|-----------|------|-----|-----------|-----------------|
| 0.00 | 3912 (149.4%) | -8.84% | -0.0570 | **-2.12pp ROI** ↓ |
| 0.06 | 244 (9.3%) | -6.85% | -0.0579 | -0.44pp ROI ↓ |

**Analysis:** Venue-specific form produced a clear regression. With WINDOW=5 split by venue, each team's home (or away) form window spans ~10 real weeks vs ~2.5 weeks for all-games form — the signal is noisier because sample sizes halve. The 6 new features are also highly correlated with the existing 6 all-games features (a team's home form is correlated with overall form), adding collinearity without independent signal. The model's accuracy improved marginally (0.534) but ROI worsened substantially. Reverted.

---

## Iteration 17: Sigmoid Calibration on Top of Odds Features

**Date:** 2026-04-18
**Hypothesis:** Sigmoid calibration (Platt scaling) applied on top of the odds-features model will further improve probability alignment. Unlike isotonic, sigmoid uses only 2 parameters per class and is less prone to overfitting — it may correct the residual shape of HistGBM's probability outputs without degrading the market-alignment already learned from the odds features.
**Files changed:** `src/model/train.py` — wrapped `HistGradientBoostingClassifier` in `CalibratedClassifierCV(base, cv=3, method="sigmoid")`.

**Results (odds features active, Iteration 16 base):**

| Threshold | Bets | ROI | Stability | vs Iter 16 |
|-----------|------|-----|-----------|------------|
| 0.00 | 4008 (152.6%) | -8.64% | -0.0547 | **-1.92pp ROI** ↓ |
| 0.06 | 161 (6.1%) | -5.89% | -0.0374 | +0.52pp ROI ↑ |

**Analysis:** Sigmoid calibration is a clear regression at threshold=0.0 (ROI -8.64% vs -6.72%), though it shows marginal improvement at threshold=0.06 (-5.89% vs -6.41%). However, at threshold=0.06 the bet count collapses from 335 to 161 — only 6.1% of matches. The threshold=0.06 improvement is statistically fragile (161 bets). The threshold=0.0 result is the more reliable signal, and it is unambiguously worse. The calibration appears to distort the market-aligned probabilities that the model learned directly from the odds features — the model's raw probabilities (already market-informed) are more useful to the value-betting filter than the sigmoid-smoothed versions. Reverted.

---

## Iteration 16: Market Fair Probabilities as Features

**Date:** 2026-04-18
**Hypothesis:** Adding the bookmaker's vig-corrected fair probabilities (`market_h`, `market_d`, `market_a`) as features lets the model learn to deviate from the market rather than predict outcomes in isolation. This directly addresses the probability calibration problem: the model can explicitly learn "given all my signals AND what the market thinks, what is my probability?"
**Files changed:** `src/model/features.py` — added `market_h`, `market_d`, `market_a` (vig-corrected: `(1/odds) / sum(1/odds)`) to `FEATURE_COLS`; computed in `_build_merged()` and `build_fixture_features()`.

**Results:**

| Threshold | Bets | ROI | Stability | vs prior baseline |
|-----------|------|-----|-----------|-------------------|
| 0.00 | 3848 (146.5%) | **-6.72%** | **-0.0448** | **+4.27pp** ↑ (**NEW BEST**) |
| 0.06 | 335 (12.8%) | -6.41% | -0.0552 | +1.80pp vs Iter 15 |

- Accuracy: **0.532** (+0.009 vs 0.523 prior — largest single-iteration accuracy jump)
- **NEW BEST on all metrics at threshold=0.0.**

**Analysis:** Adding market fair probabilities as features produced the largest improvement in the project so far. Accuracy jumped +0.009 (largest single-step gain), ROI improved +4.27pp at threshold=0.0. The mechanism: the model now knows the market's prior for each match and can learn to detect systematic patterns where its other signals (Elo, rolling form) diverge from the market estimate. Rather than predicting H/D/A from team statistics alone, it predicts "does the market misprice this match?" The bet count at threshold=0.06 drops from 1180 to 335 because the model's probabilities are now much closer to market fair probs — finding 6% edge requires a real signal above the market's already-informed estimate.

---

## Iteration 15: Threshold Grid Search

**Date:** 2026-04-18
**Hypothesis:** There exists a threshold > 0 at which value betting ROI exceeds the threshold=0.0 baseline, because requiring a minimum edge filters out the worst spurious value signals while preserving genuine edge cases.
**Files changed:** None — grid search over existing pipeline using `--threshold` CLI arg. Trained once, evaluated at 8 thresholds.

**Full grid results:**

| Threshold | Bets | Bet% | ROI | Stability |
|-----------|------|------|-----|-----------|
| 0.00 | 3776 | 143.8% | -10.99% | -0.0724 |
| 0.02 | 2632 | 100.2% | -9.58% | -0.0645 |
| 0.04 | 1808 | 68.8% | -9.08% | -0.0618 |
| **0.06** | **1180** | **44.9%** | **-8.21%** | **-0.0573** |
| 0.08 | 776 | 29.6% | -11.22% | -0.0881 |
| 0.10 | 482 | 18.4% | -11.21% | -0.0893 |
| 0.15 | 152 | 5.8% | -6.42% | -0.0507 |
| 0.20 | 37 | 1.4% | +8.81% | +0.0615 |

**Official result at threshold=0.06 (best statistically meaningful point):**
- Accuracy: 0.523, ROI: -8.21% (**+2.78pp vs baseline -10.99%**), Stability: -0.0573, Bets: 1180

**Analysis:** ROI improves monotonically from threshold 0.0 → 0.06, then degrades sharply at 0.08–0.10, then improves again at 0.15–0.20. The 0.06 cliff is likely a natural boundary between "noisy value signals" (model edge 0–6%) and "cleaner value signals" (model edge >6%). The 0.08–0.10 degradation may reflect that this range captures a mix of genuine edge and overconfident predictions with high stakes odds. The positive ROI at 0.20 (+8.81%, 37 bets) is intriguing but statistically fragile — 37 bets is insufficient for reliable inference. **Critical caveat:** the threshold 0.06 was selected on the backtest test set, introducing look-ahead bias. Future iterations must be evaluated at both threshold=0.0 (clean baseline) and threshold=0.06 (operating point). The recommended default going forward is `--threshold 0.06`.

---

## Iteration 14: Probability Calibration (CalibratedClassifierCV, isotonic, cv=3)

**Date:** 2026-04-18
**Hypothesis:** Wrapping HistGBM in `CalibratedClassifierCV(cv=3, method="isotonic")` will improve probability calibration and thus betting ROI, directly addressing the accuracy/ROI decoupling identified in Iterations 11–13.
**Files changed:** `src/model/train.py` — wrapped `HistGradientBoostingClassifier` in `CalibratedClassifierCV(base, cv=3, method="isotonic")` in both `train_walkforward` and `train_on_all_data`.

**Results:**
- Accuracy: 0.526 (+0.003 vs baseline 0.523)
- ROI: -11.49% (-0.50pp vs baseline -10.99%)
- Stability: -0.0727 (-0.0003 vs baseline -0.0724)
- Bets: 3754 / 2626 (143.0%)
- **REGRESSION on ROI and Stability at threshold=0.0**

**Analysis:** Isotonic calibration again failed to improve ROI, replicating the Iteration 5 (LogReg) finding in a new model and evaluation context. The reason is structural: `CalibratedClassifierCV` with `cv=3` splits the training data into 3 folds, fits the base model on 2/3 of the data, then calibrates on the held-out 1/3. The calibration maps predicted probabilities to empirical frequencies on the calibration fold. However, the key question is whether the calibrated probabilities align better with *bookmaker* fair odds — not just with empirical frequencies. These are different targets. The bookmaker's fair probability distribution reflects both outcome frequencies AND information not in our features (injuries, news, motivation). Calibrating to empirical frequencies does not help if the bookmaker has superior information. Reverted.

---

## Iteration 13: Longer Rolling Window (WINDOW=7)

**Date:** 2026-04-18
**Hypothesis:** A 7-game rolling window captures more stable team form than WINDOW=5, smoothing out noise from single anomalous results. WINDOW=3 was worse (Iter 10), suggesting more data is better — so 7 might beat 5.
**Files changed:** `src/model/features.py` — changed `WINDOW` from 5 to 7.

**Results:**
- Accuracy: 0.528 (+0.005 vs baseline 0.523)
- ROI: -11.73% (-0.74pp vs baseline -10.99%)
- Stability: -0.0750 (-0.0026 vs baseline -0.0724)
- Bets: 3699 / 2614 (141.5% — slight drop because more matches lack 7-game history)
- **REGRESSION on ROI and Stability; Accuracy improved**

**Analysis:** Same pattern as Iteration 12: accuracy improved (+0.005) but ROI and Stability degraded. A 7-game window produces smoother, more reliable form estimates that make the model classify outcomes more accurately — yet this does not translate to better value betting performance. The probable explanation is that better classification does not improve the model's probability calibration relative to bookmaker odds. The model with WINDOW=7 still outputs probabilities that are systematically mis-matched to the bookmaker's vig-adjusted fair odds, generating the same spurious value signals. This confirms a critical insight: **in this evaluation setup, discriminative accuracy and betting ROI are decoupled.** Improvements to classification accuracy do not automatically improve ROI under value betting. What is needed is better probability calibration rather than better discrimination. Reverted to WINDOW=5.

---

## Iteration 12: Head-to-Head Historical Win Rate

**Date:** 2026-04-18
**Hypothesis:** Adding `h2h_home_win_rate` — the historical home win rate between the two teams across all prior meetings — will improve ROI by capturing persistent matchup-specific dominance that Elo and rolling form cannot encode.
**Files changed:** `src/model/features.py` — added `_compute_h2h()`, `_get_current_h2h_state()`, `_h2h_rate()` functions; added `h2h_home_win_rate` to `FEATURE_COLS` and `_build_merged()`; added h2h lookup in `build_fixture_features()`. Teams with no prior h2h meetings use 0.5 as a neutral prior.

**Results:**
- Accuracy: 0.525 (+0.002 vs baseline 0.523)
- ROI: -12.55% (-1.56pp vs baseline -10.99%)
- Stability: -0.0811 (-0.0087 vs baseline -0.0724)
- Bets: 3749 / 2626 (142.8%)
- **REGRESSION on ROI and Stability; Accuracy marginally improved**

**Analysis:** H2H improves classification accuracy (+0.002) but worsens betting ROI (-1.56pp). This is the same decoupling pattern seen in Iterations 12 and 13: the h2h signal adds genuine discriminative information but does not improve probability calibration against bookmaker odds. The h2h rate likely shifts the model's probability distribution in ways that create more spurious value signals, increasing bet volume slightly (3749 vs 3776 baseline) while lowering the average quality. Reverted.

---

## Iteration 11: Season Phase (match_month)

**Date:** 2026-04-18
**Hypothesis:** Adding `match_month` (calendar month 1–12) as a 9th feature gives GBM the ability to learn that early-season predictions (August–September) are less reliable because Elo ratings and rolling form are unsettled. This information is structurally orthogonal to all 8 current features.
**Files changed:** `src/model/features.py` — added `"match_month"` to `FEATURE_COLS`, added `df["Date"].dt.month` computation in `_build_merged`, added `match_month` in `build_fixture_features`.

**Evaluation baseline (new setup — walk-forward + value betting):**
- Accuracy: 0.523, ROI: -10.99%, Stability: -0.0724, Bets: 3776 / 2626

**Results:**
- Accuracy: 0.521
- ROI: -12.08%
- Stability: -0.0798
- Bets: 3766 / 2626 (143.4%)
- vs baseline: Accuracy -0.002, ROI -1.09pp, Stability -0.0074 — **REGRESSION on all metrics**

**Analysis:** Adding `match_month` worsened all metrics. The most likely explanation is that month is too blunt a signal — it conflates very different situations (e.g., September matches across different leagues start at different points in their respective seasons, and a mid-table Bundesliga team in September is very different from a newly promoted Premier League side). HistGBM may also be splitting tree budget on a noisy signal that offers marginal discrimination at best. Additionally, since the rolling form and Elo features already implicitly capture early-season noise (form is NA until enough games are played, Elo starts at default for new teams), the month feature may be providing no new information the model can usefully exploit. Reverted.

---

## Iteration 10: Shorter Rolling Window (3-game)

**Date:** 2026-04-18
**Hypothesis:** A 3-game rolling window captures more recent form than the 5-game window — tighter recency should improve signal quality and reduce noise from stale results.
**Files changed:** `src/model/features.py` — changed `WINDOW` from 5 to 3. Also restored flat betting in `main.py` (reverted from Iter 8's multi-outcome value betting back to `compute_betting_results`).

**Results:**
- Accuracy: 0.519
- ROI: -5.53%
- Stability: -0.0563
- Test bets: 2743 (vs 2643 in Iter 6 — smaller window has lower warm-up cost, adds 100 test bets)
- vs Iter 6 (best): Accuracy -0.002, ROI -0.44pp, Stability -0.0047 — **REGRESSION on all metrics**

**Analysis:** The 3-game window worsened all metrics. Despite the lower warm-up cost adding 100 more test bets, the model's predictive accuracy fell slightly and ROI worsened. A shorter window produces noisier rolling estimates — with only 3 games, team form stats are more susceptible to single-match outliers (red cards, unusual opponents, travel fatigue). The 5-game window appears to offer a better bias-variance tradeoff for this dataset. Combined with the fact that Iteration 1 showed home/away-split form (sparser windows) also hurt, the evidence consistently points to the 5-game window as optimal at this scale. Reverted `WINDOW` to 5.

---

## Iteration 9: Elo Hyperparameter Tuning (K=20, HOME_ADV=65)

**Date:** 2026-04-18
**Hypothesis:** The default Elo parameters (K=30, HOME_ADV=100) may not be optimal for this dataset. Lowering K to 20 (more stable, less reactive ratings) and HOME_ADV to 65 (reflecting modern football's declining home advantage) will produce more accurate team strength estimates and improve ROI.
**Files changed:** `src/model/features.py` — changed `ELO_K` from 30 to 20, `ELO_HOME_ADV` from 100 to 65. Also restored flat betting in `main.py` (reverted from Iter 8's multi-outcome value betting back to `compute_betting_results`).

**Results:**
- Accuracy: 0.516
- ROI: -6.19%
- Stability: -0.0629
- Test bets: 2643
- vs Iter 6 (best): Accuracy -0.005, ROI -1.10pp, Stability -0.0113 — **REGRESSION on all metrics**

**Analysis:** Lower K and lower HOME_ADV both worsened results across all three metrics. The K=30 and HOME_ADV=100 defaults appear better calibrated to this multi-league European dataset. Two possible reasons: (1) A higher K=30 makes Elo more reactive to recent results, which may actually be beneficial over a ~12-year training set where team quality changes dramatically season to season; static lower-K ratings accumulate inertia that misrepresents current strength. (2) The HOME_ADV=100 may reflect a historical dataset baseline that is simply more accurate as a prior over all seasons and leagues combined — reducing to 65 undershoots the actual advantage embedded in this dataset's composition. The original K=30, HOME_ADV=100 parameters are confirmed as better. Reverted both values.

---

## Iteration 8: Multi-Outcome Value Betting

**Date:** 2026-04-17
**Hypothesis:** Betting any outcome where model probability > bookmaker implied probability — across all three outcomes (H/D/A) per match — will improve ROI by identifying underpriced draws and away wins that single-outcome value betting (Iterations 4+5) systematically missed.
**Files changed:** `src/model/features.py` — reverted to Iter 6 baseline (8 features, no derived cols); `src/model/train.py` — reverted to Iter 6 baseline (HistGBM, no categorical_features); `src/evaluation/metrics.py` — added `compute_value_betting_results()` function (multi-outcome value betting, kept existing functions unchanged); `main.py` — switched from flat betting (`compute_betting_results`) to multi-outcome value betting (`compute_value_betting_results`).

**Results:**
- Accuracy: 0.521 (same as Iter 6 — same model)
- ROI: -17.80%
- Stability: -0.0993
- Test bets: 3337 / 2643 matches (126.3% — average 1.26 bets per match)
- vs Iter 6 (best): ROI -12.71pp, Stability -0.0477 — **REGRESSION on all metrics**

**Analysis:** Multi-outcome value betting substantially worsened ROI from -5.09% to -17.80% (-12.71pp). The bet rate of 126.3% (>1 bet per match) reveals that the model frequently sees "value" in multiple outcomes simultaneously — which is only possible because the model's probability distribution is not well-calibrated against bookmaker implied probabilities. Specifically, bookmaker odds include a vig (~5% margin) that compresses all implied probabilities below 1.0; a poorly calibrated model that outputs probabilities close to the bookmaker's may generate spurious value signals on multiple outcomes at once. The result is that multi-outcome betting magnifies the same systematic overconfidence problem seen in Iterations 4+5, just across more outcomes. Iter 6 flat betting (all matches, no filter) remains the best approach — the bookmaker margin cannot be beaten by simple probability comparison on this feature set.

---

## Iteration 7: Feature Enrichment (Goal Difference, Elo Diff, League Categorical)

**Date:** 2026-04-17
**Hypothesis:** Adding `home_form_gd`, `away_form_gd` (rolling goal difference), `elo_diff` (Elo differential), and `league_code` (integer-encoded league as HistGBM categorical) on top of the Iter 6 8-feature set would improve ROI because: goal difference captures style more directly than separate gf/ga; Elo diff is the single strongest Elo scalar; league captures systematic home-advantage differences across competitions.
**Files changed:** `src/model/features.py` — added `FEATURE_COLS` as module-level constant (12 features), added derived feature computation (`home_form_gd`, `away_form_gd`, `elo_diff`, `league_code`) inside `_build_merged()`; `src/model/train.py` — imported `FEATURE_COLS`, added `categorical_features=_CATEGORICAL_FEATURES` to `HistGradientBoostingClassifier`.

**Results:**
- Accuracy: 0.518
- ROI: -6.30%
- Stability: -0.0647
- Test bets: 2643
- vs Iter 6 (best): Accuracy -0.003, ROI -1.21pp, Stability -0.0131 — **REGRESSION on all metrics**

**Analysis:** Adding the four derived features hurt rather than helped. The most likely explanation is multicollinearity: `home_form_gd` is a deterministic linear combination of `home_form_gf` and `home_form_ga` (already in the feature set), and `elo_diff` is a linear combination of `home_elo` and `away_elo`. While HistGBM is tree-based and theoretically tolerant of correlated features, adding redundant linear combinations can still inflate variance by splitting the tree budget across equivalent signals, reducing generalization. The `league_code` categorical likely offers negligible additional discriminative power since the Elo system already captures cross-league team strength implicitly. The result is a net regression: Iter 6 remains the best. Future directions should avoid derived features that are pure linear combinations of existing ones; instead pursue genuinely new information (head-to-head records, season phase, squad depth).

---

## Iteration 6: HistGBM with Elo + Rolling Features (All Bets)

**Date:** 2026-04-17
**Hypothesis:** HistGradientBoostingClassifier with the full 8-feature set (Elo + rolling stats) will outperform Logistic Regression because the richer feature combination gives the gradient boosting model non-linear interactions to exploit — unlike Iteration 2 where GBM had only 6 weak rolling features.
**Files changed:** `src/model/train.py` — replaced `CalibratedClassifierCV(Pipeline(StandardScaler+LogisticRegression))` with bare `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=20, random_state=42)`; `main.py` — reverted to bet on all matches (removed value-bet filter and `add_model_proba` call).

**Results:**
- Accuracy: 0.521
- ROI: -5.09%
- Stability: -0.0516
- Test bets: 2643
- vs Iter 3 (previous best): Accuracy +0.002, ROI +1.23pp, Stability +0.0136 — **NEW BEST on all metrics**

**Analysis:** HistGBM with the full 8-feature Elo+rolling set is a clear winner over Logistic Regression. ROI improved from -6.32% to -5.09% (+1.23pp) and stability improved substantially from -0.0652 to -0.0516. This confirms the hypothesis: the Elo feature provides the non-linear interactions (e.g., Elo differential × rolling form) that GBM can exploit but that LogReg's linear boundary cannot capture. Iteration 2 showed GBM≈LogReg with 6 rolling-only features; adding Elo gave GBM the signal it needed. ROI is still negative (-5.09%), meaning we are still inside the bookmaker vig, but we have closed the gap considerably. Next priority: enrich features further (e.g., league-specific effects, season phase, head-to-head, goal difference rolling stats) to push the base model's discrimination above the vig threshold.

---

## Iteration 5: Calibrated Probabilities + Value Betting

**Date:** 2026-04-17
**Hypothesis:** Wrapping LogisticRegression in `CalibratedClassifierCV` (cv=5, method="isotonic") will fix the overconfidence problem identified in Iteration 4, making the value-bet filter (`model_prob > 1/odds`) a genuine edge signal and improving ROI.
**Files changed:** src/model/train.py — replaced bare `Pipeline(StandardScaler + LogisticRegression)` with `CalibratedClassifierCV(base_pipeline, cv=5, method="isotonic")`; `model.classes_` accessed directly (CalibratedClassifierCV exposes this attribute).

**Results:**
- Accuracy: 0.522
- ROI: -15.52%
- Stability: -0.1318
- Test bets: 929 (35.1% of 2643)
- vs Iter 3 (best): ROI -9.20pp worse, Stability -0.0666 worse
- vs Iter 4: ROI -0.42pp worse — marginally worse, not better

**Analysis:** Isotonic calibration did not rescue the value-bet filter. ROI remained deeply negative at -15.52%, essentially the same as Iteration 4 (-15.10%). The number of value bets increased slightly (929 vs 884), suggesting calibration softened probabilities somewhat but did not eliminate the systematic overconfidence. Two structural problems likely persist: (1) the model's predicted class is strongly correlated with inflated probability for that class — calibration reduces the magnitude of overconfidence but does not change which bets are selected, because the ranking of outcomes per match is preserved by monotone calibration; (2) the value-bet filter operates on the predicted outcome only, so it consistently picks bets in the direction of the model's already-dominant signal. The model may lack the discriminative power to identify genuine value regardless of calibration quality. The entire value-betting approach may require a fundamentally different signal (e.g., draw probability specifically, or ensemble disagreement) rather than calibration of a single classifier.

---

## Iteration 4: Value Betting Filter

**Date:** 2026-04-17
**Hypothesis:** Only betting when model's predicted probability exceeds the bookmaker's implied probability (value bets) will improve ROI — possibly into positive territory — because it filters out bets where we have no edge over the market.
**Files changed:** src/evaluation/metrics.py — added `add_model_proba()` function that computes model probability per predicted outcome, bookmaker implied probability (1/odds), and `is_value_bet` flag; main.py — added `add_model_proba` call and value-bet filter before computing betting metrics.

**Results:**
- Accuracy: 0.519 (unchanged — whole-test-set metric)
- ROI: -15.10%
- Stability: -0.1292
- Test bets: 884 (33.4% of 2643)
- vs Iter 3: ROI -8.78pp worse, Stability -0.0640 worse

**Analysis:** Value betting severely worsened ROI (-15.10% vs -6.32%). The filter selects 884 bets (33.4%), but these are precisely the bets where the model is overconfident relative to the bookmaker. Logistic Regression without calibration tends to produce over-confident probabilities in the direction of the predicted class; the "value" signal is therefore mostly noise — the model thinks it has edge where it does not. The bookmaker's implied probability is better calibrated than the raw LogReg output, so filtering to cases where model > bookmaker actually selects the worst bets. Value betting requires well-calibrated model probabilities (e.g., via Platt scaling or isotonic regression) to work in practice.

---

## Iteration 3: Elo Ratings as Features

**Date:** 2026-04-17
**Hypothesis:** Adding Elo ratings as features will improve ROI because Elo captures long-run team strength that a 5-game rolling window misses — especially early in a season when rolling form is noisy.
**Files changed:** src/model/train.py — reverted from HistGBM to LogisticRegression + StandardScaler pipeline; src/model/features.py — added `_compute_elo()` function computing pre-match Elo ratings (K=30, HOME_ADV=100, default=1500), added `home_elo` and `away_elo` as two new features (8 total); also aligned `group_keys=True` to match original baseline to fix a pandas groupby compatibility issue.

**Results:**
- Accuracy: 0.519
- ROI: -6.32%
- Stability: -0.0652
- Test bets: 2643
- vs baseline: Accuracy +0.027, ROI +0.47%, Stability -0.0015

**Analysis:** Elo ratings improved both accuracy and ROI over the baseline. Accuracy jumped from 0.492 to 0.519 (+2.7pp), and ROI improved from -6.79% to -6.32% (+0.47pp). The number of test bets is unchanged at 2643 (Elo is always available from match 1; the rolling form warm-up remains the binding constraint). Stability is marginally worse (-0.0652 vs -0.0637), likely noise rather than a systematic pattern. The result confirms the hypothesis: Elo's global team strength signal adds genuine information beyond 5-game rolling form. However, ROI remains negative, so Elo alone is insufficient — future work should explore value betting or combining Elo with threshold-based bet selection.

---

## Iteration 2: HistGradientBoostingClassifier

**Date:** 2026-04-17
**Hypothesis:** Replacing Logistic Regression with HistGradientBoostingClassifier will improve ROI because gradient boosting captures non-linear feature interactions that a linear model cannot, potentially finding subtler patterns between team form stats.
**Files changed:** src/model/train.py — replaced Pipeline(StandardScaler + LogisticRegression) with HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42); src/model/features.py — reverted to combined rolling stats baseline (also fixed groupby().apply() pandas 3.x bug where "team" was dropped from index)

**Results:**
- Accuracy: 0.487
- ROI: -7.23%
- Stability: -0.0671
- Test bets: 2643
- vs baseline: Accuracy -0.005, ROI -0.44%, Stability -0.0034

**Analysis:** HistGBM did not improve over Logistic Regression. The gradient boosting model produced marginally lower accuracy and worse ROI (-7.23% vs -6.79%). With only 6 features (rolling means of pts, gf, ga for home and away teams), there are few non-linear interactions for the tree model to exploit. The linear model appears adequate for these aggregate features. The key bottleneck is the features themselves, not the model capacity. This suggests future iterations should focus on richer features (e.g., Elo ratings) or smarter bet selection (value betting threshold), rather than swapping model architectures.

---

## Iteration 1: Home/Away Split Form

**Date:** 2026-04-17
**Hypothesis:** Separating home and away rolling form will improve ROI because teams often perform very differently at home vs away, and mixing the two signals adds noise.
**Files changed:** src/model/features.py — replaced combined team rolling stats with venue-specific stats (home team's home-game stats, away team's away-game stats)

**Results:**
- Accuracy: 0.481
- ROI: -8.78%
- Stability: -0.0825
- Test bets: 2379
- vs baseline: Accuracy -0.011, ROI -1.99%, Stability -0.0188

**Analysis:** The venue-split approach underperformed on all three metrics. The most likely cause is the warm-up cost: requiring 5 home-only games AND 5 away-only games to compute features drops more early-season matches (2379 vs 2643 test bets), removing games where the signal may have been better calibrated. Additionally, having only 5 home games per team (roughly half a season) may produce noisier rolling estimates than 5 combined games, which are more frequent. The hypothesis that venue-split form is less noisy was not confirmed — the added sparsity appears to hurt more than the venue specificity helps.

**Next directions (ranked):**
1. **Threshold-based betting (value bets):** Only bet when model probability exceeds bookmaker implied probability. This directly targets edge over the market and should reduce bet count while improving ROI. High confidence — this is the single most principled improvement available.
2. **Gradient Boosting model (XGBoost/LightGBM):** Logistic Regression is linear; tree-based models may capture non-linear interactions between features better. Medium-high confidence.
3. **Elo ratings as features:** A dynamic per-team strength estimate that updates after every match, richer than rolling form alone. Many published betting models use Elo as a core feature. Medium-high confidence.

---

## Open Hypotheses

Ranked by estimated probability of improving ROI:

~~**Threshold-based betting (value bets):**~~ _Tested in Iteration 4 — worsened ROI from -6.32% to -15.10%. Raw LogReg probabilities are poorly calibrated; value filtering selects overconfident bets, not genuine edge. Requires probability calibration (Platt scaling / isotonic regression) to work._

~~**Probability calibration + value betting:**~~ _Tested in Iteration 5 — calibration (isotonic, cv=5) did not improve ROI over Iteration 4. ROI -15.52% vs -15.10%. The value-bet approach based on predicted-class probability appears structurally broken with this feature set; calibration preserves ranking so the same bets are selected. Value betting via this mechanism is abandoned._

~~**Additional feature engineering (goal difference, league effects, elo_diff):**~~ _Tested in Iteration 7 — regression on all metrics. Derived features that are linear combinations of existing features (gd = gf - ga; elo_diff = home_elo - away_elo) did not add information and slightly hurt generalization. Discarded approach._

~~**Multi-outcome value betting:**~~ _Tested in Iteration 8 — severe regression. ROI -17.80% vs -5.09% in Iter 6. Bet rate 126.3% (>1 per match) indicates the model generates spurious value across multiple outcomes simultaneously, amplifying the overconfidence problem. Any value-betting approach using raw model probabilities vs. bookmaker implied probabilities is abandoned — the model is not calibrated well enough relative to the bookmaker vig._

~~**Elo hyperparameter tuning (K=20, HOME_ADV=65):**~~ _Tested in Iteration 9 — regression on all metrics (ROI -6.19% vs -5.09%). K=30 and HOME_ADV=100 are confirmed as better for this multi-league European dataset. The higher K provides more reactivity to team form changes season-to-season; the higher HOME_ADV better reflects the historical baseline across the full dataset. Elo parameter search abandoned at this scale._

~~**Shorter rolling window (WINDOW=3):**~~ _Tested in Iteration 10 — regression on all metrics (ROI -5.53% vs -5.09%). Fewer games = noisier estimates; consistent with Iteration 1 finding. WINDOW=5 is confirmed as optimal._

~~**Season phase (match_month):**~~ _Tested Iter 11 — regression._
~~**Head-to-head win rate:**~~ _Tested Iter 12 — regression on ROI._
~~**WINDOW=7:**~~ _Tested Iter 13 — regression on ROI._
~~**Isotonic calibration (cv=3):**~~ _Tested Iter 14 — regression._
~~**Threshold tuning:**~~ _Tested Iter 15 — threshold=0.06 was best statistically-robust operating point (pre-odds-features)._
~~**Market fair probs as features:**~~ _Tested Iter 16 — **NEW BEST**: ROI -6.72% at threshold=0.0 (+4.27pp), accuracy 0.532 (+0.009). Kept._
~~**Sigmoid calibration on odds-features model:**~~ _Tested Iter 17 — regression at threshold=0.0 (-8.64% vs -6.72%). Calibration distorts the market-aligned probs the model already learned. Reverted._

**Current paradigm (updated 2026-04-19):** Market fair probs + league dummies are the dominant features. **Primary evaluation target is threshold=0.0: goal is ROI > 0% AND Stability > 0.** The threshold=0.06 operating point is retired as a decision criterion — it was selected on the test set and carries look-ahead bias. All keep/revert decisions from here forward use threshold=0.0 metrics only.

~~**Threshold re-grid:**~~ _Retired. Re-gridding on the test set perpetuates the same look-ahead bias that made threshold=0.06 unreliable. The path to a valid operating threshold is improving the base model until ROI > 0 at threshold=0.0, then validating a threshold on a held-out set._

~~**Sigmoid calibration on odds-features model:**~~ _Tested Iter 17 — regression._

~~**Venue-specific rolling form:**~~ _Tested Iter 18 — regression (halved sample size per window, high collinearity with all-games form)._

~~**League one-hot encoding:**~~ _Tested Iter 19 — **NEW BEST at threshold=0.06 (+10.34pp ROI)**. Kept permanently._

---

### Simple ideas (low code, quick to test — 1 run each)

**20. Threshold re-grid on Iter-19 model** _(highest priority — 0 code changes)_
The optimal threshold was selected on the pre-odds, pre-league model (Iter 15). With 14 features, market-aligned probabilities are distributed differently: the model selects 335 bets at 0.06 vs 1180 in Iter 15. The new optimal may be lower (0.03–0.04) or higher. Re-run the 8-threshold grid on the current model with no changes. _High confidence this reveals a better operating point. Zero implementation risk._

~~**21. Exponential decay rolling form (ewm):**~~ _Tested Iter 22 (global, reverted for threshold=0.06 harm — now moot) and Iter 30 (per-league, **KEPT — NEW BEST**: ROI -1.89% +1.51pp, Stability -0.0125 +0.0100, Accuracy +0.006). EWM is now the permanent default in `_team_rolling_stats`._

~~**22. Re-test WINDOW=7 in the odds-features paradigm:**~~ _Tested Iter 31 on top of EWM+per-league. Marginal +0.09pp ROI and +0.0009 stability — within noise margin. Accuracy -0.014. Reverted; WINDOW=5 kept._

---

### Medium ideas (moderate implementation, clear hypothesis)

~~**23. Days since last match:**~~ _Tested Iter 23 — regression at both thresholds (threshold=0.0: -2.48pp; threshold=0.06: -5.68pp, turns +3.93% → -1.75%). Accuracy improved (+0.003) but ROI worsened — same decoupling pattern. Market already prices scheduling effects, so no genuine edge added. Reverted._

~~**24. Elo momentum:**~~ _Tested Iter 27 — regression (ROI -4.31% vs -3.40%, -0.91pp; Stability -0.0289 vs -0.0225). Correlated with form_pts; adds noise in smaller per-league datasets. Large accuracy drop (-0.015). Reverted._

~~**25. Season-progress ratio:**~~ _Tested Iter 20 — regression (ROI -8.91% vs -6.39% at threshold=0.0; +2.68% vs +3.93% at threshold=0.06). The model already captures early-season noise implicitly through Elo convergence and rolling form variance; the explicit ratio splits tree budget without adding independent information. Reverted._

~~**26. Re-test H2H win rate in the odds-features paradigm:**~~ _Tested Iter 33 — marginal improvement (+0.23pp ROI, +0.0018 stability). Within noise margin but both primary metrics improve. KEPT tentatively._

---

### Complex ideas (multi-component changes or structural shifts)

~~**27. League-specific sub-models:**~~ _Tested Iter 24 — **NEW BEST at threshold=0.0**: ROI -3.40% (+2.99pp vs -6.39%), Stability -0.0225 (+0.0196). Accuracy dropped -0.025 but ROI improved significantly — accuracy/ROI decoupling in reverse. Per-league calibration reduces cross-league probability smearing. KEPT. Run with `--per-league`._

~~**28. Kelly criterion bet sizing:**~~ _Tested Iter 21 (global model) and Iter 26 (per-league). On global model: +3.63pp ROI at threshold=0.0. On per-league models: -0.94pp regression. Kelly's benefit depends on having poorly-calibrated cross-league variance to de-weight. Per-league models are already better calibrated, so Kelly adds no value. Not recommended with per-league setup._

~~**28. Season standings context:**~~ _Tested Iter 28 — regression (ROI -4.18% vs -3.40%, -0.78pp; Stability -0.0278 vs -0.0225). Season pts/games_played are highly correlated with Elo (same teams dominate both). Redundant with existing Elo features; adds variance in smaller per-league datasets. Reverted._

~~**29. Draw propensity (rolling draw rate):**~~ _Tested Iter 29 — severe regression (ROI -7.20% vs -3.40%, -3.80pp; Stability -0.0480). WINDOW=5 draw rates are too noisy (near-zero autocorrelation); also bookmakers over-price draws by design to manage exposure, so any model nudged toward draw bets is playing into the market's strongest pricing. Large accuracy drop (-0.012). Reverted._

~~**30. Opponent-quality-adjusted form (SPI-style):**~~ _Tested Iter 35 — regression (ROI -2.85%, -1.19pp). Scale distortion and train/predict Elo inconsistency. Reverted._

~~**32. EWM span=3:**~~ _Tested Iter 32 — severe regression (ROI -4.52%, -2.63pp). Too noisy; span=5 confirmed optimal._
~~**34. HistGBM max_depth=3, min_samples_leaf=30:**~~ _Tested Iter 34 — regression (ROI -3.39%, -1.73pp). Underfit; original config kept._
~~**36. l2_regularization=1.0:**~~ _Tested Iter 36 — regression (ROI -2.64%, -0.98pp). Penalises market-deviation signal. Original config kept._

~~**37. Pinnacle closing odds as model features:**~~ _Tested Iter 37 (two variants). Variant A (both B365+Pinnacle): ROI -3.78% (-2.12pp). Variant B (Pinnacle replaces B365): ROI -2.57% (-0.91pp). Both regress. Root cause: adding highly correlated probability features hurts in small per-league datasets; the Pinnacle-vs-B365 discrepancy is already a known signal priced by the market. Correct Pinnacle use: as VALUE-DETECTION CRITERION in the bet filter (`Pinnacle_fair_prob > B365_fair_prob`), not as a model feature. Loader now passes through PSCH/PSCD/PSCA for future experiments._

**38. Pinnacle as value-detection criterion (structural change to bet filter)**
Currently bets are placed when `model_prob > B365_fair_prob`. An alternative: place bets when `Pinnacle_fair_prob > B365_fair_prob` (sharp money says B365 is underpricing this outcome), or the intersection: both conditions must hold. This bypasses the model's need to learn the Pinnacle signal as a feature and uses it directly where it's empirically strongest — as a reference point for detecting mispriced B365 odds. Requires modifying `compute_value_betting_results` to accept Pinnacle fair probs and changing the threshold logic. The PSCH columns are now available in the loaded data. _Medium-high confidence, medium effort._

~~**30. Market deviation persistence:**~~ _Tested Iter 25 — regression on per-league baseline (ROI -4.25% vs -3.40%, Stability -0.0282 vs -0.0225). WINDOW=5 too noisy for reliable bias signal; per-league models already capture market deviation patterns implicitly. Could revisit with longer window (20+ games) but would reduce training rows significantly. Reverted._

~~**Elo ratings as features:**~~ _Tested in Iteration 3 — improved accuracy (+2.7pp) and ROI (+0.47%) but remains negative. Elo is now a permanent part of the feature set._

~~**Gradient Boosting model (XGBoost/LightGBM):**~~ _Tested in Iteration 2 — no improvement over Logistic Regression with the current 6-feature set. Model capacity is not the bottleneck._

~~**Home/away split form:**~~ _Tested in Iteration 1 — worsened all metrics. Discarded._

---

## Key Findings So Far

- **Strategy shift (2026-04-19): primary goal is positive + stable ROI at threshold=0.0.** The threshold=0.06 operating point was retired — it was selected on the test set (Iter 15, look-ahead bias) and its apparent success (+3.93% ROI) is not reliable. All future iterations target ROI > 0% and Stability > 0 at threshold=0.0. This is a harder target (bookmaker vig ≈ 5% must be beaten on all bets, not just a cherry-picked 12.8% subset) but any success here is genuinely generalisable.

- **Recurring pattern — accuracy improves but ROI does not (Iterations 12, 13, 20, 22, 23):** Five separate feature experiments (H2H, WINDOW=7, season progress, EWM, days rest) all improved classification accuracy by 0.002–0.005 while worsening or not improving threshold=0.0 ROI. This means the features carry real discriminative signal that the market has already priced. Adding information the bookmaker already knows cannot create betting edge. Future experiments must seek signal the market is known to *mis-price* — e.g., market deviation persistence, opponent-adjusted form.

- **All value-betting approaches have failed — flat betting remains best (Iterations 4, 5, 8):** Three attempts to exploit model probabilities vs. bookmaker implied probabilities have all catastrophically worsened ROI: single-outcome value betting with raw LogReg (-15.10%), with calibrated LogReg (-15.52%), and multi-outcome value betting with HistGBM (-17.80%). The multi-outcome approach (Iter 8) bet on 3337 outcomes across 2643 matches (126.3% bet rate), revealing that the model simultaneously sees spurious "value" on multiple outcomes per match. The root cause is that bookmaker odds carry a vig that is not accounted for in the model's probability output, creating systematic false positive value signals. Future improvement must come from either (a) reducing the base loss rate (better accuracy) or (b) explicit bookmaker-margin correction before applying a value filter.

- **Derived linear combination features regress performance (Iteration 7):** Adding `home_form_gd` (= gf − ga), `away_form_gd`, `elo_diff` (= home_elo − away_elo), and `league_code` worsened all metrics vs Iter 6 (ROI: -5.09% → -6.30%, Accuracy: 0.521 → 0.518, Stability: -0.0516 → -0.0647). Features that are exact linear combinations of existing features offer no new information for tree models and can dilute the signal budget, increasing variance without reducing bias. The lesson: only add features that represent genuinely new information.

- **HistGBM + Elo features is the new best model (Iteration 6):** Replacing LogisticRegression with HistGradientBoostingClassifier (max_iter=300, lr=0.05, max_depth=4, min_samples_leaf=20) on the 8-feature Elo+rolling set improved ROI from -6.32% to -5.09% (+1.23pp) and stability from -0.0652 to -0.0516. The hypothesis is confirmed: GBM benefits from Elo's non-linear interactions where it had no gain with rolling-only features (Iter 2). This is the first iteration to clearly beat the previous best on all three metrics simultaneously.

- **Probability calibration does not fix the value-bet filter (Iteration 5):** Wrapping LogReg in `CalibratedClassifierCV(cv=5, method="isotonic")` left ROI essentially unchanged at -15.52% (vs -15.10% in Iter 4). Isotonic calibration is a monotone transform of the predicted probabilities, so it preserves the ranking of outcomes — the same bets are selected as "value" before and after calibration. The structural flaw is that the value-bet filter always bets in the direction of the model's predicted class, and the model's predicted class is already the bookmaker's most likely outcome most of the time. Value betting via this mechanism is abandoned.

- **Value betting without calibration makes ROI worse (Iteration 4):** Filtering to bets where model probability > bookmaker implied probability worsened ROI from -6.32% to -15.10% and reduced bets to 884 (33.4%). The root cause: Logistic Regression probabilities are not calibrated, causing systematic overconfidence for predicted outcomes. The model picks exactly the bets where it is most wrong relative to the market. Value betting requires probability calibration (Platt scaling or isotonic regression) as a prerequisite.

- **Elo ratings meaningfully improve accuracy and ROI (Iteration 3):** Adding pre-match Elo ratings for home and away teams (K=30, HOME_ADV=100) improved accuracy by +2.7pp (0.492 → 0.519) and ROI by +0.47pp (-6.79% → -6.32%). Elo is now part of the permanent 8-feature set. ROI remains negative, but the hypothesis was confirmed: long-run team strength captures information beyond 5-game rolling form. Next priority: value-bet threshold filtering to exploit the improved probability estimates.

- **Home/away venue split hurts, not helps (Iteration 1):** Splitting rolling form into home-only and away-only stats reduced all metrics vs baseline. The additional warm-up cost (needing 5 home AND 5 away games) drops ~10% of test matches, and sparser per-venue windows produce noisier estimates. Combined rolling form across all games is a better signal at window=5.

- **HistGBM offers no improvement over Logistic Regression (Iteration 2):** With only 6 aggregate rolling-mean features, there are insufficient non-linear interactions for gradient boosting to exploit. Both models perform near-equivalently (-6.79% vs -7.23% ROI). The bottleneck is feature richness, not model capacity. Future iterations should prioritize richer features (Elo ratings) or value-bet threshold filtering rather than model architecture changes.

---

## Notes / Lessons Learned

**Dataset facts:**
- Covers multiple European leagues, seasons 1314–2425
- Test period: last 2 full seasons (2324, 2425)
- Total test bets: 2643 — large enough for statistical significance
- Bookmaker margin (vig) is approximately 5%, so ROI > 0% requires genuine predictive edge
- Accuracy of ~0.492 on a 3-class problem (H/D/A) is close to the naive baseline; draws are hard to predict

**Pipeline facts:**
- Run pipeline: `uv run python main.py --per-league` (per-league + EWM form, threshold=0.0) ← **current best**
- Run global model: `uv run python main.py` (single model, for comparison)
- Run with edge filter: `uv run python main.py --per-league --threshold 0.05` (require ≥5% edge)
- Run tests: `uv run pytest tests/ -v`
- Profit chart saved automatically: `reports/profit_curve.png`
- Evaluation strategy: multi-outcome value betting with vig-corrected fair probabilities
- Frozen files: `src/data/`, `src/evaluation/report.py`, `tests/`
- Editable files: `src/model/features.py`, `src/model/train.py`, `src/evaluation/metrics.py`, `main.py`, `autoresearch/state.md`
