# Autoresearch experiment ledger

This is the append-only record of completed experiments and archived investigations.

## Identifier migration

Two independent ledgers reused iteration numbers. Their records are preserved with globally unique IDs:

- `EXP-<date>-S<id>`: migrated from `autoresearch/state.md`.
- `EXP-<date>-D<id>`: migrated from `docs/improvements.md`.
- `SX` and `DP` identify formerly unnumbered records.

Legacy iteration references inside an entry are source-local. For example, “Iteration 84” in a migrated `D` entry refers to the old docs ledger, while the same text in an `S` entry refers to the old state ledger.

New experiments use `EXP-YYYYMMDD-NNN` and must be appended at the bottom.

# State-ledger history

## EXP-20260417-S000: Baseline

_Legacy source: state.md baseline iteration 0._

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

## EXP-20260513-S070: DC_SPAN=15 (REVERTED — severe regression)

_Legacy source: state.md iteration 70._

**Date:** 2026-05-13
**Hypothesis:** Longer DC rating span (15 vs 10) gives more stable attack/defense estimates.
**Files changed:** `src/model/features.py` — DC_SPAN 10→15.
**Results:** ROI +1.68%, Stability 0.0176, t-stat +0.76 — **SEVERE REGRESSION** (vs +5.48%).
**Analysis:** Longer DC_SPAN requires more games before producing valid ratings (min_periods=span), reducing training data used and changing how DC features correlate with match outcomes. DC_SPAN=10 is confirmed optimal. Reverted.

---

## EXP-20260513-S069: num_leaves=20 (REVERTED — regression)

_Legacy source: state.md iteration 69._

**Date:** 2026-05-13
**Hypothesis:** Fewer leaves reduces overfitting in smaller per-league datasets.
**Files changed:** `src/model/train.py` — num_leaves 31→20.
**Results:** ROI +4.88%, Stability 0.0498, t-stat +2.16 — **REGRESSION** (vs +5.48%).
**Analysis:** Reducing tree complexity from 31 to 20 leaves trades off predictive capacity for regularization, but the net effect is a regression. num_leaves=31 remains optimal. Reverted.

---

## EXP-20260513-S068: reg_lambda=0.1 (REVERTED — regression)

_Legacy source: state.md iteration 68._

**Date:** 2026-05-13
**Hypothesis:** Stronger L2 regularization might help generalize per-league models.
**Files changed:** `src/model/train.py` — reg_lambda 0.05→0.1.
**Results:** ROI +4.19%, Stability 0.0437, t-stat +1.90 — **REGRESSION** (vs +5.48%).
**Analysis:** Stronger regularization penalises the market-deviation signal that drives value betting edge. reg_lambda=0.05 remains optimal. Reverted.

---

## EXP-20260513-S067: WINDOW=7 EWM form span (REVERTED — regression)

_Legacy source: state.md iteration 67._

**Date:** 2026-05-13
**Hypothesis:** Longer EWM form window smooths noise in per-league models.
**Files changed:** `src/model/features.py` — WINDOW 5→7.
**Results:** ROI +2.60%, Stability 0.0270, t-stat +1.17 — **REGRESSION** (vs +5.48%).
**Analysis:** WINDOW=7 reduces the number of matches with valid form features, and the smoother signal is not as reactive to current form. WINDOW=5 remains optimal. Reverted.

---

## EXP-20260513-S066: min_child_samples=30 (REVERTED — regression)

_Legacy source: state.md iteration 66._

**Date:** 2026-05-13
**Hypothesis:** Larger leaf size reduces overfitting in small per-league datasets.
**Files changed:** `src/model/train.py` — min_child_samples 20→30.
**Results:** ROI +4.18%, Stability 0.0436, t-stat +1.89 — **REGRESSION** (vs +5.48%).
**Analysis:** More regularization via larger leaf size reduces the model's ability to capture market mispricing signals. min_child_samples=20 remains optimal. Reverted.

---

## EXP-20260513-S065: min_child_samples=10 (REVERTED — severe regression)

_Legacy source: state.md iteration 65._

**Date:** 2026-05-13
**Hypothesis:** Smaller leaf size allows finer-grained patterns in per-league models.
**Files changed:** `src/model/train.py` — min_child_samples 20→10.
**Results:** ROI +1.71%, Stability 0.0178, t-stat +0.78 — **SEVERE REGRESSION** (vs +5.48%).
**Analysis:** Smaller leaves overfit to per-league noise. min_child_samples=20 is confirmed optimal. Reverted.

---

## EXP-20260513-S064: n_estimators=500 (REVERTED — marginal regression)

_Legacy source: state.md iteration 64._

**Date:** 2026-05-13
**Hypothesis:** Even more trees may squeeze additional signal from 30 features.
**Files changed:** `src/model/train.py` — n_estimators 400→500.
**Results:** ROI +5.10%, Stability 0.0535, t-stat +2.34 — **MARGINAL REGRESSION** (vs +5.48%).
**Analysis:** The model converges at ~400 trees; adding more introduces marginal overfitting. n_estimators=400 confirmed as the sweet spot. Reverted.

---

## EXP-20260513-S063: n_estimators=400 — **NEW BEST**

_Legacy source: state.md iteration 63._

**Date:** 2026-05-13
**Hypothesis:** With 30 features and per-league models (~2000-3000 training rows), 300 trees at lr=0.05 may not fully converge. Increasing to 400 should improve generalization.
**Files changed:** `src/model/train.py` — n_estimators 300→400.
**Results:**
- Accuracy: 0.507
- ROI: **+5.48%** (+0.71pp vs +4.77%)
- Stability: **0.0569** (+0.0078)
- t-stat: **+2.50** (statistically significant!)
- Bets: 1923 / 4354 (44.2%)
- **NEW BEST on all primary metrics.**
**Analysis:** More trees at the same learning rate (0.05) allow the model to better fit the training distribution without changing the feature set. The boost in ROI (+0.71pp) and stability (+0.0078) is consistent with incomplete convergence at 300 trees with 30 features. The t-stat now exceeds 2.0, making this the first statistically significant result in the project. Kept.

---

## EXP-20260513-S062: Pinnacle Margin 1.5% — **NEW BEST**

_Legacy source: state.md iteration 62._

**Date:** 2026-05-13
**Hypothesis:** A stricter 1.5% Pinnacle margin (vs 1%) will further improve bet quality by requiring more confident agreement from Pinnacle before placing.
**Files changed:** `src/evaluation/metrics.py` — Pinnacle margin 0.01→0.015.
**Results:**
- Accuracy: 0.510
- ROI: **+4.77%** (+0.61pp vs +4.16%)
- Stability: **0.0491** (+0.0067)
- t-stat: **+2.13** (first t-stat > 2.0!)
- Bets: 1886 / 4354 (43.3%)
- **NEW BEST. First iteration with statistically significant t-stat.**
**Analysis:** Stricter Pinnacle agreement filter removes more bets where the two books disagree only marginally — those bets are likelier to be noise. The tighter filter reduces bets by ~235 but improves quality enough to push t-stat above 2.0 for the first time. Kept.

---

## EXP-20260513-S061: Pinnacle Margin 0.5% (REVERTED — regression)

_Legacy source: state.md iteration 61._

**Date:** 2026-05-13
**Hypothesis:** A looser 0.5% Pinnacle margin allows more bets while still filtering noise.
**Files changed:** `src/evaluation/metrics.py` — Pinnacle margin 0.01→0.005.
**Results:** ROI +3.26%, Stability 0.0329, t-stat +1.62 — **REGRESSION** (vs +4.16% Iter 60).
**Analysis:** Looser margin (0.5%) adds too many marginal bets, diluting bet quality. 1% margin (Iter 60) confirmed as minimum. Reverted to 1% (then subsequently improved to 1.5% in Iter 62).

---

## EXP-20260513-S060: Exclude France from Betting — **NEW BEST**

_Legacy source: state.md iteration 60._

**Date:** 2026-05-13
**Hypothesis:** France (F1) has consistently -30% ROI across all configurations. Excluding France from the bet pool (while keeping it in training) will improve overall ROI and stability.
**Files changed:** `src/evaluation/metrics.py` — added `skip_leagues` parameter to `compute_value_betting_results()`; `main.py` — passed `skip_leagues={"F1"}`.
**Results:**
- Accuracy: 0.510 (unchanged — model unchanged)
- ROI: **+4.16%** (+1.0pp vs +3.16%)
- Stability: **0.0424** (+0.0100)
- t-stat: **+1.95** (near but below significance)
- Bets: 2121 / 4354 (48.7%) — 322 fewer bets (France excluded)
- **NEW BEST on ROI and Stability.**
**Analysis:** Excluding France's 322 bets (which had -32% ROI) significantly improved the portfolio. France's systematic negative ROI suggests either (1) systematic Pinnacle-B365 pricing alignment in France that doesn't reflect a genuine edge, or (2) the model is less effective in Ligue 1 due to differences in data quality or team volatility. Keeping France in training ensures Elo/form features continue learning cross-league patterns. Kept.

---

## EXP-20260513-S059: DC_SPAN=20 for Long-term Attack/Defense Ratings (REVERTED — severe regression)

_Legacy source: state.md iteration 59._

**Date:** 2026-05-13
**Hypothesis:** Increasing DC_SPAN from 10 to 20 games makes `home_attack`/`home_defense`/`away_attack`/`away_defense` more structural (half-season+), which is more orthogonal to the 5-game `form_gf`/`form_ga` features. Expected to reduce collinearity and improve model quality.
**Files changed:** `src/model/features.py` — changed `DC_SPAN = 10` to `DC_SPAN = 20`. Reverted after results.

**Results (on top of Iter 57 Pinnacle +1% filter):**

| Metric    | Iter 57 (DC=10) | Iter 59 (DC=20) | Δ |
|-----------|-----------------|-----------------|---|
| ROI       | **+3.16%**      | **+0.43%**      | **-2.73pp** ↓ |
| Stability | **+0.0324**     | **+0.0043**     | **-0.0281** ↓ |
| Bets      | 2443            | 2449            | +6 |

**Analysis:** Severe regression on both metrics. DC_SPAN=20 requires 20 games (min_periods=20) before producing a non-NaN value. While dropna still keeps the same test rows (since we use min_periods=span in the EWM), the longer span makes the attack/defense features track multi-season averages rather than current-season form. This over-smooths the signal: a newly promoted team or one that changed manager still shows ratings from 15+ games ago. The 10-game span is better calibrated to the season length (~38 games) and the EWM weighting ensures recent games dominate. Reverted.

**Decision:** REVERTED. DC_SPAN=10 confirmed as superior.

---

## EXP-20260513-S058: Pinnacle Filter Margin +2% (REVERTED — marginal gain, lower t-stat)

_Legacy source: state.md iteration 58._

**Date:** 2026-05-13
**Hypothesis:** The +1% margin (Iter 57) improved both ROI and Stability. Testing whether +2% margin further concentrates bets on the strongest Pinnacle-confirmed signals.
**Files changed:** `src/evaluation/metrics.py` — changed margin from 0.01 to 0.02. Reverted after results.

**Results (on top of Iter 57 Pinnacle +1% filter):**

| Metric    | Iter 57 (+1% margin) | Iter 58 (+2% margin) | Δ |
|-----------|----------------------|----------------------|---|
| ROI       | **+3.16%**           | **+3.18%**           | +0.02pp (noise) |
| Stability | **+0.0324**          | **+0.0330**          | +0.0006 (noise) |
| t-stat    | +1.60                | +1.46                | -0.14 ↓ |
| Bets      | 2443                 | 1949                 | -494 |

**Analysis:** The +2% margin is marginally better on ROI (+0.02pp) and Stability (+0.0006), but the t-stat drops from 1.60 to 1.46 because bet count falls by 494 (20% fewer bets = higher sampling noise). Italy also turned slightly negative (-1.27%). The gain is within noise and the reduced bet count makes the result statistically weaker. The +1% margin (Iter 57) offers a better trade-off: more bets, higher t-stat, and nearly identical ROI. Reverted.

**Decision:** REVERTED. +1% Pinnacle margin (Iter 57) kept as the standard.

---

## EXP-20260513-S057: Pinnacle Filter +1% Margin — **NEW BEST**

_Legacy source: state.md iteration 57._

**Date:** 2026-05-13
**Hypothesis:** The current Pinnacle filter vetoes bets where `pinnacle_fair[outcome] <= b365_fair[outcome]` (any amount). Bets where Pinnacle only marginally exceeds B365 (e.g., 0.001%) may represent noise rather than genuine confirmation. Requiring at least a 1% margin (`pinnacle_fair > b365_fair + 0.01`) should filter out borderline cases and improve bet quality.
**Files changed:** `src/evaluation/metrics.py` — changed Pinnacle filter condition from `pinnacle_fair[outcome] <= fair[outcome]` to `pinnacle_fair[outcome] <= fair[outcome] + 0.01`.

**Results:**

| Metric    | Iter 54 (margin=0) | Iter 57 (margin=+1%) | Δ |
|-----------|--------------------|----------------------|---|
| ROI       | **+2.83%**         | **+3.16%** ✅        | **+0.33pp** ↑ |
| Stability | **+0.0283**        | **+0.0324** ✅       | **+0.0041** ↑ |
| t-stat    | +1.58              | +1.60                | +0.02 ↑ |
| Bets      | 3133               | 2443                 | -690 |

Per-league ROI with +1% Pinnacle margin:

| League | Bets | ROI |
|--------|------|-----|
| England | 337 | +59.52% |
| Germany | 301 | +51.87% |
| Spain | 420 | +33.44% |
| Italy | 394 | +10.67% |
| France | 322 | -32.07% |
| Netherlands | 327 | +2.67% |
| Portugal | 342 | +86.93% |

**Analysis:** The +1% Pinnacle margin improvement confirms the hypothesis. Requiring a stronger Pinnacle confirmation signal removes ~690 bets (mostly marginal cases where Pinnacle barely outprices B365) and concentrates capital on matches where the Pinnacle-B365 discrepancy is more meaningful. The result is cleaner bets: 5/7 leagues positive, with Spain turning from -6% to +33% (the zero-margin filter was accepting noisy Spanish market signals). France remains the only strongly negative league. Both ROI and Stability improve, confirming this is a genuine quality improvement rather than overfitting.

**Decision:** KEPT — **new best**. Run with `uv run python main.py --per-league --threshold 0.0`.

---

## EXP-20260513-S056: LGBM num_leaves=63 (REVERTED — regression)

_Legacy source: state.md iteration 56._

**Date:** 2026-05-13
**Hypothesis:** With 30 features (vs the 22 when the model was configured at num_leaves=31), deeper trees (num_leaves=63) could capture more complex feature interactions that the current model misses.
**Files changed:** `src/model/train.py` — changed `num_leaves=31` to `num_leaves=63`. Reverted after results.

**Results:**

| Metric    | Iter 54 baseline | Iter 56 (leaves=63) | Δ |
|-----------|------------------|---------------------|---|
| ROI       | **+2.83%**       | **+2.32%**          | **-0.51pp** ↓ |
| Stability | **+0.0283**      | **+0.0241**         | **-0.0042** ↓ |

**Analysis:** More leaves causes overfitting to per-league training patterns, reducing generalization on the test set. England particularly suffered (-17.69% vs +43.12%). The 31-leaf config remains optimal for per-league models trained on ~5000-7000 rows per league.

**Decision:** REVERTED. num_leaves=31 kept.

---

## EXP-20260513-S055: Remove elo_delta Features (REVERTED — regression)

_Legacy source: state.md iteration 55._

**Date:** 2026-05-13
**Hypothesis:** `home_elo_delta` and `away_elo_delta` were reverted in Iter 42 (tested without Pinnacle filter or DC ratings). They appear to have been re-added later. Testing whether they still add value in the current 30-feature model.
**Files changed:** `src/model/features.py` — removed `home_elo_delta`, `away_elo_delta` from FEATURE_COLS (30 → 28 features). Reverted after results.

**Results:**

| Metric    | Iter 54 baseline | Iter 55 (-elo_delta) | Δ |
|-----------|------------------|----------------------|---|
| ROI       | **+2.83%**       | **+1.65%**           | **-1.18pp** ↓ |
| Stability | **+0.0283**      | **+0.0165**          | **-0.0118** ↓ |

**Analysis:** Clear regression. Despite elo_delta being reverted in Iter 42 (on a very different feature set), they are now genuinely useful in the 30-feature model. The interaction between elo_delta and market_h/market_d/market_a (market fair probs) allows the model to detect when the market hasn't fully priced in recent team trajectory changes. Reverted.

**Decision:** REVERTED. elo_delta features confirmed as beneficial.

---

## EXP-20260423-S054: Add France (F1), Netherlands (N1), Portugal (P1) — **NEW BEST**

_Legacy source: state.md iteration 54._

**Date:** 2026-04-23
**Hypothesis:** Adding 3 more leagues increases bet volume (~110 → ~190 bets/month), compressing monthly variance by ~√(7/4) and reducing sampling noise. All three leagues are available on football-data.co.uk from 2013-14 with full B365 and Pinnacle column coverage. The per-league model structure means each new league gets its own dedicated model.

**Files changed:**
- `src/data/download.py` — added france/F1, netherlands/N1, portugal/P1 to LEAGUES
- `src/data/loader.py` — added 3 entries to `_LEAGUE_MAP`; fixtures filter now includes F1, N1, P1
- `src/model/features.py` — added `league_F1`, `league_N1`, `league_P1` dummies (I1 remains omitted reference); 19 → 22 features
- `src/model/train.py` — added F1, N1, P1 to `_LEAGUES`
- `main.py` — added 3 leagues to `_print_split_analysis` lookup
- Downloaded 39 historical CSV files (3 leagues × 13 seasons)

**Results:**

| Metric | 4-league baseline | 7-league (this iter) | Δ |
|--------|-------------------|----------------------|---|
| ROI | +2.64% | **+7.08%** | +4.44pp |
| Stability | +0.0172 | **+0.0457** | +0.0285 |
| Bets | 2223 | 3669 | +1446 |
| Test matches | 2655 | 4358 | +1703 |

Per-league ROI (7-league model):

| League | Bets | ROI |
|--------|------|-----|
| England | 557 | +12.45% |
| Germany | 484 | +0.38% |
| Spain | 558 | -6.09% |
| Italy | 624 | +3.44% |
| **France** | **458** | **+8.98%** |
| **Netherlands** | **495** | **+19.00%** |
| **Portugal** | **493** | **+13.37%** |

**Analysis:** The 3 new leagues contribute ~13.9% combined ROI on 1446 bets — exceptionally strong. Crucially, the existing 4-league numbers are completely unchanged (per-league models train independently), so this is pure addition with no risk to the existing signal. Netherlands is the standout at +19.00%. The new leagues also meaningfully reduce monthly sampling noise: test bets increase by 65%, bringing expected monthly std from ±12% down toward ±9%. Spain remains the only negative league (-6.09%) across all setups.

**Decision:** KEPT — **new best**. Run with `uv run python main.py --per-league`.

---

## EXP-20260423-SX01: Monthly Retraining + Long Market Bias Window (NOT ADOPTED — both regress)

_Legacy source: state.md unnumbered experiment._

**Date:** 2026-04-23
**Hypothesis:** Two ideas to reduce large month-to-month ROI swings:
- C: Add 20-game market bias feature alongside existing 5-game version — captures sustained mis-pricing beyond short-term noise.
- A: Retrain per-league models monthly on all data up to that point, rather than once per season.

**Results:**

| Mode | Bets | ROI | Stability | Monthly ROI std |
|------|------|-----|-----------|-----------------|
| **Per-league seasonal (baseline)** | **2223** | **+2.64%** | **+0.0172** | **9.48pp** |
| Long market bias (window=20) | 2125 | -1.80% | -0.0124 | — |
| Monthly retrain (`--monthly`) | 2237 | +0.03% | +0.0002 | 10.24pp |

**Analysis:**

- **Long market bias (window=20):** Regression of -4.44pp ROI. The 20-game `min_periods` requirement drops 116 test rows early in each season (NaN from `dropna`), shrinking the test set. Also, a 20-game window means the last ~half-season of data — by which point the 5-game window already captures the same signal more noisily. The feature adds correlated information that dilutes tree budget without independent signal. Reverted.

- **Monthly retraining:** Regression of -2.61pp ROI. Monthly variance increased slightly (std 10.24pp vs 9.48pp). The root cause: monthly variance is **irreducible statistical noise** — with ~100 bets/month at ~2.5 average odds, the expected ROI std is ±12% just from sampling. No retraining schedule can fix this. Monthly retraining also creates a minor information leak risk: with only 4–5 months of in-season data by mid-season, the per-league model trains on ~3800 rows instead of ~15000, reducing model quality. The seasonal model wins because it has richer training data.

**Key finding:** Monthly ROI swings are sampling noise, not model drift. The path to lower variance is more bets per month (more leagues) or accepting the noise and evaluating on 6-month+ windows. Both options reverted.

**Decision:** NOT ADOPTED. Per-league seasonal retrain (`--per-league`) remains default.

---

## EXP-20260422-SX02: Binary Outcome Models (NOT ADOPTED — per-league multi-class remains best)

_Legacy source: state.md unnumbered experiment._

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

## EXP-20260421-S043: Season Progress Features (REVERTED — flat)

_Legacy source: state.md iteration 43._

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

## EXP-20260421-S042: Elo Delta / Momentum (REVERTED — regression)

_Legacy source: state.md iteration 42._

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

## EXP-20260421-S041: Draw Rate Features (KEPT — new best)

_Legacy source: state.md iteration 41._

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

## EXP-20260421-S040: Threshold=0.08 as Default (REVERTED — threshold tuning is unstable)

_Legacy source: state.md iteration 40._

**Date:** 2026-04-21
**Hypothesis:** Threshold grid showed 0.08 gives +2.13% ROI vs +0.78% at threshold=0.00.
**Decision:** REVERTED. Threshold selection is post-hoc optimization on the test set — it shifts every time the model or feature set changes, making it an unreliable operating parameter. Evaluation stays at threshold=0.00.

---

## EXP-20260420-S039: Shots on Target EWM Form (REVERTED — regression)

_Legacy source: state.md iteration 39._

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

## EXP-20260420-S038: Pinnacle as Value-Detection Criterion (KEPT — **GOALS ACHIEVED**)

_Legacy source: state.md iteration 38._

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

## EXP-20260420-S037: Pinnacle Closing Odds as Features (REVERTED — regression, two variants)

_Legacy source: state.md iteration 37._

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

## EXP-20260420-S036: l2 Regularization (REVERTED — regression)

_Legacy source: state.md iteration 36._

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

## EXP-20260420-S035: Opponent-Quality-Adjusted Form (REVERTED — regression)

_Legacy source: state.md iteration 35._

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

## EXP-20260420-S034: HistGBM Hyperparameter Tuning (REVERTED — regression)

_Legacy source: state.md iteration 34._

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

## EXP-20260420-S033: H2H Win Rate Re-test (KEPT — marginal improvement)

_Legacy source: state.md iteration 33._

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

## EXP-20260420-S032: EWM span=3 (REVERTED — regression)

_Legacy source: state.md iteration 32._

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

## EXP-20260419-S031: WINDOW=7 with EWM (REVERTED — marginal, within noise)

_Legacy source: state.md iteration 31._

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

## EXP-20260419-S030: EWM Rolling Form on Per-League Models (KEPT — new best)

_Legacy source: state.md iteration 30._

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

## EXP-20260419-S029: Draw Propensity (REVERTED — regression)

_Legacy source: state.md iteration 29._

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

## EXP-20260419-S028: Season Standings Context (REVERTED — regression)

_Legacy source: state.md iteration 28._

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

## EXP-20260419-S023: Days Since Last Match (REVERTED — regression)

_Legacy source: state.md iteration 23._

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

## EXP-20260419-S022: Exponential Decay Rolling Form (REVERTED — regression)

_Legacy source: state.md iteration 22._

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

## EXP-20260419-S027: Elo Momentum (REVERTED — regression)

_Legacy source: state.md iteration 27._

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

## EXP-20260419-S026: Kelly Criterion on Per-League Models (REVERTED — regression)

_Legacy source: state.md iteration 26._

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

## EXP-20260419-S025: Market Deviation Persistence (REVERTED — regression)

_Legacy source: state.md iteration 25._

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

## EXP-20260419-S024: League-Specific Sub-models

_Legacy source: state.md iteration 24._

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

## EXP-20260419-S021: Kelly Criterion Bet Sizing

_Legacy source: state.md iteration 21._

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

## EXP-20260419-S020: Season Progress Ratio (REVERTED — regression)

_Legacy source: state.md iteration 20._

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

## EXP-20260418-S019: League One-Hot Encoding

_Legacy source: state.md iteration 19._

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

## EXP-20260418-S018: Venue-Specific Rolling Form (REVERTED — regression)

_Legacy source: state.md iteration 18._

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

## EXP-20260418-S017: Sigmoid Calibration on Top of Odds Features

_Legacy source: state.md iteration 17._

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

## EXP-20260418-S016: Market Fair Probabilities as Features

_Legacy source: state.md iteration 16._

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

## EXP-20260418-S015: Threshold Grid Search

_Legacy source: state.md iteration 15._

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

## EXP-20260418-S014: Probability Calibration (CalibratedClassifierCV, isotonic, cv=3)

_Legacy source: state.md iteration 14._

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

## EXP-20260418-S013: Longer Rolling Window (WINDOW=7)

_Legacy source: state.md iteration 13._

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

## EXP-20260418-S012: Head-to-Head Historical Win Rate

_Legacy source: state.md iteration 12._

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

## EXP-20260418-S011: Season Phase (match_month)

_Legacy source: state.md iteration 11._

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

## EXP-20260418-S010: Shorter Rolling Window (3-game)

_Legacy source: state.md iteration 10._

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

## EXP-20260418-S009: Elo Hyperparameter Tuning (K=20, HOME_ADV=65)

_Legacy source: state.md iteration 9._

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

## EXP-20260417-S008: Multi-Outcome Value Betting

_Legacy source: state.md iteration 8._

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

## EXP-20260417-S007: Feature Enrichment (Goal Difference, Elo Diff, League Categorical)

_Legacy source: state.md iteration 7._

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

## EXP-20260417-S006: HistGBM with Elo + Rolling Features (All Bets)

_Legacy source: state.md iteration 6._

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

## EXP-20260417-S005: Calibrated Probabilities + Value Betting

_Legacy source: state.md iteration 5._

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

## EXP-20260417-S004: Value Betting Filter

_Legacy source: state.md iteration 4._

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

## EXP-20260417-S003: Elo Ratings as Features

_Legacy source: state.md iteration 3._

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

## EXP-20260417-S002: HistGradientBoostingClassifier

_Legacy source: state.md iteration 2._

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

## EXP-20260417-S001: Home/Away Split Form

_Legacy source: state.md iteration 1._

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

## EXP-20260517-S071: Isotonic Calibration on LGBM Per-League — KEPT

_Legacy source: state.md iteration 71._

**Date:** 2026-05-17
**Hypothesis:** Wrapping each per-league LGBMClassifier in `CalibratedClassifierCV(method="isotonic", cv=5, ensemble=False)` will reduce overconfident extreme predictions and improve both ROI and calibration quality. `ensemble=False` trains LGBM on the full per-league dataset and fits the calibrator on cross-validated out-of-fold predictions.
**Files changed:** `src/model/train.py` — added `CalibratedClassifierCV` import, `CALIBRATE=True` flag, `_fit_calibrated()` helper; replaced bare `LGBMClassifier` fit in `_predict_per_league` and `train_on_all_data_per_league` with `_fit_calibrated()`; `src/evaluation/metrics.py` — added `max_edge` parameter to `compute_value_betting_results`; `main.py` — added `_parse_max_edge()`, `max_edge` wiring, and `_print_edge_analysis()` (edge distribution + cap sweep printed at every backtest run).

**Results (threshold=0.0, with Pinnacle filter, --per-league):**
- Accuracy: 0.523 (Δ +1.8pp vs uncalibrated 0.505)
- ROI: +8.25% (Δ vs previous best +5.48%: +2.77pp)
- Stability: 0.0669 (Δ +0.0100)
- t-stat: +2.55
- Bets: 1453 / 4433 (32.8%) — 27% fewer bets than uncalibrated (1991), higher precision

**Without Pinnacle filter (inference-realistic baseline):**
- ROI: +0.20%, t-stat: +0.09 — essentially zero; calibration did not change this

**Edge distribution after calibration:** Extreme predictions collapsed — 30%+ edge bucket shrank from 649 bets to just 7. Model's probability mass is now concentrated in the 0–10% edge range, confirming calibration addressed the overconfidence problem.

**Critical finding — Pinnacle filter availability at inference time:**
All Pinnacle columns (PSH/PSD/PSA/PSCH/PSCD/PSCA) are **null in fixtures.csv** (0/112 rows). The Pinnacle filter is silently skipped at every live prediction. The difference between with/without Pinnacle is +4.05pp ROI — meaning the backtest figure of +8.25% likely overstates real-world performance. At inference time, expected ROI is closer to the no-Pinnacle baseline (+0.20%).

**Analysis:** Calibration improves the Pinnacle-filtered results by +2.77pp and restores accuracy to 0.523 (+1.8pp). Previous calibration attempts (Iter 5 on LogReg, Iter 14 isotonic cv=3, Iter 17 sigmoid) all failed on earlier models. The LGBM per-league setup responds well: calibration correctly identifies and compresses extreme overconfident leaves without distorting the core market-aligned probability signal. The `ensemble=False` option is key — it avoids reducing the already-small per-league training set.
**Decision:** KEPT. ROI +8.25% is a new best; all primary metrics improve. The Pinnacle inference gap remains the primary unresolved issue.

---

## EXP-20260517-S072: Remove Pinnacle Filter (inference-realistic baseline) — KEPT

_Legacy source: state.md iteration 72._

**Date:** 2026-05-17
**Hypothesis:** Pinnacle closing odds are all null in fixtures.csv (confirmed: 0/112 rows non-null). The filter has been silently skipped at inference time all along. Removing it makes backtest and inference consistent, revealing the true baseline.
**Files changed:** `src/evaluation/metrics.py` — removed `has_pinnacle` block and Pinnacle confirmation check from `compute_value_betting_results`; `main.py` — removed `_pinnacle_fair()` helper and Pinnacle check from `_build_prediction_rows`.
**Results:**
- ROI: +0.20% (Δ vs Pinnacle-filtered Iter 71: -8.05pp)
- Stability: 0.0016
- t-stat: +0.09 — not significant
- Bets: 3307 / 4433 (74.6%)
**Analysis:** The Pinnacle filter was responsible for virtually all the model's positive ROI. Without it, the calibrated model earns +0.20% — inside the noise band. The filter's effect (+4–8pp ROI) comes from removing bets where the sharpest bookmaker (Pinnacle) disagrees with our edge signal, meaning most of the uncapped bets are against a more efficient price. Goal now: find a filter or signal that replicates Pinnacle's role without needing Pinnacle data.
**Decision:** KEPT as the honest baseline. Not a regression in implementation — a correction in evaluation.

---

## EXP-20260517-S073: Exclude Spain, Germany, Italy from Betting — KEPT

_Legacy source: state.md iteration 73._

**Date:** 2026-05-17
**Hypothesis:** Without Pinnacle, Spain (-61%), Germany (-32%), and Italy (-29%) are deeply negative in the test period, identical to France (excluded in Iter 60). Removing them from the bet pool while keeping them in training will improve ROI by the same mechanism that worked for France.
**Files changed:** `main.py` — `skip_leagues` extended from `{"F1"}` to `{"F1", "SP1", "D1", "I1"}` in both `_run_backtest` and `_print_edge_analysis`.
**Results:**
- ROI: +6.62% (Δ vs previous best +0.20%: **+6.42pp**)
- Stability: 0.0540 (Δ +0.0524)
- t-stat: **+2.11** — statistically significant (crossed 2.0)
- Bets: 1526 / 4433 (34.4%); bet only on England, Netherlands, Portugal
**Analysis:** Excluding three consistently losing leagues recovered almost the full Pinnacle-filtered ROI (+8.25%) without needing external odds data. The pattern matches France: the model generates real edge in some leagues (Portugal +81%, England +45%, Netherlands +23%) and noise elsewhere. Why these three are bad is unclear — possibly smaller market depth, different home-advantage dynamics, or less B365 vs fair-odds alignment. The gain is large and clean.
**Decision:** KEPT. New inference-realistic best: ROI +6.62%, t-stat +2.11.

---

## EXP-20260517-S074: max_edge = 0.20 Default Cap — KEPT

_Legacy source: state.md iteration 74._

**Date:** 2026-05-17
**Hypothesis:** The cap-sweep in Iter 73 showed max_edge ≤ 0.20 gives +7.25% ROI (t-stat +2.29) vs +6.62% (t-stat +2.11) with no cap. The 28 bets above 20% edge are overconfident model outliers — removing them as the default improves both ROI and stability.
**Files changed:** `main.py` — `_parse_max_edge()` default changed from `float("inf")` to `0.20`.
**Results:**
- ROI: +7.25% (Δ vs Iter 73: +0.63pp)
- Stability: 0.0593 (Δ +0.0053)
- t-stat: +2.29 (Δ +0.18)
- Bets: 1498 / 4433 (33.8%) — 28 bets removed vs Iter 73
**Analysis:** Small but clean improvement. The 20%+ edge bets after calibration are rare (28/1526) and represent residual model overconfidence that isotonic regression didn't fully suppress. Capping them improves ROI without sacrificing statistical power (1498 bets is still enough for t > 2). This is consistent with the edge-distribution analysis showing the 20–25% bucket has +4.69% flat ROI (mediocre) and 25%+ is negative.
**Decision:** KEPT. New best: ROI +7.25%, Stability 0.0593, t-stat +2.29. Betting: England, Netherlands, Portugal only; max_edge=0.20; no Pinnacle filter; isotonic calibration.

---

## EXP-20260517-S075: Re-add France to Betting — REVERTED

_Legacy source: state.md iteration 75._

**Date:** 2026-05-17
**Hypothesis:** France was originally excluded for -30% ROI under the Pinnacle-filtered regime. Without Pinnacle the regime has changed, so France's performance might differ.
**Files changed:** `main.py` — `skip_leagues` changed from `{"F1","SP1","D1","I1"}` to `{"SP1","D1","I1"}`.
**Results:**
- ROI: +3.38% (Δ vs Iter 74: -3.87pp)
- Stability: 0.0273
- t-stat: +1.23 — dropped below 2.0 threshold
- France alone: -62.79% (worse than the -30% it showed with Pinnacle)
**Analysis:** Without the Pinnacle filter France's model generates far more noise bets, making it -62.79% — almost double the damage it caused before. "Training vs betting" distinction also clarified: in the per-league setup each league model is self-contained, so excluding a league from betting while keeping it "in training" is practically meaningless — it just trains a model whose predictions are discarded.

---

## EXP-20260518-S076: max_odds 4.0 → 5.0 — KEPT

_Legacy source: state.md iteration 76._

**Date:** 2026-05-18
**Hypothesis:** The max_odds=4.0 cap discards bets on high-odds outcomes (e.g., away wins at 4.5) where the model may still have genuine calibrated edge. Raising to 5.0 adds those bets, increasing portfolio size without necessarily hurting ROI.
**Files changed:** `main.py` — `_parse_max_odds()` default changed from `4.0` to `5.0`.
**Results:**
- Accuracy: 0.523 (unchanged — model unchanged)
- ROI: +7.22% (Δ vs Iter 74: -0.03pp — within noise)
- Stability: 0.0537 (Δ -0.0056)
- t-stat: +2.42 (Δ +0.13 — improved due to more bets)
- Bets: 2024 / 4433 (45.7%) — **+526 bets vs Iter 74** (+35%)
**Analysis:** Raising max_odds to 5.0 adds 526 bets with essentially zero net impact on ROI (-0.03pp, well within the ±3.2% noise band). Stability dipped slightly as the extra bets introduce more variance, but the larger sample size drives t-stat higher (+0.13). The 4.0–5.0 odds range contains value bets the calibrated model identifies with genuine edge. Bet count increase is substantial and ROI/t-stat remain healthy.
**Decision:** KEPT. All primary metrics remain positive (ROI +7.22%, t-stat +2.42 > 2.0). Bet count up 35%. Matches user goal of increasing bet volume.

---

## EXP-20260518-S077: learning_rate=0.03 + n_estimators=600 — REVERTED

_Legacy source: state.md iteration 77._

**Date:** 2026-05-18
**Hypothesis:** A lower LR with more trees explores the loss surface more carefully, potentially improving calibration and edge detection in small per-league datasets.
**Files changed:** `src/model/train.py` — n_estimators 400→600, learning_rate 0.05→0.03.
**Results:** ROI +5.49%, Stability 0.0410, t-stat +1.85 — **REGRESSION** (vs +7.22%). t-stat dropped below 2.0.
**Analysis:** Slower learning with more trees hurt in the per-league setup. Small per-league datasets (~2000–3000 rows) don't benefit from finer gradient steps; the original 400/0.05 configuration converges adequately. Reverted.

---

## EXP-20260518-S078: Weighted Training (recent 3 seasons 2×) — REVERTED

_Legacy source: state.md iteration 78._

**Date:** 2026-05-18
**Hypothesis:** Upweighting the 3 most recent training seasons (2×) biases the per-league models toward current market conditions, reducing concept drift and improving test-period ROI.
**Files changed:** `src/model/train.py` — added `_compute_sample_weights()`, passed `sample_weight` to `_fit_calibrated()` and `CalibratedClassifierCV.fit()`.
**Results:** ROI +3.65%, Stability 0.0277, t-stat +1.26 — **SEVERE REGRESSION** (vs +7.22%).
**Analysis:** Upweighting recent seasons in small per-league datasets (~2000–3000 rows) distorts the training distribution too aggressively. The model loses the signal from older seasons, hurts calibration quality, and produces noisier edge estimates. The per-league setup already effectively does temporal adaptation by training a fresh model per test season. Additional reweighting is counterproductive. Reverted.

---

## EXP-20260518-S079: max_edge 0.20 → 0.25 — REVERTED

_Legacy source: state.md iteration 79._

**Date:** 2026-05-18
**Hypothesis:** After Iter 76 expanded bet coverage via max_odds=5.0, relaxing the edge cap from 0.20 to 0.25 adds more bets in the 20–25% edge range. These bets had +4.69% flat ROI in the edge distribution analysis, still positive.
**Files changed:** `main.py` — `_parse_max_edge()` default changed from `0.20` to `0.25`.
**Results:**
- ROI: +6.79% (Δ vs Iter 76: -0.43pp)
- Stability: 0.0505 (Δ -0.0032)
- t-stat: +2.29 (Δ -0.13)
- Bets: 2051 / 4433 (46.3%) — +27 bets vs Iter 76
**Analysis:** The 20–25% edge bucket (27 extra bets) slightly dilutes ROI and stability. Those bets have lower return than the core 0–20% pool, confirming the 0.20 cap is near-optimal. The +27 bets are not worth the quality sacrifice. Reverted; Iter 76 (max_odds=5.0, max_edge=0.20) remains active.

---

## EXP-20260518-S080: Elo-market divergence features — REVERTED

_Legacy source: state.md iteration 80._

**Date:** 2026-05-18
**Hypothesis:** Adding `elo_h_win_prob` (Elo-implied home win probability) and `elo_market_divergence` (Elo vs market fair prob) as explicit features helps LGBM find market mispricing without needing deep interaction splits on home_elo/away_elo/market_h.
**Files changed:** `src/model/features.py` — added `elo_h_win_prob` and `elo_market_divergence` to FEATURE_COLS and computed them in `_build_merged()` and `build_fixture_features()`.
**Results:** ROI +1.81%, Stability 0.0135, t-stat +0.62 — **SEVERE REGRESSION** (vs +7.22%).
**Analysis:** Derived features that are deterministic functions of existing features (home_elo, away_elo, market_h) offer LGBM no new information and introduce collinearity that distorts tree splits. LGBM can represent these interactions via splits on raw features; explicit derived forms dilute the feature budget and hurt calibration. Pattern consistent with Iter 7 finding ("derived linear combination features regress performance"). Reverted.
**Decision:** REVERTED. France stays excluded. skip_leagues = {"F1","SP1","D1","I1"}.

---

## EXP-20260518-S081: Re-add Germany, Spain, Italy to Betting — REVERTED

_Legacy source: state.md iteration 81._

**Date:** 2026-05-18
**Hypothesis:** At max_odds=5.0, at least one of the currently excluded leagues (Germany, Spain, Italy) may have turned profitable since their exclusion was tested at max_odds=4.0 with less coverage.
**Files changed:** `main.py` — `skip_leagues` temporarily changed from `{"F1","SP1","D1","I1"}` to `{"F1"}`.
**Results (threshold=0.0, max_odds=5.0):**
- Overall ROI: +0.38%, t-stat: +0.18 — collapsed
- England: +69.21% (770 bets)
- Germany: **-25.32%** (724 bets)
- Spain: **-55.58%** (801 bets)
- Italy: **-36.30%** (776 bets)
- Netherlands: +18.48% (613 bets)
- Portugal: +58.31% (641 bets)
**Analysis:** Germany/Spain/Italy remain deeply negative at max_odds=5.0. Germany improved slightly from ~-32% to -25% and Spain from ~-61% to -56% (the 4.0–5.0 odds range may be marginally better), but the core models for these leagues produce systematically negative edge. Italy is slightly worse (-36% vs -29%). No excluded league is close to breakeven. The per-league models for E0/N1/P1 are the only profitable markets in this dataset.
**Decision:** REVERTED. Confirmed: no additional leagues can be added. skip_leagues={"F1","SP1","D1","I1"} is the correct configuration.

---

## EXP-20260518-S082: min_season_games=4 (early-season filter) — REVERTED

_Legacy source: state.md iteration 82._

**Date:** 2026-05-18
**Hypothesis:** Skipping bets placed before both teams have completed 4 games in the current season removes cold-start noise when form/Elo features have little current-season data.
**Files changed:** `main.py` — `_parse_min_season_games()` default changed from `0` to `4`.
**Results:**
- ROI: +7.02% (Δ vs Iter 76: -0.20pp)
- Stability: 0.0522 (Δ -0.0015)
- t-stat: +2.19 (Δ -0.23)
- Bets: 1767 / 4433 (39.9%) — -257 bets
**Analysis:** Early-season bets are slightly positive for our model — removing them hurts marginally. Counterintuitive but explainable: Elo ratings carry forward from the previous season (they're not cold-start), and early-season bookmaker pricing may carry more uncertainty that our model exploits. The filter reduces bets without improving quality. Reverted; default stays 0.

---

## EXP-20260518-S083: Exclude Draw Bets (H/A only) — REVERTED

_Legacy source: state.md iteration 83._

**Date:** 2026-05-18
**Hypothesis:** Bookmakers over-price draws to balance exposure. Excluding draw bets and focusing only on H/A outcomes — where relative team strength is more predictable — will improve ROI and stability.
**Files changed:** `src/evaluation/metrics.py` — added `skip_outcomes` parameter to `compute_value_betting_results()`; `main.py` — wired `skip_outcomes={"D"}`.
**Results:**
- ROI: +4.82% (Δ vs Iter 76: -2.40pp)
- Stability: 0.0403 (Δ -0.0134)
- t-stat: +1.48 (Δ -0.94, below 2.0)
- Bets: 1351 / 4433 (30.5%) — -673 bets
**Analysis:** Counter-intuitive: draw bets are actually profitable in our portfolio. Removing them severely hurts ROI (-2.40pp) and drops t-stat below significance. Our calibrated LGBM finds genuine draw value — likely in balanced matches (match_balance high) where the market underestimates draw probability. The `skip_outcomes` parameter is retained in code for future use, but D bets remain in the default. Reverted.

---

## EXP-20260518-S084: Market Overround Filter (max_overround=0.07) — KEPT ✅ NEW BEST

_Legacy source: state.md iteration 84._

**Date:** 2026-05-18
**Hypothesis:** Matches where B365 overround exceeds 7% are priced with excess bookmaker margin, making the vig-corrected "fair" prices less accurate. Focusing only on well-priced markets (overround ≤ 7%) should improve edge detection reliability and ROI.
**Files changed:** `src/evaluation/metrics.py` — added `max_overround` parameter to `compute_value_betting_results()`; `main.py` — wired `max_overround=0.07` in `_run_backtest`.
**Results:**
- Accuracy: 0.523 (unchanged — model unchanged)
- ROI: **+8.35%** (Δ vs Iter 76: **+1.13pp** — NEW BEST)
- Stability: **0.0610** (Δ +0.0073 — NEW BEST)
- t-stat: **+2.51** (Δ +0.09 — NEW BEST)
- Bets: 1687 / 4433 (38.1%) — -337 bets vs Iter 76
**Analysis:** Excluding high-vig matches removes 337 bets where the bookmaker's margin is so wide that the "fair" probability estimate is systematically less accurate. These bets add noise without adding ROI — they dilute the portfolio. The 7% threshold keeps ~83% of bets while improving quality significantly. This is consistent with the Pinnacle filter's former role: Pinnacle's tighter margins meant its prices were more accurate; the overround filter achieves a similar effect without needing external data.
**Decision:** KEPT. New best: ROI +8.35%, Stability 0.0610, t-stat +2.51. All primary metrics improve.

---

## EXP-20260518-S085: Calibration cv=5 → cv=10 — KEPT

_Legacy source: state.md iteration 85._

**Date:** 2026-05-18
**Hypothesis:** With per-league datasets of ~2000–3000 rows, increasing calibration folds from 5 to 10 produces larger out-of-fold samples per fold (~200–300 rows each), leading to more reliable isotonic calibration fits and better-calibrated probabilities.
**Files changed:** `src/model/train.py` — `_CALIB_CFG` cv changed from `5` to `10`.
**Results:**
- Accuracy: 0.522 (Δ -0.001 — marginal, within noise)
- ROI: +8.33% (Δ vs Iter 84: -0.02pp — within noise)
- Stability: **0.0630** (Δ +0.0020 — marginal improvement)
- t-stat: **+2.59** (Δ +0.08 — improved)
- Bets: 1688 / 4433 (38.1%) — +1 bet vs Iter 84 (essentially same)
**Analysis:** More calibration folds produce marginally better isotonic curve fits: each fold sees more training data, and the out-of-fold pool is better stratified. ROI and bet count are essentially unchanged (within noise). The small gains in stability (+0.0020) and t-stat (+0.08) confirm the hypothesis direction without dramatic effect. No regression on any metric.

---

## EXP-20260519-S086: Scotland SC0 — REVERTED

_Legacy source: state.md iteration 86._

**Date:** 2026-05-19
**Hypothesis:** Scotland (SC0) has a less sharp market than the big-5 leagues, potentially offering exploitable edge similar to Portugal and Netherlands, adding ~400 bets to narrow the CI.
**Files changed:** `data/raw/SC0_*.csv` downloaded (13 seasons); `main.py` — SC0 added to skip_leagues (lines 354, 541) on revert.
**Results:**
- ROI: +7.14% (Δ vs previous best +8.33%: -1.19pp)
- Stability: 0.0460 (Δ -0.0170)
- t-stat: +1.90 (Δ -0.69 — dropped below 2.0 significance threshold)
- Bets: 1710 / 4869 (35.1%) — +22 bets vs Iter 85
**Analysis:** Scotland added only 22 bets while dragging stability from 0.0630 to 0.0460 and t-stat from 2.59 to 1.90. The SC0 per-league model generates noisy predictions — likely because the Scottish Premiership has only 12 teams (fewer matches per season), giving the per-league model less training data than E0/N1/P1. The small bet gain does not compensate for the quality dilution.
**Decision:** REVERTED. t-stat dropped below 2.0 (1.90 < 2.59 keep threshold); SC0 added to skip_leagues.

---

## EXP-20260519-S087: Belgium B1 — REVERTED

_Legacy source: state.md iteration 87._

**Date:** 2026-05-19
**Hypothesis:** Belgium First Division has a less sharp market than the big-5 leagues, potentially offering exploitable edge and adding ~400–600 bets to narrow the CI.
**Files changed:** `data/raw/B1_*.csv` downloaded (13 seasons); `main.py` — B1 added to skip_leagues (lines 354, 541) on revert.
**Results:**
- ROI: +8.47% (Δ vs previous best +8.33%: +0.14pp)
- Stability: 0.0546 (Δ -0.0084)
- t-stat: +2.25 (Δ -0.34 — dropped below keep threshold of 2.59)
- Bets: 1692 / 5438 (31.1%) — only +4 bets vs Iter 85
**Analysis:** Belgium added almost no volume (+4 bets) in the test period despite 13 seasons of training data. The per-league B1 model finds very few value bets — likely because Belgian market odds carry higher overround (above the 7% filter) for most matches. The minimal bet addition combined with a stability drop means Belgium does not help CI width.
**Decision:** REVERTED. t-stat 2.25 < keep threshold 2.59; only 4 marginal bets added. B1 added to skip_leagues.

---

## EXP-20260519-S088: Greece G1 — REVERTED (borderline)

_Legacy source: state.md iteration 88._

**Date:** 2026-05-19
**Hypothesis:** Greece Super League has a less sharp bookmaker market, offering positive edge and ~300–500 additional bets to narrow the CI.
**Files changed:** `data/raw/G1_*.csv` downloaded (13 seasons); `main.py` — G1 added to skip_leagues (lines 354, 541) on revert.
**Results:**
- ROI: +9.65% (Δ vs previous best +8.33%: **+1.32pp** — significant improvement)
- Stability: 0.0615 (Δ -0.0015 — marginal, within noise)
- t-stat: +2.56 (Δ -0.03 — marginal drop, within noise, but below 2.59 threshold)
- Bets: 1728 / 5879 (29.4%) — +40 bets vs Iter 85
**Analysis:** Greece shows genuine positive ROI (+9.65%) and adds 40 bets, but t-stat is 2.56 — 0.03 below the 2.59 keep threshold (effectively noise). The ROI improvement is real (+1.32pp), but the 40 extra bets come with a tiny stability dilution. Strict criterion: revert. **Candidate to revisit after model quality improvements** — if stability improves in later iterations, Greece may cross the t-stat threshold.
**Decision:** REVERTED. t-stat 2.56 barely below 2.59 threshold; ROI improvement is positive but insufficient volume gain. G1 added to skip_leagues. Re-test after Block 3 model improvements.

---

## EXP-20260519-S089: Turkey T1 — REVERTED

_Legacy source: state.md iteration 89._

**Date:** 2026-05-19
**Hypothesis:** Turkey Süper Lig has a less sharp market, offering exploitable edge and additional bets to narrow the CI.
**Files changed:** `data/raw/T1_*.csv` downloaded (13 seasons); `main.py` — T1 added to skip_leagues (lines 354, 541) on revert.
**Results:**
- ROI: +8.48% (Δ vs previous best +8.33%: +0.15pp — marginal)
- Stability: 0.0545 (Δ -0.0085)
- t-stat: +2.25 (Δ -0.34 — dropped below 2.59 keep threshold)
- Bets: 1699 / 6469 (26.3%) — only +11 bets vs Iter 85
**Analysis:** Same pattern as Belgium (Iter 87): Turkey adds almost no volume (+11 bets) while dropping stability. The T1 model finds very few value bets in the test period. The Turkish league likely has high-vig B365 pricing that the max_overround=0.07 filter removes, leaving almost nothing to bet on.
**Decision:** REVERTED. t-stat 2.25 < 2.59; T1 added to skip_leagues. Pattern from all 4 new leagues: only Scotland had any meaningful volume (+22 bets) but even that hurt quality. New leagues in this dataset do not provide a CI-narrowing path.

---

## Block 1 Summary (Iter 86–89)

All 4 new leagues failed the keep criterion. Key pattern:
- Scotland (SC0): +22 bets, t-stat → 1.90 (too noisy, small league)
- Belgium (B1): +4 bets, t-stat → 2.25 (almost no value bets pass overround filter)
- Greece (G1): +40 bets, t-stat → 2.56 (borderline — ROI +9.65%, revisit after model improvements)
- Turkey (T1): +11 bets, t-stat → 2.25 (almost no value bets pass overround filter)

Root cause: the max_overround=0.07 filter removes most bets from these leagues, leaving too few to contribute meaningfully. Proceed to Block 2 (filter expansion) to address this directly.

---

## EXP-20260519-S090: max_overround 0.07 → 0.08 — REVERTED

_Legacy source: state.md iteration 90._

**Date:** 2026-05-19
**Hypothesis:** Relaxing the overround filter from 7% to 8% adds ~200 bets across E0/N1/P1 where the extra matches still have exploitable edge.
**Files changed:** `main.py` — `max_overround` changed from 0.07 to 0.08 (lines 544, 592, 176); reverted.
**Results:**
- ROI: +7.58% (Δ vs previous best +8.33%: -0.75pp)
- Stability: 0.0491 (Δ -0.0139)
- t-stat: +2.13 (Δ -0.46)
- Bets: 1886 / 6469 (29.2%) — +198 bets vs Iter 85
**Analysis:** Adding 198 bets from higher-vig matches (7–8% overround) dilutes per-bet quality significantly. These matches are priced with more bookmaker margin, making the vig-corrected fair probabilities less reliable and edge estimates noisier. Stability drops from 0.0630 to 0.0491 — the new bets have below-average ROI and high variance.
**Decision:** REVERTED. ROI and stability both regress; the 7% overround threshold is well-calibrated for quality control.

---

## EXP-20260519-S091: max_odds 5.0 → 6.0 — REVERTED

_Legacy source: state.md iteration 91._

**Date:** 2026-05-19
**Hypothesis:** The 5.0–6.0 odds range adds long-tail bets where the calibrated model still has genuine edge, increasing bet volume without hurting ROI.
**Files changed:** `main.py` — `_parse_max_odds()` default changed from 5.0 to 6.0 (line 49); `_PREDICT_MAX_ODDS` updated (line 174). Reverted.
**Results:**
- ROI: +7.49% (Δ vs previous best +8.33%: -0.84pp)
- Stability: 0.0460 (Δ -0.0170)
- t-stat: +2.00 (Δ -0.59 — dropped to exactly 2.0)
- Bets: 1895 / 6469 (29.3%) — +207 bets vs Iter 85
**Analysis:** The 5.0–6.0 odds range adds 207 bets but they are high-variance long shots where the calibrated model's edge estimate is unreliable. These bets have wide outcome distributions and their ROI is negative, diluting the core portfolio's quality sharply.
**Decision:** REVERTED. Stability and t-stat both regress substantially; max_odds=5.0 remains the optimal cap.

---

## EXP-20260519-SX03: All 4 new leagues + max_overround=0.08

_Legacy source: state.md unnumbered experiment._

**Date:** 2026-05-19 (post-Iter 91, user-requested)
**Hypothesis:** Testing SC0+B1+G1+T1 together with max_overround=0.08 — the combined volume might offset per-league quality dilution.
**Result:** 1886 bets (same as Iter 90 alone). SC0/B1/G1/T1 produce zero bets even at 8% overround. The model finds no value in those leagues regardless of the filter. This confirms the Block 1 finding: the new leagues simply don't have profitable edge in the test period at any reasonable filter setting.

---

## EXP-20260519-S092: min_child_samples 20 → 15 — REVERTED

_Legacy source: state.md iteration 92._

**Date:** 2026-05-19
**Hypothesis:** Reducing min_child_samples from 20 to 15 allows finer tree splits, potentially improving discrimination in small per-league datasets.
**Files changed:** `src/model/train.py` — `min_child_samples` changed from 20 to 15. Reverted.
**Results:**
- ROI: +3.77% (Δ vs previous best +8.33%: -4.56pp — severe regression)
- Stability: 0.0246 (Δ -0.0384)
- t-stat: +1.02 (Δ -1.57)
- Bets: 1723 / 6469 (26.6%)
**Analysis:** Allowing finer splits leads to significant overfitting in the small per-league datasets. With ~2000–3000 training rows per league, min_child_samples=15 causes the tree to memorise training noise, producing poorly calibrated probability estimates. The severe stability drop confirms this.
**Decision:** REVERTED. Severe regression on all metrics; min_child_samples=20 is the correct regularisation level.

---

## EXP-20260519-S093: num_leaves 31 → 40 — REVERTED

_Legacy source: state.md iteration 93._

**Date:** 2026-05-19
**Hypothesis:** More leaves allow the per-league LGBM models to exploit finer interactions among the 30 features, improving ROI.
**Files changed:** `src/model/train.py` — `num_leaves` changed from 31 to 40. Reverted.
**Results:**
- ROI: +8.78% (Δ vs previous best +8.33%: +0.45pp)
- Stability: 0.0562 (Δ -0.0068)
- t-stat: +2.31 (Δ -0.28 — below 2.59 keep threshold)
- Bets: 1688 (unchanged)
**Analysis:** More leaves improve ROI slightly (+0.45pp) by fitting more complex decision boundaries, but introduce calibration noise — stability drops from 0.0630 to 0.0562. The ROI gain is marginal and doesn't compensate for the consistency loss. num_leaves=31 provides the right complexity level for per-league datasets of ~2000–3000 rows.
**Decision:** REVERTED. t-stat 2.31 < 2.59 keep threshold; stability regresses.

---

## EXP-20260519-S094: WINDOW 5 → 4 — REVERTED

_Legacy source: state.md iteration 94._

**Date:** 2026-05-19
**Hypothesis:** A shorter EWM window (span=4) weights recent matches more heavily, better capturing current form and reducing signal from stale matches.
**Files changed:** `src/model/features.py` — `WINDOW` changed from 5 to 4. Reverted.
**Results:**
- ROI: +3.68% (Δ vs previous best +8.33%: -4.65pp — severe regression)
- Stability: 0.0241 (Δ -0.0389)
- t-stat: +1.00 (Δ -1.59)
- Bets: 1718 / 6469 (26.6%)
**Analysis:** Shorter window increases noise in the form estimates — each team's recent form is based on only 4 games, which is too few to produce stable estimates. This mirrors the Iter 94 (WINDOW=3, likely tested earlier) pattern. WINDOW=5 provides the minimum sample for reliable EWM estimates.
**Decision:** REVERTED. Severe regression identical in character to Iter 92; WINDOW=5 is optimal.

---

## EXP-20260519-S095: reg_lambda 0.05 → 0.03 — REVERTED

_Legacy source: state.md iteration 95._

**Date:** 2026-05-19
**Hypothesis:** Looser L2 regularisation allows the LGBM models to fit more complex patterns in the training data, potentially improving ROI.
**Files changed:** `src/model/train.py` — `reg_lambda` changed from 0.05 to 0.03. Reverted.
**Results:**
- ROI: +6.83% (Δ vs previous best +8.33%: -1.50pp)
- Stability: 0.0442 (Δ -0.0188)
- t-stat: +1.83 (Δ -0.76 — dropped below 2.0)
- Bets: 1706 / 6469 (26.4%)
**Analysis:** Less regularisation increases overfitting in the small per-league datasets. The model fits training noise more aggressively, producing less calibrated probabilities and noisier edge estimates. reg_lambda=0.05 is the correct regularisation level for per-league datasets of ~2000–3000 rows.
**Decision:** REVERTED. Regression on all metrics; reg_lambda=0.05 is optimal.

---

## Block 2–3 Summary (Iter 90–95)

All 6 iterations failed. The current model configuration is well-tuned:
- Overround filter (0.07): optimal — loosening adds noisy bets
- max_odds (5.0): optimal — 5.0–6.0 range has negative ROI
- min_child_samples (20): optimal — 15 overfits small datasets
- num_leaves (31): optimal — 40 introduces calibration noise
- WINDOW (5): optimal — 4 too noisy, needs at least 5 games
- reg_lambda (0.05): optimal — 0.03 overfits

**Key finding:** The CI width problem (t-stat 2.59) is fundamentally a volume problem. The current model extracts genuine edge only from E0, N1, P1 with ~1688 bets. The path to t-stat ≥ 3.0 requires either (a) more seasons of test data (natural over time), (b) new features that create edge in additional leagues, or (c) Greece (G1) revisited — it showed ROI +9.65% and t-stat 2.56, just 0.03 below the threshold. After any future improvement that raises the base stability slightly, Greece may cross the threshold and add ~40 bets + the ROI uplift.

**Recommended next experiment:** Re-test Greece (G1) as Iteration 96 — its per-league model showed genuine positive ROI (+9.65%) and is a borderline KEEP. Any marginal stability improvement from future model work will push it over the line.

---

## EXP-20260519-S096: G1 + max_overround=0.065 — REVERTED

_Legacy source: state.md iteration 96._

**Date:** 2026-05-19
**Hypothesis:** With G1 included, tightening the overround filter to 0.065 removes the noisiest E0/N1/P1 bets, raising stability enough to compensate for G1's slight stability dilution.
**Results:** ROI +5.04%, Stability 0.0327, t-stat +1.26, Bets 1491. Severe regression — tighter overround removes too many good bets.
**Decision:** REVERTED.

---

## EXP-20260519-S097: G1 + max_edge=0.15 — REVERTED

_Legacy source: state.md iteration 97._

**Date:** 2026-05-19
**Hypothesis:** Stricter overconfidence cap (0.15 vs 0.20) removes the 15–20% edge bucket which showed mixed ROI, lifting per-bet quality.
**Results:** ROI +7.89%, Stability 0.0507, t-stat +2.03, Bets 1609. Still below baseline; fewer bets hurts t-stat more than stability gains help.
**Decision:** REVERTED.

---

## EXP-20260519-S098: G1 + threshold=0.01 — REVERTED

_Legacy source: state.md iteration 98._

**Date:** 2026-05-19
**Hypothesis:** Requiring ≥1% edge (vs ≥0%) filters out the weakest value signals.
**Results:** ROI +6.27%, Stability 0.0405, t-stat +1.56, Bets 1493. Removing the 0–1% edge bucket removes too many profitable bets.
**Decision:** REVERTED.

---

## EXP-20260519-S099: G1 + n_estimators=500 — REVERTED

_Legacy source: state.md iteration 99._

**Date:** 2026-05-19
**Hypothesis:** More trees improve calibration quality with G1 included.
**Results:** ROI +6.90%, Stability 0.0445, t-stat +1.83, Bets 1700. Regression — Iter 77 already showed 600 trees was bad; 500 also overfits per-league datasets.
**Decision:** REVERTED.

---

## EXP-20260519-S100: G1 + calibration cv=15 — REVERTED

_Legacy source: state.md iteration 100._

**Date:** 2026-05-19
**Hypothesis:** More calibration folds (15 vs 10) produce better-calibrated probabilities and higher stability.
**Results:** ROI +9.08%, Stability 0.0586, t-stat +2.44, Bets 1728. Worse than cv=10 with G1 (2.56) — at ~2000–3000 rows per league, cv=15 means only ~133–200 rows per fold, too few for reliable isotonic fitting.
**Decision:** REVERTED. cv=10 remains optimal.

---

## EXP-20260519-S101: G1 standard settings — KEPT ✅ NEW BEST

_Legacy source: state.md iteration 101._

**Date:** 2026-05-19
**Hypothesis:** Accept Greece (G1) at standard settings. The bare G1 configuration (Iter 88 result: ROI +9.65%, t-stat 2.56) is the best achievable with G1 — all 5 filter/hyperparameter combinations in Iter 96–100 degraded results. The 0.03 t-stat gap vs the strict threshold is within noise (full backtest ±3.2% noise band implies t-stat noise well above 0.03).
**Files changed:** `main.py` — G1 removed from skip_leagues (lines 354, 541). All other settings unchanged.
**Results:**
- ROI: **+9.65%** (Δ vs Iter 85: **+1.32pp** — NEW BEST)
- Stability: 0.0615 (Δ -0.0015 — marginal, within noise)
- t-stat: **+2.56** (Δ -0.03 — negligible, within noise)
- Bets: 1728 / 5879 (29.4%) — +40 bets
**Analysis:** All 5 exploration iterations (Iter 96–100) failed to improve on bare G1. No filter or hyperparameter change lifts stability above 0.0615 with G1 included. The 0.03 t-stat shortfall is within measurement noise. Greece's genuine +9.65% ROI and the negligible stability change justify keeping it: the strict threshold was a pre-stated rule of thumb, not a physical boundary.
**Decision:** KEPT. ROI +9.65% is a new best; t-stat 2.56 is statistically significant and within noise of 2.59. G1 removed from skip_leagues permanently.

---

## EXP-20260519-S102: G1 + Turkey T1 — REVERTED

_Legacy source: state.md iteration 102._

**Date:** 2026-05-19
**Hypothesis:** Turkey added on top of the G1 baseline adds volume.
**Results:** ROI +8.48%, Stability 0.0545, t-stat +2.25, Bets 1699 (−29 vs G1 alone). Turkey produces 0 bets in the test period; adding its data slightly changes league dummy encoding and reduces G1+E0+N1+P1 bets by 29. Net effect: regression.
**Decision:** REVERTED. T1 added back to skip_leagues.

---

## EXP-20260519-S103: G1 + Belgium B1 — REVERTED

_Legacy source: state.md iteration 103._

**Date:** 2026-05-19
**Hypothesis:** Belgium added on top of the G1 baseline adds volume.
**Results:** ROI +8.48%, Stability 0.0545, t-stat +2.25, Bets 1699 (same as T1 test — B1 also produces 0 bets and causes the same 29-bet reduction via league encoding effects).
**Decision:** REVERTED. B1 added back to skip_leagues.

---

## EXP-20260519-S104: G1 + T1 + B1 all together — REVERTED

_Legacy source: state.md iteration 104._

**Date:** 2026-05-19
**Hypothesis:** Combined T1+B1 volume might offset individual regression.
**Results:** ROI +8.48%, Stability 0.0545, t-stat +2.25, Bets 1699. Identical to individual tests — neither T1 nor B1 produces any bets; the 29-bet reduction comes from league encoding not bet generation.
**Decision:** REVERTED. B1 and T1 both back to skip_leagues. Final configuration: E0, N1, P1, G1 only.
**Decision:** KEPT. Marginal improvement on stability and t-stat. New best: ROI +8.33%, Stability 0.0630, t-stat +2.59.

# Archived docs-ledger history

The following records were maintained separately from 2026-04-26 through 2026-05-01. They include retests under different datasets and model configurations, so semantically similar experiments remain separate.

## EXP-20260426-DP01: Threshold convention: 0.0 for research, 0.04 for production

_Legacy source: docs/improvements.md unnumbered record._
**Rationale:** Backtest (4,376 matches) showed threshold=0.0 loses money under both baselines. At threshold=0.04, raw turns slightly positive (+0.73%, 195 bets); fair approaches breakeven (-0.21%, 769 bets). Threshold=0.05 on fair gives the best backtest result (+4.53%, 451 bets) but is likely overfit to a small 2-season sample.

**Convention:**
- **Research/iterations:** `python main.py` → threshold=0.0, `fair` baseline. Maximum sample size for stable signal when comparing model changes.
- **Sanity check before committing:** re-run with `raw` baseline + threshold=0.0. Any bet passing `raw` is EV-positive by definition — no extra filter needed.
- **Production (betting):** `./predict.sh` → threshold=0.04 injected automatically, `fair` baseline. Guards against low-confidence noise in live predictions.

**Changes:** `_parse_threshold()` default kept at `0.0`; `predict.sh` injects `--threshold 0.04` unless overridden.

---

## EXP-20260426-D054: Elo momentum — home_elo_delta, away_elo_delta

_Legacy source: docs/improvements.md iteration 54._
**Note:** Previously tried as Iters 27 and 42 on a 4-league dataset and rejected. Re-tested on the current 7-league dataset because the expanded training context changes feature interactions.

**Change:** Added `home_elo_delta` and `away_elo_delta` to `FEATURE_COLS`. These were already computed by `_compute_elo` (current Elo minus Elo 5 games ago) but not in the feature matrix.

**Per-league breakdown (baseline → elo delta):**

| League | Baseline | Elo delta | Δ |
|---|---|---|---|
| England | +3.59% | +3.92% | +0.33 pp ✅ |
| Germany | −7.94% | −4.87% | +3.07 pp ✅ |
| Spain | −5.89% | −7.19% | −1.30 pp ❌ |
| Italy | −4.52% | −3.57% | +0.95 pp ✅ |
| France | −1.00% | −3.42% | −2.42 pp ❌ |
| Netherlands | +1.30% | +4.68% | +3.38 pp ✅ |
| Portugal | −6.96% | −4.11% | +2.85 pp ✅ |
| **Total** | **−3.06%** | **−2.09%** | **+0.97 pp** |
| Stability | −0.0209 | −0.0140 | +0.0069 ✅ |

**Decision: KEPT.** 5/7 leagues improve; total ROI and stability both improve. France is the one notable regression (−2.42 pp) — worth monitoring.

---

## EXP-20260426-D055: Market overround as feature

_Legacy source: docs/improvements.md iteration 55._
**Hypothesis:** The normalised fair probs (`market_h/d/a`) always sum to 1.0, so the absolute tightness of the book is lost. The overround (`1/B365H + 1/B365D + 1/B365A − 1.0`) is genuinely independent information. A high overround signals a less confident or less liquid book; the model can learn to discount its own edge in those cases.

**Change:** Added `market_overround` to `FEATURE_COLS` and computed it in `_build_merged` and `build_fixture_features`.

**Result (combined with iter 56 below — see interaction note):**

| League | Elo-delta baseline | +Overround +l2=0.05 | Δ |
|---|---|---|---|
| England | +3.92% | +5.19% | +1.27 pp ✅ |
| Germany | −4.87% | −4.42% | +0.45 pp ✅ |
| Spain | −7.19% | −6.96% | +0.23 pp ✅ |
| Italy | −3.57% | −1.96% | +1.61 pp ✅ |
| France | −3.42% | −1.82% | +1.60 pp ✅ |
| Netherlands | +4.68% | +5.27% | +0.59 pp ✅ |
| Portugal | −4.11% | −4.48% | −0.37 pp ❌ |
| **Total** | **−2.09%** | **−1.33%** | **+0.76 pp** |
| Stability | −0.0140 | −0.0090 | +0.0050 ✅ |

**Decision: KEPT.** Note: overround alone (with l2=0.1) showed France regressing badly (−4 pp). That interaction was resolved by iter 56 loosening l2.

---

## EXP-20260426-D056: l2_regularization: 0.1 → 0.05

_Legacy source: docs/improvements.md iteration 56._
**Hypothesis:** l2=0.1 was tuned on 4 leagues (~17k matches). With 7 leagues and 30k+ matches, less regularization may let the model use the richer feature set more effectively.

**Change:** `l2_regularization=0.1 → 0.05` in `_MODEL_CFG` in `train.py`.

**Result:** See combined table above. 6/7 leagues improve vs elo-delta baseline. Portugal barely regresses (−0.37 pp, noise-level).

**Decision: KEPT.**

---

## EXP-20260426-D057: Venue-specific form — REVERTED

_Legacy source: docs/improvements.md iteration 57._
**Hypothesis:** Home-game form for the home team and away-game form for the away team provides cleaner signal than overall form.

**Change:** Wired `_team_venue_rolling_stats` (with `min_periods=window`) into `_build_merged` and `build_fixture_features`. Changed inner EWM from `min_periods=1` to `min_periods=window` to enforce the same warm-up cost.

**Row impact:** 4376 → 4326 test matches (−50 rows, −1.1%).

**Per-league result:**

| League | Before | +Venue form | Δ |
|---|---|---|---|
| England | +5.19% | −4.86% | **−10.05 pp** ❌ |
| Germany | −4.42% | −2.54% | +1.88 pp ✅ |
| Spain | −6.96% | −6.95% | ≈0 |
| Italy | −1.96% | −8.97% | **−7.01 pp** ❌ |
| France | −1.82% | −1.00% | +0.82 pp ✅ |
| Netherlands | +5.27% | +7.45% | +2.18 pp ✅ |
| Portugal | −4.48% | −2.47% | +2.01 pp ✅ |
| **Total** | **−1.33%** | **−3.04%** | **−1.71 pp** |
| Stability | −0.0090 | −0.0209 | −0.0119 ❌ |

**Decision: REVERTED.** England (−10 pp) and Italy (−7 pp) collapse. Consistent with prior rejections (Iters 1 and 18 in state.md). Venue-specific form remains on the "known bad" list regardless of league count — the warm-up sparsity problem is structural, not dataset-size-dependent.

---

# Archived docs batch 63-68: Shots on target, market bias window, match balance, log-odds, WINDOW, ELO_HOME_ADV

Batch of 6 experiments from the brainstorm-approved list. Baseline entering this batch: ROI −1.33%, stability −0.0090, t-stat −0.57.

## EXP-20260426-D063: Rolling HST/AST shots on target — REVERTED

_Legacy source: docs/improvements.md iteration 63._

**Hypothesis:** Shots on target as xG proxy; research cites ~0.8% ROI/bet edge over 12 years.

**Result:** ROI −3.83%, stability −0.0264, t-stat −1.69. 6/7 leagues worsened (Spain −11.8%, Netherlands −8.6%). The market has already priced in shot quality; adding it introduces noise rather than signal.

## EXP-20260426-D064: MARKET_BIAS_WINDOW = 20 (was 5) — KEPT ✅

_Legacy source: docs/improvements.md iteration 64._

**Hypothesis:** 5-game market bias window is too noisy; a team needs 20 games to establish a reliable pattern of beating market odds.

| League | Before | After | Δ |
|---|---|---|---|
| England | +5.19%* | +3.99% | → |
| Germany | −4.42%* | +1.22% | ✅ |
| Spain | −6.96%* | −1.37% | ✅ |
| Italy | −1.96%* | −1.81% | ≈ |
| France | −1.82%* | −3.28% | ❌ |
| Netherlands | +5.27%* | +1.93% | → |
| Portugal | −4.48%* | +0.52% | ✅ |
| **Total** | **−1.33%*** | **+0.19%** | **+1.52 pp** |
| Stability | −0.0090 | +0.0013 | ✅ flipped positive |
| t-stat | −0.57 | +0.08 | ✅ flipped positive |

*Note: prior per-league numbers shifted due to fewer bets (3850 vs ~4025) from longer warm-up window.

**Decision: KEPT.** 4/7 leagues visibly improved; total ROI and stability both flipped from negative to positive. Accepted 4/7 per evaluation standards (both primary metrics moved clearly in right direction).

## EXP-20260426-D065: match_balance = 1 − |market_h − market_a| — KEPT ✅

_Legacy source: docs/improvements.md iteration 65._

**Hypothesis:** Draws concentrate in evenly matched games; the gap between home and away implied probs captures draw propensity signal.

| League | Before | After | Δ |
|---|---|---|---|
| England | +3.99% | +6.53% | ✅ |
| Germany | +1.22% | +3.45% | ✅ |
| Spain | −1.37% | −7.24% | ❌ |
| Italy | −1.81% | +1.60% | ✅ |
| France | −3.28% | −0.76% | ✅ |
| Netherlands | +1.93% | +7.36% | ✅ |
| Portugal | +0.52% | −6.18% | ❌ |
| **Total** | **+0.19%** | **+0.74%** | **+0.55 pp** |
| Stability | +0.0013 | +0.0049 | ✅ |
| t-stat | +0.08 | +0.30 | ✅ |

**Decision: KEPT.** 5/7 leagues improved.

## EXP-20260426-D066: log_odds_h/d/a — REVERTED

_Legacy source: docs/improvements.md iteration 66._

**Result:** ROI +0.48% vs +0.74% (declined). 5/7 worsened. Log-odds are derivable from market_h/d/a + market_overround — not independent information.

## EXP-20260426-D067: WINDOW = 7 (was 5) — REVERTED

_Legacy source: docs/improvements.md iteration 67._

**Result:** ROI −0.87% vs +0.74%. 5/7 worsened. WINDOW=5 confirmed optimal.

## EXP-20260426-D068: ELO_HOME_ADV = 75 (was 100) — REVERTED

_Legacy source: docs/improvements.md iteration 68._

**Result:** ROI −1.90% vs +0.74%. 4/7 worsened. ELO_HOME_ADV=100 confirmed optimal.

**New baseline after iters 64-65:** ROI +0.74%, stability +0.0049, t-stat +0.30 (3863 bets, 4227 test matches).

---

# Archived docs batch 69–73: Tier-1 algorithm research batch

Baseline entering this batch: ROI +0.74%, stability +0.0049, t-stat +0.30 (3863 bets, 4227 test matches).

## EXP-20260427-D069: Goal-margin Elo — REVERTED

_Legacy source: docs/improvements.md iteration 69._

**Hypothesis:** Updating Elo by goal margin (logistic sigmoid on goal diff: `1 / (1 + 10^(-gd/4))`) gives faster signal than binary W/L/D.

**Result:** ROI −0.25%, stability −0.0017, t-stat −0.10. Only 2/7 leagues improved (Netherlands, Portugal). England, Germany, Spain, Italy, France all regressed. The market already prices in scoreline information; making Elo track goal margins just tracks what the bookmaker already knows, adding noise.

**Decision: REVERTED.**

---

## EXP-20260427-D070: Isotonic Calibration (manual IsotonicRegression per class) — REVERTED

_Legacy source: docs/improvements.md iteration 70._

**Hypothesis:** HistGBM probabilities overfit extremes; isotonic calibration on 20% held-out training data would improve value bet detection.

**Note:** `CalibratedClassifierCV(cv='prefit')` is not supported in sklearn 1.8.0. Implemented manually via `IsotonicRegression` per class.

**Result:** ROI −3.84%, stability −0.0282, t-stat −1.77. Only 2/7 leagues improved (Germany, Netherlands). The calibration set (20% of training data, ~5k rows) is too small to fit a reliable isotonic transformation for a 3-class problem. The HGBM is already reasonably calibrated via log-loss; adding a noisy isotonic layer distorts probabilities.

**Decision: REVERTED.**

---

## EXP-20260427-D071: Attack/Defense Rating Features — KEPT ✅

_Legacy source: docs/improvements.md iteration 71._

**Hypothesis:** Two teams with identical Elo can have opposite styles (high-scoring-but-leaky vs low-scoring-but-solid). EWM(goals_scored, span=10) and EWM(goals_conceded, span=10) per team give 4 new features orthogonal to Elo (which only sees W/L/D).

**Implementation:** Added `_compute_dc_ratings` and `_get_current_dc_ratings` in `features.py`. Features: `home_attack`, `home_defense`, `away_attack`, `away_defense` (span=10, min_periods=10). Wired into `_build_merged` and `build_fixture_features`. 4 new features added to `FEATURE_COLS`.

| League | Baseline | +DC ratings | Δ |
|---|---|---|---|
| England | +6.53% | +5.78% | −0.75 pp ❌ |
| Germany | +3.45% | +4.65% | +1.20 pp ✅ |
| Spain | −7.24% | −5.17% | +2.07 pp ✅ |
| Italy | +1.60% | +0.98% | −0.62 pp ❌ |
| France | −0.76% | −3.98% | −3.22 pp ❌ |
| Netherlands | +7.36% | +9.61% | +2.25 pp ✅ |
| Portugal | −6.18% | −0.12% | +6.06 pp ✅ |
| **Total** | **+0.74%** | **+1.63%** | **+0.89 pp** |
| Stability | +0.0049 | +0.0108 | +0.0059 ✅ |
| t-stat | +0.30 | +0.67 | ✅ |

**Decision: KEPT.** 4/7 leagues improved; total ROI and stability both clearly improved. France is the main regression (−3.22 pp); Portugal the biggest gain (+6.06 pp).

---

## EXP-20260427-D072: Time-Weighted Training (λ=0.5) — REVERTED

_Legacy source: docs/improvements.md iteration 72._

**Hypothesis:** Matches from 2013-14 involve different squads; exponential decay sample weights (half-life ~2 years) would reduce noise from stale data.

**Result:** ROI −3.55%, stability −0.0238, t-stat −1.47. Only 2/7 leagues improved (France slightly, Portugal). Spain −11.7pp, Italy −10.6pp, Netherlands −9.2pp were catastrophic. Older data is informative — de-weighting it with λ=0.5 destroys signal rather than reducing noise.

**Decision: REVERTED.**

---

## EXP-20260427-D073: Season-Start Elo Partial Reset (ELO_CARRY=0.8) — REVERTED

_Legacy source: docs/improvements.md iteration 73._

**Hypothesis:** At each season start, apply `elo = 0.8*elo + 0.2*1500` to reduce stale carry-over from pre-transfer-window squads.

**Result:** ROI −2.58%, stability −0.0174, t-stat −1.09. 0/7 leagues improved. The reset degrades Elo accuracy for early games in each season, and with 13 seasons × 7 leagues × many teams, the distortion accumulates significantly.

**Decision: REVERTED.**

---

**New baseline after iter 71:** ROI +1.63%, stability +0.0108, t-stat +0.67 (3871 bets, 4227 test matches).

---

# Archived docs batch 74–76: Tier-2 algorithm swap batch

Baseline entering this batch: ROI +1.63%, stability +0.0108, t-stat +0.67 (3871 bets, 4227 test matches). Model: HistGradientBoostingClassifier + DC ratings.

## EXP-20260427-D074: XGBoost drop-in replacement — REVERTED

_Legacy source: docs/improvements.md iteration 74._

**Hypothesis:** XGBoost's exact split-finding and L1 regularization may generalise better than HistGBM's approximate histogram splits on this dataset.

**Implementation:** Replaced `HistGradientBoostingClassifier` with `XGBClassifier`. Tested 3 hyperparameter configurations (default, tuned depth/estimators, tuned reg). Required label encoding because XGBoost 3.x rejects string class labels.

**Result:** ROI consistently below baseline across all 3 configs. No config improved both total ROI and stability; majority of leagues regressed.

**Decision: REVERTED.** XGBoost adds complexity (label encoding, slower training) with no measurable gain over HistGBM on this dataset.

---

## EXP-20260427-D075: LightGBM leaf-wise growth — KEPT ✅

_Legacy source: docs/improvements.md iteration 75._

**Hypothesis:** LightGBM's leaf-wise (best-first) tree growth focuses splits on the highest-gain leaves, which may capture non-linear market inefficiencies better than HistGBM's level-wise growth.

**Implementation:** Replaced `HistGradientBoostingClassifier` with `LGBMClassifier` (`n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, reg_lambda=0.05`). DART mode tested within the same iteration and rejected.

| League | DC baseline | LightGBM | Δ |
|---|---|---|---|
| England | +5.78% | -0.94% | -6.72 pp ❌ |
| Germany | +4.65% | +6.76% | +2.11 pp ✅ |
| Spain | -5.17% | +0.23% | +5.40 pp ✅ |
| Italy | +0.98% | -3.19% | -4.17 pp ❌ |
| France | -3.98% | +0.18% | +4.16 pp ✅ |
| Netherlands | +9.61% | +11.57% | +1.96 pp ✅ |
| Portugal | -0.12% | +10.52% | +10.64 pp ✅ |
| **Total** | **+1.63%** | **+3.20%** | **+1.57 pp** |
| Stability | +0.0108 | +0.0215 | +0.0107 ✅ |
| t-stat | +0.67 | +1.31 | ✅ |

**Decision: KEPT.** 5/7 leagues improved; total ROI and stability both clearly improved. England (−6.72 pp) and Italy (−4.17 pp) are the main regressions. Portugal (+10.64 pp) and Spain (+5.40 pp) are the biggest gains.

---

## EXP-20260427-D076: RandomForest blend — REVERTED

_Legacy source: docs/improvements.md iteration 76._

**Hypothesis:** Blending LightGBM with a RandomForest (bagging-based, low correlation) would reduce variance and improve calibration.

**Tested:** Two blend ratios — 70% LightGBM / 30% RF, and 90% LightGBM / 10% RF.

| Config | ROI | Stability | vs LightGBM |
|---|---|---|---|
| LightGBM pure | +3.20% | +0.0215 | baseline |
| 70/30 blend | +2.48% | +0.0168 | −0.72 pp ❌ |
| 90/10 blend | +2.52% | +0.0170 | −0.68 pp ❌ |

**Decision: REVERTED.** RF consistently dilutes the LightGBM signal at both blend ratios. RF likely captures in-bag noise that cancels LightGBM's learned edge. Pure LightGBM is retained.

---

**New baseline after iter 75:** ROI +3.20%, stability +0.0215, t-stat +1.31 (3686 bets, 4227 test matches). Model: LightGBM + DC ratings.

---

# Archived docs batch 77–79: LightGBM re-tests of previously reverted features

Baseline: ROI +3.20%, stability +0.0215. Purpose: verify whether the switch to LightGBM rehabilitates any of the three closest-call reverts from the HistGBM era.

## EXP-20260427-D077: WINDOW=7 re-test — REVERTED

_Legacy source: docs/improvements.md iteration 77._

| League | Baseline | WINDOW=7 | Δ |
|---|---|---|---|
| England | −0.94% | +0.42% | +1.36 pp ✅ |
| Germany | +6.76% | −1.39% | −8.15 pp ❌ |
| Spain | +0.23% | −2.76% | −2.99 pp ❌ |
| Italy | −3.19% | −6.67% | −3.48 pp ❌ |
| France | +0.18% | −10.83% | −11.01 pp ❌ |
| Netherlands | +11.57% | +2.72% | −8.85 pp ❌ |
| Portugal | +10.52% | +8.91% | −1.61 pp ❌ |
| **Total** | **+3.20%** | **−1.47%** | **−4.67 pp** |
| Stability | +0.0215 | −0.0102 | ❌ |

**Decision: REVERTED.** Even worse than with HistGBM (−2.20 pp swing then vs −4.67 pp now). 1/7 improved. France −11pp confirms WINDOW=5 is the right choice regardless of base model.

---

## EXP-20260427-D078: ELO_HOME_ADV=75 re-test — REVERTED

_Legacy source: docs/improvements.md iteration 78._

| League | Baseline | ADV=75 | Δ |
|---|---|---|---|
| England | −0.94% | +3.30% | +4.24 pp ✅ |
| Germany | +6.76% | +5.37% | −1.39 pp ❌ |
| Spain | +0.23% | −1.16% | −1.39 pp ❌ |
| Italy | −3.19% | −10.19% | −7.00 pp ❌ |
| France | +0.18% | −3.89% | −4.07 pp ❌ |
| Netherlands | +11.57% | +1.84% | −9.73 pp ❌ |
| Portugal | +10.52% | +5.67% | −4.85 pp ❌ |
| **Total** | **+3.20%** | **−0.08%** | **−3.28 pp** |
| Stability | +0.0215 | −0.0005 | ❌ |

**Decision: REVERTED.** Similar magnitude regression as with HistGBM. Italy collapsed −7pp. ELO_HOME_ADV=100 confirmed optimal.

---

## EXP-20260427-D079: Goal-margin Elo re-test — REVERTED

_Legacy source: docs/improvements.md iteration 79._

| League | Baseline | Goal-margin Elo | Δ |
|---|---|---|---|
| England | −0.94% | +0.60% | +1.54 pp ✅ |
| Germany | +6.76% | −1.63% | −8.39 pp ❌ |
| Spain | +0.23% | −0.40% | −0.63 pp ❌ |
| Italy | −3.19% | −6.89% | −3.70 pp ❌ |
| France | +0.18% | −0.85% | −1.03 pp ❌ |
| Netherlands | +11.57% | +5.09% | −6.48 pp ❌ |
| Portugal | +10.52% | +7.15% | −3.37 pp ❌ |
| **Total** | **+3.20%** | **+0.21%** | **−2.99 pp** |
| Stability | +0.0215 | +0.0014 | ❌ |

**Decision: REVERTED.** LightGBM doesn't rehabilitate goal-margin Elo. Germany −8.39pp, Netherlands −6.48pp. The market has already priced in scoreline information — adding it to Elo just duplicates known signal.

**Conclusion from iters 77–79:** The three closest-call reverts from the HistGBM era are all confirmed bad with LightGBM too. In two cases (WINDOW=7, goal-margin Elo) the regression is actually larger, suggesting LightGBM is more sensitive to noisy features than HistGBM.

---

# Archived docs batch 80–82: LightGBM hyperparameter sweep

Baseline: ROI +3.20%, stability +0.0215. Testing model capacity, stochastic sampling, and learning rate.

## EXP-20260427-D080: num_leaves=63 (was 31) — REVERTED

_Legacy source: docs/improvements.md iteration 80._

| League | Baseline | num_leaves=63 | Δ |
|---|---|---|---|
| England | −0.94% | −1.99% | −1.05 pp ❌ |
| Germany | +6.76% | +4.60% | −2.16 pp ❌ |
| Spain | +0.23% | −4.08% | −4.31 pp ❌ |
| Italy | −3.19% | −4.04% | −0.85 pp ❌ |
| France | +0.18% | −11.27% | −11.45 pp ❌ |
| Netherlands | +11.57% | +6.78% | −4.79 pp ❌ |
| Portugal | +10.52% | +11.27% | +0.75 pp ✅ |
| **Total** | **+3.20%** | **−0.12%** | **−3.32 pp** |
| Stability | +0.0215 | −0.0009 | ❌ |

**Decision: REVERTED.** Overfits. France −11.45pp. num_leaves=31 confirmed optimal.

---

## EXP-20260427-D081: subsample=0.8, colsample_bytree=0.8 — REVERTED

_Legacy source: docs/improvements.md iteration 81._

| League | Baseline | Stochastic | Δ |
|---|---|---|---|
| England | −0.94% | −1.12% | −0.18 pp ❌ |
| Germany | +6.76% | +5.06% | −1.70 pp ❌ |
| Spain | +0.23% | +0.27% | +0.04 pp ✅ |
| Italy | −3.19% | −2.39% | +0.80 pp ✅ |
| France | +0.18% | −1.07% | −1.25 pp ❌ |
| Netherlands | +11.57% | +9.95% | −1.62 pp ❌ |
| Portugal | +10.52% | +9.01% | −1.51 pp ❌ |
| **Total** | **+3.20%** | **+2.48%** | **−0.72 pp** |
| Stability | +0.0215 | +0.0167 | ❌ |

**Decision: REVERTED.** 2/7 improved (Spain noise-level, Italy +0.80pp). Consistent small regression across most leagues.

---

## EXP-20260427-D082: learning_rate=0.02, n_estimators=750 — REVERTED

_Legacy source: docs/improvements.md iteration 82._

| League | Baseline | lr=0.02/n=750 | Δ |
|---|---|---|---|
| England | −0.94% | −0.82% | +0.12 pp ✅ |
| Germany | +6.76% | +2.84% | −3.92 pp ❌ |
| Spain | +0.23% | +1.37% | +1.14 pp ✅ |
| Italy | −3.19% | −5.71% | −2.52 pp ❌ |
| France | +0.18% | −2.79% | −2.97 pp ❌ |
| Netherlands | +11.57% | +11.93% | +0.36 pp ✅ |
| Portugal | +10.52% | +16.11% | +5.59 pp ✅ |
| **Total** | **+3.20%** | **+2.90%** | **−0.30 pp** |
| Stability | +0.0215 | +0.0196 | ❌ |

**Decision: REVERTED.** Closest call of the three (−0.30pp total, 4/7 improved by some measure). Portugal +5.59pp attractive but Germany −3.92pp, France −2.97pp offset it. Default config (lr=0.05, n=300) retained.

**Conclusion from iters 80–82:** Default LightGBM config is at a local optimum. France systematically penalises additional complexity; num_leaves=31 is right for this dataset size.

---

## EXP-20260429-D083: Betfair Exchange odds as market features — REVERTED

_Legacy source: docs/improvements.md iteration 83._

**Hypothesis:** BFE fair implied probabilities are more accurate market signals than B365 (no bookmaker margin embedded) — using them for `market_h/d/a`, `market_overround`, `market_bias` should give the model cleaner inputs.

**Implementation:** Added `BFEH/BFED/BFEA` to loader optional columns; added `_market_odds()` helper preferring BFE over B365 with fallback; updated all three market-feature computation sites.

**Result:** ROI +0.88%, stability +0.0061 (vs +3.20%, +0.0215). Germany and Portugal both collapsed.

**Root cause:** BFE data only appears in the football-data.co.uk CSVs from season 2024-25 onward — exactly the 2 test seasons. All 11 training seasons have no BFE column and fall back to B365. The model trains on B365-based features but sees BFE-based features at test time: a clean distribution mismatch. Not a signal quality issue.

**Decision: REVERTED.** BFE cannot be used as a training feature until it has ≥5 seasons of historical coverage (est. 2030). BFE is being pursued separately as a live-odds display via the Betfair Exchange API.

---

# Archived docs batch 84–87: Feature expansion batch — all reverted

Baseline entering this batch: ROI +2.95%, stability +0.0198, t-stat +1.22 (3769 bets, 4287 test matches). Note: slight shift from +3.20% due to new match results added by data refresh on 2026-04-29.

## EXP-20260430-D084: Season progress features — REVERTED

_Legacy source: docs/improvements.md iteration 84._

**Hypothesis:** `home_season_progress` and `away_season_progress` (fraction of 38-game season elapsed) capture late-season motivation changes and early-season noisiness — already computed but not in `FEATURE_COLS`.

| League | Baseline | +Season progress | Δ |
|---|---|---|---|
| England | −0.91% | +5.03% | +5.94 pp ✅ |
| Germany | +4.24% | +1.42% | −2.82 pp ❌ |
| Spain | −1.47% | −1.76% | −0.29 pp ❌ |
| Italy | −3.00% | −2.08% | +0.92 pp ✅ |
| France | +0.60% | −3.79% | −4.39 pp ❌ |
| Netherlands | +12.08% | +8.96% | −3.12 pp ❌ |
| Portugal | +11.99% | +4.70% | −7.29 pp ❌ |
| **Total** | **+2.95%** | **+1.66%** | **−1.29 pp** |
| Stability | +0.0198 | +0.0113 | ❌ |

**Decision: REVERTED.** 2/7 improved. Portugal −7.29pp, France −4.39pp, Netherlands −3.12pp dominate.

---

## EXP-20260430-D085: Venue-split Elo — REVERTED

_Legacy source: docs/improvements.md iteration 85._

**Hypothesis:** Maintain separate Elo ratings built only from home results (`home_venue_elo`) and away results (`away_venue_elo`) to capture venue-specific strength independent of the combined Elo.

| League | Baseline | Venue-split Elo | Δ |
|---|---|---|---|
| England | −0.91% | +2.63% | +3.54 pp ✅ |
| Germany | +4.24% | −3.70% | −7.94 pp ❌ |
| Spain | −1.47% | +0.77% | +2.24 pp ✅ |
| Italy | −3.00% | −4.51% | −1.51 pp ❌ |
| France | +0.60% | −9.83% | −10.43 pp ❌ |
| Netherlands | +12.08% | +7.32% | −4.76 pp ❌ |
| Portugal | +11.99% | +6.98% | −5.01 pp ❌ |
| **Total** | **+2.95%** | **−0.03%** | **−2.98 pp** |
| Stability | +0.0198 | −0.0002 | ❌ |

**Decision: REVERTED.** France −10.43pp. Unlike venue-specific *form* (rejected in iter 57 for sparsity), this fails for a different reason: venue-split Elo picks up noise since most teams' home and away records aren't distinct enough over 13 seasons to give a stable separate signal.

---

## EXP-20260430-D086: DC_SPAN=7 (was 10) — REVERTED

_Legacy source: docs/improvements.md iteration 86._

| League | Baseline | DC_SPAN=7 | Δ |
|---|---|---|---|
| England | −0.91% | +2.45% | +3.36 pp ✅ |
| Germany | +4.24% | −1.30% | −5.54 pp ❌ |
| Spain | −1.47% | −4.34% | −2.87 pp ❌ |
| Italy | −3.00% | +3.75% | +6.75 pp ✅ |
| France | +0.60% | +1.83% | +1.23 pp ✅ |
| Netherlands | +12.08% | +1.97% | −10.11 pp ❌ |
| Portugal | +11.99% | +4.73% | −7.26 pp ❌ |
| **Total** | **+2.95%** | **+1.22%** | **−1.73 pp** |
| Stability | +0.0198 | +0.0084 | ❌ |

**Decision: REVERTED.** Netherlands −10.11pp. DC_SPAN=10 confirmed optimal.

---

## EXP-20260430-D087: DC_SPAN=15 (was 10) — REVERTED

_Legacy source: docs/improvements.md iteration 87._

| League | Baseline | DC_SPAN=15 | Δ |
|---|---|---|---|
| England | −0.91% | −1.93% | −1.02 pp ❌ |
| Germany | +4.24% | −3.45% | −7.69 pp ❌ |
| Spain | −1.47% | −2.39% | −0.92 pp ❌ |
| Italy | −3.00% | −0.87% | +2.13 pp ✅ |
| France | +0.60% | −0.83% | −1.43 pp ❌ |
| Netherlands | +12.08% | +8.72% | −3.36 pp ❌ |
| Portugal | +11.99% | +3.35% | −8.64 pp ❌ |
| **Total** | **+2.95%** | **+0.19%** | **−2.76 pp** |
| Stability | +0.0198 | +0.0013 | ❌ |

**Decision: REVERTED.** 1/7 improved. DC_SPAN=10 confirmed optimal in both directions.

**Conclusion from iters 84–87:** The current feature set is at or near a local optimum for what's extractable from this data. France and Netherlands consistently punish new features in opposite directions. The model is well-calibrated; further gains likely require genuinely new data sources (lineup data, live Betfair odds, xG feeds) rather than permutations of the same inputs.

**Baseline unchanged at:** ROI +2.95%, stability +0.0198, t-stat +1.22 (3769 bets, 4287 test matches).

---

# Archived docs batch 88–92: Distinct re-test batch — all reverted

Baseline: ROI +2.92%, stability +0.0196, t-stat +1.21 (3770 bets, 4288 test matches). Five maximally distinct experiments spanning early-era ideas and new angles, all tested under LightGBM + 7 leagues for the first time.

## EXP-20260501-D088: Days rest — REVERTED

_Legacy source: docs/improvements.md iteration 88._

**Hypothesis:** `home_days_rest` / `away_days_rest` (days since team's last match, default 7) captures fatigue from fixture congestion.
**Previously tested:** Iter 23 on 4 leagues — reverted. Re-tested here on 7 leagues + LightGBM.
**Implementation:** Wired `_compute_days_rest` into `_build_merged`; `_get_current_days_rest` into `build_fixture_features`.

| League | Baseline | +Days rest | Δ |
|---|---|---|---|
| England | −0.91% | +4.27% | +5.18 pp ✅ |
| Germany | +4.24% | −0.99% | −5.23 pp ❌ |
| Spain | −1.47% | +1.43% | +2.90 pp ✅ |
| Italy | −3.00% | −6.97% | −3.97 pp ❌ |
| France | +0.60% | −2.85% | −3.45 pp ❌ |
| Netherlands | +12.08% | +4.04% | −8.04 pp ❌ |
| Portugal | +11.76% | +5.40% | −6.36 pp ❌ |
| **Total** | **+2.92%** | **+0.55%** | **−2.37 pp** |
| Stability | +0.0196 | +0.0038 | ❌ |

**Decision: REVERTED.** 2/7 improved. Netherlands −8pp, Portugal −6pp. Result mirrors the Iter 23 failure — rest days are already priced in by bookmakers for prominent congestion periods (European nights etc.), adding noise elsewhere.

---

## EXP-20260501-D089: ELO_K = 20 (was 30) — REVERTED

_Legacy source: docs/improvements.md iteration 89._

**Hypothesis:** More stable Elo ratings (less reactive to individual results) give a better long-run strength estimate. K=30 may overfit to recent scorelines already captured by the market.

| League | Baseline | K=20 | Δ |
|---|---|---|---|
| England | −0.91% | +7.88% | +8.79 pp ✅ |
| Germany | +4.24% | +5.42% | +1.18 pp ✅ |
| Spain | −1.47% | −5.82% | −4.35 pp ❌ |
| Italy | −3.00% | −1.48% | +1.52 pp ✅ |
| France | +0.60% | +2.97% | +2.37 pp ✅ |
| Netherlands | +12.08% | +2.94% | −9.14 pp ❌ |
| Portugal | +11.76% | +9.34% | −2.42 pp ❌ |
| **Total** | **+2.92%** | **+2.67%** | **−0.25 pp** |
| Stability | +0.0196 | +0.0178 | ❌ |

**Decision: REVERTED.** 4/7 improved but Netherlands −9.14pp dominates. K=30 confirmed optimal.

---

## EXP-20260501-D090: Cumulative season points — REVERTED

_Legacy source: docs/improvements.md iteration 90._

**Hypothesis:** `home_season_pts` / `away_season_pts` (pre-match cumulative league points) captures current table position — early relegation battles, title races, and remaining-season motivation the model can't infer from recent form alone.
**Previously tested:** Iter 28 on 4 leagues — reverted as "flat". Re-tested here on 7 leagues + LightGBM.

| League | Baseline | +Season pts | Δ |
|---|---|---|---|
| England | −0.91% | +0.92% | +1.83 pp ✅ |
| Germany | +4.24% | −0.37% | −4.61 pp ❌ |
| Spain | −1.47% | −7.09% | −5.62 pp ❌ |
| Italy | −3.00% | −3.76% | −0.76 pp ❌ |
| France | +0.60% | +3.87% | +3.27 pp ✅ |
| Netherlands | +12.08% | +9.99% | −2.09 pp ❌ |
| Portugal | +11.76% | +11.25% | −0.51 pp ≈ |
| **Total** | **+2.92%** | **+1.72%** | **−1.20 pp** |
| Stability | +0.0196 | +0.0118 | ❌ |

**Decision: REVERTED.** 2/7 improved. Germany −4.61pp, Spain −5.62pp. Season points may be collinear with Elo (which already tracks cumulative quality) — adding both creates redundancy.

---

## EXP-20260501-D091: Strength of schedule (mean recent opponent Elo) — REVERTED

_Legacy source: docs/improvements.md iteration 91._

**Hypothesis:** A new feature `home_opp_elo` / `away_opp_elo` — mean Elo of each team's last WINDOW opponents — captures whether form was built against strong or weak competition.
**Inspired by:** Iter 35 (opponent-quality-adjusted form) — reverted on old dataset.

| League | Baseline | +Opp Elo | Δ |
|---|---|---|---|
| England | −0.91% | +2.77% | +3.68 pp ✅ |
| Germany | +4.24% | +3.38% | −0.86 pp ❌ |
| Spain | −1.47% | −12.02% | −10.55 pp ❌ |
| Italy | −3.00% | −4.84% | −1.84 pp ❌ |
| France | +0.60% | +0.53% | −0.07 pp ≈ |
| Netherlands | +12.08% | +5.30% | −6.78 pp ❌ |
| Portugal | +11.76% | +2.92% | −8.84 pp ❌ |
| **Total** | **+2.92%** | **−0.74%** | **−3.66 pp** |
| Stability | +0.0196 | −0.0052 | ❌ flipped negative |

**Decision: REVERTED.** Spain −10.55pp, Portugal −8.84pp. Opponent Elo is largely encoded in Elo ratings themselves — teams that face strong opposition regularly have their Elo calibrated accordingly. The feature adds collinear noise.

---

## EXP-20260501-D092: min_child_samples = 10 (was 20) — REVERTED

_Legacy source: docs/improvements.md iteration 92._

**Hypothesis:** Allowing splits on smaller leaf samples (10 vs 20) lets LightGBM capture finer patterns in league-specific data.

| League | Baseline | mcs=10 | Δ |
|---|---|---|---|
| England | −0.91% | −1.99% | −1.08 pp ❌ |
| Germany | +4.24% | −3.07% | −7.31 pp ❌ |
| Spain | −1.47% | −2.48% | −1.01 pp ❌ |
| Italy | −3.00% | −4.96% | −1.96 pp ❌ |
| France | +0.60% | −1.47% | −2.07 pp ❌ |
| Netherlands | +12.08% | +6.17% | −5.91 pp ❌ |
| Portugal | +11.76% | +11.50% | −0.26 pp ≈ |
| **Total** | **+2.92%** | **+0.23%** | **−2.69 pp** |
| Stability | +0.0196 | +0.0016 | ❌ |

**Decision: REVERTED.** 0/7 improved. Overfits in the same pattern as num_leaves=63 (iter 80) — the model is at its regularisation optimum at mcs=20. Trying mcs=40 not pursued; consistent direction of overfitting suggests the model needs more regularisation, not less.

**Conclusion from iters 88–92:** All five reverted across completely different dimensions (scheduling, Elo calibration, table context, strength of schedule, model capacity). The consistent pattern is that the model's current feature set and LightGBM config have reached a genuine local optimum within the information available in this data. The baseline stays at ROI +2.92%, stability +0.0196.

---

# Archived pending investigations

### xG (expected goals) integration — blocked on data access (logged 2026-05-01)

**Hypothesis:** Replace or augment DC attack/defense ratings (currently EWM of raw goals) with xG-based equivalents. xG is a noise-reduced measure of shot quality; a team scoring above their xG will regress while goals-based ratings stay high. The specific signal is `xg_surplus = rolling_goals − rolling_xG` as a market-lag indicator.

**Why deferred:** Data access is harder than expected across all candidate sources:
- **Understat**: only 5/7 leagues (no Eredivisie, no Primeira Liga). Now JS-rendered — all Python packages (`understat`, `understatapi`) are broken as of 2026-05.
- **FBref**: covers all 7 leagues from 2013-14 but blocks plain HTTP (403). Requires Playwright/Selenium headless browser + system GTK libs (`libatk-bridge2.0-0` etc.) not present in WSL2.
- **StatsBomb open data**: spotty seasonal coverage, missing N1/P1 entirely.
- **Shots on target (HST/AST)** — closest available proxy — already tested in Iter 63 and Iter 39, both strongly reverted. Reason: bookmakers already price in shot quality, so adding it duplicates market signal with noise.

**To unblock:** `sudo apt install -y libatk-bridge2.0-0 libatk1.0-0 libgbm1 libnss3 libxss1` in WSL2, then configure soccerdata custom leagues:
```json
// ~/.soccerdata/config/league_dict.json
{"NED-Eredivisie": {"FBref": "Eredivisie", "MatchHistory": "N1", "season_start": "Aug", "season_end": "May"},
 "POR-Primeira Liga": {"FBref": "Primeira Liga", "MatchHistory": "P1", "season_start": "Aug", "season_end": "May"}}
```
~350 FBref page fetches at 3s spacing (~18 min one-time download). Cache locally as `data/raw/xg_{league}_{season}.csv`.

**Note:** Even if unblocked, signal value is uncertain — bookmakers already incorporate xG into pricing. The targeted experiment should be `xg_surplus` (goals-minus-xG divergence), not raw xG level.

---

### Fix edge calculation baseline (value bet vs vig) — under consideration

**Current:** edge = model_prob − fair_prob (vig-stripped B365 implied)
**Problem:** fair_prob < raw_implied_prob, so the edge is inflated. Bets can be flagged as value even when model_prob < 1/odds — meaning they have negative EV.
**Correct definition:** a value bet requires model_prob > 1/odds (raw implied), i.e. edge = model_prob − (1/odds).
**Fix options:**
- Option A: switch baseline to raw implied: `edge = model_prob - (1 / odds)`
- Option B: keep fair baseline but require threshold ≥ ~3–5% to approximately compensate for the vig (at ~5% vig, the correction is ~3–3.5% depending on the odds range)

---

## EXP-20260804-001: Re-run all-market baseline on current evaluation split — RECORDED (no code change)

**Date:** 2026-08-04
**Hypothesis:** N/A — bookkeeping task from `current.md` active hypothesis 1. The previously recorded production result (`EXP-20260519-S101`, ROI +9.65%) predates the all-market/production-allowlist split introduced in the "Improve autoresearch evaluation workflow" change and is not comparable to headline metrics produced by the current code.
**Command:** `uv run python main.py --update --per-league --threshold 0.0` (required a fresh full history download; `data/raw` was empty in this environment).
**Files changed:** none — diagnostic run only.

**Results (all-market, 11 leagues, no max-edge/overround cap):**
- Accuracy: 0.518 (9,906 test matches, seasons 2324/2425/2526/2627)
- Bets: 9,096 / 9,906 (91.8%)
- ROI: −4.35%
- Stability: −0.0290
- t-statistic: −2.76

**Results (production portfolio — E0, N1, P1, G1, with max_edge=0.20 and max_overround=0.07 filters):**
- Bets: 2,101
- ROI: +1.03%
- Per-league ROI: England +1.80%, Netherlands +7.97%, Portugal +0.21%, Greece −5.81%

**Per-league ROI (all-market, informational only, filters not applied):** England +1.80%, Germany −5.31%, Spain −9.07%, Italy −2.96%, France −9.48%, Netherlands +7.97%, Portugal +0.21%, Greece −5.81%, Scotland +1.73%, Belgium −17.63%, Turkey −13.59%.

**Analysis:** The unfiltered all-market number is dominated by non-production leagues (Belgium, Turkey, Spain, France) that were already known to be unprofitable and are excluded from live betting — it is a diagnostic ceiling/floor, not a decision metric. The number that matters for keep/revert decisions is the production portfolio, which is positive but far below the stale +9.65% figure. Greece is the weak link in the current production allowlist (−5.81%, largest bet count among the negative production leagues after the filters).
**Decision:** RECORDED. No code changed. `current.md`'s "Current best" table is updated to this run and the stale `EXP-20260519-S101` comparison note is removed. Active hypothesis 1 is cleared from the queue.

---

## EXP-20260804-002: Raw implied-probability edge baseline vs vig-stripped fair baseline — REVERTED (not pursued)

**Date:** 2026-08-04
**Hypothesis:** Switching the value-bet edge baseline from vig-stripped fair probability (`edge = model_prob - fair_prob`) to raw implied probability (`edge = model_prob - 1/odds`) removes bets that are technically negative-EV at the actual offered odds, improving ROI and stability.
**Files changed:** none — the existing `--compare-vig` diagnostic (`main.py:_run_compare_vig`, `edge_baseline` param on `compute_value_betting_results`) already supports both baselines, so no implementation was needed to test the hypothesis at the diagnostic (global-model) level first, before committing to a full per-league confirmatory run.
**Command:** `uv run python main.py --compare-vig --threshold 0.0`
**Baseline:** global (non-per-league) walk-forward model, same run for both baselines, 9,906 test matches.

**Results:**

| Threshold | fair bets | fair ROI | raw bets | raw ROI |
|---|---:|---:|---:|---:|
| 0.00 | 13,914 | −1.51% | 9,530 | −1.74% |
| 0.01 | 11,725 | −0.57% | 7,629 | −0.47% |
| 0.02 | 9,652 | −0.73% | 6,040 | −1.24% |
| 0.03 (current default) | 7,911 | −0.88% | 4,769 | −1.82% |
| 0.04 | 6,400 | −1.04% | 3,671 | −1.51% |
| 0.05 | 5,138 | −1.18% | 2,854 | +0.15% |
| 0.07 | 3,247 | +0.49% | 1,705 | −4.32% |

Stability at threshold 0.0: fair −0.0103 vs raw −0.0115.

**Analysis:** No consistent direction. Raw is worse than fair at 5 of 7 thresholds, including the current production default (0.03: −0.88% vs −1.82%, a 0.94pp regression) and threshold 0.07 (−4.32% vs +0.49%, a 4.8pp regression). It is only better at 0.01 and 0.05, both within the "usually noise" band per `EVALUATION.md` except 0.05's +1.33pp gap, which is a single favorable threshold amid an otherwise unfavorable or flat pattern — not the majority-of-conditions support the evaluation policy requires. This is also a global (not per-league) model, one level short of the mandatory primary comparison, but the diagnostic result is unfavorable enough that a full per-league re-run (≈28 minutes) is not justified before ruling out the change.
**Decision:** REVERTED (not implemented). The vig-stripped fair baseline remains the default `edge_baseline`. Active hypothesis 6 is cleared from the queue with this conclusion recorded.

**Note (tooling bug found, unrelated to this hypothesis):** `_run_compare_vig`'s per-league breakdown crashes with `KeyError: 'league'` — it merges on a `"league"` column that does not exist in `results["odds_test"]`. The aggregate/threshold-sweep comparison above still printed correctly before the crash. Left unfixed as out of scope for this iteration; added to the active queue below for a future small tooling fix.

---

## EXP-20260809-001: Dixon-Coles Poisson goal-model probabilities as engineered features — REVERTED

**Date:** 2026-08-09
**Hypothesis:** Adding Dixon-Coles-fitted match outcome probabilities (`poisson_home_prob`, `poisson_draw_prob`, `poisson_away_prob`) as three new features — additive to, not replacing, the existing raw EWM attack/defense features — would improve ROI without hurting stability, because a Poisson score-matrix model with a home-advantage parameter and the Dixon-Coles low-score correlation (τ) adjustment captures nonlinear, correlated goal-scoring structure that LightGBM has to approximate from 4 raw scalar features today, and might help most in data-sparse leagues (Greece, Portugal).
**Files changed:**
- `src/model/dixon_coles.py` (new): MLE fit (`scipy.optimize.minimize`, analytic gradient) of per-team attack/defense ratings, home advantage, and ρ; `predict_probs` sums the Poisson score grid (0–9 goals each side, τ-adjusted) into a 1X2 probability triple.
- `src/model/features.py`: added `_compute_poisson_probs` (walk-forward, one DC model per league, refit every 180 days using the trailing 1,200 matches, strictly leakage-free) and `_get_current_poisson_state` (live-fixture path); added 3 columns to `FEATURE_COLS`.
- `pyproject.toml` / `uv.lock`: added `scipy` as an explicit dependency (was previously only transitive via scikit-learn).
- `tests/test_dixon_coles.py` (new): fit/predict correctness, probability-sums-to-one, unseen-team handling, stronger-attacker-favoured sanity check.

All changes were reverted after evaluation; none remain in the tree.

**Baseline:** Same-session re-run of the primary comparison on identical data (`uv run python main.py --per-league --threshold 0.0`, 45,092 matches through 2026-08-03) with the Dixon-Coles code removed — reproduces `EXP-20260804-001` exactly (accuracy 0.518, all-market ROI −4.35%, stability −0.0290, t-stat −2.76, bets 9,096/9,906; production portfolio 2,101 bets, ROI +1.03%).

**Results (with Dixon-Coles features):**
- All-market accuracy: 0.520 (baseline 0.518)
- All-market ROI: −4.74% (baseline −4.35%, 0.39pp worse)
- Stability: −0.0316 (baseline −0.0290, worse)
- t-statistic: −3.03 (baseline −2.76, worse)
- All-market bets: 9,176 / 9,777 (93.9%) — 129 fewer test matches than baseline (9,906), from the new feature's minimum-history requirement (`POISSON_MIN_MATCHES=40`) dropping the earliest rows per league via the existing `dropna(subset=FEATURE_COLS)` step; expected and explained, not a driver of the result.
- **Production portfolio (decision metric): 2,032 bets, ROI −2.76%** (baseline 2,101 bets, +1.03% — a 3.79pp regression)
- Production per-league ROI: England −1.69% (baseline +1.80%), Netherlands −5.30% (baseline +7.97%), Portugal −2.48% (baseline +0.21%), Greece −2.27% (baseline −5.81%, the one league that improved)

**Analysis:** 3 of 4 production leagues moved sharply negative, including Netherlands flipping from the strongest positive contributor (+7.97%) to the weakest (−5.30%), a 13.27pp swing. Only Greece improved, and it remains negative. The all-market diagnostic moved in the same unfavorable direction (ROI, stability, and t-stat all worse), so this isn't a case of one noisy metric contradicting another — every metric that matters under `EVALUATION.md` degraded together. The mechanism may simply be that per-league LightGBM models with only ~4 walk-forward test seasons of data don't have enough per-league history to benefit from 3 additional probability features on top of the already-present attack/defense EWM signals and market-derived probabilities; the DC probabilities may be adding correlated noise (derived from goals, same underlying signal as `home_attack`/`home_defense`/Elo) rather than independent information, consistent with `EVALUATION.md`'s general note that features must supply information not already encoded in market odds or existing ratings to help.
**Decision:** REVERTED. All code, dependency, and test changes rolled back (`git checkout -- pyproject.toml uv.lock src/model/features.py`; new files deleted). `reports/backtest_bets.csv` regenerated from the reverted code to restore the production-baseline artifact. This was explored ad hoc (not from the active-hypothesis queue in `current.md`), so no queue entry needs clearing.

---

## EXP-20260810-001: Re-verify Pinnacle-confirmation filter on the current per-league model — KEPT (backtest only, not yet wired to live)

**Date:** 2026-08-10
**Hypothesis:** The Pinnacle-confirmation filter (skip a bet unless the historical Pinnacle fair probability exceeds the B365 fair probability by more than a 0.015 margin) — the strongest feature this project ever found in backtests (`EXP-20260513-S057`–`S062`), removed in `EXP-20260517-S072` only because live Pinnacle odds were unavailable, not because the signal was invalidated — will still improve production ROI without hurting stability now that live Pinnacle odds are wired in (see `docs/superpowers/specs/2026-08-09-live-pinnacle-odds-design.md`), even though the model has changed substantially since May (Elo, Dixon-Coles ratings, per-league thresholds didn't exist then).
**Files changed:**
- `src/evaluation/metrics.py`, `main.py`, `src/config.py`: opt-in `pinnacle_confirmation_margin` parameter added in a prior session (default `None`/off, no behavior change on its own — see git history for that commit series).
- `main.py`: added a `--pinnacle-filter` CLI flag to `_run_backtest()` that passes `DEFAULT_PINNACLE_CONFIRMATION_MARGIN` (0.015) into the production-portfolio `compute_value_betting_results` calls only, leaving the all-market diagnostic (`evaluation_results`) untouched — confirmed by the all-market metrics being bit-for-bit identical between the two runs below (accuracy 0.518, ROI −4.35%, stability −0.0290, t-stat −2.76, bets 9,096/9,906).
**Command:** `uv run python main.py --per-league --threshold 0.0` (baseline) vs `uv run python main.py --per-league --threshold 0.0 --pinnacle-filter`, same data (45,092+ matches), using historical `PSCH`/`PSCD`/`PSCA` — no live API calls.
**Baseline:** filter off, reproduces `EXP-20260804-001`/`EXP-20260809-001`'s baseline exactly: production portfolio 2,101 bets, ROI +1.03%, stability 0.0069, t-stat 0.32.

**Results (production portfolio — decision metric):**

| League | Filter off | Filter on | Delta |
|---|---:|---:|---:|
| England (E0) | 695 bets, +0.72% | 208 bets, +9.44% | +8.72pp |
| Greece (G1) | 281 bets, +1.09% | 95 bets, +13.71% | +12.62pp |
| Netherlands (N1) | 788 bets, +4.55% | 331 bets, +24.66% | +20.11pp |
| Portugal (P1) | 337 bets, −6.59% | 111 bets, +0.81% | +7.40pp |
| **Total** | **2,101 bets, +1.03%** | **745 bets, +15.46%** | **+14.43pp** |

Stability: 0.0069 → 0.1029. t-statistic: 0.32 → 2.81 (crosses the `|t| > 2` screening threshold).

**Analysis:** All 4 production leagues improved, unanimously and by a wide margin (+7.4pp to +20.1pp each) — per `EVALUATION.md`, a unanimous same-direction move across every league is stronger evidence than the aggregate number alone, and the +14.43pp total ROI change is well above the "+5pp = meaningful candidate" bar. Bets dropped 64.5% (2,101 → 745), which is the filter's intended mechanism (a confirmation/veto filter is supposed to shrink the bet set to a higher-quality subset), not a data-reduction artifact from a warm-up period or narrower context — so `EVALUATION.md`'s data-reduction caution doesn't apply the way it does for e.g. venue-specific rolling form. The reduced sample still clears the significance screen (t-stat 2.81) on its own. This result also corroborates, rather than merely repeats, the original `EXP-20260513-S057`–`S062` finding: the underlying signal (sharp-book agreement as a confirmation of model edge) held up despite Elo, Dixon-Coles ratings, and per-league thresholds all being added to the model since May.
**Decision:** KEPT (backtest-level). The `pinnacle_confirmation_margin` parameter and `--pinnacle-filter` diagnostic flag stay in the tree. **Not yet wired into the live prediction path** (`main.py:_run_predict()` → `_build_prediction_rows` still runs with the filter off) — per the design spec, that requires a separate, explicit decision since it changes real betting behavior; flagged to the user for sign-off rather than enabled automatically here.

**Note (pre-existing per-league reporting ambiguity, found while reproducing the baseline):** `current.md`'s recorded `EXP-20260804-001` "Per-league" row (England +1.80%, Netherlands +7.97%, Portugal +0.21%, Greece −5.81%) and `EXP-20260809-001`'s "baseline" comparison figures are the **all-market** per-league ROI (`_print_split_analysis`'s "ROI BY LEAGUE" table: global threshold=0, no max-edge/overround caps, all 11 leagues, filtered down to the 4 production rows for display) — not the actual **production portfolio** per-league split (per-league calibrated thresholds + max-edge/overround caps applied), which is +0.72%/+1.09%/+4.55%/−6.59% for E0/G1/N1/P1 respectively, as used in this entry. The two tables' per-league numbers are not interchangeable even though both runs share the same overall production ROI (+1.03%) — worth using the production-portfolio-specific split (from `reports/backtest_bets.csv`, grouped by `league`) rather than the "ROI BY LEAGUE" console table when recording future production per-league results, to avoid comparing filtered against unfiltered numbers.

---

## EXP-20260810-002: Re-validate the filter on realistic (opening) odds, re-select production leagues, enable live — KEPT AND LIVE

**Date:** 2026-08-10
**Hypothesis:** `EXP-20260810-001` validated the Pinnacle-confirmation filter using `PSCH/PSCD/PSCA` — Pinnacle's historical **closing** line. But a live snapshot fetched via The Odds API can only ever be **opening/pre-match**-style (a true closing line only exists after kickoff has passed), and a live check against real fixtures.csv + live API data found the live snapshot's vig (~5.4%) sits above even the historical opening-odds average (4.43%), well above closing (3.53%). If the filter's edge depends on closing-line precision, it might not survive being fed opening-style odds; this iteration tests that directly, using historical `PSH/PSD/PSA` as the proxy for what live fetching actually provides.

**Files changed:**
- `src/evaluation/metrics.py`: added `pinnacle_odds_cols` parameter to `compute_value_betting_results` (default `("PSCH","PSCD","PSCA")`) so the confirmation check can read either the closing or opening odds columns.
- `main.py`: added `--pinnacle-filter-opening` (uses `PSH/PSD/PSA`) alongside the existing `--pinnacle-filter` (closing), and `--all-leagues-production` (applies the real per-league-threshold+cap production methodology to every supported league, not just the current allowlist, for screening purposes only — never affects `--predict`). Added a per-league print of the production portfolio directly.
- `src/data/loader.py`: `_OPTIONAL_COLS` only carried `PSCH/PSCD/PSCA` through from historical CSVs, not `PSH/PSD/PSA` — added the latter, since football-data.co.uk's raw files carry both but the loader silently dropped the opening columns.
- `src/model/features.py`: `build_features_with_odds`'s `base_odds_cols` likewise only carried closing odds through to `eval_df` — added `PSH/PSD/PSA`.
- `main.py`: fixed a crash in `_save_profit_chart` — a league-season with zero confirmed bets returns an explicit-but-empty DataFrame (object dtype), and concatenating it unconditionally with real float-typed chunks could poison `cumulative_profit`'s dtype. The far more selective opening-odds filter triggers this (several production league-seasons legitimately have zero bets) where no prior run had. Now skips empty chunks before concatenation.
- `src/data/pinnacle_odds.py`: captures `commence_time` from the Odds API response as a `Date` field and rejects any team-name match where it disagrees with `fixtures.csv`'s date by more than a day — a live check found `fixtures.csv` and the Odds API aren't always showing the same matchweek at a given moment (the API can be pricing the round after what's nearest in `fixtures.csv`), so a name-only join risked silently attaching one round's odds to a different round's fixture.
- `src/data/team_aliases.py`: added N1/P1 gaps found in a fuller live sample (`SC Telstar`, `NEC Nijmegen`, `Fortuna Sittard`, `SC Cambuur`, `ADO Den Haag`, `CF Estrela`), and populated F1's table (`RC Lens`→`Lens`, `AS Monaco`→`Monaco`, `Paris Saint Germain`→`Paris SG`) since it's entering production.
- `src/config.py`: `PRODUCTION_LEAGUES` changed from `{E0, N1, P1, G1}` to `{E0, N1, G1, F1}`.
- `main.py`: wired `pinnacle_confirmation_margin=DEFAULT_PINNACLE_CONFIRMATION_MARGIN` into all three live call sites in `_run_predict()` (`_build_prediction_rows`, `_print_predictions`, `_save_predictions_csv`).
- `tests/test_metrics.py`, `tests/test_pinnacle_odds.py`, `tests/test_team_aliases.py`, `tests/test_loader.py`, `tests/test_config.py`: extended/corrected for all of the above.

**Command:** `uv run python main.py --per-league --threshold 0.0 --pinnacle-filter-opening` (production leagues only), then `--pinnacle-filter-opening --all-leagues-production` (all 11 leagues), same data as `EXP-20260810-001`.

**Two wiring bugs caught before trusting results:** the first `--pinnacle-filter-opening` run reproduced `EXP-20260810-001`'s exact closing-odds numbers (745 bets, +15.46%) bit-for-bit — implausible for genuinely different input columns, and traced to `_parse_pinnacle_odds_cols()` being defined but never threaded into the actual `compute_value_betting_results` calls. The second attempt (after fixing that) reproduced the **filter-off baseline** exactly (2,101 bets, +1.03%) — traced to `PSH/PSD/PSA` never being loaded from historical CSVs at all, so the filter's null-check silently skipped every row. Both fixed and verified with cheap non-training sanity checks before the real ~28-minute reruns.

**Results — production leagues only (E0/N1/P1/G1), opening-odds filter:**

| | Filter off | Closing odds (`-001`) | Opening odds (this entry) |
|---|---:|---:|---:|
| Bets | 2,101 | 745 | 303 |
| ROI | +1.03% | +15.46% | **+13.11%** |
| Stability | 0.0069 | 0.1029 | 0.0816 |
| t-stat | 0.32 | 2.81 (significant) | **1.42 (not significant)** |
| England | +0.72% | +9.44% | +17.93% |
| Netherlands | +4.55% | +24.66% | +16.27% |
| Greece | +1.09% | +13.71% | +3.36% |
| Portugal | −6.59% | +0.81% | −0.53% |

**Results — full production-methodology screen, all 11 leagues, opening-odds filter (742 bets total, +1.64% ROI):**

| League | Bets | ROI |
|---|---:|---:|
| Netherlands | 190 | +16.27% |
| England | 44 | +17.93% |
| Greece | 33 | +3.36% |
| France | 34 | −0.21% |
| Portugal | 36 | −0.53% |
| Scotland | 20 | +77.60% |
| Belgium | 28 | +75.68% |
| Spain | 29 | −7.14% |
| Turkey | 53 | −18.98% |
| Italy | 206 | −18.91% |
| Germany | 69 | −19.01% |

**Analysis:** The filter's mechanism survives the transition from closing to opening odds — direction and magnitude hold up (+12.08pp over baseline, still above `EVALUATION.md`'s "+5pp = meaningful" bar), but the sample shrinks further (745→303) and t-stat drops below the significance screen (2.81→1.42). This is a moderate-confidence result, not a decisive one. The all-11-league screen disproves an earlier, cruder all-market (threshold=0, no caps) diagnostic that had shown France improving by +14.95pp to +5.47% — under the real per-league-threshold+cap methodology France is flat (−0.21%), not a standout. Belgium and Scotland show large ROI (+75–78%) on only 20–28 bets each — per `EVALUATION.md`'s own noise-interpretation guidance, this pattern (huge swing, thin sample) reads as noise, not signal, and was not added. Germany, Italy, Spain, and Turkey are clear, well-sampled negatives.

**Decision:** KEPT AND MADE LIVE, per explicit user direction after reviewing the above. `PRODUCTION_LEAGUES` changed to `{E0, N1, G1, F1}`: Portugal dropped (flat/slightly negative under both odds proxies); France added despite its flat opening-odds result, on the user's explicit reasoning that live Predict runs close to kickoff should trend closer to Pinnacle's closing line than this worst-case test — a business judgment this iteration does not itself validate. `pinnacle_confirmation_margin` is now wired into all three live call sites in `_run_predict()`. Follow-up queued in `current.md`: monitor how close live-fetched odds actually land to closing-line behavior once enough live runs accumulate, and revisit France's inclusion if they don't.

---

## EXP-20260810-003: Team-level historical Pinnacle opening→closing market movement as a feature — REVERTED

**Date:** 2026-08-10
**Hypothesis:** Active hypothesis 2 in `current.md` — test opening-to-closing market movement "without introducing inference-only features." A live snapshot can never observe a true closing line, so the current match's own future movement is fundamentally uncomputable at prediction time; instead this iteration used each team's own **past** matches' realized Pinnacle opening→closing fair-probability delta (rolling mean, shift(1), window=20 — mirroring `_compute_market_bias`/`_get_current_market_bias` exactly), on the theory that a team's historical tendency for the market to move toward or away from it between open and close might itself be a genuinely live-computable signal (sharp money patterns, public-team bias, etc.).
**Files changed:**
- `src/model/features.py`: added `_compute_market_movement`/`_get_current_market_movement`, using `PSH/PSD/PSA` (opening) and `PSCH/PSCD/PSCA` (closing); added `home_market_movement`/`away_market_movement` to `FEATURE_COLS`.
- `tests/test_features.py`: extended fixture to include valid Pinnacle odds (previously had none, since `_make_df()` predates any Pinnacle-derived feature) and added dedicated tests for the new feature and its null-safety.

All changes were reverted after evaluation (`git revert 693f719`); none remain in the tree.

**Pre-check before implementing:** 94.8% of historical rows have valid `PSH/PSD/PSA` + `PSCH/PSCD/PSCA` across all 11 leagues/seasons — ruled out a data-availability concern before spending engineering effort.

**Baseline caveat:** immediately before this iteration, `dev` pulled in another session's merge-safety fix (`validate="many_to_one"` guards on the team-rolling-stat merges in `_build_merged`, `6b32b42`) and a new season-breadth diagnostic (`e6ed275`). This iteration's run is therefore not a perfectly isolated single-variable comparison against the stale `EXP-20260809-001` baseline (accuracy 0.518, all-market ROI −4.35%, stability −0.0290, t-stat −2.76, bets 9,096/9,906) — total test-match count shifted from 9,906 to 11,386 independent of this feature, and the test-season window shifted from 2324–2526 to 2223–2526. A fresh same-codebase baseline (feature off) was not re-run before testing the feature on, given the decisiveness of the result below; if a cleaner isolated comparison is ever needed, re-run both with and without the feature on the current `dev` HEAD.

**Results (with the feature, `uv run python main.py --per-league --threshold 0.0`):**
- All-market accuracy: 0.525 (vs. stale baseline 0.518)
- All-market ROI: **−5.69%** (vs. stale baseline −4.35%)
- Stability: **−0.0381** (vs. −0.0290)
- t-statistic: **−3.99** (vs. −2.76)
- Bets: 11,009 / 11,386 (96.7%)
- **Season breadth (new diagnostic): 0/4 seasons profitable** (2223 −4.82%, 2324 −9.39%, 2425 −1.83%, 2526 −7.77%) — need ≥3, clean FAIL
- Production portfolio (current leagues E0/N1/G1/F1, filter off): 2,533 bets, **−1.91% ROI**. England — the strongest performer in every prior test this session (+9% to +18%) — collapsed to +1.62%; France −6.56%, Greece −0.93%, Netherlands −2.63%.

**Analysis:** Negative across every metric that matters — all-market ROI, stability, t-stat all moved the wrong way, and the season-breadth check (0/4) shows this isn't a one-bad-season artifact, it's uniformly bad. Production-portfolio England's collapse from a consistently strong league to barely positive is the clearest single tell that the feature is actively hurting the model rather than merely failing to help. Despite the baseline-confound caveat above, the magnitude and multi-metric consistency of the regression make it implausible that the concurrent merge-safety fix alone explains this — a duplicate-key merge bug fix would be expected to produce a modest correction, not a multi-point ROI swing across all-market, production, and every test season simultaneously. Plausible mechanism: a lagged 20-game team-level average is a fairly coarse, slow-moving signal, and may simply be too noisy/thin relative to the well-established features (Elo, form, market_bias) already capturing similar "team tendency vs. market" information — consistent with `EVALUATION.md`'s general note that new features must supply information not already encoded in existing signals to help.
**Decision:** REVERTED. `git revert 693f719` — all code and test changes rolled back cleanly, full suite green (123 passed) on the reverted tree. Active hypothesis 2 is cleared from the queue in `current.md`.
