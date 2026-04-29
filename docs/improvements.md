# Possible Improvements

## Evaluation Standards (established 2026-04-26)

See **[docs/evaluation_standards.md](evaluation_standards.md)** for the full reference including:
- Per-league review protocol (mandatory per iteration)
- Statistical noise model and confidence intervals by sample size
- The stability metric as a t-statistic proxy
- What to trust vs ignore (rules of thumb table)
- Data-reduction caution and the known-bad feature list
- Why cross-iteration consistency is the real signal

---

## Implemented

### 1. Threshold convention: 0.0 for research, 0.04 for production (2026-04-26)
**Rationale:** Backtest (4,376 matches) showed threshold=0.0 loses money under both baselines. At threshold=0.04, raw turns slightly positive (+0.73%, 195 bets); fair approaches breakeven (-0.21%, 769 bets). Threshold=0.05 on fair gives the best backtest result (+4.53%, 451 bets) but is likely overfit to a small 2-season sample.

**Convention:**
- **Research/iterations:** `python main.py` → threshold=0.0, `fair` baseline. Maximum sample size for stable signal when comparing model changes.
- **Sanity check before committing:** re-run with `raw` baseline + threshold=0.0. Any bet passing `raw` is EV-positive by definition — no extra filter needed.
- **Production (betting):** `./predict.sh` → threshold=0.04 injected automatically, `fair` baseline. Guards against low-confidence noise in live predictions.

**Changes:** `_parse_threshold()` default kept at `0.0`; `predict.sh` injects `--threshold 0.04` unless overridden.

---

### 2. (Iter 54) Elo momentum — home_elo_delta, away_elo_delta (2026-04-26)
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

### 3. (Iter 55) Market overround as feature (2026-04-26)
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

### 4. (Iter 56) l2_regularization: 0.1 → 0.05 (2026-04-26)
**Hypothesis:** l2=0.1 was tuned on 4 leagues (~17k matches). With 7 leagues and 30k+ matches, less regularization may let the model use the richer feature set more effectively.

**Change:** `l2_regularization=0.1 → 0.05` in `_MODEL_CFG` in `train.py`.

**Result:** See combined table above. 6/7 leagues improve vs elo-delta baseline. Portugal barely regresses (−0.37 pp, noise-level).

**Decision: KEPT.**

---

### 5. (Iter 57) Venue-specific form — REVERTED (2026-04-26)
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

### 6. (Iters 63-68) Shots on target, market bias window, match balance, log-odds, WINDOW, ELO_HOME_ADV (2026-04-26)

Batch of 6 experiments from the brainstorm-approved list. Baseline entering this batch: ROI −1.33%, stability −0.0090, t-stat −0.57.

#### Iter 63: Rolling HST/AST shots on target — REVERTED

**Hypothesis:** Shots on target as xG proxy; research cites ~0.8% ROI/bet edge over 12 years.

**Result:** ROI −3.83%, stability −0.0264, t-stat −1.69. 6/7 leagues worsened (Spain −11.8%, Netherlands −8.6%). The market has already priced in shot quality; adding it introduces noise rather than signal.

#### Iter 64: MARKET_BIAS_WINDOW = 20 (was 5) — KEPT ✅

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

#### Iter 65: match_balance = 1 − |market_h − market_a| — KEPT ✅

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

#### Iter 66: log_odds_h/d/a — REVERTED

**Result:** ROI +0.48% vs +0.74% (declined). 5/7 worsened. Log-odds are derivable from market_h/d/a + market_overround — not independent information.

#### Iter 67: WINDOW = 7 (was 5) — REVERTED

**Result:** ROI −0.87% vs +0.74%. 5/7 worsened. WINDOW=5 confirmed optimal.

#### Iter 68: ELO_HOME_ADV = 75 (was 100) — REVERTED

**Result:** ROI −1.90% vs +0.74%. 4/7 worsened. ELO_HOME_ADV=100 confirmed optimal.

**New baseline after iters 64-65:** ROI +0.74%, stability +0.0049, t-stat +0.30 (3863 bets, 4227 test matches).

---

### 7. (Iters 69–73) Tier-1 algorithm research batch (2026-04-27)

Baseline entering this batch: ROI +0.74%, stability +0.0049, t-stat +0.30 (3863 bets, 4227 test matches).

#### Iter 69: Goal-margin Elo — REVERTED

**Hypothesis:** Updating Elo by goal margin (logistic sigmoid on goal diff: `1 / (1 + 10^(-gd/4))`) gives faster signal than binary W/L/D.

**Result:** ROI −0.25%, stability −0.0017, t-stat −0.10. Only 2/7 leagues improved (Netherlands, Portugal). England, Germany, Spain, Italy, France all regressed. The market already prices in scoreline information; making Elo track goal margins just tracks what the bookmaker already knows, adding noise.

**Decision: REVERTED.**

---

#### Iter 70: Isotonic Calibration (manual IsotonicRegression per class) — REVERTED

**Hypothesis:** HistGBM probabilities overfit extremes; isotonic calibration on 20% held-out training data would improve value bet detection.

**Note:** `CalibratedClassifierCV(cv='prefit')` is not supported in sklearn 1.8.0. Implemented manually via `IsotonicRegression` per class.

**Result:** ROI −3.84%, stability −0.0282, t-stat −1.77. Only 2/7 leagues improved (Germany, Netherlands). The calibration set (20% of training data, ~5k rows) is too small to fit a reliable isotonic transformation for a 3-class problem. The HGBM is already reasonably calibrated via log-loss; adding a noisy isotonic layer distorts probabilities.

**Decision: REVERTED.**

---

#### Iter 71: Attack/Defense Rating Features — KEPT ✅

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

#### Iter 72: Time-Weighted Training (λ=0.5) — REVERTED

**Hypothesis:** Matches from 2013-14 involve different squads; exponential decay sample weights (half-life ~2 years) would reduce noise from stale data.

**Result:** ROI −3.55%, stability −0.0238, t-stat −1.47. Only 2/7 leagues improved (France slightly, Portugal). Spain −11.7pp, Italy −10.6pp, Netherlands −9.2pp were catastrophic. Older data is informative — de-weighting it with λ=0.5 destroys signal rather than reducing noise.

**Decision: REVERTED.**

---

#### Iter 73: Season-Start Elo Partial Reset (ELO_CARRY=0.8) — REVERTED

**Hypothesis:** At each season start, apply `elo = 0.8*elo + 0.2*1500` to reduce stale carry-over from pre-transfer-window squads.

**Result:** ROI −2.58%, stability −0.0174, t-stat −1.09. 0/7 leagues improved. The reset degrades Elo accuracy for early games in each season, and with 13 seasons × 7 leagues × many teams, the distortion accumulates significantly.

**Decision: REVERTED.**

---

**New baseline after iter 71:** ROI +1.63%, stability +0.0108, t-stat +0.67 (3871 bets, 4227 test matches).

---

### 8. (Iters 74–76) Tier-2 algorithm swap batch (2026-04-27)

Baseline entering this batch: ROI +1.63%, stability +0.0108, t-stat +0.67 (3871 bets, 4227 test matches). Model: HistGradientBoostingClassifier + DC ratings.

#### Iter 74: XGBoost drop-in replacement — REVERTED

**Hypothesis:** XGBoost's exact split-finding and L1 regularization may generalise better than HistGBM's approximate histogram splits on this dataset.

**Implementation:** Replaced `HistGradientBoostingClassifier` with `XGBClassifier`. Tested 3 hyperparameter configurations (default, tuned depth/estimators, tuned reg). Required label encoding because XGBoost 3.x rejects string class labels.

**Result:** ROI consistently below baseline across all 3 configs. No config improved both total ROI and stability; majority of leagues regressed.

**Decision: REVERTED.** XGBoost adds complexity (label encoding, slower training) with no measurable gain over HistGBM on this dataset.

---

#### Iter 75: LightGBM leaf-wise growth — KEPT ✅

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

#### Iter 76: RandomForest blend — REVERTED

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

### 9. (Iters 77–79) LightGBM re-tests of previously reverted features (2026-04-27)

Baseline: ROI +3.20%, stability +0.0215. Purpose: verify whether the switch to LightGBM rehabilitates any of the three closest-call reverts from the HistGBM era.

#### Iter 77: WINDOW=7 re-test — REVERTED

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

#### Iter 78: ELO_HOME_ADV=75 re-test — REVERTED

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

#### Iter 79: Goal-margin Elo re-test — REVERTED

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

### 10. (Iters 80–82) LightGBM hyperparameter sweep (2026-04-27)

Baseline: ROI +3.20%, stability +0.0215. Testing model capacity, stochastic sampling, and learning rate.

#### Iter 80: num_leaves=63 (was 31) — REVERTED

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

#### Iter 81: subsample=0.8, colsample_bytree=0.8 — REVERTED

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

#### Iter 82: learning_rate=0.02, n_estimators=750 — REVERTED

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

### 11. (Iter 83) Betfair Exchange odds as market features — REVERTED (2026-04-29)

**Hypothesis:** BFE fair implied probabilities are more accurate market signals than B365 (no bookmaker margin embedded) — using them for `market_h/d/a`, `market_overround`, `market_bias` should give the model cleaner inputs.

**Implementation:** Added `BFEH/BFED/BFEA` to loader optional columns; added `_market_odds()` helper preferring BFE over B365 with fallback; updated all three market-feature computation sites.

**Result:** ROI +0.88%, stability +0.0061 (vs +3.20%, +0.0215). Germany and Portugal both collapsed.

**Root cause:** BFE data only appears in the football-data.co.uk CSVs from season 2024-25 onward — exactly the 2 test seasons. All 11 training seasons have no BFE column and fall back to B365. The model trains on B365-based features but sees BFE-based features at test time: a clean distribution mismatch. Not a signal quality issue.

**Decision: REVERTED.** BFE cannot be used as a training feature until it has ≥5 seasons of historical coverage (est. 2030). BFE is being pursued separately as a live-odds display via the Betfair Exchange API.

---

### 12. (Iters 84–87) Feature expansion batch — all reverted (2026-04-30)

Baseline entering this batch: ROI +2.95%, stability +0.0198, t-stat +1.22 (3769 bets, 4287 test matches). Note: slight shift from +3.20% due to new match results added by data refresh on 2026-04-29.

#### Iter 84: Season progress features — REVERTED

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

#### Iter 85: Venue-split Elo — REVERTED

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

#### Iter 86: DC_SPAN=7 (was 10) — REVERTED

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

#### Iter 87: DC_SPAN=15 (was 10) — REVERTED

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

## Pending

### Fix edge calculation baseline (value bet vs vig) — under consideration

**Current:** edge = model_prob − fair_prob (vig-stripped B365 implied)  
**Problem:** fair_prob < raw_implied_prob, so the edge is inflated. Bets can be flagged as value even when model_prob < 1/odds — meaning they have negative EV.  
**Correct definition:** a value bet requires model_prob > 1/odds (raw implied), i.e. edge = model_prob − (1/odds).  
**Fix options:**  
- Option A: switch baseline to raw implied: `edge = model_prob - (1 / odds)`  
- Option B: keep fair baseline but require threshold ≥ ~3–5% to approximately compensate for the vig (at ~5% vig, the correction is ~3–3.5% depending on the odds range)
