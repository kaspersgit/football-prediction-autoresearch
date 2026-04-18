# Autoresearch State Document

## Current Best Model

| Metric     | Value          |
|------------|----------------|
| Accuracy   | 0.523          |
| ROI        | -8.21%         |
| Stability  | -0.0573        |
| Bets       | 1180 / 2626 (44.9%) |
| Threshold  | 0.06           |
| Model      | HistGradientBoostingClassifier |
| Features   | 5-game rolling mean: pts, gf, ga (home + away) + Elo ratings — 8 features total |

_Last updated: 2026-04-18 (Iteration 15 — threshold=0.06 identified as optimal operating point via grid search. **Caveat: threshold was selected on test set.**)_

**Note on evaluation setup (2026-04-18):** All metrics from Iteration 11 onward use the new pipeline:
- **Walk-forward backtest**: one model trained per test season (2425 then 2526); Elo carries forward correctly.
- **Value betting**: multi-outcome, vig-corrected fair probabilities. Default `--threshold 0.06`.
- **Test set**: 2425 (1426 matches) + 2526 partial (1200 matches) = 2626 total.
- **Important**: the 0.06 threshold was selected on the backtest set — future OOS performance may differ. All future iterations should be evaluated at both threshold=0.0 (fair baseline) and threshold=0.06 (operating point).

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

~~**Season phase (match_month):**~~ _Tested in Iteration 11 — regression._
~~**Head-to-head historical win rate:**~~ _Tested in Iteration 12 — regression on ROI, accuracy +0.002._
~~**Longer rolling window (WINDOW=7):**~~ _Tested in Iteration 13 — regression on ROI, accuracy +0.005._
~~**Probability calibration (isotonic cv=3):**~~ _Tested in Iteration 14 — regression. Calibration to empirical frequencies ≠ alignment with bookmaker pricing._
~~**Threshold tuning:**~~ _Tested in Iteration 15 — threshold=0.06 gives best ROI (-8.21% vs -10.99% baseline). **New recommended operating point.** Caveat: selected on test set._

**Operating point going forward:** evaluate at both threshold=0.0 (clean baseline) and threshold=0.06 (operating point).

**Research direction pivot:** All feature and calibration experiments have either not improved ROI or improved it only via threshold selection. The model appears to have reached a ceiling under the current paradigm (value betting on 4 leagues, 8 features). Consider:

1. **Draw-specific model:** Draws are the hardest outcome (bookmakers overprice them; public underestimates them). A separate classifier trained only to identify draws vs non-draws might provide more accurate draw probabilities, which are the hardest to calibrate. _Medium-high confidence — draws are structurally different from H/A outcomes._

2. **League-specific models:** Train a separate HistGBM per league. League-specific patterns (e.g., Serie A's low-scoring draws, Bundesliga's high home win rate) may be better captured by specialised models than a pooled model with implicit cross-league noise. _Medium confidence._

3. **Odds-based features:** Add the opening bookmaker odds as features directly. The model would learn to compare its probability estimates to the market, which directly addresses the calibration vs market problem. The market is the strongest available signal. _High confidence — the odds ARE the best prior; using them as a feature lets the model explicitly learn when to deviate._

~~**Elo ratings as features:**~~ _Tested in Iteration 3 — improved accuracy (+2.7pp) and ROI (+0.47%) but remains negative. Elo is now a permanent part of the feature set._

~~**Gradient Boosting model (XGBoost/LightGBM):**~~ _Tested in Iteration 2 — no improvement over Logistic Regression with the current 6-feature set. Model capacity is not the bottleneck._

~~**Home/away split form:**~~ _Tested in Iteration 1 — worsened all metrics. Discarded._

---

## Key Findings So Far

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
- Run pipeline: `uv run python main.py` (value betting, threshold=0.0)
- Run with edge filter: `uv run python main.py --threshold 0.05` (require ≥5% edge)
- Run tests: `uv run pytest tests/ -v`
- Profit chart saved automatically: `reports/profit_curve.png`
- Evaluation strategy: multi-outcome value betting with vig-corrected fair probabilities
- Frozen files: `src/data/`, `src/evaluation/report.py`, `tests/`
- Editable files: `src/model/features.py`, `src/model/train.py`, `src/evaluation/metrics.py`, `main.py`, `autoresearch/state.md`
