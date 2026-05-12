# Research Report: Alternative Models & Algorithms
**Date:** 2026-04-27  
**Status:** Approved for implementation planning

---

## 1. Current State

| Metric | Value |
|---|---|
| Model | `HistGradientBoostingClassifier` (sklearn) |
| Mode | Single global model (or `--per-league` for best historical result) |
| ROI (current baseline) | +0.74% (iters 64-65, threshold=0.0) |
| t-stat | +0.30 (not significant; need \|t\| > 2) |
| Bets | ~3,863 / 4,227 test matches |
| Best ever | +7.08% ROI with `--per-league` flag (pre-iter-54 feature set) |
| Features (24) | Form EWM, Elo + delta, market H/D/A + overround, match balance, H2H, draw rates, market bias (w=20), 6 league dummies |
| Available libs | scikit-learn, pandas, numpy, scipy (xgboost/lightgbm NOT installed) |

**What has failed (permanent no-list):**
- Venue-specific form (failed iters 1, 18, 57 — structural warm-up sparsity)
- Shots on target as xG proxy (iter 63 — market already prices this in)
- Log-odds features (iter 66 — not independent from market_h/d/a + overround)
- WINDOW=7 (iter 67), ELO_HOME_ADV=75 (iter 68)
- Monthly retraining (earlier experiment — sampling noise is irreducible)
- Binary outcome decomposition (3 binary classifiers — weaker than per-league multi-class)

---

## 2. Literature Survey

### 2.1 Statistical / Domain-Specific Models

**Dixon-Coles (1997)**  
The foundational paper for football outcome modelling. Models home and away goals as independent Poisson random variables with team-specific parameters:
- `attack_i`: team i's scoring rate
- `defense_j`: team j's goals-conceded rate
- `P(home_goals=k, away_goals=m)` modeled via bivariate Poisson with small-score correction factor ρ

Key insight: **Elo gives one scalar per team; Dixon-Coles gives two** (attack rate, defense rate). A team that scores 3 but concedes 2 has completely different DC parameters than a team that scores 1 but concedes 0, despite potentially the same Elo. The attack/defense decoupling surfaces information the market misprices when teams with asymmetric profiles meet.

**pi-rating / Soccer Power Index (SPI)**  
FiveThirtyEight's SPI (Nate Silver, 2015) and Constantinou's pi-rating (2013) both extend Elo by:
1. Separately tracking offensive and defensive strength per team
2. Updating based on actual goals vs expected goals (not just W/L/D)
3. Applying a home advantage correction per league

Result: 4 features per match (home_off, home_def, away_off, away_def) instead of Elo's 2. The goal-margin update rule gives faster signal from dominant wins (4-0 updates more than 1-0).

**Bivariate Poisson**  
Extension of Dixon-Coles allowing correlation between home and away scores (correlated Poisson). More complex optimization with no clear benefit for match-outcome prediction (correlation parameter ρ in basic DC already handles the low-score anomaly adequately).

**Bradley-Terry Model**  
Pairwise ranking model: P(i beats j) = p_i / (p_i + p_j). Less suitable here — doesn't natively handle draws and requires all teams to be in one connected graph. Elo is functionally similar and already used.

### 2.2 Machine Learning Alternatives

**XGBoost**  
Gradient boosting with exact split finding (vs HistGBM's histogram approximation). Key differences:
- L1 regularization (`alpha`) in addition to L2 (`lambda`)
- Column subsampling per tree (`colsample_bytree`) and per level (`colsample_bylevel`) — forces more feature diversity
- `dart` booster: applies dropout to trees (prevents late trees from over-specializing)
- Must install: `uv add xgboost`

**LightGBM**  
Leaf-wise tree growth (vs HistGBM's level-wise). Grows the leaf with maximum loss reduction regardless of level, creating asymmetric trees. Typically faster and lower memory. Has DART mode. Must install: `uv add lightgbm`.

**CatBoost**  
Handles categorical features via ordered target encoding. Less relevant here since league dummies are already one-hot encoded. Must install separately.

**RandomForest**  
Bagging ensemble (independent trees with bootstrap + feature subsampling). Already available in sklearn. Advantages over boosting for noisy data: each tree sees less of the noise, so the ensemble variance is lower. Historically underperforms boosting on structured data, but football outcomes are close to maximum noise.

**MLP / Neural Network**  
With ~30k training rows and 24 features, a 2-3 layer MLP could capture higher-order interactions. Key risk: football prediction data is very noisy (~51% accuracy ceiling), so the signal-to-noise ratio is too low for neural nets to learn complex patterns reliably. Would need torch or keras; likely overfits badly.

### 2.3 Training Strategy Variants

**Probability Calibration (Isotonic Regression)**  
`HistGBM` optimizes log-loss, which encourages well-calibrated probabilities in theory — but in practice, gradient boosting is known to overfit probability extremes (predicts 0.85 when true rate is 0.65). Poor calibration directly inflates perceived edge: if model says 40% but true rate is 33%, we think we have +7% edge over 30% fair odds but actually have 0%.

sklearn's `CalibratedClassifierCV(cv='prefit', method='isotonic')` wraps the trained model with a monotonic calibration step fitted on a held-out fold. Already imported in `train.py` but **never actually used**.

**Time-Weighted Training (Sample Weights)**  
Weight training samples by recency: `w = exp(-λ * days_from_latest)`. The model ignores stale data (teams from 10 years ago with completely different squads) and focuses on recent patterns. Both HistGBM and XGBoost accept `sample_weight` directly. Risk: reducing effective training N from ~30k to ~15k-equivalent increases variance.

**Soft Target Labeling (Market Consensus Regularization)**  
Instead of binary outcome labels (`y ∈ {0, 1}`), use market-fair probabilities as soft labels (`y_soft_H = fair_h`). This regularizes the model to start from market consensus and learn residual edges rather than learning the full probability surface from scratch. Implemented by re-framing as regression with MSE loss, or by directly scaling the cross-entropy gradient. High implementation complexity.

**Goal-Margin Elo (Score-Based Update)**  
Instead of W/L/D → {1, 0.5, 0} outcomes for Elo, weight by goal margin: `actual_home = (home_goals - away_goals + max_goals) / (2 * max_goals)`. Gives more update signal from dominant wins, converges faster. Easy to implement in `_compute_elo`.

**Direct ROI Optimization (Custom Loss)**  
Train the model to maximize expected betting profit rather than log-loss. For XGBoost/LightGBM, this means a custom gradient/hessian. Risk: the loss surface is discontinuous (bet/no-bet is a step function) and the model can exploit it trivially. Needs careful constraint design. High complexity, high risk.

### 2.4 Betting / Ensemble Strategies

**Probability Blending (Dixon-Coles + HGBM)**  
Weighted average of probabilities from two independent models: `p_blend = α * p_hgbm + (1-α) * p_dc`. If the models have independent errors, blending reduces variance. Works best when the two models exploit different information sources (DC: goal distributions, HGBM: market + form + Elo).

**Two-Stage Stacking (HGBM → Calibrating LogReg)**  
Train 7 league-specific HGBM models (Stage 1). Then train a logistic regression meta-learner (Stage 2) using Stage 1 probabilities + market probabilities as features. The meta-learner learns to weight the league models and blend with market. Avoids data leakage via cross-validated Stage 1 predictions. Moderate complexity.

**Kelly Criterion Sizing**  
Rather than flat 1-unit bets, size bets proportional to edge: `f = edge / (odds - 1)`. The `--kelly` flag already exists in `main.py`. Quarter-Kelly (`f/4`) is standard practice to account for model uncertainty. Already implementable without code changes.

---

## 3. Evaluation Framework Compatibility

All proposals must satisfy the existing evaluation rules:
- Report ROI + stability + t-stat per iteration
- Require 4+/7 leagues to improve (5+/7 preferred)
- No data leakage: all feature states computed strictly before each match
- No row reduction beyond acceptable limits (<2% acceptable, >5% requires justification)
- t-stat gate: trust cross-league direction count over raw ROI delta

---

## 4. Proposed Iteration Plan

Iterations are grouped by confidence tier and ordered within each tier by implementation ease.

### Tier 1 — High Confidence, Zero New Dependencies (Iters 69–73)

These use only sklearn/scipy (already installed) and have clear mechanistic motivation.

---

**Iter 69: Goal-Margin Elo**  
*Hypothesis:* Updating Elo by goal margin rather than W/L/D gives faster, more accurate team strength updates. A 4-0 win provides more information than a 1-0 win — both are currently treated identically.  
*Implementation:* Change `_compute_elo` in `features.py`: replace `actual_home = 1.0/0.5/0.0` with a continuous goal-margin score, e.g. `actual_home = (FTHG - FTAG) / (FTHG + FTAG + 1)` mapped to [0,1].  
*Risk:* Low. The update logic is isolated to Elo computation; all other features unchanged.  
*Row impact:* None.  
*Expected gain:* +0.5–2pp ROI if Elo is currently underweighting decisive wins.

---

**Iter 70: Probability Calibration (Isotonic)**  
*Hypothesis:* HistGBM's raw probabilities overfit probability extremes, inflating perceived edge on high-odds bets. Isotonic calibration will reduce false positives in value bet detection.  
*Implementation:* Wrap each league model with `CalibratedClassifierCV(model, cv='prefit', method='isotonic')` fitted on a 20% held-out chunk of training data. Note: `CalibratedClassifierCV` is already imported in `train.py`.  
*Risk:* Medium. Calibration reduces the held-out training data by 20%. Need to verify calibration doesn't over-smooth probabilities.  
*Row impact:* None on test set; slight reduction in effective training N.  
*Expected gain:* Hard to predict — calibration removes false edges. Could improve or reduce ROI; primary benefit is making edge estimates more trustworthy.

---

**Iter 71: Attack/Defense Rating Features (Dixon-Coles inspired)**  
*Hypothesis:* Two teams with identical Elo can have very different playing styles (high-scoring vs defensive). Separate attack and defense ratings capture this asymmetry and provide independent signal from Elo.  
*Implementation:* Add `_compute_dc_ratings` in `features.py`. For each match (processed in date order), maintain rolling attack/defense estimates per team updated via a simple exponential smoothing of goals scored/conceded (pure Python/numpy, no scipy optimization needed for the feature version):  
  - `home_attack_rating ← EWM(goals_scored, span=10)`  
  - `home_defense_rating ← EWM(goals_conceded, span=10)`  
  - Add 4 features: `home_attack`, `home_defense`, `away_attack`, `away_defense`  
*Risk:* Low-medium. Not correlated with Elo (Elo uses W/L/D; these use raw goals). Might overlap with `form_gf`/`form_ga` already in the feature set — may be redundant.  
*Row impact:* None (EWM, same warm-up as existing form features).  
*Expected gain:* +0.5–3pp if goal rate signal is not fully captured by current form features.

---

**Iter 72: Time-Weighted Training (Sample Weights)**  
*Hypothesis:* Matches from 10+ years ago (2013-14) feature different team squads, tactics, and market efficiency than recent seasons. De-weighting old matches reduces noise in learned feature-outcome relationships.  
*Implementation:* Compute `sample_weight = exp(-λ * (latest_date - match_date).days / 365)` with `λ = 0.5` (half-life ~2 years). Pass to `model.fit(X_train, y_train, sample_weight=w)`.  
*Risk:* Low-medium. Reduces effective training N. Must test multiple λ values.  
*Row impact:* None.  
*Expected gain:* Uncertain. If older data is informative, this hurts. If team dynamics have shifted significantly, it helps.

---

**Iter 73: Season-Start Elo Partial Reset**  
*Hypothesis:* Elo ratings carry full history across seasons, but squads change significantly in summer transfer windows. A partial regression toward the mean at season start (e.g., `elo = 0.8 * elo + 0.2 * 1500`) would reduce the "stale star team" problem where relegated/promoted teams are badly mispriced in season openers.  
*Implementation:* In `_compute_elo`, detect season transitions and apply partial regression.  
*Risk:* Low. Season transitions are well-defined. Effect is isolated to ~8 games post-season-start.  
*Row impact:* None.  
*Expected gain:* Small but plausible. Season openers are known to have higher bookmaker edge.

---

### Tier 2 — Medium Confidence, Requires New Library (Iters 74–76)

Requires `uv add xgboost` or `uv add lightgbm`. Low installation risk but more hyperparameter tuning needed.

---

**Iter 74: XGBoost Drop-In Replacement**  
*Hypothesis:* XGBoost's L1 regularization and column subsampling produce more diverse trees than HistGBM, reducing correlation between trees and improving generalization on noisy football data.  
*Implementation:* Replace `HistGradientBoostingClassifier` with `xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.05, colsample_bytree=0.8, use_label_encoder=False, eval_metric='mlogloss')`.  
*Hyperparameters to validate:* `colsample_bytree ∈ {0.6, 0.8, 1.0}`, `reg_alpha ∈ {0.0, 0.1, 0.5}`.  
*Risk:* Low for installation. Medium for performance — XGBoost may not outperform HistGBM on tabular data.  
*Row impact:* None.  
*Expected gain:* Unclear. Could be ±2pp. Worth testing to rule out model class as bottleneck.

---

**Iter 75: LightGBM Leaf-Wise Growth**  
*Hypothesis:* LightGBM's leaf-wise tree growth creates deeper, more asymmetric trees that better capture the tail behavior in football outcomes (unexpected results).  
*Implementation:* Replace with `lgb.LGBMClassifier(num_leaves=31, n_estimators=300, learning_rate=0.05, min_child_samples=20, reg_lambda=0.05)`. Also test DART booster mode (`boosting_type='dart'`).  
*Risk:* Same as XGBoost. Leaf-wise growth is more prone to overfitting; `num_leaves` must be carefully tuned.  
*Row impact:* None.  
*Expected gain:* Unclear. LightGBM sometimes outperforms HistGBM by 0.5–2pp on structured data.

---

**Iter 76: RandomForest Ensemble Blend**  
*Hypothesis:* Blending HistGBM probabilities with RandomForest probabilities (`p_blend = 0.7 * p_hgbm + 0.3 * p_rf`) reduces variance from boosting's sequential error-focusing. RF's independent bagged trees make different errors than HGBM.  
*Implementation:* `RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=20)` from sklearn. Blend probabilities before value bet computation.  
*Risk:* Low (sklearn already available). The blend weight (0.7/0.3) needs tuning.  
*Row impact:* None.  
*Expected gain:* Small (+0.5–1pp). Ensembles typically help more when base models are diverse and independently trained.

---

### Tier 3 — Lower Confidence, High Implementation Effort (Iters 77–79)

These are more experimental and harder to validate within the existing eval framework.

---

**Iter 77: Full Dixon-Coles Probability Blend**  
*Hypothesis:* Dixon-Coles models goal distributions explicitly using Poisson rates, producing probabilities from a different information source than HGBM. Blending their outputs gives a diversified probability estimate.  
*Implementation:*  
  1. Implement `DixonColesModel` class using `scipy.optimize.minimize` to fit attack/defense parameters per team per season  
  2. Apply time-decay weight (ξ = 0.0065 as in original paper)  
  3. Compute P(H), P(D), P(A) from summed Poisson probabilities  
  4. Blend: `p_final = α * p_hgbm + (1-α) * p_dc` with `α ∈ {0.6, 0.7, 0.8}`  
*Risk:* High implementation complexity. DC model must be trained per-season walk-forward to avoid leakage. scipy.optimize convergence is not guaranteed.  
*Row impact:* None.  
*Expected gain:* Medium-high in theory. DC and HGBM exploit different information (goal distributions vs tabular features). Blending reduces each model's idiosyncratic errors.

---

**Iter 78: Two-Stage Stacking (Per-League HGBM → Meta LogReg)**  
*Hypothesis:* A logistic regression meta-learner can learn to weight the 7 per-league models against each other and against the market, improving calibration and generalization.  
*Implementation:*  
  1. Train 7 league-specific HGBM models (Stage 1) using cross-validated out-of-fold predictions on training data  
  2. Train LogReg (Stage 2) on: [stage1_proba_H, stage1_proba_D, stage1_proba_A, market_H, market_D, market_A, league_dummies]  
  3. Stage 2 predicts final probabilities used for value betting  
*Risk:* High. Leakage risk if Stage 1 predictions on training data are not done properly via cross-validation. Complex walk-forward integration.  
*Row impact:* None.  
*Expected gain:* Moderate. Stacking typically helps 1–3pp in well-controlled setups.

---

**Iter 79: Direct ROI Loss (XGBoost Custom Objective)**  
*Hypothesis:* Log-loss encourages accurate probability estimates but doesn't optimize for betting profitability. A custom objective that penalizes missed value bets and rewards found ones would directly improve ROI.  
*Implementation:*  
  - Custom gradient: for each match, compute expected profit given current proba; gradient = -(profit - threshold)  
  - Requires XGBoost (iter 74 must pass first)  
  - Needs careful regularization to prevent the model from assigning 99% to everything  
*Risk:* Very high. Custom objectives are hard to validate. The betting function is non-differentiable (threshold step). Would need smooth approximation.  
*Row impact:* None.  
*Expected gain:* High if implemented correctly. Unknown in practice.

---

## 5. Recommended Execution Order

```
Tier 1 (run in order, revert losers, keep winners):
  69 → 70 → 71 → 72 → 73

Tier 2 (only if Tier 1 gains are insufficient for significance):
  74 → 75 → 76

Tier 3 (only after establishing t-stat > 1.5):
  77 → 78 → 79
```

**Stopping condition:** If any iteration brings `|t-stat| > 2.0`, that is the primary goal achieved. Run 3 more iterations to confirm stability, then switch focus to production (threshold=0.04) tuning.

---

## 6. Quick Reference: What Each Approach Adds

| Approach | New Information Source | Deps | Effort |
|---|---|---|---|
| Goal-margin Elo | Goal score (vs W/L/D for Elo) | None | XS |
| Calibration (isotonic) | Better probability estimates | None | S |
| Attack/defense ratings | Goals as independent offensive/defensive signal | None | S |
| Time-weighted training | Recency-weighted sample importance | None | S |
| Season-start Elo reset | Transfer window team changes | None | XS |
| XGBoost | Different tree-building strategy | xgboost | M |
| LightGBM | Leaf-wise growth + DART | lightgbm | M |
| RandomForest blend | Ensemble diversity | None | S |
| Dixon-Coles blend | Poisson goal model probabilities | scipy | XL |
| Stacking | Meta-learner calibration | None | XL |
| Direct ROI loss | Betting-objective training | xgboost | XL |
