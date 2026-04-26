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

## Pending

### Fix edge calculation baseline (value bet vs vig) — under consideration

**Current:** edge = model_prob − fair_prob (vig-stripped B365 implied)  
**Problem:** fair_prob < raw_implied_prob, so the edge is inflated. Bets can be flagged as value even when model_prob < 1/odds — meaning they have negative EV.  
**Correct definition:** a value bet requires model_prob > 1/odds (raw implied), i.e. edge = model_prob − (1/odds).  
**Fix options:**  
- Option A: switch baseline to raw implied: `edge = model_prob - (1 / odds)`  
- Option B: keep fair baseline but require threshold ≥ ~3–5% to approximately compensate for the vig (at ~5% vig, the correction is ~3–3.5% depending on the odds range)
