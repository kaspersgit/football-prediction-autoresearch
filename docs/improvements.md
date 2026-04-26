# Possible Improvements

## Implemented

### 1. Threshold convention: 0.0 for research, 0.04 for production (2026-04-26)
**Rationale:** Backtest (4,376 matches) showed threshold=0.0 loses money under both baselines. At threshold=0.04, raw turns slightly positive (+0.73%, 195 bets); fair approaches breakeven (-0.21%, 769 bets). Threshold=0.05 on fair gives the best backtest result (+4.53%, 451 bets) but is likely overfit to a small 2-season sample.

**Convention:**
- **Research/iterations:** `python main.py` → threshold=0.0, `fair` baseline. Maximum sample size for stable signal when comparing model changes.
- **Sanity check before committing:** re-run with `raw` baseline + threshold=0.0. Any bet passing `raw` is EV-positive by definition — no extra filter needed.
- **Production (betting):** `./predict.sh` → threshold=0.04 injected automatically, `fair` baseline. Guards against low-confidence noise in live predictions.

**Changes:** `_parse_threshold()` default kept at `0.0`; `predict.sh` injects `--threshold 0.04` unless overridden.

---

### 3. Elo momentum (home_elo_delta, away_elo_delta) (2026-04-26)
**Change:** Added `home_elo_delta` and `away_elo_delta` to `FEATURE_COLS`. These were already computed by `_compute_elo` but not used. Delta = current Elo minus Elo 5 games ago.
**Result:** ROI improved from −3.06% → −2.09% (+0.97 pp), accuracy 0.535 → 0.537, bets 4082 → 4015.

### 4. Venue-specific form (home_vform_*, away_vform_*) (2026-04-26)
**Change:** Wired `_team_venue_rolling_stats` into `_build_merged` and `build_fixture_features`. Adds 6 new features: home team's EWM form in home games only, away team's EWM form in away games only.
**Result:** ROI improved from −2.09% → −1.31% (+0.78 pp), bets 4015 → 4035.

**Combined gain (iterations 1+2):** −3.06% → −1.31% (+1.75 pp at threshold=0.0).

---

## Pending

## 2. Fix edge calculation baseline (value bet vs vig) — under consideration

**Current:** edge = model_prob − fair_prob (vig-stripped B365 implied)  
**Problem:** fair_prob < raw_implied_prob, so the edge is inflated. Bets can be flagged as value even when model_prob < 1/odds — meaning they have negative EV.  
**Correct definition:** a value bet requires model_prob > 1/odds (raw implied), i.e. edge = model_prob − (1/odds).  
**Fix options:**  
- Option A: switch baseline to raw implied: `edge = model_prob - (1 / odds)`  
- Option B: keep fair baseline but require threshold ≥ ~3–5% to approximately compensate for the vig (at ~5% vig, the correction is ~3–3.5% depending on the odds range)
