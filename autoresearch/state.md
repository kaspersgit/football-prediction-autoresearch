# Autoresearch State Document

## Current Best Model

| Metric     | Value          |
|------------|----------------|
| Accuracy   | 0.492          |
| ROI        | -6.79%         |
| Stability  | -0.0637        |
| Model      | Logistic Regression + StandardScaler |
| Features   | 5-game rolling mean: pts, gf, ga (home + away) — 6 features total |

_Last updated: 2026-04-17 (Iteration 0 — baseline)_

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

_(No iterations yet — start here.)_

---

## Open Hypotheses

Ranked by estimated probability of improving ROI:

1. **Threshold-based betting (value bets):** Only bet when model probability exceeds bookmaker implied probability. This directly targets edge over the market and should reduce bet count while improving ROI. _High confidence._

2. **Gradient Boosting model (XGBoost/LightGBM):** Logistic Regression is linear and likely underfit given the non-linear interactions between team features. Tree-based models may capture these better. _Medium-high confidence._

3. **Elo ratings as features:** Elo gives a dynamic per-team strength estimate that updates after every match, which is a richer signal than rolling form. Many published betting models use Elo as a core feature. _Medium-high confidence._

4. **Home/away split form:** The current rolling stats mix home and away performance. A team may have very different home vs away form, and separating these could improve predictions. _Medium confidence._

5. **Weighted rolling average:** Weight recent games more heavily in the rolling mean (e.g., exponential decay). More recent form may be more predictive than older games. _Medium confidence._

6. **Shorter rolling window (3 games):** A 3-game window may capture more recent form changes. Worth comparing to 5 and 10. _Low-medium confidence._

---

## Key Findings So Far

_(Empty — no completed iterations yet.)_

---

## Notes / Lessons Learned

**Dataset facts:**
- Covers multiple European leagues, seasons 1314–2425
- Test period: last 2 full seasons (2324, 2425)
- Total test bets: 2643 — large enough for statistical significance
- Bookmaker margin (vig) is approximately 5%, so ROI > 0% requires genuine predictive edge
- Accuracy of ~0.492 on a 3-class problem (H/D/A) is close to the naive baseline; draws are hard to predict

**Pipeline facts:**
- Run pipeline: `uv run python main.py`
- Run tests: `uv run pytest tests/ -v`
- Frozen files: `src/data/`, `src/evaluation/`, `main.py`, `tests/`
- Editable files: `src/model/features.py`, `src/model/train.py`, `autoresearch/state.md`
