# CI-Narrowing Iteration Plan (Iter 86–95)

**Date:** 2026-05-19
**Goal:** Keep ROI > 5% at threshold=0.0 while narrowing the 95% CI (raising t-stat above ~3.0) by increasing bet volume.

---

## Problem

After excluding four consistently negative leagues (F1, SP1, D1, I1) the portfolio shrank from ~3300 bets to 1688, roughly halving bet volume. The t-stat is 2.59 — statistically significant but narrow margin above 2.0. The formula is:

```
t = stability × √N_bets
```

Narrowing the CI requires more bets at equal or better per-bet quality. The only lever that adds 1000+ bets is adding new profitable leagues.

## Current best (Iter 85)

| Metric | Value |
|--------|-------|
| ROI | +8.33% |
| Stability | 0.0630 |
| t-stat | +2.59 |
| Bets | 1688 / 4433 (38.1%) |
| Leagues | E0, N1, P1 only |

## Technical enabler: no code changes needed for new leagues

`loader.py:load_all_data()` auto-loads every `{code}_{season}.csv` in `data/raw/`. Adding new league CSV files is sufficient for backtest inclusion. The only frozen part (`_LEAGUE_MAP`) gates fixture prediction, not backtesting. New leagues can be downloaded using the existing `download_season(code, season)` function without touching `src/data/`.

## 10-Iteration Plan

### Block 1: New leagues (Iter 86–89)

Each iteration follows the same pattern:
1. Download all seasons (1314–2526) for the league code using `download_season()`
2. Remove the league code from `skip_leagues` in `main.py`
3. Run `uv run python main.py --per-league`
4. **Keep if:** overall ROI ≥ 5% AND combined t-stat ≥ 2.59
5. **Revert if:** either condition fails

| Iter | League | Code | Expected bets |
|------|--------|------|---------------|
| 86 | Scotland Premiership | SC0 | +400–600 |
| 87 | Belgium First Division | B1 | +400–600 |
| 88 | Greece Super League | G1 | +300–500 |
| 89 | Turkey Süper Lig | T1 | +400–600 |

If a league is reverted, the next league is still attempted — a negative result is still a useful data point.

### Block 2: Filter expansion (Iter 90–91)

With bet volume established, loosen existing filters for marginal additional coverage across all kept leagues.

| Iter | Change | Hypothesis |
|------|--------|------------|
| 90 | max_overround 0.07 → 0.08 | Slightly higher vig markets may still have exploitable edge; extra 50–150 bets |
| 91 | max_odds 5.0 → 6.0 | High-odds long-tail bets where calibrated model finds edge |

**Keep criterion:** t-stat improves AND ROI does not drop below 5%.

### Block 3: Model quality (Iter 92–95)

These are the top entries from the "next hypotheses" list in state.md. Improving per-bet consistency raises stability and therefore t-stat independently of volume.

| Iter | Change | From state.md priority |
|------|--------|------------------------|
| 92 | min_child_samples 20 → 15 | #2 — bias-variance tuning |
| 93 | num_leaves 31 → 40 | #3 — model complexity |
| 94 | EWM span 5 → 4 | #5 — recency-noise tradeoff |
| 95 | reg_lambda 0.05 → 0.03 | Looser regularisation with more training data from added leagues |

**Keep criterion per iteration:** ROI ≥ 5% AND stability does not regress.

## Success definition

After Iter 95, the target state is:
- ROI > 5% (floor maintained)
- t-stat ≥ 3.0 (meaningful CI improvement — roughly 1000+ additional good bets needed)
- Bets ≥ 2500

Partial success (t-stat 2.6–2.9) is acceptable if all 4 new leagues failed — it means the CI problem requires a different long-term approach (e.g., more historical seasons, second-tier leagues).

## Constraints

- `src/data/`, `src/evaluation/report.py`, `tests/` are frozen
- Editable: `src/model/features.py`, `src/model/train.py`, `src/evaluation/metrics.py`, `main.py`, `autoresearch/state.md`
- Each iteration must pass `uv run pytest tests/ -v` before being recorded
- Follow the full iteration protocol in `autoresearch/GUIDE.md`
