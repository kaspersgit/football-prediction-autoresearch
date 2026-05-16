# Autoresearch Guide: Football Prediction Improvement

## Mission

You are an autoresearch LLM tasked with iteratively improving the football match prediction model's **ROI** and **profit stability**. Your job is to run hypothesis-driven experiments, measure their impact using the established evaluation pipeline, and document findings.

## Single Source of Truth

**`autoresearch/state.md` is the only file used to track iteration results.**

- `docs/improvements.md` is archived — do not read or write to it.
- `autoresearch/state.md` is structured with the **Current Best** at the top (lines 1–30) and the **iteration log** growing at the bottom (chronological, newest last).

## Constraints

**DO NOT touch:**
- `src/data/` — data loading and download logic is frozen
- `src/evaluation/report.py` — HTML report generation is frozen
- `tests/` — existing tests must continue to pass

**You MAY modify:**
- `src/model/features.py` — add or change features
- `src/model/train.py` — change model type, hyperparameters, training strategy
- `src/evaluation/metrics.py` — betting strategy and metric computation
- `main.py` — pipeline wiring
- `autoresearch/state.md` — update after every iteration

## Iteration Protocol

Each iteration MUST follow these steps exactly:

### 1. Read Current State

Read `autoresearch/state.md` in two parts:

```python
# Part 1 — Current Best (always at the top)
Read(file_path="autoresearch/state.md", offset=0, limit=30)

# Part 2 — Recent iterations (last 300 lines = newest experiments)
# First get the line count:
#   wc -l autoresearch/state.md  → N lines
# Then read:
Read(file_path="autoresearch/state.md", offset=N-300, limit=300)
```

Do NOT read the entire file — it is large. The two reads above give you everything you need: the current best metrics and the recent experimental context.

### 2. Form a Hypothesis

Write a clear, falsifiable hypothesis. Example:
> "Adding Elo ratings as features will improve ROI by at least 2% because Elo captures relative team strength better than simple rolling form."

### 3. Implement

Make minimal changes. One hypothesis per iteration — do not bundle unrelated changes.

### 4. Run the Pipeline

```bash
uv run python main.py --per-league
```

Record the output metrics (Accuracy, Bets, ROI, Stability, t-stat).

### 5. Run Tests

```bash
uv run pytest tests/ -v
```

All tests must pass. Fix failures before recording results.

### 6. Analyse Results

Compare to the current best (from step 1). Consider:
- Did ROI improve? By how much?
- Did stability improve or degrade?
- Is the result statistically meaningful (t-stat > 2.0)?
- Is the improvement likely to generalize, or could it be overfitting?

**Keep if:** ROI improves AND stability does not significantly degrade.
**Revert if:** Either metric regresses. Use `git checkout` to revert files.

### 7. Update state.md

**Append** the new iteration entry at the **bottom** of `autoresearch/state.md`. Do not insert it near the top or in the middle.

Also update the **Current Best** block at the very top of the file if this iteration beats the record.

Use this format for the appended entry:

```markdown
## Iteration N: [Short Name] — [KEPT / REVERTED]

**Date:** YYYY-MM-DD
**Hypothesis:** One sentence.
**Files changed:** List of files and what changed.
**Results:**
- ROI: +X.XX% (Δ vs previous best: +X.XXpp)
- Stability: X.XXXX
- t-stat: +X.XX
- Bets: NNNN / MMMM (XX.X%)
**Analysis:** 2–3 sentences on why it worked or didn't.
**Decision:** KEPT / REVERTED. [One sentence reason.]
```

For reverted iterations with no notable findings, a compact 3-line entry is fine.

## What Good Looks Like

- **ROI > 0%** at `threshold=0.0` — beating the bookmakers on all bets
- **t-stat > 2.0** — statistically significant (t = stability × √N_bets)
- **Both metrics improving** is the goal — a high ROI from few lucky bets is not enough

Bookmakers have ~5% margin (vig), so ROI > 0% is genuinely hard.

## Ideas to Explore

You are not limited to these — they are jumping-off points:

**Feature Engineering:**
- More leagues (Belgium B1, Greece G1, Scotland SC0, Turkey T1)
- xG-based features (blocked on data access — see state.md notes)
- Referee ID or referee tendency features
- Market movement features (opening vs closing odds)

**Model / Training:**
- Weighted training (up-weight recent seasons)
- Ensemble of league models
- Calibrated probabilities

**Betting Strategy:**
- Pinnacle margin tuning (currently 1.5% — try 1.0%, 2.0%)
- Maximum odds cap adjustment (currently 4.0)
- Staking strategy experiments
