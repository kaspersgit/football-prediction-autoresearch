# Autoresearch Guide: Football Prediction Improvement

## Mission

You are an autoresearch LLM tasked with iteratively improving the football match prediction model's **ROI** and **profit stability**. Your job is to run hypothesis-driven experiments, measure their impact using the established evaluation pipeline, and document findings.

## Constraints

**DO NOT touch:**
- `src/data/` — data loading and download logic is frozen
- `src/evaluation/report.py` — HTML report generation is frozen
- `tests/` — existing tests must continue to pass

**You MAY modify:**
- `src/model/features.py` — add or change features
- `src/model/train.py` — change model type, hyperparameters, training strategy
- `src/evaluation/metrics.py` — betting strategy and metric computation (unfrozen as of Iteration 4)
- `main.py` — pipeline wiring, to connect new metric signatures (unfrozen as of Iteration 4)
- `autoresearch/state.md` — update after every iteration

## Iteration Protocol

Each iteration MUST follow these steps exactly:

### 1. Read Current State
Read `autoresearch/state.md` to understand what has been tried, what worked, and what the current best metrics are.

### 2. Form a Hypothesis
Write a clear, falsifiable hypothesis. Example:
> "Adding Elo ratings as features will improve ROI by at least 2% because Elo captures relative team strength better than simple rolling form."

### 3. Implement
Make minimal changes to `src/model/features.py` and/or `src/model/train.py`. Keep changes focused on testing one hypothesis at a time. Do not make multiple independent changes in one iteration — you won't know which helped.

### 4. Run the Pipeline

```bash
uv run python main.py                   # value betting, threshold=0.0 (any positive edge)
uv run python main.py --threshold 0.05  # require at least 5% edge over fair odds
```

Record the output metrics (Accuracy, Threshold, Bets, ROI, Stability).
A profit curve PNG is saved to `reports/profit_curve.png` automatically on each run.

### 5. Run Tests

```bash
uv run pytest tests/ -v
```

All tests must pass. If they fail, fix the issue before recording results.

### 6. Analyse Results
Compare new metrics to the baseline and best-so-far. Consider:
- Did ROI improve? By how much?
- Did stability improve or degrade?
- Were the changes in the expected direction? Why or why not?
- Is the improvement likely to generalize, or could it be overfitting to the test period?

### 7. Update state.md
Add an entry to `autoresearch/state.md` following the format in that document. Update the "Current Best" section if this iteration beats the record.

### 8. Propose Next Directions
At the end of your state.md update, list 2-3 concrete next hypotheses to try, ranked by your confidence they will improve ROI.

## Ideas to Explore (Starting Points)

You are not limited to these — they are jumping-off points:

**Feature Engineering:**
- Elo rating system (track per-team Elo updated after each match)
- Head-to-head historical record between the two teams
- Days since last match (fatigue proxy)
- Home/away split form (separate rolling stats for home games vs away games)
- League position / points in current season
- Goal difference rolling average (not just goals for/against separately)
- Weighted rolling average (recent games weighted more heavily)

**Model Architecture:**
- Random Forest or Gradient Boosting (XGBoost, LightGBM)
- Separate models per league
- Calibrated probabilities (CalibratedClassifierCV)
- Threshold-based betting: only bet when model confidence exceeds a threshold

**Betting Strategy:**
- Kelly criterion bet sizing instead of flat 1 unit
- Only bet on outcomes where model probability > bookmaker implied probability (value bets)
- Only bet on matches where model disagrees strongly with the market

**Data:**
- Longer or shorter rolling window (try 3, 7, 10 games)
- Season-start correction (treat first N games of season differently)

## What Good Looks Like

- **ROI > 0%** means you're making money (beating the bookmakers)
- **Stability > 0.05** means profits are consistent rather than from a few lucky bets
- **Both improving** is the goal — a high ROI from 10 lucky bets is not good

Bookmakers have ~5% margin (vig), so achieving ROI > 0% is genuinely hard and means you've found edge.

## Output Format for Each Iteration

When you complete an iteration, output a summary in this format:

```
## Iteration N: [Hypothesis Name]

**Hypothesis:** [One sentence]
**Changes:** [What files were changed and how]
**Results:**
  - Accuracy: X.XXX (baseline: 0.XXX)
  - ROI: +X.XX% (baseline: X.XX%)
  - Stability: X.XXXX (baseline: X.XXXX)
**Analysis:** [2-3 sentences on why it worked or didn't]
**Next directions:** [2-3 ranked ideas]
```
