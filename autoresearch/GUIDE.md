# Autoresearch guide

## Mission

Run one hypothesis-driven experiment at a time to improve football betting ROI and profit stability. Use the inference-realistic configuration and preserve enough evidence to distinguish an observed result from a plausible explanation.

## Sources of truth

Read these files before starting an iteration:

1. `autoresearch/current.md` for the latest verified configuration and active hypotheses.
2. `autoresearch/EVALUATION.md` for the mandatory comparison and keep/revert rules.
3. The most recent entries in `autoresearch/experiments.md` for experimental context.

Do not add current-state summaries or experiment results elsewhere.

## File boundaries

Do not modify:

- `src/data/`
- `src/evaluation/report.py`
- existing tests, except when an intentional behaviour change requires a regression test

The normal experiment scope is:

- `src/model/features.py`
- `src/model/train.py`
- `src/evaluation/metrics.py`
- `src/config.py`
- `main.py`
- new or directly relevant tests
- `autoresearch/current.md`
- `autoresearch/experiments.md`

## Iteration protocol

### 1. Establish the baseline

Read the current state, evaluation policy, and recent experiment entries. Confirm the working tree and run the relevant tests before editing.

### 2. Form one hypothesis

State a falsifiable expected outcome and mechanism. Identify the primary metrics and any expected change in rows or bets.

Example:

> Reducing `min_child_samples` from 20 to 15 will improve ROI without reducing stability because it permits moderately finer splits while avoiding the overfitting observed at 10.

### 3. Implement the smallest change

Do not bundle unrelated features, training changes, and betting filters. Add or update tests when runtime behaviour changes.

### 4. Run the primary comparison

```bash
uv run python main.py --per-league --threshold 0.0
```

Record accuracy, ROI, stability, t-statistic, bets, test matches, and the per-league breakdown. Secondary thresholds may be diagnostic only.

### 5. Run tests

```bash
uv run pytest tests/ -v
```

Separate pre-existing failures from failures introduced by the iteration. Do not record an iteration as kept while related tests fail.

### 6. Apply the evaluation policy

Use `autoresearch/EVALUATION.md`. Explain metric deltas, league concentration, sample changes, statistical uncertainty, and whether the mechanism is likely to generalize.

### 7. Keep or revert

Keep the implementation only when it satisfies the evaluation policy. Otherwise revert only the experiment's changes; preserve unrelated work in the repository.

### 8. Record the result

Append the experiment to `autoresearch/experiments.md` using the next globally unique ID:

```markdown
## EXP-YYYYMMDD-NNN: Short name — KEPT / REVERTED

**Date:** YYYY-MM-DD
**Hypothesis:** One sentence.
**Files changed:** File and purpose.
**Baseline:** The exact experiment ID and metrics used for comparison.
**Results:**
- ROI: +X.XX% (change: +X.XX percentage points)
- Stability: X.XXXX
- t-statistic: +X.XX
- Bets: N / M (X.X%)
- Leagues improved: N / M
**Analysis:** Why the observed result supports or rejects the hypothesis, including caveats.
**Decision:** KEPT / REVERTED and why.
```

If kept, update `autoresearch/current.md`. Remove the hypothesis from the active queue whether it was kept or reverted.
