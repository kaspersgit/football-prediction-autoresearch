# Evaluation policy

Use this policy for every autoresearch iteration. Dated results and sample sizes belong in `current.md` and `experiments.md`.

## Research configuration

Run the primary comparison with:

```bash
uv run python main.py --per-league --threshold 0.0
```

This uses the largest available bet sample and avoids selecting an edge threshold on the test set. Additional thresholds may be reported as secondary diagnostics, but they must not determine whether a model change is kept.

## Keep or revert

Keep a change only when:

1. Total ROI improves.
2. Stability improves or remains effectively neutral.
3. A majority of evaluated leagues improve. Accept a smaller majority only when total ROI and stability move clearly in the right direction.
4. Any change in usable rows or bets has been explained.

Revert when ROI or stability materially regresses. A gain driven by one or two outlier leagues is a warning that the result may be sampling noise.

## Required metrics

Record these values for every iteration:

- Accuracy across all test matches.
- ROI for the default betting portfolio.
- Stability (`mean_profit / std_profit`).
- t-statistic (`stability × sqrt(number_of_bets)`).
- Bets and total test matches.
- Per-league ROI and direction of change.

Use `|t| > 2` as an approximate 95% significance threshold. This interpretation assumes sufficiently independent and representative bets; it is a screening rule, not proof that an observed edge will generalize.

## Interpreting noisy results

Football bets have high outcome variance. Use these rules as rough guidance:

| Observation | Treatment |
|---|---|
| Total ROI change below 1 percentage point | Usually noise. Do not keep on this alone. |
| Total ROI change of 2–4 percentage points | Weak signal. Require support from stability and league directions. |
| Total ROI change above 5 percentage points | Meaningful candidate. Still verify league concentration and sample changes. |
| A clear majority of supported leagues move in the same direction | Stronger evidence than one aggregate result. |
| One league moves by more than 8 percentage points | Potential league-specific signal; do not generalize automatically. |
| Weekly or monthly streak | Expected noise unless it persists over a substantially larger sample. |

## Data-reduction checks

Before adding a feature with a warm-up period or narrower context:

1. Count the training and test rows before and after the change.
2. Report the change in bets.
3. Check whether the apparent improvement remains when compared on an equivalent sample.

Venue-specific rolling form is a known failure under several historical configurations because it reduces usable data and makes form estimates noisier. Do not repeat it without a materially different data setup and a stated reason.

## Cross-iteration evidence

Prefer conclusions supported by multiple independent experiments. Stronger evidence combines:

- the same direction across several changes or datasets;
- improvement in most leagues;
- stability moving with ROI;
- a plausible mechanism that uses information not already encoded in market odds; and
- consistent results after accounting for changes in sample size.

Do not re-test a known-bad feature unless the model, dataset, or evaluation setup changed enough to invalidate the earlier conclusion. Explain that change in the new hypothesis.
