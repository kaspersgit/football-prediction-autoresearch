# Forward shadow evaluation

## Goal

Add a forward-only record of production predictions, the prices available when they
were generated, and their later outcomes. This provides an execution-realistic check
alongside the historical walk-forward evaluation without turning the new data into
another repeatedly tuned development set.

## Relationship with normal development

Normal model development continues to use the four-season, all-market walk-forward
evaluation. It remains the fast comparison mechanism for features, model parameters,
and league-level behavior.

The shadow ledger is an after-the-fact production monitor. It records only the model
version actually used by the scheduled prediction workflow. Shadow observations do
not become training features, threshold-calibration input, or evidence for the normal
autoresearch keep/revert rule. They may identify a production mismatch or support a
rollback, but a replacement model must still be selected through the historical
evaluation.

The report will group observations by Git commit so that a production model change is
visible. Comparisons between commits remain observational because they cover different
calendar periods.

Shadow reporting starts immediately, but sparse data is explicitly labelled
`informational`. There is no fixed bet-count threshold that turns it into a release
gate. The weekly block-bootstrap confidence interval shows when the evidence becomes
more precise.

## Storage design

Two CSV files under `data/shadow/` form the durable audit trail:

- `predictions.csv` contains immutable prediction-time observations.
- `settlements.csv` contains immutable outcomes linked by prediction ID.

Both files are committed by the prediction workflow. A workflow concurrency group
prevents simultaneous append operations. Appending an existing ID is a no-op, while a
conflicting record with an existing ID is an error. Settlement never updates a row in
`predictions.csv`.

This design is preferred over a mutable combined CSV because it preserves the original
prediction snapshot. It is preferred over one file per run because the two ledgers are
easier to analyze and do not create an unbounded number of small files.

## Prediction records

Each production fixture produces three prediction records, one for each of `H`, `D`,
and `A`. Recording all outcomes supports probability calibration and avoids retaining
only successful-looking value selections.

Each record contains:

- `prediction_id`: SHA-256 of run ID, fixture identity, and outcome;
- `run_id`: timestamp-based identifier shared by one prediction execution;
- `model_commit`: the latest Git commit that changed production code or configuration
  (`main.py`, `predict.sh`, `src/`, `models/`, the prediction workflow, or the Python
  dependency files). This remains stable across scheduled ledger-only commits;
- `fetched_at`: UTC timestamp associated with the odds snapshot;
- fixture date, league, home team, away team, and outcome;
- model probability and vig-normalized B365 fair probability;
- captured B365 odds;
- captured best available odds and bookmaker name;
- model edge, applied league threshold, and `is_value_bet`. The flag is true only when
  the live threshold, maximum edge, maximum B365 odds, and fixture-overround filters all
  pass;
- production-market eligibility.

The ledger records the odds delivered by the fixture feed at prediction time. It does
not claim that a human placed the bet or that the price accepted a specific stake.

## Settlement records

Settlement runs after completed results have been refreshed and before a new prediction
snapshot is appended. A prediction is settled when league, fixture date, home team, and
away team match a completed result.

Each settlement contains:

- `prediction_id` and UTC `settled_at`;
- actual match result and whether the predicted outcome won;
- hypothetical flat-stake profit at the captured best price;
- result-file B365 odds and best available odds for the predicted outcome;
- closing-line value based on captured versus result-file price.

Closing-line value is calculated as:

```text
captured_best_odds / result_file_best_odds - 1
```

A positive value means the captured price was better than the result-file closing-price
proxy. The label remains explicit because football-data.co.uk does not guarantee that
every historical bookmaker field represents the exact market close.

Predictions without a matching completed result remain pending. Re-running settlement
does not duplicate a settlement. Conflicting duplicate settlements fail loudly.

## Statistical reporting

The shadow report includes:

- prediction runs, settled and pending fixtures;
- settled qualifying value bets;
- flat-stake profit and ROI;
- win rate;
- mean and median closing-line value;
- cumulative profit over time;
- results by league and model commit;
- a 95% weekly block-bootstrap interval for ROI.

The historical evaluation report uses the same weekly interval. Bets are grouped by ISO
year and week, and complete weeks are sampled with replacement. This retains correlation
between bets from the same matchweek. Bootstrap output uses a fixed random seed for
reproducible reports. An interval is omitted when fewer than two distinct weeks exist.

The shadow report states that its result is hypothetical execution at the captured
quote, not evidence of accepted real wagers.

## Forward-only boundary

The earliest allowed prediction timestamp is `2026-08-04T00:00:00Z`. The system never
synthesizes earlier records from historical match files. Historical results are used
only to settle predictions that already exist in the forward ledger.

The autoresearch policy states that shadow rows are monitoring data and must not be used
to select features, tune thresholds, or choose production leagues. This protects the
only forward observation period from the repeated-experiment selection bias present in
the historical evaluation.

## Automation and CLI behavior

`main.py --predict` performs the following sequence:

1. refresh completed match results;
2. settle pending shadow predictions;
3. train production models and calculate upcoming predictions;
4. append all fixture-outcome records;
5. generate the existing prediction report and the shadow evaluation report.

`main.py --settle-shadow` refreshes results, settles pending rows, and regenerates the
shadow report without producing a new prediction snapshot. This provides a manual
recovery path when a scheduled prediction run is unavailable.

The prediction workflow checks out the current branch ref after the concurrency wait,
with full Git history, and commits changes under `data/shadow/` after a successful run.
This lets a queued run include the previous run's ledger append and provides enough
history to resolve `model_commit`. A non-fast-forward push remains a hard failure.
Generated HTML remains deployed through GitHub Pages and is not used as the durable
source of record.

## Failure handling

- Missing ledger files are treated as empty ledgers and created on the first append.
- Malformed ledgers, non-canonical column order, empty or duplicate IDs, invalid value
  domains, orphan settlements, or ambiguous completed fixture matches fail the run.
- Missing results leave predictions pending.
- Missing closing-price fields still permit result and profit settlement; closing-line
  value is left empty.
- No shadow storage operation reads an environment variable other than `GITHUB_RUN_ID`,
  and that value is not rendered as a secret.

## Testing

Automated tests cover:

- deterministic prediction IDs and conversion from prediction rows;
- append-only idempotency and conflicting duplicate rejection;
- completed, pending, and ambiguous settlement behavior;
- hypothetical profit and closing-line-value calculations;
- ISO-week block construction and deterministic bootstrap intervals;
- insufficient-week behavior;
- shadow report content and sparse-data label;
- prediction workflow integration and manual settlement routing;
- the existing historical evaluation and prediction regression suite.

## Acceptance criteria

- Every scheduled production prediction run appends its exact fixture-outcome snapshot.
- Existing prediction and settlement records are never mutated or duplicated.
- Completed predictions settle automatically from refreshed results.
- Shadow reporting includes price movement and weekly ROI uncertainty.
- Historical evaluation reports the same weekly ROI uncertainty method.
- Shadow data begins on or after 2026-08-04 and is excluded from model development.
- The prediction workflow persists ledger changes safely.
