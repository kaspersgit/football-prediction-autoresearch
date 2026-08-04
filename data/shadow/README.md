# Forward shadow ledger

`predictions.csv` and `settlements.csv` are the durable, append-only record for forward
shadow evaluation. The scheduled prediction workflow owns these files and commits them
after a successful run; generated reports are not the source of record.

Each prediction run stores all three outcomes for every eligible production fixture with
the model probabilities and prices captured at fetch time. `is_value_bet` is true only
when the outcome passes the live threshold, maximum edge, maximum B365 odds, and fixture
overround filters. A matching settlement is written separately once a completed result
is available. Existing IDs are idempotent on append; duplicate rows already stored in a
ledger are malformed, and a record with the same ID but different content is an error.

`model_commit` identifies the latest Git commit that changed production code or
configuration (`main.py`, `predict.sh`, `src/`, `models/`, the prediction workflow, or
the Python dependency files). Scheduled ledger-only commits therefore remain grouped
under the same deployed model revision. The workflow fetches full Git history to resolve
this value deterministically.

This data is monitoring-only. We do not use it to train models, calibrate thresholds,
select production leagues, or decide autoresearch keep/revert outcomes. It begins at
`2026-08-04T00:00:00Z`; historical match files only settle predictions already present
in this ledger.
