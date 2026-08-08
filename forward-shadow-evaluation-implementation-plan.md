# Forward Shadow Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record, settle, and report immutable forward production predictions with closing-line movement and weekly block-bootstrap ROI uncertainty.

**Architecture:** Keep prediction snapshots and settlements in separate append-only CSV ledgers under `data/shadow/`. A focused shadow module owns record construction, deduplicated appends, and settlement; an uncertainty module owns weekly resampling; a report module joins the ledgers for monitoring. `main.py` only orchestrates these units around the existing prediction path.

**Tech Stack:** Python 3.11, pandas, NumPy, pytest, GitHub Actions, static HTML.

## Global Constraints

- Shadow data begins at `2026-08-04T00:00:00Z`; do not synthesize older predictions.
- Shadow records are monitoring-only and never enter training, threshold calibration, production-league selection, or autoresearch keep/revert decisions.
- `predictions.csv` and `settlements.csv` are append-only; identical IDs are idempotent and conflicting IDs raise `ValueError`.
- Persist only `data/shadow/` from scheduled prediction runs; never stage generated reports or unrelated workspace changes.
- Preserve the existing unstaged changes in `autoresearch/current.md`, `autoresearch/experiments.md`, and `reports/backtest_bets.csv`.
- Amend all implementation work into the existing feature commit `c5eb00e`; do not create additional final commits.

---

### Task 1: Weekly block-bootstrap ROI uncertainty

**Files:**
- Create: `src/evaluation/uncertainty.py`
- Create: `tests/test_uncertainty.py`

**Interfaces:**
- Produces: `weekly_roi_interval(bets: pd.DataFrame, confidence: float = 0.95, n_resamples: int = 10_000, seed: int = 42) -> tuple[float, float] | None`.
- Consumes columns: `Date`, `profit`, and optional `stake` (defaults to one unit).

- [ ] **Step 1: Write failing tests for insufficient weeks and deterministic weekly sampling**

```python
def test_weekly_roi_interval_requires_two_weeks():
    bets = pd.DataFrame({"Date": ["2026-08-04"], "profit": [1.0], "stake": [1.0]})
    assert weekly_roi_interval(bets) is None


def test_weekly_roi_interval_is_deterministic_and_contains_observed_roi():
    bets = pd.DataFrame({
        "Date": ["2026-08-04", "2026-08-05", "2026-08-11", "2026-08-12"],
        "profit": [1.0, -1.0, 2.0, -1.0],
        "stake": [1.0, 1.0, 1.0, 1.0],
    })
    first = weekly_roi_interval(bets, n_resamples=2_000, seed=7)
    second = weekly_roi_interval(bets, n_resamples=2_000, seed=7)
    assert first == second
    assert first[0] <= 25.0 <= first[1]
```

- [ ] **Step 2: Run `uv run pytest tests/test_uncertainty.py -v` and verify import failure**
- [ ] **Step 3: Implement ISO-year/week aggregation and resample complete weekly profit/stake blocks**
- [ ] **Step 4: Run `uv run pytest tests/test_uncertainty.py -v` and verify both tests pass**

### Task 2: Immutable prediction ledger

**Files:**
- Create: `src/evaluation/shadow.py`
- Create: `tests/test_shadow.py`
- Create: `data/shadow/README.md`

**Interfaces:**
- Produces: `build_prediction_records(fixture_features, y_proba, classes, fetched_at, threshold, league_thresholds, run_id, model_commit) -> pd.DataFrame`.
- Produces: `append_ledger(records: pd.DataFrame, path: Path, id_column: str) -> int` returning appended-row count.
- Produces constants `PREDICTION_COLUMNS`, `SETTLEMENT_COLUMNS`, `SHADOW_START_UTC`, `PREDICTIONS_PATH`, and `SETTLEMENTS_PATH`.

- [ ] **Step 1: Write a failing test that one fixture creates canonical `A`, `D`, `H` records with deterministic IDs and captured prices**

```python
def test_build_prediction_records_captures_all_outcomes_and_prices():
    records = build_prediction_records(
        fixture_features=_fixture_features(),
        y_proba=np.array([[0.20, 0.25, 0.55]]),
        classes=["A", "D", "H"],
        fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
        threshold=0.03,
        league_thresholds={"E0": 0.04},
        run_id="run-1",
        model_commit="abc123",
    )
    assert records["outcome"].tolist() == ["A", "D", "H"]
    assert records["prediction_id"].nunique() == 3
    assert records.loc[records.outcome == "H", "captured_best_odds"].item() == 2.0
    assert records.loc[records.outcome == "H", "applied_threshold"].item() == 0.04
```

- [ ] **Step 2: Run the focused test and verify failure because the module is absent**
- [ ] **Step 3: Implement UTC boundary validation, fair probabilities, production filters, best-price fallback, and SHA-256 IDs**
- [ ] **Step 4: Run the focused test and verify pass**
- [ ] **Step 5: Write failing append tests for new rows, identical reruns, and conflicting duplicate IDs**
- [ ] **Step 6: Implement header-preserving CSV append with strict column validation and canonical duplicate comparison**
- [ ] **Step 7: Run `uv run pytest tests/test_shadow.py -v` and verify the ledger tests pass**
- [ ] **Step 8: Document ledger ownership and monitoring-only policy in `data/shadow/README.md`**

### Task 3: Idempotent result settlement and closing-line value

**Files:**
- Modify: `src/evaluation/shadow.py`
- Modify: `tests/test_shadow.py`

**Interfaces:**
- Produces: `build_settlement_records(predictions: pd.DataFrame, settlements: pd.DataFrame, results: pd.DataFrame, settled_at) -> pd.DataFrame`.
- Produces: `settle_shadow_predictions(results: pd.DataFrame, predictions_path: Path = PREDICTIONS_PATH, settlements_path: Path = SETTLEMENTS_PATH, settled_at=None) -> int`.

- [ ] **Step 1: Write failing tests for a completed winning outcome, captured-price profit, and CLV**

```python
def test_build_settlements_uses_captured_profit_and_result_file_closing_proxy():
    settlements = build_settlement_records(
        predictions=_prediction_rows(),
        settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
        results=_completed_result(ftr="H", custom_max_h=1.80),
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )
    home = settlements[settlements.outcome == "H"].iloc[0]
    assert home.profit == pytest.approx(1.0)
    assert home.closing_line_value == pytest.approx(2.0 / 1.8 - 1.0)
```

- [ ] **Step 2: Run the focused test and verify the missing settlement function failure**
- [ ] **Step 3: Implement fixture-key matching, outcome settlement, flat-stake profit, and closing-price fallback**
- [ ] **Step 4: Add failing tests proving pending fixtures remain absent, existing settlements are skipped, and ambiguous results raise**
- [ ] **Step 5: Implement pending/idempotent/ambiguous behavior and append through `append_ledger`**
- [ ] **Step 6: Run `uv run pytest tests/test_shadow.py -v` and verify all settlement tests pass**

### Task 4: Shadow monitoring report

**Files:**
- Create: `src/evaluation/shadow_report.py`
- Create: `tests/test_shadow_report.py`

**Interfaces:**
- Produces: `generate_shadow_report(predictions: pd.DataFrame, settlements: pd.DataFrame, output_path: Path) -> Path`.
- Consumes: `weekly_roi_interval` from Task 1 and the two schemas from Task 2.

- [ ] **Step 1: Write a failing sparse-data report test**

```python
def test_shadow_report_labels_sparse_results_as_informational(tmp_path):
    output = generate_shadow_report(_predictions(), _settlements(), tmp_path / "shadow.html")
    html = output.read_text()
    assert "Forward shadow evaluation" in html
    assert "Informational" in html
    assert "monitoring only" in html.lower()
    assert "closing-line value" in html.lower()
```

- [ ] **Step 2: Run the focused test and verify import failure**
- [ ] **Step 3: Implement joined qualifying-bet metrics, pending counts, ROI/CLV summaries, league and commit tables, and cumulative-profit data**
- [ ] **Step 4: Add a test with at least two ISO weeks and assert the rendered weekly 95% interval**
- [ ] **Step 5: Run `uv run pytest tests/test_shadow_report.py -v` and verify pass**

### Task 5: Historical evaluation weekly interval

**Files:**
- Modify: `src/evaluation/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Consumes: `weekly_roi_interval(results_df)`.
- Produces: an initial all-market summary card labelled `Weekly 95% ROI interval`; interactive filters do not relabel the static interval as filtered.

- [ ] **Step 1: Add a failing report test with bets spanning two ISO weeks**
- [ ] **Step 2: Run `uv run pytest tests/test_report.py -v` and verify the missing interval text**
- [ ] **Step 3: Compute and render the interval, or `Not enough completed weeks` when unavailable**
- [ ] **Step 4: Run `uv run pytest tests/test_report.py -v` and verify pass**

### Task 6: Prediction and settlement orchestration

**Files:**
- Modify: `main.py`
- Create: `tests/test_shadow_cli.py`

**Interfaces:**
- Produces: `_settle_and_report_shadow(results_df, settled_at) -> None`.
- Produces: `_record_shadow_predictions(fixture_features, y_proba, classes, fetched_at, threshold, league_thresholds) -> int`.
- Adds CLI mode: `--settle-shadow`.

- [ ] **Step 1: Write a failing routing test showing `--settle-shadow` invokes settlement without inference**
- [ ] **Step 2: Run the focused test and verify the route is absent**
- [ ] **Step 3: Add the manual route: refresh results, load history, settle, and generate `reports/shadow_evaluation.html`**
- [ ] **Step 4: Write a failing orchestration test proving prediction snapshots contain three outcomes per retained production fixture**
- [ ] **Step 5: Integrate settlement immediately after `load_all_data()` and append predictions after probability generation**
- [ ] **Step 6: Generate the shadow report on normal, empty-fixture, and manual-settlement paths**
- [ ] **Step 7: Run `uv run pytest tests/test_shadow_cli.py tests/test_predictions_report.py -v` and verify pass**

### Task 7: Scheduled persistence and policy documentation

**Files:**
- Modify: `.github/workflows/predict.yml`
- Modify: `README.md`
- Modify: `autoresearch/EVALUATION.md`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Workflow appends and commits only `data/shadow/predictions.csv` and `data/shadow/settlements.csv`.
- Workflow deploys `reports/shadow_evaluation.html` as `docs/shadow.html`.

- [ ] **Step 1: Write a failing workflow test that parses YAML text and requires a concurrency group, shadow-only `git add`, and shadow Pages staging**
- [ ] **Step 2: Run `uv run pytest tests/test_workflows.py -v` and verify failure**
- [ ] **Step 3: Add workflow concurrency, commit identity, guarded append-only commit/push, and shadow report staging**
- [ ] **Step 4: Document local `--settle-shadow`, the shadow URL/artifacts, hypothetical execution, and the monitoring-only development boundary**
- [ ] **Step 5: Run `uv run pytest tests/test_workflows.py -v` and verify pass**

### Task 8: Full verification and single-commit delivery

**Files:**
- Amend: existing commit `c5eb00e`
- Preserve unstaged: `autoresearch/current.md`, `autoresearch/experiments.md`, `reports/backtest_bets.csv`

- [ ] **Step 1: Run `uv run pytest tests/ -v` and require zero failures**
- [ ] **Step 2: Run Ruff on every changed Python file and require zero errors**
- [ ] **Step 3: Run `python -m py_compile` on every changed Python module**
- [ ] **Step 4: Parse both workflow YAML files and run `git diff --check`**
- [ ] **Step 5: Confirm no changed production file reads or prints environment values other than `GITHUB_SHA` and `GITHUB_RUN_ID`**
- [ ] **Step 6: Stage only planned feature files, verify the three unrelated artifacts remain unstaged, and amend `c5eb00e`**
- [ ] **Step 7: Fetch `origin/dev`, then push with an explicit force-with-lease only if the remote still points to `1170e81`**
- [ ] **Step 8: Verify `origin/dev` points to the amended feature commit and report the test evidence**
