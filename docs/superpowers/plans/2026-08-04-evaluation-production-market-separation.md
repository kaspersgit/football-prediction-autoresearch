# Evaluation and Production Market Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate every supported 1X2 league continuously while allowing live predictions only for an explicit production league subset.

**Architecture:** `src/config.py` owns the supported league metadata and production allowlist. Training, downloads, evaluation, reports, and inference consume that configuration. Evaluation produces an all-market dataset at the research threshold plus a separate production simulation, while the report dynamically renders every observed league and defaults to all markets.

**Tech Stack:** Python 3.11, pandas, NumPy, LightGBM, pytest, embedded HTML/JavaScript reports.

## Global Constraints

- `SUPPORTED_LEAGUES` contains E0, D1, SP1, I1, F1, N1, P1, G1, SC0, B1, and T1.
- `PRODUCTION_LEAGUES` contains E0, N1, P1, and G1.
- Evaluation must not apply the production allowlist.
- Production inference must not emit fixtures outside `PRODUCTION_LEAGUES`.
- The repository supports 1X2 outcomes only; no new betting type is added.
- Preserve the branch as one commit by amending only after all verification passes.

---

### Task 1: Canonical League Configuration

**Files:**
- Modify: `src/config.py`
- Modify: `src/data/download.py`
- Modify: `src/data/loader.py`
- Modify: `src/model/train.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `LEAGUE_NAMES: dict[str, str]`, `SUPPORTED_LEAGUES: tuple[str, ...]`, `PRODUCTION_LEAGUES: frozenset[str]`, and derived `EXCLUDED_BETTING_LEAGUES`.
- Consumes: Existing betting limits in `src/config.py`.

- [ ] **Step 1: Add failing configuration tests**

```python
from src.config import (
    EXCLUDED_BETTING_LEAGUES,
    LEAGUE_NAMES,
    PRODUCTION_LEAGUES,
    SUPPORTED_LEAGUES,
)


def test_production_leagues_are_supported():
    assert PRODUCTION_LEAGUES <= set(SUPPORTED_LEAGUES)
    assert EXCLUDED_BETTING_LEAGUES == set(SUPPORTED_LEAGUES) - PRODUCTION_LEAGUES


def test_all_known_leagues_are_supported():
    assert set(SUPPORTED_LEAGUES) == {
        "E0", "D1", "SP1", "I1", "F1", "N1", "P1", "G1", "SC0", "B1", "T1"
    }
    assert set(LEAGUE_NAMES) == set(SUPPORTED_LEAGUES)
```

- [ ] **Step 2: Run the configuration tests and verify they fail**

Run: `uv run pytest tests/test_config.py -v`

Expected: import failures for the new configuration names.

- [ ] **Step 3: Implement the canonical configuration**

```python
LEAGUE_NAMES = {
    "E0": "England", "D1": "Germany", "SP1": "Spain", "I1": "Italy",
    "F1": "France", "N1": "Netherlands", "P1": "Portugal", "G1": "Greece",
    "SC0": "Scotland", "B1": "Belgium", "T1": "Turkey",
}
SUPPORTED_LEAGUES = tuple(LEAGUE_NAMES)
PRODUCTION_LEAGUES = frozenset({"E0", "N1", "P1", "G1"})
EXCLUDED_BETTING_LEAGUES = frozenset(set(SUPPORTED_LEAGUES) - PRODUCTION_LEAGUES)
```

Replace local league lists in the downloader, fixture loader, and trainer with imports
from `src.config`. The downloader's public `LEAGUES` mapping remains available but is
derived as `{name.lower(): code for code, name in LEAGUE_NAMES.items()}`.

- [ ] **Step 4: Run the configuration and loader tests**

Run: `uv run pytest tests/test_config.py tests/test_loader.py -v`

Expected: all tests pass.

---

### Task 2: Production Fixture Boundary

**Files:**
- Modify: `main.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `_filter_production_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame`.
- Consumes: `PRODUCTION_LEAGUES` from `src.config`.

- [ ] **Step 1: Add a failing behavior test**

```python
def test_production_fixture_filter_uses_allowlist():
    fixtures = pd.DataFrame({"league": ["E0", "D1", "G1", "B1"], "id": [1, 2, 3, 4]})
    result = main._filter_production_fixtures(fixtures)
    assert result["id"].tolist() == [1, 3]
```

This test catches any production change that accidentally emits a supported but
unapproved league.

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run pytest tests/test_config.py::test_production_fixture_filter_uses_allowlist -v`

Expected: `_filter_production_fixtures` is missing.

- [ ] **Step 3: Implement and use the production boundary**

```python
def _filter_production_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    return fixtures[fixtures["league"].isin(PRODUCTION_LEAGUES)].reset_index(drop=True)
```

Call the helper in `_run_predict` after feature construction. Keep all supported league
models trained; only filter immediately before probability generation and report output.
If no fixture remains, return with a clear message instead of passing an empty frame to
the model loop.

- [ ] **Step 4: Run the focused test**

Run: `uv run pytest tests/test_config.py -v`

Expected: all tests pass.

---

### Task 3: All-Market Evaluation and Separate Artifacts

**Files:**
- Modify: `main.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: all-market `evaluation_results`, production `production_results`, `reports/evaluation_bets.csv`, and `reports/backtest_bets.csv`.
- Consumes: walk-forward output, `SUPPORTED_LEAGUES`, `PRODUCTION_LEAGUES`, and existing betting limits.

- [ ] **Step 1: Add a failing evaluation boundary test**

Extract a small helper with this contract:

```python
def _compute_market_bets(eval_df, y_proba, classes, threshold, max_odds, max_edge,
                         min_season_games, leagues=None):
    ...
```

Test it with E0 and D1 rows:

```python
all_bets = _compute_market_bets(..., leagues=None)
production_bets = _compute_market_bets(..., leagues=PRODUCTION_LEAGUES)
assert set(all_bets["league"]) == {"E0", "D1"}
assert set(production_bets["league"]) == {"E0"}
```

The helper must attach league metadata before returning.

- [ ] **Step 2: Run the new test and verify it fails**

Run: `uv run pytest tests/test_config.py::test_evaluation_keeps_non_production_leagues -v`

Expected: `_compute_market_bets` is missing.

- [ ] **Step 3: Implement the helper and restructure `_run_backtest`**

Use `compute_value_betting_results` with no `skip_leagues` for all-market evaluation.
When `leagues` is supplied, derive the skip set as
`set(SUPPORTED_LEAGUES) - set(leagues)`. Always merge the league lookup into the result.

In `_run_backtest`:

1. Keep prior-season threshold calibration, but calculate thresholds for
   `SUPPORTED_LEAGUES`.
2. Build the production simulation using calibrated thresholds and only
   `PRODUCTION_LEAGUES`.
3. Build the primary evaluation using all supported leagues and the fixed CLI research
   threshold.
4. Calculate printed and report metrics from the all-market evaluation.
5. Write all-market bets to `reports/evaluation_bets.csv`.
6. Write production simulation bets to `reports/backtest_bets.csv`.
7. Pass all-market results to `generate_report`.

- [ ] **Step 4: Run focused evaluation tests**

Run: `uv run pytest tests/test_config.py tests/test_metrics.py tests/test_threshold_selector.py -v`

Expected: all tests pass.

---

### Task 4: Dynamic All-Market Report Controls

**Files:**
- Modify: `src/evaluation/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `generate_report(..., production_leagues: set[str] | None = None)` with
  dynamic league controls.
- Consumes: league values embedded in `results_df` and `all_predictions`.

- [ ] **Step 1: Add a failing report behavior test**

Extend the report fixture with E0, D1, G1, and B1 rows, then call:

```python
generate_report(..., production_leagues={"E0", "G1"})
```

Assert the generated HTML contains one button for each observed league, initializes each
league to `true`, and embeds the production list separately:

```python
for league in ["E0", "D1", "G1", "B1"]:
    assert f'id="btn-lg-{league}"' in content
assert 'var _productionLeagues=["E0","G1"]' in content
assert "showProductionMarkets()" in content
```

- [ ] **Step 2: Run the report test and verify it fails**

Run: `uv run pytest tests/test_report.py -v`

Expected: G1/B1 buttons or the production preset are absent.

- [ ] **Step 3: Generate controls from data**

Add helpers that derive sorted observed leagues, render the buttons, and serialize the
initial/production state. Replace the seven hardcoded buttons and `_activeLeagues`
literal with template placeholders. Every observed league starts active.

Add two preset buttons:

```javascript
function showAllMarkets() {
  Object.keys(_activeLeagues).forEach(function(league) { _activeLeagues[league] = true; });
  syncLeagueButtons();
  applyFilters();
}

function showProductionMarkets() {
  Object.keys(_activeLeagues).forEach(function(league) {
    _activeLeagues[league] = _productionLeagues.indexOf(league) >= 0;
  });
  syncLeagueButtons();
  applyFilters();
}
```

Pass `production_leagues=PRODUCTION_LEAGUES` from `_run_backtest`.

- [ ] **Step 4: Run report tests**

Run: `uv run pytest tests/test_report.py tests/test_predictions_report.py -v`

Expected: all tests pass.

---

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `autoresearch/EVALUATION.md`
- Modify: `autoresearch/current.md`

**Interfaces:**
- Consumes: final behavior and artifact names from Tasks 1–4.
- Produces: user-facing commands and evaluation/production contract.

- [ ] **Step 1: Update documentation**

Document that evaluation always includes all supported leagues at the research threshold,
the report defaults to all markets, production uses its allowlist, and the two bet CSVs
have separate purposes.

- [ ] **Step 2: Run static checks**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 3: Run the complete test suite**

Run: `uv run pytest`

Expected: 28 or more tests pass with no failures.

- [ ] **Step 4: Review the final diff against the design**

Run: `git diff --stat` and inspect `git diff` for `src/config.py`, `main.py`,
`src/evaluation/report.py`, and the updated tests. Confirm that production filtering is
absent from the all-market evaluation path and present in live inference.

- [ ] **Step 5: Preserve the single commit and publish**

Stage the implementation, amend `Improve autoresearch evaluation workflow`, run the
verification commands again against the amended tree, and push with
`git push --force-with-lease origin dev`.
