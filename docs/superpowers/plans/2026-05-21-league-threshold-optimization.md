# League-Specific Threshold Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single global edge threshold with a per-league threshold dict, optimized on prior out-of-sample walk-forward data, leaving the LGBM model and feature pipeline untouched.

**Architecture:** `train_walkforward` is extended to return per-season OOS data. `_run_backtest` loops over seasons, calibrating per-league thresholds from all prior test seasons before evaluating each new season. A new pure function `select_league_thresholds` handles the sweep. Live prediction loads thresholds from `models/league_thresholds.json`.

**Tech Stack:** Python, pandas, numpy, LightGBM (existing), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/evaluation/threshold_selector.py` | `select_league_thresholds` — pure threshold sweep function |
| Create | `tests/test_threshold_selector.py` | Unit tests for threshold selector |
| Modify | `src/model/train.py` | Expose `season_results` + `eval_df` in return dict; `TEST_SEASONS = 4` |
| Modify | `main.py` | Per-season calibration loop; save `league_thresholds.json`; per-league threshold in `_run_predict` / `_build_prediction_rows` |
| Modify | `predict.sh` | Remove `--threshold 0.04` injection |

---

## Task 1: Expose per-season OOS data from `train_walkforward`

**Files:**
- Modify: `src/model/train.py`

The `season_results` list is already built inside `train_walkforward` but never returned. Each entry needs an `eval_df` (odds_test with `y_true` added) and the `season` label. This is needed by `select_league_thresholds` and the per-season calibration loop.

Also change `TEST_SEASONS = 4` — more test seasons give better calibration signal and reduce noise.

- [ ] **Step 1: Change TEST_SEASONS and add eval_df to season_results in all four code paths**

In `src/model/train.py`, make the following changes:

```python
TEST_SEASONS = 4  # was 2; 4 seasons gives 2 seasons of calibration + 2 of evaluation
```

In the `per_league and binary_outcomes` branch (first `if`), change:
```python
season_results.append({
    "y_pred": y_pred, "y_proba": y_proba, "y_true": y_true,
    "odds_test": odds_test, "classes": _CLASSES,
})
```
to:
```python
_eval_df = odds_test.reset_index(drop=True).copy()
_eval_df["y_true"] = y_true
season_results.append({
    "y_pred": y_pred, "y_proba": y_proba, "y_true": y_true,
    "odds_test": odds_test, "classes": _CLASSES,
    "eval_df": _eval_df, "season": test_season,
})
```

Apply the same `eval_df` / `season` additions to the other three branches (`elif per_league`, `elif binary_outcomes`, `else`). Each already builds `odds_test` and has `y_true` available — just replicate the two new lines above `season_results.append(...)` in each branch.

- [ ] **Step 2: Return season_results from train_walkforward**

At the end of `train_walkforward`, add `"season_results": season_results` to the return dict:

```python
return {
    "y_pred": y_pred_all,
    "y_proba": y_proba_all,
    "y_test": pd.Series(y_true_all),
    "classes": season_results[-1]["classes"],
    "odds_test": odds_all,
    "accuracy": accuracy,
    "season_results": season_results,   # NEW
}
```

- [ ] **Step 3: Verify the change hasn't broken anything**

```bash
uv run pytest tests/ -x -q
```

Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/model/train.py
git commit -m "feat: expose season_results from train_walkforward; TEST_SEASONS=4"
```

---

## Task 2: Create `select_league_thresholds`

**Files:**
- Create: `src/evaluation/threshold_selector.py`
- Create: `tests/test_threshold_selector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_threshold_selector.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.evaluation.threshold_selector import select_league_thresholds


def _make_season_data(n: int, league: str, edge: float, classes=None) -> dict:
    """
    Synthetic season: `n` matches where the model always has `edge` advantage over fair.
    All bets are correct (profit = odds - 1) to produce positive ROI.
    Odds fixed at 2.0 for simplicity (overround ~0 for test purposes — set overround filter high).
    """
    if classes is None:
        classes = np.array(["A", "D", "H"])
    rng = np.random.default_rng(42)
    n_classes = len(classes)
    # fair prob = 1/3 for each outcome; model gives +edge to "H"
    y_proba = np.full((n, n_classes), 1 / n_classes)
    h_idx = list(classes).index("H")
    y_proba[:, h_idx] += edge
    # model always predicts H and is always correct
    eval_df = pd.DataFrame({
        "Date": pd.date_range("2023-08-01", periods=n, freq="7D"),
        "HomeTeam": [f"Home{i}" for i in range(n)],
        "AwayTeam": [f"Away{i}" for i in range(n)],
        "league": league,
        "season": "2023-24",
        "y_true": "H",
        "B365H": 2.0,
        "B365D": 3.4,
        "B365A": 4.0,
        "PSCH": 2.0,
        "PSCD": 3.4,
        "PSCA": 4.0,
    })
    return {"eval_df": eval_df, "y_proba": y_proba, "classes": classes}


def test_picks_higher_threshold_when_it_improves_stability():
    """When a higher threshold improves ROI×√bets, it should be picked."""
    # 100 bets with edge 0.03: at threshold=0.0 all 100 bets pass.
    # At threshold=0.025, all 100 still pass (edge 0.03 > 0.025).
    # At threshold=0.04, 0 bets pass (edge 0.03 < 0.04) → below min_bets.
    # So both 0.0 and 0.025 qualify; same bets, same stability → lower wins (0.0).
    data = [_make_season_data(100, "E0", edge=0.03)]
    result = select_league_thresholds(data, leagues=["E0"], grid=[0.0, 0.025, 0.04])
    assert result["E0"] == 0.0  # tie broken by lower threshold


def test_falls_back_to_default_when_min_bets_not_met():
    """If no threshold produces >= min_bets bets, return the default."""
    data = [_make_season_data(5, "E0", edge=0.03)]  # only 5 bets, min_bets=20
    result = select_league_thresholds(
        data, leagues=["E0"], grid=[0.0, 0.02, 0.04], min_bets=20, default_threshold=0.0
    )
    assert result["E0"] == 0.0


def test_returns_default_when_no_prior_data():
    """Empty prior_season_data → all leagues get default threshold."""
    result = select_league_thresholds(
        [], leagues=["E0", "N1"], grid=[0.0, 0.02], default_threshold=0.0
    )
    assert result == {"E0": 0.0, "N1": 0.0}


def test_per_league_thresholds_are_independent():
    """Each league gets its own threshold independently."""
    data_e0 = _make_season_data(100, "E0", edge=0.06)
    data_n1 = _make_season_data(3, "N1", edge=0.06)  # too few for N1
    data = [
        {
            "eval_df": pd.concat([data_e0["eval_df"], data_n1["eval_df"]]).reset_index(drop=True),
            "y_proba": np.vstack([data_e0["y_proba"], data_n1["y_proba"]]),
            "classes": data_e0["classes"],
        }
    ]
    result = select_league_thresholds(
        data, leagues=["E0", "N1"], grid=[0.0, 0.02, 0.04], min_bets=20, default_threshold=0.0
    )
    # E0 has enough bets at any threshold; N1 does not → N1 gets default
    assert result["N1"] == 0.0
    assert "E0" in result


def test_missing_league_gets_default():
    """A league with no rows in prior data gets the default threshold."""
    data = [_make_season_data(50, "E0", edge=0.03)]
    result = select_league_thresholds(
        data, leagues=["E0", "G1"], grid=[0.0, 0.02], default_threshold=0.0
    )
    assert result["G1"] == 0.0
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
uv run pytest tests/test_threshold_selector.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.evaluation.threshold_selector'`

- [ ] **Step 3: Implement `select_league_thresholds`**

Create `src/evaluation/threshold_selector.py`:

```python
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_roi, compute_value_betting_results


def select_league_thresholds(
    prior_season_data: list[dict],
    leagues: list[str],
    grid: list[float],
    min_bets: int = 20,
    default_threshold: float = 0.0,
    max_odds: float = float("inf"),
    max_overround: float = float("inf"),
    max_edge: float = float("inf"),
) -> dict[str, float]:
    """
    For each league, sweep `grid` thresholds on accumulated prior-season OOS data
    and return the threshold that maximises ROI × √bets (stability), subject to
    a minimum of `min_bets` bets over the calibration window.

    Falls back to `default_threshold` when:
    - `prior_season_data` is empty, or
    - the league has no rows, or
    - no threshold in `grid` produces >= `min_bets` bets.

    Tie-breaking: lower threshold wins (preserves more bets, more conservative).

    Each entry in prior_season_data must have:
        eval_df  — DataFrame with y_true, B365H/D/A, Date, league, season columns
        y_proba  — ndarray shape (n_rows, n_classes), row-aligned to eval_df
        classes  — ndarray of class labels (e.g. ['A','D','H'])
    """
    if not prior_season_data:
        return {lg: default_threshold for lg in leagues}

    combined_df = pd.concat(
        [sd["eval_df"].reset_index(drop=True) for sd in prior_season_data]
    ).reset_index(drop=True)
    combined_proba = np.vstack([sd["y_proba"] for sd in prior_season_data])
    classes = prior_season_data[-1]["classes"]

    result: dict[str, float] = {}
    all_leagues_in_data = set(combined_df["league"].unique())

    for league in leagues:
        if league not in all_leagues_in_data:
            result[league] = default_threshold
            continue

        mask = combined_df["league"].values == league
        league_df = combined_df[mask].reset_index(drop=True)
        league_proba = combined_proba[mask]

        best_threshold = default_threshold
        best_stability = -np.inf

        for t in sorted(grid):  # ascending → lower threshold wins on equal stability
            bets = compute_value_betting_results(
                league_df,
                league_proba,
                classes,
                threshold=t,
                max_odds=max_odds,
                max_overround=max_overround,
                max_edge=max_edge,
            )
            n = len(bets)
            if n < min_bets:
                continue
            roi = compute_roi(bets)
            stability = roi * (n ** 0.5)
            if stability > best_stability:
                best_stability = stability
                best_threshold = t

        result[league] = best_threshold

    return result
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
uv run pytest tests/test_threshold_selector.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/threshold_selector.py tests/test_threshold_selector.py
git commit -m "feat: add select_league_thresholds for per-league threshold calibration"
```

---

## Task 3: Wire per-season calibration loop into `_run_backtest`

**Files:**
- Modify: `main.py`

Replace the single `compute_value_betting_results` call in `_run_backtest` with a loop over `results["season_results"]`. For each season: calibrate per-league thresholds from all prior seasons, apply those thresholds by calling `compute_value_betting_results` once per active league. Concatenate per-season bet chunks.

The `all_leagues_bets` block (used for the interactive HTML report) is left using the global `threshold` — it is a visualization aid and does not drive betting decisions.

- [ ] **Step 1: Add imports and constants near the top of `main.py`**

Add after the existing imports:

```python
from src.evaluation.threshold_selector import select_league_thresholds
```

Add these constants near `_parse_threshold` (around line 38):

```python
_THRESHOLD_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
_ACTIVE_LEAGUES = ["E0", "N1", "P1", "G1"]   # leagues not in skip_leagues
_ALL_KNOWN_LEAGUES = {"E0", "D1", "SP1", "I1", "F1", "N1", "P1", "G1", "SC0", "B1", "T1"}
```

- [ ] **Step 2: Replace the single compute_value_betting_results call with a per-season loop**

Note: `train_walkforward_monthly` does not return `season_results`. Wrap the calibration loop in `if "season_results" in results` and fall back to the old single-call approach when it is absent (monthly mode). The old fallback block is:

```python
# fallback: monthly mode (no season_results) — use global threshold
betting_results = compute_value_betting_results(
    eval_df,
    results["y_proba"],
    results["classes"],
    threshold=threshold,
    kelly_fraction=kelly,
    inv_odds_factor=inv_odds_factor,
    min_stake=min_stake,
    max_odds=max_odds,
    skip_leagues={"F1", "SP1", "D1", "I1", "SC0", "B1", "T1"},
    max_edge=max_edge,
    min_season_games=min_season_games,
    max_overround=0.07,
)
final_threshold_map = {lg: threshold for lg in _ACTIVE_LEAGUES}
```

In `_run_backtest`, find and replace the block that currently reads:

```python
betting_results = compute_value_betting_results(
    eval_df,
    results["y_proba"],
    results["classes"],
    threshold=threshold,
    kelly_fraction=kelly,
    inv_odds_factor=inv_odds_factor,
    min_stake=min_stake,
    max_odds=max_odds,
    skip_leagues={"F1", "SP1", "D1", "I1", "SC0", "B1", "T1"},
    max_edge=max_edge,
    min_season_games=min_season_games,
    max_overround=0.07,
)
```

Replace it with the following block. The outer `if/else` handles the monthly fallback:

```python
if "season_results" in results:
    # --- Per-season threshold calibration ---
# For each test season, calibrate per-league thresholds from all PRIOR test seasons,
# then apply them. Season 1 (no prior data) uses the global default threshold.
prior_season_data: list[dict] = []
per_season_chunks: list[pd.DataFrame] = []
final_threshold_map: dict[str, float] = {lg: threshold for lg in _ACTIVE_LEAGUES}

for sr in results["season_results"]:
    if prior_season_data:
        final_threshold_map = select_league_thresholds(
            prior_season_data,
            leagues=_ACTIVE_LEAGUES,
            grid=_THRESHOLD_GRID,
            max_odds=max_odds,
            max_overround=0.07,
            max_edge=max_edge,
        )

    # Apply the (possibly per-league) thresholds: one compute call per active league.
    leagues_in_season = set(sr["eval_df"]["league"].unique())
    season_chunks: list[pd.DataFrame] = []
    for league in _ACTIVE_LEAGUES:
        if league not in leagues_in_season:
            continue
        league_threshold = final_threshold_map.get(league, threshold)
        skip_all_but_this = _ALL_KNOWN_LEAGUES - {league}
        lg_bets = compute_value_betting_results(
            sr["eval_df"],
            sr["y_proba"],
            sr["classes"],
            threshold=league_threshold,
            kelly_fraction=kelly,
            inv_odds_factor=inv_odds_factor,
            min_stake=min_stake,
            max_odds=max_odds,
            skip_leagues=skip_all_but_this,
            max_edge=max_edge,
            min_season_games=min_season_games,
            max_overround=0.07,
        )
        season_chunks.append(lg_bets)

    prior_season_data.append(sr)
    if season_chunks:
        per_season_chunks.append(
            pd.concat(season_chunks).sort_values("Date").reset_index(drop=True)
        )

    if per_season_chunks:
        betting_results = pd.concat(per_season_chunks).sort_values("Date").reset_index(drop=True)
    else:
        betting_results = pd.DataFrame(
            columns=["Date", "HomeTeam", "AwayTeam", "profit", "cumulative_profit"]
        )
else:
    # Monthly mode — no season_results; use global threshold unchanged
    betting_results = compute_value_betting_results(
        eval_df,
        results["y_proba"],
        results["classes"],
        threshold=threshold,
        kelly_fraction=kelly,
        inv_odds_factor=inv_odds_factor,
        min_stake=min_stake,
        max_odds=max_odds,
        skip_leagues={"F1", "SP1", "D1", "I1", "SC0", "B1", "T1"},
        max_edge=max_edge,
        min_season_games=min_season_games,
        max_overround=0.07,
    )
    final_threshold_map = {lg: threshold for lg in _ACTIVE_LEAGUES}
```

- [ ] **Step 3: Print the calibrated thresholds in the backtest output**

After the existing `print(f"t-stat: ...")` line in `_run_backtest`, add:

```python
print(f"League thresholds (last calibration): " +
      ", ".join(f"{lg}={v:+.2f}" for lg, v in sorted(final_threshold_map.items())))
```

- [ ] **Step 4: Run the full backtest to verify it completes without errors**

```bash
uv run python main.py 2>&1 | tail -20
```

Expected: backtest completes, prints `League thresholds (last calibration): ...`, shows ROI/stability/t-stat.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: per-season threshold calibration loop in _run_backtest"
```

---

## Task 4: Save `league_thresholds.json` after backtest

**Files:**
- Modify: `main.py`

After the backtest loop, persist `final_threshold_map` so the live prediction path can consume it.

- [ ] **Step 1: Add JSON save after the threshold print line**

In `_run_backtest`, directly after the `print(f"League thresholds ...")` line added in Task 3, add:

```python
import json as _json
_thresholds_path = Path("models/league_thresholds.json")
_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
_thresholds_path.write_text(_json.dumps(final_threshold_map, indent=2))
print(f"League thresholds saved to {_thresholds_path}")
```

- [ ] **Step 2: Run backtest and confirm the file is created**

```bash
uv run python main.py 2>&1 | grep "League thresholds"
cat models/league_thresholds.json
```

Expected: JSON file with one key per active league, values between 0.0 and 0.10.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: save league_thresholds.json after backtest"
```

---

## Task 5: Apply per-league thresholds in live prediction

**Files:**
- Modify: `main.py`
- Modify: `predict.sh`

The live prediction path (`_run_predict`) currently applies a single `threshold`. Update it to load `models/league_thresholds.json` and pass a per-league threshold dict to `_build_prediction_rows`. Update `predict.sh` to remove the hardcoded `--threshold 0.04` injection.

- [ ] **Step 1: Update `_build_prediction_rows` to accept per-league thresholds**

Change the function signature from:
```python
def _build_prediction_rows(fixture_features, y_proba, classes, threshold: float) -> list[dict]:
```
to:
```python
def _build_prediction_rows(
    fixture_features, y_proba, classes, threshold: float,
    league_thresholds: dict | None = None,
) -> list[dict]:
```

Inside the function, replace the line:
```python
            if edge <= threshold:
                continue
```
with:
```python
            league = row.get("league", "")
            t = (league_thresholds or {}).get(league, threshold)
            if edge <= t:
                continue
```

The `threshold` parameter becomes the fallback when a league has no entry in `league_thresholds`.

- [ ] **Step 2: Load `league_thresholds.json` in `_run_predict` and pass it through**

In `_run_predict`, after `threshold = _parse_threshold()`, add:

```python
import json as _json
_lt_path = Path("models/league_thresholds.json")
league_thresholds: dict | None = _json.loads(_lt_path.read_text()) if _lt_path.exists() else None
if league_thresholds:
    print(f"Using per-league thresholds: " +
          ", ".join(f"{lg}={v:+.2f}" for lg, v in sorted(league_thresholds.items())))
else:
    print(f"No league_thresholds.json found — using global threshold {threshold:+.2f}")
```

Then update every call to `_build_prediction_rows` and `_print_predictions` in `_run_predict` to pass `league_thresholds`:

```python
pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold, league_thresholds=league_thresholds)
```

```python
_print_predictions(fixture_features, y_proba, classes, threshold, fetched_at)
```

Note: `_print_predictions` calls `_build_prediction_rows` internally with just `threshold`. Update `_print_predictions` to also accept and forward `league_thresholds`:

Change its signature:
```python
def _print_predictions(fixture_features, y_proba, classes, threshold: float, fetched_at=None, league_thresholds: dict | None = None) -> None:
```

And its internal call:
```python
pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold, league_thresholds=league_thresholds)
```

Then in `_run_predict`, pass it:
```python
_print_predictions(fixture_features, y_proba, classes, threshold, fetched_at, league_thresholds=league_thresholds)
```

Also update `_save_predictions_csv` the same way (it also calls `_build_prediction_rows` internally):

```python
def _save_predictions_csv(fixture_features, y_proba, classes, threshold: float, fetched_at, league_thresholds: dict | None = None) -> None:
    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold, league_thresholds=league_thresholds)
```

And in `_run_predict`:
```python
_save_predictions_csv(fixture_features, y_proba, classes, threshold, fetched_at, league_thresholds=league_thresholds)
```

- [ ] **Step 3: Update `predict.sh` to remove the hardcoded threshold injection**

Replace the entire content of `predict.sh` with:

```bash
#!/usr/bin/env bash
# Run predictions for the current game week across all tracked leagues.
# Per-league thresholds are loaded from models/league_thresholds.json (generated by backtest).
# Pass --threshold X to override the fallback for leagues without a calibrated threshold.
set -euo pipefail
cd "$(dirname "$0")"
uv run python main.py --predict "$@" 2>/dev/null
```

- [ ] **Step 4: Run the full test suite to verify nothing is broken**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main.py predict.sh
git commit -m "feat: apply per-league thresholds in live prediction; update predict.sh"
```

---

## Task 6: End-to-end smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the full backtest and confirm the output is sensible**

```bash
uv run python main.py 2>&1 | grep -E "ROI|Stability|t-stat|League thresholds|Bets"
```

Expected output includes:
- `ROI: +X.XX%`
- `Stability: X.XXXX`
- `t-stat: +X.XX`
- `League thresholds (last calibration): E0=+X.XX, G1=+X.XX, N1=+X.XX, P1=+X.XX`
- `League thresholds saved to models/league_thresholds.json`

- [ ] **Step 2: Verify `league_thresholds.json` contents look reasonable**

```bash
cat models/league_thresholds.json
```

Expected: a JSON object with 4 keys (E0, N1, P1, G1) and float values between 0.0 and 0.10.

- [ ] **Step 3: Run all tests one final time**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass, no warnings about missing keys.

- [ ] **Step 4: Commit final state**

```bash
git add models/league_thresholds.json
git commit -m "chore: add initial league_thresholds.json from first calibrated backtest"
```
