# CI-Narrowing Iterations 86–95 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase bet volume and model quality across 10 iterations to narrow the 95% CI (raise t-stat above 3.0) while keeping ROI > 5%.

**Architecture:** Three blocks — (1) add 4 new leagues by downloading data and removing them from skip_leagues, (2) loosen filter thresholds for marginal volume, (3) tune model hyperparameters for better per-bet consistency. Each iteration is self-contained: implement → run → keep or revert → update state.md.

**Tech Stack:** LightGBM + isotonic calibration, pandas, Python 3.11+, uv

---

## File Map

| File | What changes |
|------|-------------|
| `main.py` | `skip_leagues` sets (lines 354, 541), `max_overround` hardcodes (lines 544, 592), `_parse_max_odds()` default (line 49) |
| `src/model/train.py` | `_LGBM_PARAMS` dict (lines 14–19): `num_leaves`, `min_child_samples`, `reg_lambda` |
| `src/model/features.py` | `WINDOW` constant (line 3) |
| `autoresearch/state.md` | Append iteration entry after every run; update Current Best block if new best |
| `data/raw/` | New league CSV files downloaded here (no code change needed — loader auto-discovers) |

---

## Iteration Protocol (applies to every task)

Before each task, read current best:
```bash
head -30 autoresearch/state.md
wc -l autoresearch/state.md  # get N, then:
# read last 300 lines: offset = N-300
```

After each task, append to `autoresearch/state.md`:
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

**Keep criterion (new leagues):** overall ROI ≥ 5% AND t-stat ≥ current best (2.59 at start).
**Keep criterion (filter/model):** ROI ≥ 5% AND no metric regresses.
**Revert command:** `git checkout main.py` or `git checkout src/model/train.py` or `git checkout src/model/features.py`

---

## Task 1: Iteration 86 — Scotland (SC0)

**Files:**
- Download to: `data/raw/SC0_*.csv` (auto-discovered, no code change to loader)
- Modify: `main.py` lines 354, 541

### Step 1.1: Download Scotland data

- [ ] Run:
```bash
uv run python -c "
from src.data.download import download_season, SEASONS
for s in SEASONS:
    try:
        download_season('SC0', s)
    except Exception as e:
        print(f'Skip SC0 {s}: {e}')
"
```
Expected: files like `data/raw/SC0_1314.csv` … `SC0_2526.csv` created. Some early seasons may 404 (Scotland data starts ~1314 on football-data.co.uk) — that is fine.

### Step 1.2: Verify data loaded

- [ ] Run:
```bash
uv run python -c "
from src.data.loader import load_all_data
df = load_all_data()
print(df[df['league']=='SC0'].groupby('season').size())
"
```
Expected: rows grouped by season for SC0. If output is empty, the CSV columns differ — check `data/raw/SC0_2324.csv` manually for required columns (Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A).

### Step 1.3: Remove SC0 from skip_leagues

- [ ] Edit `main.py` line 354 — change:
```python
skip_leagues={"F1", "SP1", "D1", "I1"},
```
to:
```python
skip_leagues={"F1", "SP1", "D1", "I1", "SC0"},
```
**Do NOT remove SC0 yet** — first run with it excluded to confirm baseline is unchanged, then remove it. Actually: edit both lines (354 and 541) to **remove SC0 from the set** (i.e., allow SC0 bets):
```python
# line 354
skip_leagues={"F1", "SP1", "D1", "I1"},   # already correct — SC0 not listed = allowed

# line 541
skip_leagues={"F1", "SP1", "D1", "I1"},   # already correct — SC0 not listed = allowed
```
Scotland is not currently in skip_leagues, so no edit to main.py is needed — just downloading the data is sufficient to include it.

### Step 1.4: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```
Record: ROI, Stability, t-stat, Bets count printed in `=== BACKTEST RESULTS ===`.

### Step 1.5: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```
Expected: all PASSED.

### Step 1.6: Evaluate and decide

- [ ] Check: overall ROI ≥ 5% AND t-stat ≥ 2.59?
  - **Yes → keep.** Proceed to Step 1.7.
  - **No → revert** (Scotland hurt the portfolio):
    ```bash
    # Delete the SC0 files or add SC0 to skip_leagues:
    # Edit main.py lines 354 and 541:
    # skip_leagues={"F1", "SP1", "D1", "I1", "SC0"},
    git add main.py
    git commit -m "chore: exclude SC0 from betting (regression)"
    ```

### Step 1.7: Update state.md and commit

- [ ] Append iteration entry to bottom of `autoresearch/state.md` (see Iteration Protocol above).
- [ ] Update Current Best block at top of `autoresearch/state.md` if metrics improved.
- [ ] Run:
```bash
git add autoresearch/state.md main.py data/raw/SC0_*.csv
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 2: Iteration 87 — Belgium (B1)

**Files:**
- Download to: `data/raw/B1_*.csv`
- Modify: `main.py` lines 354, 541 (add or remove B1 from skip_leagues)

### Step 2.1: Download Belgium data

- [ ] Run:
```bash
uv run python -c "
from src.data.download import download_season, SEASONS
for s in SEASONS:
    try:
        download_season('B1', s)
    except Exception as e:
        print(f'Skip B1 {s}: {e}')
"
```

### Step 2.2: Verify data loaded

- [ ] Run:
```bash
uv run python -c "
from src.data.loader import load_all_data
df = load_all_data()
print(df[df['league']=='B1'].groupby('season').size())
"
```
Expected: rows per season for B1.

### Step 2.3: Run pipeline (B1 auto-included — not in skip_leagues)

- [ ] Run:
```bash
uv run python main.py --per-league
```
Record: ROI, Stability, t-stat, Bets.

### Step 2.4: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 2.5: Evaluate and decide

- [ ] Check: overall ROI ≥ 5% AND t-stat ≥ best so far?
  - **Yes → keep.** Proceed to Step 2.6.
  - **No → add B1 to skip_leagues:**
    ```python
    # main.py line 354 and 541 — add "B1":
    skip_leagues={"F1", "SP1", "D1", "I1", "B1"},   # adjust based on what SC0 decision was
    ```
    Then:
    ```bash
    git add main.py
    git commit -m "chore: exclude B1 from betting (regression)"
    ```

### Step 2.6: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md main.py data/raw/B1_*.csv
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 3: Iteration 88 — Greece (G1)

**Files:**
- Download to: `data/raw/G1_*.csv`
- Modify: `main.py` lines 354, 541 (if revert needed)

### Step 3.1: Download Greece data

- [ ] Run:
```bash
uv run python -c "
from src.data.download import download_season, SEASONS
for s in SEASONS:
    try:
        download_season('G1', s)
    except Exception as e:
        print(f'Skip G1 {s}: {e}')
"
```

### Step 3.2: Verify data loaded

- [ ] Run:
```bash
uv run python -c "
from src.data.loader import load_all_data
df = load_all_data()
print(df[df['league']=='G1'].groupby('season').size())
"
```

### Step 3.3: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 3.4: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 3.5: Evaluate and decide

- [ ] Check: overall ROI ≥ 5% AND t-stat ≥ best so far?
  - **Yes → keep.** Proceed to Step 3.6.
  - **No → add G1 to skip_leagues** (main.py lines 354 and 541 — add `"G1"` to both sets), then commit.

### Step 3.6: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md main.py data/raw/G1_*.csv
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 4: Iteration 89 — Turkey (T1)

**Files:**
- Download to: `data/raw/T1_*.csv`
- Modify: `main.py` lines 354, 541 (if revert needed)

### Step 4.1: Download Turkey data

- [ ] Run:
```bash
uv run python -c "
from src.data.download import download_season, SEASONS
for s in SEASONS:
    try:
        download_season('T1', s)
    except Exception as e:
        print(f'Skip T1 {s}: {e}')
"
```

### Step 4.2: Verify data loaded

- [ ] Run:
```bash
uv run python -c "
from src.data.loader import load_all_data
df = load_all_data()
print(df[df['league']=='T1'].groupby('season').size())
"
```

### Step 4.3: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 4.4: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 4.5: Evaluate and decide

- [ ] Check: overall ROI ≥ 5% AND t-stat ≥ best so far?
  - **Yes → keep.** Proceed to Step 4.6.
  - **No → add T1 to skip_leagues** (main.py lines 354 and 541), then commit.

### Step 4.6: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md main.py data/raw/T1_*.csv
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 5: Iteration 90 — max_overround 0.07 → 0.08

**Files:**
- Modify: `main.py` lines 544, 592

### Step 5.1: Read current state

- [ ] Run:
```bash
head -20 autoresearch/state.md
```
Note current ROI and t-stat as the comparison baseline.

### Step 5.2: Change max_overround

- [ ] Edit `main.py` line 544:
```python
        max_overround=0.08,
```
- [ ] Edit `main.py` line 592:
```python
        max_overround=0.08,
```
Also update the inline comment at line 176:
```python
_PREDICT_MAX_OVERROUND = 0.08  # must match backtest max_overround
```

### Step 5.3: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 5.4: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 5.5: Evaluate and decide

- [ ] Check: ROI ≥ 5% AND t-stat ≥ previous best?
  - **Yes → keep.** Proceed to Step 5.6.
  - **No → revert:**
    ```bash
    git checkout main.py
    ```

### Step 5.6: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md main.py
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 6: Iteration 91 — max_odds 5.0 → 6.0

**Files:**
- Modify: `main.py` line 49 (`_parse_max_odds` default), line 174 (`_PREDICT_MAX_ODDS`)

### Step 6.1: Change max_odds default

- [ ] Edit `main.py` line 49:
```python
    return 6.0  # Iter 91: test edge validity up to odds 6.0
```
- [ ] Edit `main.py` line 174:
```python
_PREDICT_MAX_ODDS = 6.0      # must match backtest max_odds
```

### Step 6.2: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 6.3: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 6.4: Evaluate and decide

- [ ] Check: ROI ≥ 5% AND t-stat ≥ previous best?
  - **Yes → keep.** Proceed to Step 6.5.
  - **No → revert:**
    ```bash
    git checkout main.py
    ```

### Step 6.5: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md main.py
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 7: Iteration 92 — min_child_samples 20 → 15

**Files:**
- Modify: `src/model/train.py` line 18

### Step 7.1: Change min_child_samples

- [ ] Edit `src/model/train.py` line 18:
```python
    min_child_samples=15,
```

### Step 7.2: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 7.3: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 7.4: Evaluate and decide

- [ ] Check: ROI ≥ 5% AND stability does not regress vs previous best?
  - **Yes → keep.** Proceed to Step 7.5.
  - **No → revert:**
    ```bash
    git checkout src/model/train.py
    ```

### Step 7.5: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md src/model/train.py
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 8: Iteration 93 — num_leaves 31 → 40

**Files:**
- Modify: `src/model/train.py` line 17

### Step 8.1: Change num_leaves

- [ ] Edit `src/model/train.py` line 17:
```python
    num_leaves=40,
```

### Step 8.2: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 8.3: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 8.4: Evaluate and decide

- [ ] Check: ROI ≥ 5% AND stability does not regress?
  - **Yes → keep.** Proceed to Step 8.5.
  - **No → revert:**
    ```bash
    git checkout src/model/train.py
    ```

### Step 8.5: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md src/model/train.py
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 9: Iteration 94 — EWM WINDOW 5 → 4

**Files:**
- Modify: `src/model/features.py` line 3

### Step 9.1: Change WINDOW

- [ ] Edit `src/model/features.py` line 3:
```python
WINDOW = 4
```
This affects all three EWM form features (pts, gf, ga) computed with `span=window` in `_compute_form()`.

### Step 9.2: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 9.3: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 9.4: Evaluate and decide

- [ ] Check: ROI ≥ 5% AND stability does not regress?
  - **Yes → keep.** Proceed to Step 9.5.
  - **No → revert:**
    ```bash
    git checkout src/model/features.py
    ```

### Step 9.5: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md src/model/features.py
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Task 10: Iteration 95 — reg_lambda 0.05 → 0.03

**Files:**
- Modify: `src/model/train.py` line 19

### Step 10.1: Change reg_lambda

- [ ] Edit `src/model/train.py` line 19:
```python
    reg_lambda=0.03,
```

### Step 10.2: Run pipeline

- [ ] Run:
```bash
uv run python main.py --per-league
```

### Step 10.3: Run tests

- [ ] Run:
```bash
uv run pytest tests/ -v
```

### Step 10.4: Evaluate and decide

- [ ] Check: ROI ≥ 5% AND stability does not regress?
  - **Yes → keep.** Proceed to Step 10.5.
  - **No → revert:**
    ```bash
    git checkout src/model/train.py
    ```

### Step 10.5: Update state.md and commit

- [ ] Append iteration entry to `autoresearch/state.md`.
- [ ] Update Current Best block if improved.
- [ ] Run:
```bash
git add autoresearch/state.md src/model/train.py
git commit -m "chore: update backtest bets [skip ci]"
```

---

## Success Check (after Task 10)

- [ ] Run:
```bash
head -20 autoresearch/state.md
```
Verify: ROI > 5%, t-stat ≥ 3.0 (or note partial success if all new leagues failed).

- [ ] Run:
```bash
uv run pytest tests/ -v
```
All tests must pass.
