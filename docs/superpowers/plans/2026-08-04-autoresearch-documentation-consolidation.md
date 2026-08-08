# Autoresearch documentation consolidation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one current-state document, one evaluation policy, one collision-free experiment ledger, and one source for runtime betting defaults.

**Architecture:** Separate current state, procedure, evaluation policy, and immutable history. Put shared betting defaults in `src/config.py`, import them from both backtest and prediction paths, and describe behaviour from those executable values.

**Tech Stack:** Markdown, Python 3.11, pytest, existing command-line pipeline.

## Global Constraints

- Preserve every historical experiment from `autoresearch/state.md` and `docs/improvements.md`.
- Keep `threshold=0.0` explicit for research, CLI default `0.03`, and `predict.sh` default `0.04`.
- Do not change model behaviour or generated report behaviour.
- Keep `docs/index.html` unchanged.

---

### Task 1: Shared betting configuration

**Files:**
- Create: `src/config.py`
- Modify: `main.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DEFAULT_MAX_ODDS`, `DEFAULT_MAX_EDGE`, `DEFAULT_MAX_OVERROUND`, and `EXCLUDED_BETTING_LEAGUES`.
- Consumes: existing values `5.0`, `0.20`, `0.07`, and `{F1, SP1, D1, I1}`.

- [x] **Step 1: Write failing tests for the shared constants and parser defaults.**
- [x] **Step 2: Run `uv run pytest tests/test_config.py -v` and confirm failure because `src.config` does not exist.**
- [x] **Step 3: Add `src/config.py` and replace repeated literals in `main.py`.**
- [x] **Step 4: Re-run `uv run pytest tests/test_config.py -v` and confirm it passes.**

### Task 2: Canonical research documents

**Files:**
- Create: `autoresearch/EVALUATION.md`
- Create: `autoresearch/current.md`
- Modify: `autoresearch/GUIDE.md`

**Interfaces:**
- `GUIDE.md` links to `current.md`, `EVALUATION.md`, and `experiments.md`.
- `current.md` records only the current verified snapshot and untested queue.
- `EVALUATION.md` contains stable policy without dated current metrics.

- [x] **Step 1: Move the stable evaluation rules into `autoresearch/EVALUATION.md`.**
- [x] **Step 2: Build `autoresearch/current.md` from the current code and the latest verified state block.**
- [x] **Step 3: Rewrite `GUIDE.md` around the new responsibilities and the explicit `--threshold 0.0` command.**
- [x] **Step 4: Check that completed ideas are absent from the active queue.**

### Task 3: Collision-free experiment ledger

**Files:**
- Create: `autoresearch/experiments.md`
- Delete: `autoresearch/state.md`
- Delete: `docs/improvements.md`
- Delete: `docs/evaluation_standards.md`

**Interfaces:**
- Produces unique IDs in the form `EXP-<date>-S<legacy-id>` and `EXP-<date>-D<legacy-id>`.
- Preserves the source and legacy iteration ID for traceability.

- [x] **Step 1: Extract the baseline and experiment entries from the state ledger, excluding duplicated current-state commentary.**
- [x] **Step 2: Append all archived improvement experiments, including pending xG and edge-baseline findings.**
- [x] **Step 3: Namespace colliding legacy IDs with date and source.**
- [x] **Step 4: Remove the superseded source ledgers after checking representative entries from both sources.**

### Task 4: User-facing documentation and verification

**Files:**
- Modify: `README.md`
- Modify: this plan to mark completed steps.

**Interfaces:**
- README describes actual staking, model, features, filters, and defaults.

- [x] **Step 1: Rewrite stale README sections from current code.**
- [x] **Step 2: Run the complete test suite with `uv run pytest tests/ -v`.**
- [x] **Step 3: Run `git diff --check`.**
- [x] **Step 4: Search for active references to removed files, legacy runtime claims, and duplicate experiment IDs.**
- [x] **Step 5: Inspect the final diff and report any remaining limitations.**
