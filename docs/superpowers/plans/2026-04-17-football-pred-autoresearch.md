# Football Prediction Autoresearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a football match outcome prediction system for 4 major EU leagues with automated data fetching, a baseline ML model, ROI-focused evaluation with HTML reports, and an autoresearch guide for iterative improvement.

**Architecture:** Data is downloaded from football-data.co.uk (free CSVs with results + bookmaker odds). A simple logistic regression baseline is trained on engineered rolling-window features. Evaluation generates an HTML report with ROI, profit stability charts, and accuracy metrics. An autoresearch guide directs an LLM to iterate on the model/features layer only.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, requests, matplotlib, jinja2, pytest

---

## File Structure

```
football_pred_autoresearch/
├── data/raw/                          # Downloaded CSVs (gitignored)
├── models/                            # Saved model artifacts
├── reports/                           # Generated HTML reports
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py                # Download CSVs from football-data.co.uk
│   │   └── loader.py                  # Load + unify all CSVs into one DataFrame
│   ├── model/
│   │   ├── __init__.py
│   │   ├── features.py                # Feature engineering (rolling stats)
│   │   └── train.py                   # Train, evaluate, save/load model
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                 # ROI, profit stability, accuracy metrics
│       └── report.py                  # HTML report generation with charts
├── autoresearch/
│   ├── GUIDE.md                       # Instructions for autoresearch LLM
│   └── state.md                       # Living document: current state + history
├── tests/
│   ├── test_loader.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_report.py
├── main.py                            # Full pipeline runner
├── requirements.txt
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/model/__init__.py`, `src/evaluation/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
requests>=2.31
matplotlib>=3.7
jinja2>=3.1
pytest>=7.4
joblib>=1.3
```

- [ ] **Step 2: Create .gitignore**

```
data/raw/
models/
reports/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Create directory structure and empty __init__.py files**

```bash
mkdir -p data/raw models reports src/data src/model src/evaluation autoresearch tests
touch src/__init__.py src/data/__init__.py src/model/__init__.py src/evaluation/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt .gitignore src/
git commit -m "feat: project scaffolding"
```

---

## Task 2: Data Download Script

**Files:**
- Create: `src/data/download.py`

The data source is football-data.co.uk. URL pattern:
- `https://www.football-data.co.uk/mmz4281/{SSYY}/{league}.csv`
- Season code: `1415` = 2014-15, ..., `2425` = 2024-25
- League codes: `E0` (England Premier League), `D1` (Germany Bundesliga), `SP1` (Spain La Liga), `I1` (Italy Serie A)

- [ ] **Step 1: Write src/data/download.py**

```python
import requests
from pathlib import Path

LEAGUES = {
    "england": "E0",
    "germany": "D1",
    "spain": "SP1",
    "italy": "I1",
}

# Generate season codes from 2013-14 to current
SEASONS = [
    f"{str(y)[2:]}{str(y+1)[2:]}"
    for y in range(2013, 2025)
]  # ['1314', '1415', ..., '2425']

BASE_URL = "https://www.football-data.co.uk/mmz4281"
RAW_DIR = Path("data/raw")


def download_season(league_code: str, season: str, force: bool = False) -> Path:
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    dest = RAW_DIR / f"{league_code}_{season}.csv"
    if dest.exists() and not force:
        return dest
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Downloaded {dest.name}")
    return dest


def download_all(force: bool = False) -> list[Path]:
    paths = []
    for country, code in LEAGUES.items():
        for season in SEASONS:
            try:
                p = download_season(code, season, force=force)
                paths.append(p)
            except requests.HTTPError as e:
                print(f"Skipping {code} {season}: {e}")
    return paths


def update_current_season() -> None:
    """Re-download the current season to get latest results."""
    current = SEASONS[-1]
    for code in LEAGUES.values():
        try:
            download_season(code, current, force=True)
        except requests.HTTPError as e:
            print(f"Could not update {code} {current}: {e}")


if __name__ == "__main__":
    import sys
    if "--update" in sys.argv:
        update_current_season()
    else:
        download_all()
```

- [ ] **Step 2: Run download script**

```bash
python -m src.data.download
```

Expected: Files appear in `data/raw/` like `E0_1314.csv`, `D1_1314.csv`, etc. Some older seasons may 404 — that's OK, they'll be skipped. Should download ~100+ files total.

- [ ] **Step 3: Verify a file looks right**

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/E0_2324.csv'); print(df.columns.tolist()); print(df.shape)"
```

Expected: columns include `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `B365H`, `B365D`, `B365A`. Shape ~380 rows × 50+ columns.

- [ ] **Step 4: Commit**

```bash
git add src/data/download.py
git commit -m "feat: data download script from football-data.co.uk"
```

---

## Task 3: Data Loader

**Files:**
- Create: `src/data/loader.py`
- Create: `tests/test_loader.py`

The loader reads all CSVs and returns a unified DataFrame with consistent columns, dropping rows with missing core data.

- [ ] **Step 1: Write failing test**

```python
# tests/test_loader.py
import pandas as pd
import pytest
from src.data.loader import load_all_data, REQUIRED_COLS

def test_load_all_data_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.loader.RAW_DIR", tmp_path)
    # Create a minimal valid CSV
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    for col in REQUIRED_COLS:
        assert col in df.columns

def test_load_adds_league_and_season_columns(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.loader.RAW_DIR", tmp_path)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert "league" in df.columns
    assert "season" in df.columns
    assert df["league"].iloc[0] == "E0"
    assert df["season"].iloc[0] == "2324"

def test_rows_with_missing_result_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.loader.RAW_DIR", tmp_path)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
        "13/08/2023,City,United,,,,,,\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert len(df) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: ImportError or AttributeError — `loader.py` doesn't exist yet.

- [ ] **Step 3: Write src/data/loader.py**

```python
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]

_LEAGUE_MAP = {"E0": "england", "D1": "germany", "SP1": "spain", "I1": "italy"}


def _parse_filename(path: Path) -> tuple[str, str]:
    # filename like E0_2324.csv → league=E0, season=2324
    stem = path.stem  # "E0_2324"
    parts = stem.split("_", 1)
    return parts[0], parts[1]


def _load_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    except Exception:
        return None
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None
    df = df[REQUIRED_COLS].copy()
    df = df.dropna(subset=["FTR", "FTHG", "FTAG", "B365H", "B365D", "B365A"])
    df = df[df["FTR"].isin(["H", "D", "A"])]
    league, season = _parse_filename(path)
    df["league"] = league
    df["season"] = season
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


def load_all_data() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        df = _load_file(path)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLS + ["league", "season"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Smoke test on real data**

```bash
python -c "
from src.data.loader import load_all_data
df = load_all_data()
print(f'Total rows: {len(df)}')
print(f'Leagues: {df[\"league\"].unique()}')
print(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
print(df.head())
"
```

Expected: 30,000+ rows, 4 leagues, dates from 2013 to 2025.

- [ ] **Step 6: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: data loader with unified DataFrame"
```

---

## Task 4: Feature Engineering (Baseline)

**Files:**
- Create: `src/model/features.py`
- Create: `tests/test_features.py`

Features: rolling 5-game stats per team (goals scored, conceded, points earned) for both home and away team, plus home advantage. These are "form" features computed before each match — no data leakage.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features.py
import pandas as pd
import numpy as np
import pytest
from src.model.features import build_features

def _make_df():
    rows = []
    # 10 games: team A vs team B alternating home/away
    for i in range(10):
        rows.append({
            "Date": pd.Timestamp(f"2023-08-{i+1:02d}"),
            "HomeTeam": "Arsenal" if i % 2 == 0 else "Chelsea",
            "AwayTeam": "Chelsea" if i % 2 == 0 else "Arsenal",
            "FTHG": 2, "FTAG": 1, "FTR": "H",
            "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
            "league": "E0", "season": "2324",
        })
    return pd.DataFrame(rows)

def test_build_features_returns_dataframe():
    df = _make_df()
    X, y = build_features(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_no_data_leakage():
    # Features must use only past data, so first few rows per team may be NaN-dropped
    df = _make_df()
    X, y = build_features(df)
    # We should have fewer rows than input (early rows dropped — no history yet)
    assert len(X) < len(df)

def test_feature_columns_present():
    df = _make_df()
    X, y = build_features(df)
    expected = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    for col in expected:
        assert col in X.columns, f"Missing column: {col}"

def test_target_values():
    df = _make_df()
    X, y = build_features(df)
    assert set(y.unique()).issubset({"H", "D", "A"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write src/model/features.py**

```python
import pandas as pd
import numpy as np

WINDOW = 5  # rolling window size


def _points(ftr: str, is_home: bool) -> int:
    if ftr == "H":
        return 3 if is_home else 0
    if ftr == "A":
        return 0 if is_home else 3
    return 1  # draw


def _team_rolling_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Compute rolling stats per team across all matches (home or away)."""
    records = []
    for _, row in df.iterrows():
        for team, is_home in [(row["HomeTeam"], True), (row["AwayTeam"], False)]:
            gf = row["FTHG"] if is_home else row["FTAG"]
            ga = row["FTAG"] if is_home else row["FTHG"]
            pts = _points(row["FTR"], is_home)
            records.append({"Date": row["Date"], "team": team, "gf": gf, "ga": ga, "pts": pts})
    team_df = pd.DataFrame(records).sort_values("Date")
    # Rolling mean of past `window` games (shift(1) to avoid leakage)
    team_df = team_df.groupby("team", group_keys=False).apply(
        lambda g: g.assign(
            form_pts=g["pts"].shift(1).rolling(window, min_periods=window).mean(),
            form_gf=g["gf"].shift(1).rolling(window, min_periods=window).mean(),
            form_ga=g["ga"].shift(1).rolling(window, min_periods=window).mean(),
        )
    )
    return team_df[["Date", "team", "form_pts", "form_gf", "form_ga"]]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values("Date").reset_index(drop=True)
    stats = _team_rolling_stats(df)

    # Merge home team stats
    merged = df.merge(
        stats.rename(columns={"team": "HomeTeam", "form_pts": "home_form_pts",
                               "form_gf": "home_form_gf", "form_ga": "home_form_ga"}),
        on=["Date", "HomeTeam"], how="left"
    )
    # Merge away team stats
    merged = merged.merge(
        stats.rename(columns={"team": "AwayTeam", "form_pts": "away_form_pts",
                               "form_gf": "away_form_gf", "form_ga": "away_form_ga"}),
        on=["Date", "AwayTeam"], how="left"
    )

    feature_cols = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    merged = merged.dropna(subset=feature_cols)
    X = merged[feature_cols].reset_index(drop=True)
    y = merged["FTR"].reset_index(drop=True)
    return X, y


def build_features_with_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Like build_features but also returns bookmaker odds for the kept rows."""
    df = df.sort_values("Date").reset_index(drop=True)
    stats = _team_rolling_stats(df)

    merged = df.merge(
        stats.rename(columns={"team": "HomeTeam", "form_pts": "home_form_pts",
                               "form_gf": "home_form_gf", "form_ga": "home_form_ga"}),
        on=["Date", "HomeTeam"], how="left"
    )
    merged = merged.merge(
        stats.rename(columns={"team": "AwayTeam", "form_pts": "away_form_pts",
                               "form_gf": "away_form_gf", "form_ga": "away_form_ga"}),
        on=["Date", "AwayTeam"], how="left"
    )

    feature_cols = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    merged = merged.dropna(subset=feature_cols)
    merged = merged.reset_index(drop=True)
    X = merged[feature_cols]
    y = merged["FTR"]
    odds = merged[["B365H", "B365D", "B365A", "Date", "HomeTeam", "AwayTeam", "league", "season"]]
    return X, y, odds
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/features.py tests/test_features.py
git commit -m "feat: rolling-window feature engineering baseline"
```

---

## Task 5: Baseline Model Training

**Files:**
- Create: `src/model/train.py`

Train a logistic regression classifier. Use a time-based train/test split (last 2 seasons = test). Save model with joblib.

- [ ] **Step 1: Write src/model/train.py**

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.model.features import build_features_with_odds

MODEL_PATH = Path("models/baseline.joblib")
TEST_SEASONS = 2  # number of most recent seasons held out for testing


def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season"].unique())
    test_seasons = set(seasons[-TEST_SEASONS:])
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test


def train_model(df: pd.DataFrame) -> dict:
    train_df, test_df = split_by_season(df)

    X_train, y_train, _ = build_features_with_odds(train_df)
    X_test, y_test, odds_test = build_features_with_odds(test_df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    accuracy = (y_pred == y_test.values).mean()
    print(f"Test accuracy: {accuracy:.3f}")

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
        "odds_test": odds_test,
        "accuracy": accuracy,
    }


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    from src.data.loader import load_all_data
    df = load_all_data()
    results = train_model(df)
    print(f"Accuracy: {results['accuracy']:.3f}")
```

- [ ] **Step 2: Run training**

```bash
python -m src.model.train
```

Expected: prints accuracy ~0.50-0.55 (football is hard to predict). Model file saved at `models/baseline.joblib`.

- [ ] **Step 3: Commit**

```bash
git add src/model/train.py
git commit -m "feat: baseline logistic regression model with time-based split"
```

---

## Task 6: Evaluation Metrics (ROI & Profit Stability)

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Betting strategy:** Bet 1 unit on every match on the predicted outcome. If correct, profit = odds - 1. If wrong, profit = -1. Track cumulative profit per match (ordered by date).

**ROI** = total_profit / total_staked * 100  
**Profit stability** (scalar): Sharpe-like ratio = mean(per_bet_profit) / std(per_bet_profit). Higher = more stable positive returns.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import pandas as pd
import numpy as np
from src.evaluation.metrics import compute_betting_results, compute_roi, compute_stability

def _make_results():
    return pd.DataFrame({
        "y_true": ["H", "D", "A", "H", "H"],
        "y_pred": ["H", "H", "A", "A", "H"],
        "B365H": [2.0, 3.5, 4.0, 2.0, 1.8],
        "B365D": [3.5, 3.0, 3.5, 3.5, 3.5],
        "B365A": [4.0, 2.1, 2.0, 4.0, 5.0],
        "Date": pd.date_range("2024-01-01", periods=5),
    })

def test_compute_betting_results_columns():
    df = _make_results()
    res = compute_betting_results(df)
    assert "profit" in res.columns
    assert "cumulative_profit" in res.columns
    assert len(res) == 5

def test_correct_prediction_gives_positive_profit():
    df = _make_results()
    res = compute_betting_results(df)
    # First row: pred=H, true=H, odds=2.0 → profit = 2.0 - 1 = 1.0
    assert abs(res.iloc[0]["profit"] - 1.0) < 1e-6

def test_wrong_prediction_gives_minus_one():
    df = _make_results()
    res = compute_betting_results(df)
    # Second row: pred=H, true=D → profit = -1
    assert abs(res.iloc[1]["profit"] - (-1.0)) < 1e-6

def test_roi_calculation():
    df = _make_results()
    res = compute_betting_results(df)
    roi = compute_roi(res)
    total_staked = 5.0
    total_profit = res["profit"].sum()
    assert abs(roi - (total_profit / total_staked * 100)) < 1e-6

def test_stability_is_scalar():
    df = _make_results()
    res = compute_betting_results(df)
    stab = compute_stability(res)
    assert isinstance(stab, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write src/evaluation/metrics.py**

```python
import pandas as pd
import numpy as np

_ODDS_COL = {"H": "B365H", "D": "B365D", "A": "B365A"}


def compute_betting_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have: y_true, y_pred, B365H, B365D, B365A, Date
    Returns df with profit and cumulative_profit columns, sorted by Date.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    profits = []
    for _, row in df.iterrows():
        pred = row["y_pred"]
        true = row["y_true"]
        odds_col = _ODDS_COL[pred]
        if pred == true:
            profit = float(row[odds_col]) - 1.0
        else:
            profit = -1.0
        profits.append(profit)
    df = df.copy()
    df["profit"] = profits
    df["cumulative_profit"] = df["profit"].cumsum()
    return df


def compute_roi(results: pd.DataFrame) -> float:
    """ROI as percentage: total_profit / total_staked * 100."""
    total_staked = float(len(results))
    total_profit = results["profit"].sum()
    return (total_profit / total_staked) * 100.0


def compute_stability(results: pd.DataFrame) -> float:
    """Sharpe-like ratio: mean profit per bet / std of profit per bet.
    Higher = more stable positive returns. Returns 0.0 if std is 0."""
    profits = results["profit"]
    std = profits.std()
    if std == 0:
        return 0.0
    return float(profits.mean() / std)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: betting ROI and profit stability metrics"
```

---

## Task 7: HTML Report Generation

**Files:**
- Create: `src/evaluation/report.py`
- Create: `tests/test_report.py`

The report includes: accuracy, ROI, stability score, a bar chart of per-bet profits sorted by profit value (to visualize distribution), and a cumulative profit curve over time.

- [ ] **Step 1: Write failing test**

```python
# tests/test_report.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.evaluation.report import generate_report

def _make_results():
    return pd.DataFrame({
        "y_true": ["H", "D", "A", "H", "H"] * 10,
        "y_pred": ["H", "H", "A", "A", "H"] * 10,
        "B365H": [2.0] * 50,
        "B365D": [3.5] * 50,
        "B365A": [4.0] * 50,
        "Date": pd.date_range("2024-01-01", periods=50),
        "HomeTeam": ["Arsenal"] * 50,
        "AwayTeam": ["Chelsea"] * 50,
        "league": ["E0"] * 50,
        "season": ["2324"] * 50,
        "profit": ([1.0, -1.0, 3.0, -1.0, 0.8]) * 10,
        "cumulative_profit": list(range(50)),
    })

def test_generate_report_creates_html(tmp_path):
    df = _make_results()
    out = tmp_path / "report.html"
    generate_report(
        results_df=df,
        accuracy=0.52,
        roi=-3.5,
        stability=0.12,
        output_path=out,
    )
    assert out.exists()
    content = out.read_text()
    assert "<html" in content.lower()
    assert "ROI" in content
    assert "Accuracy" in content
    assert "Stability" in content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_report.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write src/evaluation/report.py**

```python
import base64
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Football Prediction Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; background: #f9f9f9; color: #222; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 2px solid #283593; padding-bottom: 6px; }}
  .metrics {{ display: flex; gap: 24px; margin: 24px 0; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 20px 32px; box-shadow: 0 2px 8px rgba(0,0,0,.1); text-align: center; flex: 1; }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; color: {roi_color}; }}
  .metric-card:nth-child(1) .value {{ color: #1565c0; }}
  .metric-card:nth-child(3) .value {{ color: #2e7d32; }}
  .metric-card .label {{ color: #777; font-size: 0.9em; margin-top: 4px; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 16px 0; }}
  .explanation {{ background: #e8eaf6; border-left: 4px solid #3949ab; padding: 12px 18px; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Football Prediction Evaluation Report</h1>
<h2>Summary Metrics</h2>
<div class="metrics">
  <div class="metric-card"><div class="value">{accuracy:.1%}</div><div class="label">Accuracy</div></div>
  <div class="metric-card"><div class="value" style="color:{roi_color}">{roi:+.2f}%</div><div class="label">ROI (Return on Investment)</div></div>
  <div class="metric-card"><div class="value">{stability:.3f}</div><div class="label">Profit Stability (Sharpe-like)</div></div>
</div>
<div class="explanation">
  <b>ROI</b>: total profit / total staked × 100. Positive means profit over the test period.<br>
  <b>Stability</b>: mean profit per bet / std(profit per bet). Higher = more consistent returns. Above 0.05 is good.
</div>
<h2>Profit Distribution (sorted by profit)</h2>
<img src="data:image/png;base64,{bar_chart_b64}" alt="Profit distribution bar chart">
<h2>Cumulative Profit Over Time</h2>
<img src="data:image/png;base64,{cumulative_chart_b64}" alt="Cumulative profit chart">
<h2>Bet Details</h2>
<p>Total bets: {n_bets} &nbsp;|&nbsp; Correct: {n_correct} &nbsp;|&nbsp; Wrong: {n_wrong}</p>
</body>
</html>"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _bar_chart(profits: pd.Series) -> str:
    sorted_profits = profits.sort_values().values
    colors = ["#e53935" if p < 0 else "#43a047" for p in sorted_profits]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(sorted_profits)), sorted_profits, color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Bet (sorted by profit)")
    ax.set_ylabel("Profit (units)")
    ax.set_title("Per-Bet Profit Distribution (sorted)")
    green_patch = mpatches.Patch(color="#43a047", label="Win")
    red_patch = mpatches.Patch(color="#e53935", label="Loss")
    ax.legend(handles=[green_patch, red_patch])
    return _fig_to_b64(fig)


def _cumulative_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["cumulative_profit"], color="#1565c0", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(
        df["Date"], df["cumulative_profit"], 0,
        where=df["cumulative_profit"] >= 0, alpha=0.15, color="#43a047"
    )
    ax.fill_between(
        df["Date"], df["cumulative_profit"], 0,
        where=df["cumulative_profit"] < 0, alpha=0.15, color="#e53935"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit Over Time")
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def generate_report(
    results_df: pd.DataFrame,
    accuracy: float,
    roi: float,
    stability: float,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bar_b64 = _bar_chart(results_df["profit"])
    cum_b64 = _cumulative_chart(results_df)

    n_bets = len(results_df)
    n_correct = (results_df["y_true"] == results_df["y_pred"]).sum()
    n_wrong = n_bets - n_correct
    roi_color = "#2e7d32" if roi >= 0 else "#c62828"

    html = _HTML_TEMPLATE.format(
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        roi_color=roi_color,
        bar_chart_b64=bar_b64,
        cumulative_chart_b64=cum_b64,
        n_bets=n_bets,
        n_correct=n_correct,
        n_wrong=n_wrong,
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_report.py -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/report.py tests/test_report.py
git commit -m "feat: HTML evaluation report with profit charts"
```

---

## Task 8: Main Pipeline Runner

**Files:**
- Create: `main.py`

Wires together: load data → train model → compute metrics → generate report.

- [ ] **Step 1: Write main.py**

```python
#!/usr/bin/env python3
"""
Main pipeline: data → model → evaluation → HTML report.

Usage:
  python main.py              # full pipeline
  python main.py --update     # re-download current season, then full pipeline
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import load_all_data
from src.model.features import build_features_with_odds
from src.model.train import train_model, split_by_season
from src.evaluation.metrics import compute_betting_results, compute_roi, compute_stability
from src.evaluation.report import generate_report


def run_pipeline():
    if "--update" in sys.argv:
        print("Updating current season data...")
        from src.data.download import update_current_season
        update_current_season()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("Training model...")
    results = train_model(df)

    # Build evaluation DataFrame
    _, test_df = split_by_season(df)
    X_test, y_test, odds_test = build_features_with_odds(test_df)

    y_pred = results["y_pred"]
    eval_df = odds_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred

    print("Computing metrics...")
    betting_results = compute_betting_results(eval_df)
    roi = compute_roi(betting_results)
    stability = compute_stability(betting_results)
    accuracy = results["accuracy"]

    print(f"\n=== RESULTS ===")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"ROI:       {roi:+.2f}%")
    print(f"Stability: {stability:.4f}")
    print(f"Test bets: {len(betting_results)}")

    print("\nGenerating report...")
    generate_report(
        results_df=betting_results,
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        output_path=Path("reports/evaluation_report.html"),
    )
    print("Done. Open reports/evaluation_report.html to view results.")


if __name__ == "__main__":
    run_pipeline()
```

- [ ] **Step 2: Run full pipeline**

```bash
python main.py
```

Expected: prints metrics, saves `reports/evaluation_report.html`. Open it in a browser to verify charts and metrics appear correctly.

- [ ] **Step 3: Verify report opens in browser**

```bash
python -c "import webbrowser; webbrowser.open('reports/evaluation_report.html')"
```

Or just open `reports/evaluation_report.html` manually. Verify:
- Three metric cards (Accuracy, ROI, Stability) are visible
- Bar chart shows profit distribution (red/green bars)
- Cumulative profit line chart is shown
- Total bets count is non-zero

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: main pipeline runner with full end-to-end evaluation"
```

---

## Task 9: Autoresearch Guide and State Document

**Files:**
- Create: `autoresearch/GUIDE.md`
- Create: `autoresearch/state.md`

- [ ] **Step 1: Write autoresearch/GUIDE.md**

````markdown
# Autoresearch Guide: Football Prediction Improvement

## Mission

You are an autoresearch LLM tasked with iteratively improving the football match prediction model's **ROI** and **profit stability**. Your job is to run hypothesis-driven experiments, measure their impact using the established evaluation pipeline, and document findings.

## Constraints

**DO NOT touch:**
- `src/data/` — data loading and download logic is frozen
- `src/evaluation/` — evaluation metrics and report generation are frozen
- `main.py` — the pipeline runner is frozen
- `tests/` — existing tests must continue to pass

**You MAY modify:**
- `src/model/features.py` — add or change features
- `src/model/train.py` — change model type, hyperparameters, training strategy
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
python main.py
```

Record the output metrics (Accuracy, ROI, Stability).

### 5. Run Tests

```bash
pytest tests/ -v
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
````

- [ ] **Step 2: Write autoresearch/state.md**

```markdown
# Autoresearch State: Football Prediction Model

*This document is updated after every autoresearch iteration. It is the single source of truth for what has been tried and what the current best model is.*

---

## Current Best Model

| Metric | Value |
|--------|-------|
| Accuracy | *(run baseline to fill)* |
| ROI | *(run baseline to fill)* |
| Stability | *(run baseline to fill)* |
| Model | Logistic Regression (baseline) |
| Features | 5-game rolling: pts, gf, ga for home + away team |

**To reproduce:** `python main.py` with unmodified `src/model/`

---

## Baseline (Iteration 0)

**Date:** *(fill after first run)*  
**Model:** Logistic Regression, StandardScaler  
**Features:** 5-game rolling mean of points, goals for, goals against — for both home and away team (6 features total)  
**Training:** All seasons up to the last 2 (time-based split, no lookahead)  
**Test period:** Last 2 seasons across all 4 leagues  

**Results:**
- Accuracy: *(fill)*
- ROI: *(fill)*
- Stability: *(fill)*
- Total test bets: *(fill)*

**Notes:** This is the floor. Every subsequent iteration should be compared against these numbers.

---

## Iteration History

*(Entries added here after each autoresearch iteration)*

---

## Open Hypotheses (Ranked by Confidence)

1. **Home/away split form** — Use separate rolling stats for home-only and away-only games. Teams often perform very differently at home vs away.
2. **Elo ratings** — Compute Elo per team updated after each match. Captures longer-term relative strength better than a 5-game window.
3. **Value betting threshold** — Only bet when model probability > bookmaker implied probability. This filters out low-edge bets and should improve ROI even with lower total bets.

---

## Key Findings So Far

*(Populated during autoresearch)*

---

## Notes / Lessons Learned

- The data covers 4 leagues (E0, D1, SP1, I1) and ~11 seasons (2013-14 to 2024-25)
- Bookmaker odds used: Bet365 (B365H, B365D, B365A)
- Betting strategy: flat 1 unit per predicted match; profit = odds−1 if correct, −1 if wrong
- Test set = last 2 seasons; train set = all earlier seasons
- Do NOT use validation set metrics to tune — test set is locked for honest evaluation
```

- [ ] **Step 3: Run baseline and fill in state.md with results**

```bash
python main.py
```

Copy the printed Accuracy, ROI, Stability values into `autoresearch/state.md` under "Baseline (Iteration 0)".

- [ ] **Step 4: Commit**

```bash
git add autoresearch/
git commit -m "feat: autoresearch guide and living state document"
```

---

## Task 10: Run All Tests

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Run full pipeline one final time**

```bash
python main.py
```

Expected: Report generated at `reports/evaluation_report.html`.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete football prediction autoresearch project"
```

---

## Self-Review

### Spec Coverage Check

| Requirement | Task |
|---|---|
| Free data source, 10+ seasons, England/Germany/Spain/Italy | Task 2 (football-data.co.uk, seasons 2013-2025) |
| Auto-update script | Task 2 (`update_current_season()`, `--update` flag) |
| Data loaded in Part 1 | Task 3 (`load_all_data()`) |
| Preprocessing + baseline model | Tasks 4, 5 |
| Accuracy evaluation | Tasks 6, 8 |
| Bookmaker odds / ROI metric | Task 6 |
| Profit stability metric + scalar | Task 6 (`compute_stability`) |
| Profit distribution chart | Task 7 (sorted bar chart) |
| Cumulative profit chart | Task 7 |
| HTML report | Task 7 |
| Autoresearch guide | Task 9 |
| LLM iterates on Part 2 only | Task 9 (GUIDE.md constraints section) |
| Hypothesis → implement → test → analyse → next directions | Task 9 (GUIDE.md Iteration Protocol) |
| Logged, human-readable state doc | Task 9 (state.md) |

All requirements covered. ✓
