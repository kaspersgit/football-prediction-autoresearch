from pathlib import Path

import pandas as pd

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
