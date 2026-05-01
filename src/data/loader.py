from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
_OPTIONAL_COLS = ["PSCH", "PSCD", "PSCA", "HST", "AST"]  # Pinnacle closing odds + shots on target

# All non-Betfair bookmaker columns used for CustomMax computation (BFE* excluded)
_BK_COLS_H = ["B365H", "BFDH", "BMGMH", "BVH", "BWH", "CLH", "LBH", "PSH", "WHH", "VCH", "SJH", "IWH", "GBH", "SBH"]
_BK_COLS_D = ["B365D", "BFDD", "BMGMD", "BVD", "BWD", "CLD", "LBD", "PSD", "WHD", "VCD", "SJD", "IWD", "GBD", "SBD"]
_BK_COLS_A = ["B365A", "BFDA", "BMGMA", "BVA", "BWA", "CLA", "LBA", "PSA", "WHA", "VCA", "SJA", "IWA", "GBA", "SBA"]

_LEAGUE_MAP = {
    "E0": "england", "D1": "germany", "SP1": "spain", "I1": "italy",
    "F1": "france", "N1": "netherlands", "P1": "portugal",
}
_FIXTURE_ODDS_COLS = ["B365H", "B365D", "B365A"]
_FIXTURE_PINNACLE_COLS = ["PSH", "PSD", "PSA"]

# Column → readable bookmaker name (BFE* excluded throughout)
_BK_COL_TO_NAME: dict[str, str] = {}
for _suffix, _name in [
    ("B365", "Bet365"), ("BFD", "Betfair"), ("BMGM", "BetMGM"), ("BV", "Bwin"),
    ("BW", "Betway"), ("CL", "Coral"), ("LB", "Ladbrokes"), ("PS", "Pinnacle"),
    ("WH", "Wm Hill"), ("VC", "VC Bet"), ("SJ", "Stan James"),
    ("IW", "Interwetten"), ("GB", "Gamebookers"), ("SB", "Sportingbet"),
]:
    _BK_COL_TO_NAME[f"{_suffix}H"] = _name
    _BK_COL_TO_NAME[f"{_suffix}D"] = _name
    _BK_COL_TO_NAME[f"{_suffix}A"] = _name


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
    optional_present = [c for c in _OPTIONAL_COLS if c in df.columns]

    # Compute CustomMax{H/D/A}: best available non-BFE odds per row
    custom_parts = {}
    for bk_cols, custom_col in [
        (_BK_COLS_H, "CustomMaxH"),
        (_BK_COLS_D, "CustomMaxD"),
        (_BK_COLS_A, "CustomMaxA"),
    ]:
        avail = [c for c in bk_cols if c in df.columns]
        if avail:
            custom_parts[custom_col] = df[avail].apply(pd.to_numeric, errors="coerce").max(axis=1)
        else:
            custom_parts[custom_col] = float("nan")

    custom_df = pd.DataFrame(custom_parts, index=df.index)
    custom_cols = list(custom_parts.keys())
    df = pd.concat([df[REQUIRED_COLS + optional_present], custom_df], axis=1)
    for c in _OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    df = df.dropna(subset=["FTR", "FTHG", "FTAG", "B365H", "B365D", "B365A"])
    df = df[df["FTR"].isin(["H", "D", "A"])]
    league, season = _parse_filename(path)
    df["league"] = league
    df["season"] = season
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


def load_fixtures() -> pd.DataFrame:
    """Load upcoming fixtures from data/raw/fixtures.csv, filtered to tracked leagues."""
    path = RAW_DIR / "fixtures.csv"
    if not path.exists():
        raise FileNotFoundError(
            "fixtures.csv not found — run with --predict to download it first"
        )
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = df[df["Div"].isin(_LEAGUE_MAP)].copy()
    missing_odds = [c for c in _FIXTURE_ODDS_COLS if c not in df.columns]
    if missing_odds:
        raise ValueError(f"fixtures.csv missing columns: {missing_odds}")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"] + _FIXTURE_ODDS_COLS)
    for col in _FIXTURE_ODDS_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=_FIXTURE_ODDS_COLS)
    df["league"] = df["Div"].map(_LEAGUE_MAP)
    # Include pre-match Pinnacle odds when available (null-safe: filter skipped when absent)
    for c in _FIXTURE_PINNACLE_COLS:
        if c not in df.columns:
            df[c] = float("nan")

    # Compute CustomMax{H/D/A} and which bookmaker offers that best price
    custom_parts: dict[str, pd.Series] = {}
    for bk_cols, val_col, bk_col in [
        (_BK_COLS_H, "CustomMaxH", "CustomMaxBkH"),
        (_BK_COLS_D, "CustomMaxD", "CustomMaxBkD"),
        (_BK_COLS_A, "CustomMaxA", "CustomMaxBkA"),
    ]:
        avail = [c for c in bk_cols if c in df.columns]
        if avail:
            numeric = df[avail].apply(pd.to_numeric, errors="coerce")
            custom_parts[val_col] = numeric.max(axis=1)
            custom_parts[bk_col] = numeric.idxmax(axis=1).map(_BK_COL_TO_NAME).fillna("")
        else:
            custom_parts[val_col] = float("nan")
            custom_parts[bk_col] = ""

    custom_df = pd.DataFrame(custom_parts, index=df.index)
    keep_cols = ["Date", "HomeTeam", "AwayTeam", "league", "Div"] + _FIXTURE_ODDS_COLS + _FIXTURE_PINNACLE_COLS
    return pd.concat([df[keep_cols], custom_df], axis=1).reset_index(drop=True)


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
