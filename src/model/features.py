import pandas as pd

WINDOW = 5


def _points(ftr: str, is_home: bool) -> int:
    if ftr == "H":
        return 3 if is_home else 0
    if ftr == "A":
        return 0 if is_home else 3
    return 1


def _home_rolling_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling stats for each team computed from their HOME games only."""
    home_records = []
    for _, row in df.iterrows():
        home_records.append({
            "Date": row["Date"],
            "team": row["HomeTeam"],
            "gf": row["FTHG"],
            "ga": row["FTAG"],
            "pts": _points(row["FTR"], True),
        })
    home_df = pd.DataFrame(home_records).sort_values(["team", "Date"]).reset_index(drop=True)
    for col, out in [("pts", "home_form_pts"), ("gf", "home_form_gf"), ("ga", "home_form_ga")]:
        home_df[out] = home_df.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
    return home_df[["Date", "team", "home_form_pts", "home_form_gf", "home_form_ga"]]


def _away_rolling_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling stats for each team computed from their AWAY games only."""
    away_records = []
    for _, row in df.iterrows():
        away_records.append({
            "Date": row["Date"],
            "team": row["AwayTeam"],
            "gf": row["FTAG"],
            "ga": row["FTHG"],
            "pts": _points(row["FTR"], False),
        })
    away_df = pd.DataFrame(away_records).sort_values(["team", "Date"]).reset_index(drop=True)
    for col, out in [("pts", "away_form_pts"), ("gf", "away_form_gf"), ("ga", "away_form_ga")]:
        away_df[out] = away_df.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
    return away_df[["Date", "team", "away_form_pts", "away_form_gf", "away_form_ga"]]


def _merge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    home_stats = _home_rolling_stats(df)
    away_stats = _away_rolling_stats(df)

    # Home team's home stats
    merged = df.merge(
        home_stats.rename(columns={"team": "HomeTeam"}),
        on=["Date", "HomeTeam"],
        how="left",
    )
    # Away team's away stats
    merged = merged.merge(
        away_stats.rename(columns={"team": "AwayTeam"}),
        on=["Date", "AwayTeam"],
        how="left",
    )
    feature_cols = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    return merged.dropna(subset=feature_cols).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    merged = _merge_features(df)
    X = merged[feature_cols]
    y = merged["FTR"]
    return X, y


def build_features_with_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_cols = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    merged = _merge_features(df)
    X = merged[feature_cols]
    y = merged["FTR"]
    odds = merged[["B365H", "B365D", "B365A", "Date", "HomeTeam", "AwayTeam", "league", "season"]]
    return X, y, odds
