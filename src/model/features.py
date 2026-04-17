import pandas as pd

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
    def _add_rolling(g: pd.DataFrame) -> pd.DataFrame:
        return g.assign(
            form_pts=g["pts"].shift(1).rolling(window, min_periods=window).mean(),
            form_gf=g["gf"].shift(1).rolling(window, min_periods=window).mean(),
            form_ga=g["ga"].shift(1).rolling(window, min_periods=window).mean(),
        )

    team_df = team_df.groupby("team", group_keys=False)[
        ["Date", "team", "gf", "ga", "pts"]
    ].apply(_add_rolling)
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
