import pandas as pd

WINDOW = 5
ELO_K = 30
ELO_HOME_ADV = 100
ELO_DEFAULT = 1500


def _points(ftr: str, is_home: bool) -> int:
    if ftr == "H":
        return 3 if is_home else 0
    if ftr == "A":
        return 0 if is_home else 3
    return 1


def _compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-match Elo ratings for each row, updated in date order.
    Returns df with home_elo and away_elo columns added.
    No leakage: Elo recorded BEFORE the match result is applied.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    elo: dict[str, float] = {}

    home_elos = []
    away_elos = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        h_elo = elo.get(home, ELO_DEFAULT)
        a_elo = elo.get(away, ELO_DEFAULT)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Expected score for home team (with home advantage)
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo + ELO_HOME_ADV) / 400))
        expected_away = 1 - expected_home

        ftr = row["FTR"]
        if ftr == "H":
            actual_home, actual_away = 1.0, 0.0
        elif ftr == "A":
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        elo[home] = h_elo + ELO_K * (actual_home - expected_home)
        elo[away] = a_elo + ELO_K * (actual_away - expected_away)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    return df


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
    team_df = team_df.groupby("team", group_keys=True).apply(
        lambda g: g.assign(
            form_pts=g["pts"].shift(1).rolling(window, min_periods=window).mean(),
            form_gf=g["gf"].shift(1).rolling(window, min_periods=window).mean(),
            form_ga=g["ga"].shift(1).rolling(window, min_periods=window).mean(),
        )
    )
    # In some pandas versions groupby+apply promotes the group key to the index;
    # reset to ensure "team" is available as a regular column.
    if "team" not in team_df.columns:
        team_df = team_df.reset_index(level="team")
    return team_df[["Date", "team", "form_pts", "form_gf", "form_ga"]]


def _build_merged(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    df = _compute_elo(df)
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
        "home_elo", "away_elo",
    ]
    merged = merged.dropna(subset=feature_cols)
    return merged.reset_index(drop=True)


FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    merged = _build_merged(df)
    X = merged[FEATURE_COLS].reset_index(drop=True)
    y = merged["FTR"].reset_index(drop=True)
    return X, y


def build_features_with_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    merged = _build_merged(df)
    X = merged[FEATURE_COLS]
    y = merged["FTR"]
    odds = merged[["B365H", "B365D", "B365A", "Date", "HomeTeam", "AwayTeam", "league", "season"]]
    return X, y, odds
