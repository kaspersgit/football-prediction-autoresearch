import pandas as pd

WINDOW = 5
ELO_K = 30
ELO_HOME_ADV = 100
ELO_DEFAULT = 1500

_LEAGUE_TO_INT = {"E0": 0, "D1": 1, "SP1": 2, "I1": 3}

FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga", "home_form_gd",
    "away_form_pts", "away_form_gf", "away_form_ga", "away_form_gd",
    "home_elo", "away_elo", "elo_diff",
    "league_code",
]


def _points(ftr: str, is_home: bool) -> int:
    if ftr == "H":
        return 3 if is_home else 0
    if ftr == "A":
        return 0 if is_home else 3
    return 1


def _compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match Elo ratings — updated after each match, no leakage."""
    df = df.sort_values("Date").reset_index(drop=True)
    elo: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        h_elo = elo.get(home, ELO_DEFAULT)
        a_elo = elo.get(away, ELO_DEFAULT)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo + ELO_HOME_ADV) / 400))
        expected_away = 1 - expected_home
        ftr = row["FTR"]
        actual_home = 1.0 if ftr == "H" else (0.0 if ftr == "A" else 0.5)
        actual_away = 1.0 - actual_home

        elo[home] = h_elo + ELO_K * (actual_home - expected_home)
        elo[away] = a_elo + ELO_K * (actual_away - expected_away)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    return df


def _team_rolling_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling stats per team across all matches."""
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
        on=["Date", "HomeTeam"], how="left",
    )
    merged = merged.merge(
        stats.rename(columns={"team": "AwayTeam", "form_pts": "away_form_pts",
                               "form_gf": "away_form_gf", "form_ga": "away_form_ga"}),
        on=["Date", "AwayTeam"], how="left",
    )

    # Derived features
    merged["home_form_gd"] = merged["home_form_gf"] - merged["home_form_ga"]
    merged["away_form_gd"] = merged["away_form_gf"] - merged["away_form_ga"]
    merged["elo_diff"] = merged["home_elo"] - merged["away_elo"]
    merged["league_code"] = merged["league"].map(_LEAGUE_TO_INT).fillna(0).astype(int)

    base_cols = ["home_form_pts", "home_form_gf", "home_form_ga",
                 "away_form_pts", "away_form_gf", "away_form_ga"]
    merged = merged.dropna(subset=base_cols).reset_index(drop=True)
    return merged


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
