import pandas as pd

WINDOW = 5
ELO_K = 30
ELO_HOME_ADV = 100
ELO_DEFAULT = 1500

FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo",
    "market_h", "market_d", "market_a",
]


def _points(ftr: str, is_home: bool) -> int:
    if ftr == "H":
        return 3 if is_home else 0
    if ftr == "A":
        return 0 if is_home else 3
    return 1


def _compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match head-to-head home win rate for each fixture — no leakage."""
    df = df.sort_values("Date").reset_index(drop=True)
    # pair_wins[(team_a, team_b)] = [a_wins, b_wins, total] where team_a < team_b alphabetically
    pair_wins: dict[tuple, list] = {}
    rates = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key = (min(home, away), max(home, away))
        team_a = key[0]
        rec = pair_wins.get(key, [0, 0, 0])  # [a_wins, b_wins, total]

        if rec[2] == 0:
            rates.append(0.5)
        else:
            home_wins = rec[0] if home == team_a else rec[1]
            rates.append(home_wins / rec[2])

        # Update after reading (no leakage)
        ftr = row["FTR"]
        if key not in pair_wins:
            pair_wins[key] = [0, 0, 0]
        if ftr == "H":
            pair_wins[key][0 if home == team_a else 1] += 1
        elif ftr == "A":
            pair_wins[key][1 if home == team_a else 0] += 1
        pair_wins[key][2] += 1

    df = df.copy()
    df["h2h_home_win_rate"] = rates
    return df


def _get_current_h2h_state(df: pd.DataFrame) -> dict[tuple, list]:
    """Run h2h through all matches and return final [a_wins, b_wins, total] per pair."""
    df = df.sort_values("Date")
    pair_wins: dict[tuple, list] = {}
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key = (min(home, away), max(home, away))
        team_a = key[0]
        if key not in pair_wins:
            pair_wins[key] = [0, 0, 0]
        ftr = row["FTR"]
        if ftr == "H":
            pair_wins[key][0 if home == team_a else 1] += 1
        elif ftr == "A":
            pair_wins[key][1 if home == team_a else 0] += 1
        pair_wins[key][2] += 1
    return pair_wins


def _h2h_rate(state: dict, home: str, away: str) -> float:
    key = (min(home, away), max(home, away))
    rec = state.get(key, [0, 0, 0])
    if rec[2] == 0:
        return 0.5
    home_wins = rec[0] if home == key[0] else rec[1]
    return home_wins / rec[2]


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
    total_imp = 1/merged["B365H"] + 1/merged["B365D"] + 1/merged["B365A"]
    merged["market_h"] = (1/merged["B365H"]) / total_imp
    merged["market_d"] = (1/merged["B365D"]) / total_imp
    merged["market_a"] = (1/merged["B365A"]) / total_imp
    merged = merged.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return merged


def _get_current_elo_state(df: pd.DataFrame) -> dict[str, float]:
    """Run Elo through all matches and return final rating per team."""
    df = df.sort_values("Date")
    elo: dict[str, float] = {}
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        h_elo = elo.get(home, ELO_DEFAULT)
        a_elo = elo.get(away, ELO_DEFAULT)
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo + ELO_HOME_ADV) / 400))
        ftr = row["FTR"]
        actual_home = 1.0 if ftr == "H" else (0.0 if ftr == "A" else 0.5)
        elo[home] = h_elo + ELO_K * (actual_home - expected_home)
        elo[away] = a_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))
    return elo


def _get_current_team_form(df: pd.DataFrame, window: int = WINDOW) -> dict[str, dict]:
    """Return rolling form state per team based on their last `window` completed games."""
    records = []
    for _, row in df.iterrows():
        for team, is_home in [(row["HomeTeam"], True), (row["AwayTeam"], False)]:
            gf = row["FTHG"] if is_home else row["FTAG"]
            ga = row["FTAG"] if is_home else row["FTHG"]
            pts = _points(row["FTR"], is_home)
            records.append({"Date": row["Date"], "team": team, "gf": gf, "ga": ga, "pts": pts})
    team_df = pd.DataFrame(records).sort_values("Date")
    form: dict[str, dict] = {}
    for team, group in team_df.groupby("team"):
        last_n = group.tail(window)
        if len(last_n) >= window:
            form[team] = {
                "form_pts": last_n["pts"].mean(),
                "form_gf": last_n["gf"].mean(),
                "form_ga": last_n["ga"].mean(),
            }
    return form


def build_fixture_features(
    historical_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the feature matrix for upcoming fixtures using the current model state
    (Elo ratings and rolling form) derived from all historical matches.

    Teams with fewer than WINDOW completed games are dropped (insufficient history).
    Returns a DataFrame with FEATURE_COLS plus match metadata columns.
    """
    elo_state = _get_current_elo_state(historical_df)
    form_state = _get_current_team_form(historical_df)

    rows = []
    for _, row in fixtures_df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in form_state or away not in form_state:
            continue
        hf, af = form_state[home], form_state[away]
        rows.append({
            "Date": row["Date"],
            "HomeTeam": home,
            "AwayTeam": away,
            "league": row.get("league", ""),
            "B365H": row["B365H"],
            "B365D": row["B365D"],
            "B365A": row["B365A"],
            "home_form_pts": hf["form_pts"],
            "home_form_gf": hf["form_gf"],
            "home_form_ga": hf["form_ga"],
            "away_form_pts": af["form_pts"],
            "away_form_gf": af["form_gf"],
            "away_form_ga": af["form_ga"],
            "home_elo": elo_state.get(home, ELO_DEFAULT),
            "away_elo": elo_state.get(away, ELO_DEFAULT),
            "market_h": (1/row["B365H"]) / (1/row["B365H"] + 1/row["B365D"] + 1/row["B365A"]),
            "market_d": (1/row["B365D"]) / (1/row["B365H"] + 1/row["B365D"] + 1/row["B365A"]),
            "market_a": (1/row["B365A"]) / (1/row["B365H"] + 1/row["B365D"] + 1/row["B365A"]),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


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
