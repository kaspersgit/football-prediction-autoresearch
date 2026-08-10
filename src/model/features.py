import pandas as pd

WINDOW = 5
DC_SPAN = 10
MARKET_BIAS_WINDOW = 20
ELO_K = 30
ELO_HOME_ADV = 100
ELO_DEFAULT = 1500

FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo",
    "home_elo_delta", "away_elo_delta",
    "market_h", "market_d", "market_a",
    "market_overround",
    "league_E0", "league_D1", "league_SP1", "league_F1", "league_N1", "league_P1",  # I1 is omitted reference
    "h2h_home_win_rate",
    "home_draw_rate", "away_draw_rate",
    "home_market_bias", "away_market_bias",
    "match_balance",
    "home_attack", "home_defense", "away_attack", "away_defense",
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


def _compute_market_bias(df: pd.DataFrame, window: int = MARKET_BIAS_WINDOW) -> pd.DataFrame:
    """Pre-match rolling mean of (actual_result_share - market_fair_prob) per team.
    Positive = team consistently beats market expectations; negative = over-valued by market."""
    df = df.sort_values("Date").reset_index(drop=True)
    total_imp = 1 / df["B365H"] + 1 / df["B365D"] + 1 / df["B365A"]
    market_h_arr = ((1 / df["B365H"]) / total_imp).values
    market_a_arr = ((1 / df["B365A"]) / total_imp).values

    records = []
    for i, row in df.iterrows():
        actual_h = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)
        actual_a = 1.0 if row["FTR"] == "A" else (0.5 if row["FTR"] == "D" else 0.0)
        records.append({"Date": row["Date"], "team": row["HomeTeam"], "bias": actual_h - market_h_arr[i]})
        records.append({"Date": row["Date"], "team": row["AwayTeam"], "bias": actual_a - market_a_arr[i]})

    bias_df = pd.DataFrame(records).sort_values("Date")
    bias_df = bias_df.groupby("team", group_keys=True).apply(
        lambda g: g.assign(
            market_bias=g["bias"].shift(1).rolling(window, min_periods=window).mean()
        )
    )
    if "team" not in bias_df.columns:
        bias_df = bias_df.reset_index(level="team")
    bias_df = bias_df[["Date", "team", "market_bias"]]

    df = df.merge(
        bias_df.rename(columns={"team": "HomeTeam", "market_bias": "home_market_bias"}),
        on=["Date", "HomeTeam"], how="left",
    )
    df = df.merge(
        bias_df.rename(columns={"team": "AwayTeam", "market_bias": "away_market_bias"}),
        on=["Date", "AwayTeam"], how="left",
    )
    return df


def _get_current_market_bias(df: pd.DataFrame, window: int = MARKET_BIAS_WINDOW) -> dict[str, float]:
    """Return each team's current market-bias state (rolling mean over last window games)."""
    df = df.sort_values("Date").reset_index(drop=True)
    total_imp = 1 / df["B365H"] + 1 / df["B365D"] + 1 / df["B365A"]
    market_h_arr = ((1 / df["B365H"]) / total_imp).values
    market_a_arr = ((1 / df["B365A"]) / total_imp).values

    records = []
    for i, row in df.iterrows():
        actual_h = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)
        actual_a = 1.0 if row["FTR"] == "A" else (0.5 if row["FTR"] == "D" else 0.0)
        records.append({"Date": row["Date"], "team": row["HomeTeam"], "bias": actual_h - market_h_arr[i]})
        records.append({"Date": row["Date"], "team": row["AwayTeam"], "bias": actual_a - market_a_arr[i]})

    bias_df = pd.DataFrame(records).sort_values("Date")
    result = {}
    for team, group in bias_df.groupby("team"):
        last_n = group.tail(window)
        if len(last_n) >= window:
            result[team] = float(last_n["bias"].mean())
    return result


def _compute_elo(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Pre-match Elo ratings and momentum (delta over last `window` games) — no leakage."""
    df = df.sort_values("Date").reset_index(drop=True)
    elo: dict[str, float] = {}
    elo_history: dict[str, list] = {}
    home_elos, away_elos = [], []
    home_deltas, away_deltas = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        h_elo = elo.get(home, ELO_DEFAULT)
        a_elo = elo.get(away, ELO_DEFAULT)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        h_hist = elo_history.get(home, [])
        a_hist = elo_history.get(away, [])
        home_deltas.append(h_elo - h_hist[-window] if len(h_hist) >= window else 0.0)
        away_deltas.append(a_elo - a_hist[-window] if len(a_hist) >= window else 0.0)

        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo + ELO_HOME_ADV) / 400))
        ftr = row["FTR"]
        actual_home = 1.0 if ftr == "H" else (0.0 if ftr == "A" else 0.5)

        elo[home] = h_elo + ELO_K * (actual_home - expected_home)
        elo[away] = a_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))

        if home not in elo_history:
            elo_history[home] = []
        if away not in elo_history:
            elo_history[away] = []
        elo_history[home].append(h_elo)
        elo_history[away].append(a_elo)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["home_elo_delta"] = home_deltas
    df["away_elo_delta"] = away_deltas
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
            form_pts=g["pts"].shift(1).ewm(span=window, min_periods=window).mean(),
            form_gf=g["gf"].shift(1).ewm(span=window, min_periods=window).mean(),
            form_ga=g["ga"].shift(1).ewm(span=window, min_periods=window).mean(),
        )
    )
    if "team" not in team_df.columns:
        team_df = team_df.reset_index(level="team")
    return team_df[["Date", "team", "form_pts", "form_gf", "form_ga"]]


def _compute_dc_ratings(df: pd.DataFrame, span: int = DC_SPAN) -> pd.DataFrame:
    """Long-window EWM attack/defense ratings per team (span=DC_SPAN, min_periods=1).
    Uses raw goals to separate offensive quality from defensive quality independently of Elo W/L/D."""
    records = []
    for _, row in df.iterrows():
        for team, is_home in [(row["HomeTeam"], True), (row["AwayTeam"], False)]:
            gf = row["FTHG"] if is_home else row["FTAG"]
            ga = row["FTAG"] if is_home else row["FTHG"]
            records.append({"Date": row["Date"], "team": team, "gf": gf, "ga": ga})
    team_df = pd.DataFrame(records).sort_values("Date")
    team_df = team_df.groupby("team", group_keys=True).apply(
        lambda g: g.assign(
            attack=g["gf"].shift(1).ewm(span=span, min_periods=span).mean(),
            defense=g["ga"].shift(1).ewm(span=span, min_periods=span).mean(),
        )
    )
    if "team" not in team_df.columns:
        team_df = team_df.reset_index(level="team")
    return team_df[["Date", "team", "attack", "defense"]]


def _get_current_dc_ratings(df: pd.DataFrame, span: int = DC_SPAN) -> dict[str, dict]:
    """Return each team's current attack/defense rating from their last completed games."""
    records = []
    for _, row in df.iterrows():
        for team, is_home in [(row["HomeTeam"], True), (row["AwayTeam"], False)]:
            gf = row["FTHG"] if is_home else row["FTAG"]
            ga = row["FTAG"] if is_home else row["FTHG"]
            records.append({"Date": row["Date"], "team": team, "gf": gf, "ga": ga})
    team_df = pd.DataFrame(records).sort_values("Date")
    result = {}
    for team, group in team_df.groupby("team"):
        if len(group) >= span:
            ewm_vals = group[["gf", "ga"]].ewm(span=span, min_periods=span).mean().iloc[-1]
            result[team] = {"attack": float(ewm_vals["gf"]), "defense": float(ewm_vals["ga"])}
    return result


def _compute_draw_rates(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Pre-match rolling draw rate per team over last `window` games — no leakage."""
    records = []
    for _, row in df.sort_values("Date").iterrows():
        is_draw = 1.0 if row["FTR"] == "D" else 0.0
        records.append({"Date": row["Date"], "team": row["HomeTeam"], "is_draw": is_draw})
        records.append({"Date": row["Date"], "team": row["AwayTeam"], "is_draw": is_draw})
    dr_df = pd.DataFrame(records).sort_values("Date")
    dr_df = dr_df.groupby("team", group_keys=True).apply(
        lambda g: g.assign(
            draw_rate=g["is_draw"].shift(1).rolling(window, min_periods=window).mean()
        )
    )
    if "team" not in dr_df.columns:
        dr_df = dr_df.reset_index(level="team")
    dr_df = dr_df[["Date", "team", "draw_rate"]]

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.merge(
        dr_df.rename(columns={"team": "HomeTeam", "draw_rate": "home_draw_rate"}),
        on=["Date", "HomeTeam"], how="left",
    )
    df = df.merge(
        dr_df.rename(columns={"team": "AwayTeam", "draw_rate": "away_draw_rate"}),
        on=["Date", "AwayTeam"], how="left",
    )
    return df


def _get_current_draw_rates(df: pd.DataFrame, window: int = WINDOW) -> dict[str, float]:
    """Return each team's draw rate over their last `window` completed games."""
    records = []
    for _, row in df.sort_values("Date").iterrows():
        is_draw = 1.0 if row["FTR"] == "D" else 0.0
        records.append({"Date": row["Date"], "team": row["HomeTeam"], "is_draw": is_draw})
        records.append({"Date": row["Date"], "team": row["AwayTeam"], "is_draw": is_draw})
    dr_df = pd.DataFrame(records).sort_values("Date")
    result = {}
    for team, group in dr_df.groupby("team"):
        last_n = group.tail(window)
        if len(last_n) >= window:
            result[team] = float(last_n["is_draw"].mean())
    return result


def _build_merged(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    df = _compute_elo(df)
    df = _compute_h2h(df)
    df = _compute_draw_rates(df)
    df = _compute_market_bias(df)

    stats = _team_rolling_stats(df)
    dc = _compute_dc_ratings(df)

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
    merged = merged.merge(
        dc.rename(columns={"team": "HomeTeam", "attack": "home_attack", "defense": "home_defense"}),
        on=["Date", "HomeTeam"], how="left",
    )
    merged = merged.merge(
        dc.rename(columns={"team": "AwayTeam", "attack": "away_attack", "defense": "away_defense"}),
        on=["Date", "AwayTeam"], how="left",
    )
    total_imp = 1/merged["B365H"] + 1/merged["B365D"] + 1/merged["B365A"]
    merged["market_h"] = (1/merged["B365H"]) / total_imp
    merged["market_d"] = (1/merged["B365D"]) / total_imp
    merged["market_a"] = (1/merged["B365A"]) / total_imp
    merged["market_overround"] = total_imp - 1.0
    merged["match_balance"] = 1.0 - (merged["market_h"] - merged["market_a"]).abs()
    for lc in ["E0", "D1", "SP1", "F1", "N1", "P1"]:
        merged[f"league_{lc}"] = (merged["league"] == lc).astype(float)
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


def _get_current_elo_delta_state(df: pd.DataFrame, window: int = WINDOW) -> dict[str, float]:
    """Return each team's Elo delta (current Elo minus Elo `window` games ago)."""
    df = df.sort_values("Date")
    elo: dict[str, float] = {}
    elo_history: dict[str, list] = {}
    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        h_elo = elo.get(home, ELO_DEFAULT)
        a_elo = elo.get(away, ELO_DEFAULT)
        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo + ELO_HOME_ADV) / 400))
        ftr = row["FTR"]
        actual_home = 1.0 if ftr == "H" else (0.0 if ftr == "A" else 0.5)
        elo[home] = h_elo + ELO_K * (actual_home - expected_home)
        elo[away] = a_elo + ELO_K * ((1 - actual_home) - (1 - expected_home))
        for team, pre_elo in [(home, h_elo), (away, a_elo)]:
            if team not in elo_history:
                elo_history[team] = []
            elo_history[team].append(pre_elo)

    result = {}
    for team, current in elo.items():
        hist = elo_history.get(team, [])
        result[team] = current - hist[-window] if len(hist) >= window else 0.0
    return result


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
        if len(group) >= window:
            ewm_vals = group[["pts", "gf", "ga"]].ewm(span=window, min_periods=window).mean().iloc[-1]
            form[team] = {
                "form_pts": ewm_vals["pts"],
                "form_gf": ewm_vals["gf"],
                "form_ga": ewm_vals["ga"],
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
    elo_delta_state = _get_current_elo_delta_state(historical_df)
    form_state = _get_current_team_form(historical_df)
    h2h_state = _get_current_h2h_state(historical_df)
    draw_rate_state = _get_current_draw_rates(historical_df)
    market_bias_state = _get_current_market_bias(historical_df)
    dc_state = _get_current_dc_ratings(historical_df)

    rows = []
    for _, row in fixtures_df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in form_state or away not in form_state:
            continue
        hf, af = form_state[home], form_state[away]
        total_imp = 1/row["B365H"] + 1/row["B365D"] + 1/row["B365A"]
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
            "home_elo_delta": elo_delta_state.get(home, 0.0),
            "away_elo_delta": elo_delta_state.get(away, 0.0),
            "market_h": (1/row["B365H"]) / total_imp,
            "market_d": (1/row["B365D"]) / total_imp,
            "market_a": (1/row["B365A"]) / total_imp,
            "market_overround": total_imp - 1.0,
            "league_E0": float(row.get("league", "") == "E0"),
            "league_D1": float(row.get("league", "") == "D1"),
            "league_SP1": float(row.get("league", "") == "SP1"),
            "league_F1": float(row.get("league", "") == "F1"),
            "league_N1": float(row.get("league", "") == "N1"),
            "league_P1": float(row.get("league", "") == "P1"),
            "h2h_home_win_rate": _h2h_rate(h2h_state, home, away),
            "home_draw_rate": draw_rate_state.get(home, float("nan")),
            "away_draw_rate": draw_rate_state.get(away, float("nan")),
            "home_market_bias": market_bias_state.get(home, float("nan")),
            "away_market_bias": market_bias_state.get(away, float("nan")),
            "match_balance": 1.0 - abs((1/row["B365H"]) / total_imp - (1/row["B365A"]) / total_imp),
            "home_attack": dc_state.get(home, {}).get("attack", float("nan")),
            "home_defense": dc_state.get(home, {}).get("defense", float("nan")),
            "away_attack": dc_state.get(away, {}).get("attack", float("nan")),
            "away_defense": dc_state.get(away, {}).get("defense", float("nan")),
            # Pass-through metadata (not ML features)
            "PSH": row.get("PSH", float("nan")),
            "PSD": row.get("PSD", float("nan")),
            "PSA": row.get("PSA", float("nan")),
            "CustomMaxH": row.get("CustomMaxH", float("nan")),
            "CustomMaxD": row.get("CustomMaxD", float("nan")),
            "CustomMaxA": row.get("CustomMaxA", float("nan")),
            "CustomMaxBkH": row.get("CustomMaxBkH", ""),
            "CustomMaxBkD": row.get("CustomMaxBkD", ""),
            "CustomMaxBkA": row.get("CustomMaxBkA", ""),
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
    base_odds_cols = ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "Date", "HomeTeam", "AwayTeam", "league", "season"]
    custom_max_cols = [c for c in ["CustomMaxH", "CustomMaxD", "CustomMaxA"] if c in merged.columns]
    odds = merged[base_odds_cols + custom_max_cols]
    return X, y, odds
