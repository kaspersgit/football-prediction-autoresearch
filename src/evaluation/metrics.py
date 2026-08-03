import pandas as pd

_ODDS_COL = {"H": "B365H", "D": "B365D", "A": "B365A"}
_IMPLIED_COL = {"H": "B365H", "D": "B365D", "A": "B365A"}


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


def add_model_proba(
    df: pd.DataFrame,
    y_proba,
    classes,
) -> pd.DataFrame:
    """
    Add model probability column for each row's predicted outcome.
    Also adds bookmaker implied probability and an 'is_value_bet' flag.
    y_proba: 2D array shape (n, 3), classes: array of class labels ['A','D','H'] (sorted)
    """
    df = df.copy()
    class_list = list(classes)
    model_probs = []
    implied_probs = []
    for pos, (i, row) in enumerate(df.iterrows()):
        pred = row["y_pred"]
        pred_idx = class_list.index(pred)
        model_prob = float(y_proba[pos, pred_idx])
        odds = float(row[_IMPLIED_COL[pred]])
        implied_prob = 1.0 / odds if odds > 0 else 1.0
        model_probs.append(model_prob)
        implied_probs.append(implied_prob)
    df["model_prob"] = model_probs
    df["implied_prob"] = implied_probs
    df["is_value_bet"] = df["model_prob"] > df["implied_prob"]
    return df


def _build_team_game_counts(df: pd.DataFrame) -> dict:
    """
    Returns a dict mapping (season, league, team, date) -> number of games
    that team has played in that season+league *before* that date.
    Used by the early-season filter.
    """
    counts: dict = {}
    has_season = "season" in df.columns
    has_league = "league" in df.columns
    if not has_season or not has_league:
        return counts

    # Collect all match dates per (season, league, team)
    team_dates: dict = {}
    for _, row in df.iterrows():
        key_home = (row["season"], row["league"], row["HomeTeam"])
        key_away = (row["season"], row["league"], row["AwayTeam"])
        d = pd.Timestamp(row["Date"])
        team_dates.setdefault(key_home, []).append(d)
        team_dates.setdefault(key_away, []).append(d)

    for (season, league, team), dates in team_dates.items():
        sorted_dates = sorted(set(dates))
        for rank, d in enumerate(sorted_dates):
            counts[(season, league, team, d)] = rank

    return counts


def compute_value_betting_results(
    df: pd.DataFrame,
    y_proba,
    classes,
    threshold: float = 0.0,
    kelly_fraction: float = 0.0,
    edge_baseline: str = "fair",
    inv_odds_factor: float = 0.0,
    min_stake: float = 1.0,
    max_odds: float = float("inf"),
    skip_leagues: set | None = None,
    skip_outcomes: set | None = None,
    max_edge: float = float("inf"),
    min_season_games: int = 0,
    max_overround: float = float("inf"),
) -> pd.DataFrame:
    """
    Multi-outcome value betting.

    Staking (in priority order):
      inv_odds_factor > 0  — stake = max(min_stake, factor / B365_odds)
                             Bets proportionally to win-likelihood; reduces longshot variance.
                             Validated config: factor=20, min_stake=3 (see autoresearch/experiments.md).
      kelly_fraction > 0   — fractional Kelly sizing; ROI% identical to flat but scales
                             absolute bankroll growth.
      default (both = 0)   — flat 1-unit staking.

    edge_baseline:
      "fair" (default) — compare model_prob against vig-stripped fair implied prob.
      "raw"            — compare model_prob against raw 1/odds implied prob (true EV).

    max_odds: skip bets whose B365 odds exceed this value (filter by market odds,
              not execution odds; validated cutoff = 4.0, above which model signal weakens).

    threshold:   minimum edge over fair prob required to place a bet.
    max_edge:    maximum allowed edge; bets with edge > max_edge are skipped (caps extreme predictions).
    min_season_games: skip bets where either team has played fewer than this many games in the
                      current season (in that league). Requires df to have 'season' and 'league' cols.
                      Default 0 = no filter. Recommended value: 4 (skips first ~4 gameweeks).
    kelly_fraction, inv_odds_factor, min_stake: staking size controls (see above).

    df must have: y_true, B365H, B365D, B365A, Date (index aligned with y_proba rows)
    y_proba: 2D array shape (n_matches, 3)
    classes: outcome labels in y_proba column order

    Returns DataFrame of individual bets with stake, profit and cumulative_profit columns.
    """
    outcomes = list(classes)
    df = df.reset_index(drop=True)

    team_game_counts: dict = {}
    if min_season_games > 0:
        team_game_counts = _build_team_game_counts(df)

    bet_rows = []
    for i, row in df.iterrows():
        y_true = row["y_true"]

        # Skip leagues excluded from betting (but still used in training)
        if skip_leagues and row.get("league") in skip_leagues:
            continue

        # Skip high-vig markets
        raw_check = 1.0/float(row["B365H"]) + 1.0/float(row["B365D"]) + 1.0/float(row["B365A"])
        if raw_check - 1.0 > max_overround:
            continue

        # Early-season filter: skip bets until both teams have enough season games
        if min_season_games > 0 and team_game_counts:
            season = row.get("season")
            league = row.get("league")
            date = pd.Timestamp(row["Date"])
            home_games = team_game_counts.get((season, league, row.get("HomeTeam"), date), 0)
            away_games = team_game_counts.get((season, league, row.get("AwayTeam"), date), 0)
            if home_games < min_season_games or away_games < min_season_games:
                continue

        # Raw implied probabilities (include bookmaker vig, sum to ~1.05)
        raw = {
            "H": 1.0 / float(row["B365H"]),
            "D": 1.0 / float(row["B365D"]),
            "A": 1.0 / float(row["B365A"]),
        }
        total_implied = sum(raw.values())

        # Vig-corrected fair probabilities (sum to 1.0)
        fair = {outcome: raw[outcome] / total_implied for outcome in raw}

        for j, outcome in enumerate(outcomes):
            if skip_outcomes and outcome in skip_outcomes:
                continue

            model_prob = float(y_proba[i, j])
            baseline_prob = raw[outcome] if edge_baseline == "raw" else fair[outcome]

            b365_odds = float(row[_ODDS_COL[outcome]])
            if b365_odds > max_odds:
                continue

            edge = model_prob - baseline_prob
            if edge <= threshold or edge > max_edge:
                continue

            # Execution odds: best available (CustomMax) or fall back to B365
            custom_col = f"CustomMax{outcome}"
            custom_val = row.get(custom_col)
            try:
                odds = float(custom_val) if custom_val is not None and not pd.isna(custom_val) else b365_odds
            except (TypeError, ValueError):
                odds = b365_odds

            if inv_odds_factor > 0.0:
                stake = max(min_stake, inv_odds_factor / b365_odds)
            elif kelly_fraction > 0.0:
                full_kelly = model_prob - (1.0 - model_prob) / (odds - 1.0)
                stake = kelly_fraction * max(full_kelly, 0.0)
            else:
                stake = 1.0
            profit = stake * ((odds - 1.0) if y_true == outcome else -1.0)
            bet_rows.append({
                "Date": row["Date"],
                "HomeTeam": row.get("HomeTeam", ""),
                "AwayTeam": row.get("AwayTeam", ""),
                "y_true": y_true,
                "y_pred": outcome,
                "odds": odds,
                "model_prob": model_prob,
                "implied_prob": baseline_prob,
                "stake": stake,
                "profit": profit,
            })

    if not bet_rows:
        return pd.DataFrame(columns=["Date", "y_true", "y_pred", "odds",
                                     "model_prob", "implied_prob", "stake", "profit",
                                     "cumulative_profit"])

    result = pd.DataFrame(bet_rows).sort_values("Date").reset_index(drop=True)
    result["cumulative_profit"] = result["profit"].cumsum()
    return result


def compute_roi(results: pd.DataFrame) -> float:
    """ROI as percentage: total_profit / total_staked * 100.
    Uses stake column when present (Kelly sizing); falls back to flat 1 unit per bet."""
    total_staked = results["stake"].sum() if "stake" in results.columns else float(len(results))
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
