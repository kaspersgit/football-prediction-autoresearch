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


def compute_value_betting_results(
    df: pd.DataFrame,
    y_proba,
    classes,
    threshold: float = 0.0,
    kelly_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Multi-outcome value betting with vig-corrected implied probabilities.

    For each match, normalise bookmaker implied probs to sum to 1.0 (removing vig),
    then bet any outcome where model_prob > fair_implied_prob + threshold.

    threshold: minimum edge required to place a bet (e.g. 0.05 = need 5% extra edge).
    kelly_fraction: 0.0 = flat 1-unit staking; >0 = fractional Kelly sizing.
        Stake per bet = kelly_fraction * full_kelly, where
        full_kelly = model_prob - (1 - model_prob) / (odds - 1).
        Quarter-Kelly (0.25) is a common conservative choice.

    df must have: y_true, B365H, B365D, B365A, Date (index aligned with y_proba rows)
    y_proba: 2D array shape (n_matches, 3)
    classes: outcome labels in y_proba column order

    Returns DataFrame of individual bets with stake, profit and cumulative_profit columns.
    """
    outcomes = list(classes)
    df = df.reset_index(drop=True)

    has_pinnacle = all(c in df.columns for c in ("PSCH", "PSCD", "PSCA"))

    bet_rows = []
    for i, row in df.iterrows():
        y_true = row["y_true"]

        # Raw implied probabilities (include bookmaker vig, sum to ~1.05)
        raw = {
            "H": 1.0 / float(row["B365H"]),
            "D": 1.0 / float(row["B365D"]),
            "A": 1.0 / float(row["B365A"]),
        }
        total_implied = sum(raw.values())

        # Vig-corrected fair probabilities (sum to 1.0)
        fair = {outcome: raw[outcome] / total_implied for outcome in raw}

        # Pinnacle closing fair probs — None when data unavailable for this row
        pinnacle_fair = None
        if has_pinnacle:
            psch, pscd, psca = row.get("PSCH"), row.get("PSCD"), row.get("PSCA")
            try:
                ps_total = 1/float(psch) + 1/float(pscd) + 1/float(psca)
                pinnacle_fair = {
                    "H": (1/float(psch)) / ps_total,
                    "D": (1/float(pscd)) / ps_total,
                    "A": (1/float(psca)) / ps_total,
                }
            except (TypeError, ValueError, ZeroDivisionError):
                pinnacle_fair = None  # null row — no Pinnacle filter applied

        for j, outcome in enumerate(outcomes):
            model_prob = float(y_proba[i, j])
            fair_implied_prob = fair[outcome]

            # Pinnacle confirmation: only bet when Pinnacle also thinks B365 underprices this outcome.
            # When Pinnacle data is missing for this row, the filter is skipped (no data = no veto).
            if pinnacle_fair is not None and pinnacle_fair[outcome] <= fair_implied_prob:
                continue

            if model_prob > fair_implied_prob + threshold:
                odds = float(row[_ODDS_COL[outcome]])
                if kelly_fraction > 0.0:
                    # Full Kelly fraction: f* = p - (1-p)/(odds-1)
                    # Always positive when model_prob > 1/odds (guaranteed by edge > 0 over fair)
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
                    "implied_prob": fair_implied_prob,
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
