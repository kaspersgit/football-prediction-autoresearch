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
) -> pd.DataFrame:
    """
    Multi-outcome value betting: for each match, bet 1 unit on every outcome
    where model probability > bookmaker implied probability (1/odds).

    df must have: y_true, B365H, B365D, B365A, Date (and index aligned with y_proba rows)
    y_proba: 2D array shape (n_matches, 3)
    classes: array of outcome labels in the same order as y_proba columns

    Returns a DataFrame of individual bets with columns:
      Date, y_true, outcome_bet, odds, model_prob, implied_prob, profit, cumulative_profit
    One row per bet placed (can be multiple per match if multiple outcomes have edge).
    """
    outcomes = list(classes)
    df = df.reset_index(drop=True)

    bet_rows = []
    for i, row in df.iterrows():
        y_true = row["y_true"]
        for j, outcome in enumerate(outcomes):
            odds_col = _ODDS_COL[outcome]
            odds = float(row[odds_col])
            implied_prob = 1.0 / odds if odds > 0 else 1.0
            model_prob = float(y_proba[i, j])
            if model_prob > implied_prob:
                profit = (odds - 1.0) if y_true == outcome else -1.0
                bet_rows.append({
                    "Date": row["Date"],
                    "HomeTeam": row.get("HomeTeam", ""),
                    "AwayTeam": row.get("AwayTeam", ""),
                    "y_true": y_true,
                    "y_pred": outcome,
                    "odds": odds,
                    "model_prob": model_prob,
                    "implied_prob": implied_prob,
                    "profit": profit,
                })

    if not bet_rows:
        return pd.DataFrame(columns=["Date", "y_true", "y_pred", "odds",
                                     "model_prob", "implied_prob", "profit", "cumulative_profit"])

    result = pd.DataFrame(bet_rows).sort_values("Date").reset_index(drop=True)
    result["cumulative_profit"] = result["profit"].cumsum()
    return result


def compute_roi(results: pd.DataFrame) -> float:
    """ROI as percentage: total_profit / total_staked * 100."""
    total_staked = float(len(results))
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
