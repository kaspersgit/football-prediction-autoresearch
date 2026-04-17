import pandas as pd

_ODDS_COL = {"H": "B365H", "D": "B365D", "A": "B365A"}


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
