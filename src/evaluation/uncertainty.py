import numpy as np
import pandas as pd


def weekly_roi_interval(
    bets: pd.DataFrame,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float] | None:
    """Return a percentile bootstrap interval for ROI after resampling ISO weeks."""
    weekly_bets = bets[["Date", "profit"]].copy()
    weekly_bets["stake"] = bets["stake"] if "stake" in bets else 1.0
    iso_weeks = pd.to_datetime(weekly_bets["Date"]).dt.isocalendar()
    weekly_bets["iso_year"] = iso_weeks.year
    weekly_bets["iso_week"] = iso_weeks.week
    weekly_totals = weekly_bets.groupby(["iso_year", "iso_week"])[["profit", "stake"]].sum()

    if len(weekly_totals) < 2:
        return None

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(len(weekly_totals), size=(n_resamples, len(weekly_totals)))
    weekly_profits = weekly_totals["profit"].to_numpy()
    weekly_stakes = weekly_totals["stake"].to_numpy()
    sampled_roi = (
        weekly_profits[sampled_indices].sum(axis=1)
        / weekly_stakes[sampled_indices].sum(axis=1)
        * 100.0
    )
    tail_probability = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sampled_roi, [tail_probability, 1.0 - tail_probability])
    return float(lower), float(upper)
