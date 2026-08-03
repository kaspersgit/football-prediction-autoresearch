import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_roi, compute_value_betting_results


def select_league_thresholds(
    prior_season_data: list[dict],
    leagues: list[str],
    grid: list[float],
    min_bets: int = 20,
    default_threshold: float = 0.0,
    max_odds: float = float("inf"),
    max_overround: float = float("inf"),
    max_edge: float = float("inf"),
) -> dict[str, float]:
    """
    For each league, sweep `grid` thresholds on accumulated prior-season OOS data
    and return the threshold that maximises ROI × √bets (stability), subject to
    a minimum of `min_bets` bets over the calibration window.

    Falls back to `default_threshold` when:
    - `prior_season_data` is empty, or
    - the league has no rows in the prior data, or
    - no threshold in `grid` produces >= `min_bets` bets.

    Tie-breaking: lower threshold wins (preserves more bets, more conservative).

    Each entry in prior_season_data must have:
        eval_df  — DataFrame with y_true, B365H/D/A, Date, league, season columns
        y_proba  — ndarray shape (n_rows, n_classes), row-aligned to eval_df
        classes  — ndarray of class labels (e.g. ['A','D','H'])
    """
    if not prior_season_data:
        return {lg: default_threshold for lg in leagues}

    combined_df = pd.concat(
        [sd["eval_df"].reset_index(drop=True) for sd in prior_season_data]
    ).reset_index(drop=True)
    combined_proba = np.vstack([sd["y_proba"] for sd in prior_season_data])
    classes = prior_season_data[-1]["classes"]

    result: dict[str, float] = {}
    all_leagues_in_data = set(combined_df["league"].unique())

    for league in leagues:
        if league not in all_leagues_in_data:
            result[league] = default_threshold
            continue

        mask = combined_df["league"].values == league
        league_df = combined_df[mask].reset_index(drop=True)
        league_proba = combined_proba[mask]

        best_threshold = default_threshold
        best_stability = -np.inf

        for t in sorted(grid):  # ascending → lower threshold wins on equal stability
            bets = compute_value_betting_results(
                league_df,
                league_proba,
                classes,
                threshold=t,
                max_odds=max_odds,
                max_overround=max_overround,
                max_edge=max_edge,
            )
            n = len(bets)
            if n < min_bets:
                continue
            roi = compute_roi(bets)
            stability = roi * (n ** 0.5)
            if stability > best_stability:
                best_stability = stability
                best_threshold = t

        result[league] = best_threshold

    return result
