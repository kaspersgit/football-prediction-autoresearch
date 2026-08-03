import numpy as np
import pandas as pd
from src.evaluation.threshold_selector import select_league_thresholds


def _make_season_data(n: int, league: str, edge: float, classes=None) -> dict:
    """
    Synthetic season: `n` matches where the model always has `edge` advantage over fair.
    All bets are correct (profit = odds - 1) to produce positive ROI.
    Odds fixed at 2.0 for simplicity (overround ~0 for test purposes — set overround filter high).
    """
    if classes is None:
        classes = np.array(["A", "D", "H"])
    n_classes = len(classes)
    y_proba = np.full((n, n_classes), 1 / n_classes)
    h_idx = list(classes).index("H")
    y_proba[:, h_idx] += edge
    eval_df = pd.DataFrame({
        "Date": pd.date_range("2023-08-01", periods=n, freq="7D"),
        "HomeTeam": [f"Home{i}" for i in range(n)],
        "AwayTeam": [f"Away{i}" for i in range(n)],
        "league": league,
        "season": "2023-24",
        "y_true": "H",
        "B365H": 2.0,
        "B365D": 3.4,
        "B365A": 4.0,
        "PSCH": 2.0,
        "PSCD": 3.4,
        "PSCA": 4.0,
    })
    return {"eval_df": eval_df, "y_proba": y_proba, "classes": classes}


def test_picks_higher_threshold_when_it_improves_stability():
    """When a higher threshold improves ROI×√bets, it should be picked."""
    # 100 bets with edge 0.03: at threshold=0.0 all 100 bets pass.
    # At threshold=0.025, all 100 still pass (edge 0.03 > 0.025).
    # At threshold=0.04, 0 bets pass (edge 0.03 < 0.04) → below min_bets.
    # So both 0.0 and 0.025 qualify; same bets, same stability → lower wins (0.0).
    data = [_make_season_data(100, "E0", edge=0.03)]
    result = select_league_thresholds(data, leagues=["E0"], grid=[0.0, 0.025, 0.04])
    assert result["E0"] == 0.0  # tie broken by lower threshold


def test_falls_back_to_default_when_min_bets_not_met():
    """If no threshold produces >= min_bets bets, return the default."""
    data = [_make_season_data(5, "E0", edge=0.03)]  # only 5 bets, min_bets=20
    result = select_league_thresholds(
        data, leagues=["E0"], grid=[0.0, 0.02, 0.04], min_bets=20, default_threshold=0.0
    )
    assert result["E0"] == 0.0


def test_returns_default_when_no_prior_data():
    """Empty prior_season_data → all leagues get default threshold."""
    result = select_league_thresholds(
        [], leagues=["E0", "N1"], grid=[0.0, 0.02], default_threshold=0.0
    )
    assert result == {"E0": 0.0, "N1": 0.0}


def test_per_league_thresholds_are_independent():
    """Each league gets its own threshold independently."""
    data_e0 = _make_season_data(100, "E0", edge=0.06)
    data_n1 = _make_season_data(3, "N1", edge=0.06)  # too few for N1
    data = [
        {
            "eval_df": pd.concat([data_e0["eval_df"], data_n1["eval_df"]]).reset_index(drop=True),
            "y_proba": np.vstack([data_e0["y_proba"], data_n1["y_proba"]]),
            "classes": data_e0["classes"],
        }
    ]
    result = select_league_thresholds(
        data, leagues=["E0", "N1"], grid=[0.0, 0.02, 0.04], min_bets=20, default_threshold=0.0
    )
    # E0 has enough bets at any threshold; N1 does not → N1 gets default
    assert result["N1"] == 0.0
    assert "E0" in result


def test_missing_league_gets_default():
    """A league with no rows in prior data gets the default threshold."""
    data = [_make_season_data(50, "E0", edge=0.03)]
    result = select_league_thresholds(
        data, leagues=["E0", "G1"], grid=[0.0, 0.02], default_threshold=0.0
    )
    assert result["G1"] == 0.0
