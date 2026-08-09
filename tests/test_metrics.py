import numpy as np
import pandas as pd
import pytest

from src.config import EXCLUDED_BETTING_LEAGUES
from src.evaluation.metrics import (
    compute_betting_results,
    compute_roi,
    compute_stability,
    compute_value_betting_results,
)


def _make_results():
    return pd.DataFrame({
        "y_true": ["H", "D", "A", "H", "H"],
        "y_pred": ["H", "H", "A", "A", "H"],
        "B365H": [2.0, 3.5, 4.0, 2.0, 1.8],
        "B365D": [3.5, 3.0, 3.5, 3.5, 3.5],
        "B365A": [4.0, 2.1, 2.0, 4.0, 5.0],
        "Date": pd.date_range("2024-01-01", periods=5),
    })

def test_compute_betting_results_columns():
    df = _make_results()
    res = compute_betting_results(df)
    assert "profit" in res.columns
    assert "cumulative_profit" in res.columns
    assert len(res) == 5

def test_correct_prediction_gives_positive_profit():
    df = _make_results()
    res = compute_betting_results(df)
    # First row: pred=H, true=H, odds=2.0 → profit = 2.0 - 1 = 1.0
    assert abs(res.iloc[0]["profit"] - 1.0) < 1e-6

def test_wrong_prediction_gives_minus_one():
    df = _make_results()
    res = compute_betting_results(df)
    # Second row: pred=H, true=D → profit = -1
    assert abs(res.iloc[1]["profit"] - (-1.0)) < 1e-6

def test_roi_calculation():
    df = _make_results()
    res = compute_betting_results(df)
    roi = compute_roi(res)
    total_staked = 5.0
    total_profit = res["profit"].sum()
    assert abs(roi - (total_profit / total_staked * 100)) < 1e-6

def test_stability_is_scalar():
    df = _make_results()
    res = compute_betting_results(df)
    stab = compute_stability(res)
    assert isinstance(stab, float)


def test_value_bets_keep_all_markets_unless_production_filter_is_requested():
    matches = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "HomeTeam": ["Home E0", "Home D1"],
            "AwayTeam": ["Away E0", "Away D1"],
            "league": ["E0", "D1"],
            "y_true": ["H", "H"],
            "B365H": [2.0, 2.0],
            "B365D": [4.0, 4.0],
            "B365A": [4.0, 4.0],
        }
    )
    probabilities = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    evaluation = compute_value_betting_results(matches, probabilities, classes)
    production = compute_value_betting_results(
        matches,
        probabilities,
        classes,
        skip_leagues=EXCLUDED_BETTING_LEAGUES,
    )

    assert evaluation["league"].tolist() == ["E0", "D1"]
    assert production["league"].tolist() == ["E0"]
    assert evaluation["edge"].tolist() == pytest.approx([0.2, 0.2])


def _make_pinnacle_row(psch, pscd, psca):
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=1),
        "HomeTeam": ["Home"],
        "AwayTeam": ["Away"],
        "league": ["E0"],
        "y_true": ["H"],
        "B365H": [2.0],
        "B365D": [4.0],
        "B365A": [4.0],
        "PSCH": [psch],
        "PSCD": [pscd],
        "PSCA": [psca],
    })


def test_pinnacle_confirmation_filter_keeps_bet_when_pinnacle_agrees():
    matches = _make_pinnacle_row(1.8, 6.0, 6.0)  # pinnacle_fair[H] = 0.625 > 0.5 + 0.015
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(
        matches, probabilities, classes, pinnacle_confirmation_margin=0.015,
    )

    assert result["y_pred"].tolist() == ["H"]


def test_pinnacle_confirmation_filter_vetoes_bet_when_pinnacle_disagrees():
    matches = _make_pinnacle_row(3.0, 3.0, 3.0)  # pinnacle_fair[H] = 0.333 <= 0.5 + 0.015
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(
        matches, probabilities, classes, pinnacle_confirmation_margin=0.015,
    )

    assert result.empty


def test_pinnacle_confirmation_filter_skipped_when_pinnacle_odds_are_null():
    matches = _make_pinnacle_row(float("nan"), float("nan"), float("nan"))
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(
        matches, probabilities, classes, pinnacle_confirmation_margin=0.015,
    )

    assert result["y_pred"].tolist() == ["H"]


def test_pinnacle_confirmation_filter_is_off_by_default():
    matches = _make_pinnacle_row(3.0, 3.0, 3.0)  # would veto if the filter were active
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(matches, probabilities, classes)

    assert result["y_pred"].tolist() == ["H"]
