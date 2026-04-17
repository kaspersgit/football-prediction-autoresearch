import pandas as pd
import numpy as np
from src.evaluation.metrics import compute_betting_results, compute_roi, compute_stability

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
