import pandas as pd
import numpy as np
import pytest
from src.model.features import build_features

def _make_df():
    rows = []
    # 10 games: team A vs team B alternating home/away
    for i in range(10):
        rows.append({
            "Date": pd.Timestamp(f"2023-08-{i+1:02d}"),
            "HomeTeam": "Arsenal" if i % 2 == 0 else "Chelsea",
            "AwayTeam": "Chelsea" if i % 2 == 0 else "Arsenal",
            "FTHG": 2, "FTAG": 1, "FTR": "H",
            "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
            "league": "E0", "season": "2324",
        })
    return pd.DataFrame(rows)

def test_build_features_returns_dataframe():
    df = _make_df()
    X, y = build_features(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_no_data_leakage():
    # Features must use only past data, so first few rows per team may be NaN-dropped
    df = _make_df()
    X, y = build_features(df)
    # We should have fewer rows than input (early rows dropped — no history yet)
    assert len(X) < len(df)

def test_feature_columns_present():
    df = _make_df()
    X, y = build_features(df)
    expected = [
        "home_form_pts", "home_form_gf", "home_form_ga",
        "away_form_pts", "away_form_gf", "away_form_ga",
    ]
    for col in expected:
        assert col in X.columns, f"Missing column: {col}"

def test_target_values():
    df = _make_df()
    X, y = build_features(df)
    assert set(y.unique()).issubset({"H", "D", "A"})
