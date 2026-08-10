import pandas as pd
import pytest

from src.model.features import build_features


def _make_df():
    rows = []
    # 25 games: team A vs team B alternating home/away -- >= MARKET_MOVEMENT_WINDOW (20)
    # so home_market_movement/away_market_movement aren't dropped by dropna(subset=FEATURE_COLS)
    for i in range(25):
        rows.append({
            "Date": pd.Timestamp("2023-08-01") + pd.Timedelta(days=i),
            "HomeTeam": "Arsenal" if i % 2 == 0 else "Chelsea",
            "AwayTeam": "Chelsea" if i % 2 == 0 else "Arsenal",
            "FTHG": 2, "FTAG": 1, "FTR": "H",
            "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
            "PSH": 2.0, "PSD": 3.5, "PSA": 4.0,
            "PSCH": 1.9, "PSCD": 3.6, "PSCA": 4.2,
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

def test_market_movement_feature_reflects_opening_to_closing_delta():
    df = _make_df()
    X, y = build_features(df)
    assert "home_market_movement" in X.columns
    assert "away_market_movement" in X.columns
    assert X["home_market_movement"].notna().all()
    assert X["away_market_movement"].notna().all()

    def fair(psh, psd, psa, side):
        inv = 1 / psh + 1 / psd + 1 / psa
        return (1 / {"H": psh, "A": psa}[side]) / inv

    # Each team alternates home/away, so its rolling history mixes home-side deltas
    # (when it was HomeTeam) and away-side deltas (when it was AwayTeam) in equal measure.
    home_delta = fair(1.9, 3.6, 4.2, "H") - fair(2.0, 3.5, 4.0, "H")
    away_delta = fair(1.9, 3.6, 4.2, "A") - fair(2.0, 3.5, 4.0, "A")
    expected_movement = (home_delta + away_delta) / 2
    assert X["home_market_movement"].iloc[0] == pytest.approx(expected_movement)
    assert X["away_market_movement"].iloc[0] == pytest.approx(expected_movement)

def test_market_movement_is_nan_safe_without_pinnacle_columns():
    # No PSH/PSD/PSA/PSCH/PSCD/PSCA at all -- should not crash, just skip the feature's
    # contribution (handled upstream via FEATURE_COLS dropna, not asserted here).
    from src.model.features import _compute_market_movement
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3),
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
        "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
    })
    result = _compute_market_movement(df)
    assert result["home_market_movement"].isna().all()
    assert result["away_market_movement"].isna().all()
