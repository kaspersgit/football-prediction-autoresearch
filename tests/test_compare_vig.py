import pandas as pd

from main import _compare_vig_league_roi


def _bets():
    # Shaped like compute_value_betting_results output: carries its own 'league' col.
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "HomeTeam": ["A", "B", "C", "D", "E"],
        "AwayTeam": ["a", "b", "c", "d", "e"],
        "league": ["E0", "E0", "F1", "N1", "XX"],
        "profit": [1.0, -1.0, 2.0, -1.0, 5.0],
    })


def test_league_roi_reads_league_column_directly():
    out = _compare_vig_league_roi(_bets())
    assert out["England"] == (2, 0.0)          # (1 - 1) / 2 * 100
    assert out["France"] == (1, 200.0)
    assert out["Netherlands"] == (1, -100.0)
    assert "XX" not in out and len(out) == 3   # unknown league code dropped


def test_league_roi_no_keyerror_when_bets_already_have_league():
    # Regression: the old merge against an eval_df that also had 'league' produced
    # league_x/league_y and KeyError'd. Direct column access must not raise.
    bets = _bets()
    _compare_vig_league_roi(bets)  # must not raise


def test_league_roi_empty_or_missing_column():
    assert _compare_vig_league_roi(pd.DataFrame()) == {}
    assert _compare_vig_league_roi(pd.DataFrame({"profit": [1.0]})) == {}
