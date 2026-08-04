import pandas as pd

from src.evaluation.uncertainty import weekly_roi_interval


def test_weekly_roi_interval_requires_two_weeks():
    bets = pd.DataFrame({"Date": ["2026-08-04"], "profit": [1.0], "stake": [1.0]})
    assert weekly_roi_interval(bets) is None


def test_weekly_roi_interval_is_deterministic_and_contains_observed_roi():
    bets = pd.DataFrame(
        {
            "Date": ["2026-08-04", "2026-08-05", "2026-08-11", "2026-08-12"],
            "profit": [1.0, -1.0, 2.0, -1.0],
            "stake": [1.0, 1.0, 1.0, 1.0],
        }
    )
    first = weekly_roi_interval(bets, n_resamples=2_000, seed=7)
    second = weekly_roi_interval(bets, n_resamples=2_000, seed=7)
    assert first == second
    assert first[0] <= 25.0 <= first[1]


def test_weekly_roi_interval_defaults_to_one_unit_stake():
    """Requiring a stake column would reject valid flat-stake report inputs."""
    bets = pd.DataFrame(
        {"Date": ["2026-08-04", "2026-08-11"], "profit": [1.0, -1.0]}
    )

    assert weekly_roi_interval(bets, n_resamples=2_000, seed=7) == (-100.0, 100.0)


def test_weekly_roi_interval_keeps_same_week_number_in_different_iso_years_separate():
    """Grouping on week number alone would collapse two independent yearly blocks."""
    bets = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2027-01-08"],
            "profit": [1.0, -1.0],
            "stake": [1.0, 1.0],
        }
    )

    assert weekly_roi_interval(bets, n_resamples=2_000, seed=7) == (-100.0, 100.0)
