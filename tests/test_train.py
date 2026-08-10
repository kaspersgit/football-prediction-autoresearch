import pandas as pd

from src.model.train import MIN_FULL_SEASON_ROWS, _full_seasons


def test_full_seasons_excludes_in_progress_season_with_few_rows():
    """A season with only a handful of played fixtures must not count as a
    completed walk-forward test season, or it would silently displace a full
    season from the trailing TEST_SEASONS window."""
    full = ["2223", "2324", "2425"]
    rows = []
    for season in full:
        rows.extend([{"season": season}] * MIN_FULL_SEASON_ROWS)
    rows.extend([{"season": "2526"}] * 11)  # season just started
    df = pd.DataFrame(rows)

    assert _full_seasons(df) == full


def test_full_seasons_sorted_and_includes_seasons_at_the_threshold():
    df = pd.DataFrame(
        [{"season": "2425"}] * MIN_FULL_SEASON_ROWS
        + [{"season": "2324"}] * MIN_FULL_SEASON_ROWS
    )

    assert _full_seasons(df) == ["2324", "2425"]
