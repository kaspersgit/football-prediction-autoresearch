"""Guards data/historical/ against silently falling behind FINISHED_SEASONS.

Skipped entirely until the archive is first populated (see README.md) — this
isn't a bootstrap check, it's a "don't forget to archive the season that just
rolled out of update_current_season()'s window" check.
"""

import pytest

from src.data.download import FINISHED_SEASONS, HISTORICAL_DIR, LEAGUES


def test_historical_archive_has_every_finished_season_for_every_league():
    if not HISTORICAL_DIR.exists():
        pytest.skip(f"{HISTORICAL_DIR} not populated yet — run archive_finished_seasons()")

    missing = [
        f"{league}/{season}.csv"
        for league in LEAGUES.values()
        for season in FINISHED_SEASONS
        if not (HISTORICAL_DIR / league / f"{season}.csv").exists()
    ]

    assert not missing, (
        "data/historical/ is missing finished seasons — run "
        "`python -c \"from src.data.download import archive_finished_seasons as a; a()\"` "
        "and commit the result:\n" + "\n".join(missing)
    )
