from collections.abc import Iterable
from datetime import date
from pathlib import Path

import requests

from src.config import LEAGUE_NAMES

LEAGUES = {name.lower(): code for code, name in LEAGUE_NAMES.items()}

BASE_URL = "https://football-data.co.uk/mmz4281"
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
RAW_DIR = Path("data/raw")
# Finished-season archive, committed to the repo (data/historical/<league>/<season>.csv)
# so it survives without re-downloading. The current and previous season still change
# (results land, corrections happen) so they stay in RAW_DIR, refreshed at runtime by
# update_current_season() instead of being archived here.
HISTORICAL_DIR = Path("data/historical")


def _current_season_start_year() -> int:
    """Return the year the current football season started (e.g. 2025 for 2025-26)."""
    today = date.today()
    # Season starts in August; before August we're in the season that started last year
    return today.year if today.month >= 8 else today.year - 1


def _all_seasons(from_year: int = 2013) -> list[str]:
    end_year = _current_season_start_year()
    return [f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(from_year, end_year + 1)]


SEASONS = _all_seasons()
# The current and previous season may still change (results land, corrections
# happen) so update_current_season() keeps re-fetching those two into RAW_DIR.
# Everything older than that is done for good and belongs in the committed
# data/historical/ archive instead of being re-downloaded on every cache miss.
FINISHED_SEASONS = SEASONS[:-2]


def _first_data_row_div(content: bytes) -> str | None:
    """Return the 'Div' value of the first data row of a football-data.co.uk CSV,
    or None if the content isn't a parseable CSV with that column at all."""
    lines = content.decode("utf-8-sig", errors="replace").splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    if not header or header[0] != "Div":
        return None
    first_row = lines[1].split(",")
    return first_row[0] if first_row else None


def download_season(
    league_code: str,
    season: str,
    force: bool = False,
    dest_dir: Path | None = None,
    dest_name: str | None = None,
) -> Path:
    """Download one league-season CSV.

    football-data.co.uk sometimes hasn't published a not-yet-started season's
    file for a given league yet: the URL can 301-redirect to a *different*
    league's file (observed: SP1 2627 -> P1's data), or serve an ambiguous
    "300 Multiple Choices" HTML page saved verbatim as if it were the CSV.
    Both would silently corrupt training data with another league's results
    under the wrong label, so the response's own 'Div' column must match the
    requested league code before it's trusted and written to disk.
    """
    dest_dir = dest_dir if dest_dir is not None else RAW_DIR
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    dest = dest_dir / (dest_name or f"{league_code}_{season}.csv")
    if dest.exists() and not force:
        return dest
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    div = _first_data_row_div(response.content)
    if div != league_code:
        raise requests.HTTPError(
            f"{league_code} {season} not yet available "
            f"(got Div={div!r} instead of {league_code!r}, likely not published yet)"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Downloaded {dest.name}")
    return dest


def download_all(leagues: Iterable[str] | None = None, force: bool = False) -> list[Path]:
    """Download every season for `leagues` (default: all tracked leagues)."""
    codes = list(leagues) if leagues is not None else list(LEAGUES.values())
    paths = []
    for code in codes:
        for season in SEASONS:
            try:
                p = download_season(code, season, force=force)
                paths.append(p)
            except requests.HTTPError as e:
                print(f"Skipping {code} {season}: {e}")
    return paths


def archive_finished_seasons(leagues: Iterable[str] | None = None) -> list[Path]:
    """Download every finished season straight into the committed
    data/historical/<league>/<season>.csv archive.

    Idempotent: already-archived files are left untouched (force=False), so this
    is safe to re-run after a season boundary shifts FINISHED_SEASONS forward —
    it will only fetch the newly-finished season, not re-fetch everything.
    """
    codes = list(leagues) if leagues is not None else list(LEAGUES.values())
    paths = []
    for code in codes:
        league_dir = HISTORICAL_DIR / code
        for season in FINISHED_SEASONS:
            try:
                p = download_season(code, season, dest_dir=league_dir, dest_name=f"{season}.csv")
                paths.append(p)
            except requests.HTTPError as e:
                print(f"Skipping {code} {season}: {e}")
    return paths


def update_current_season(leagues: Iterable[str] | None = None) -> None:
    """Re-download the current (and previous) season for `leagues` (default: all) to
    get latest results."""
    codes = list(leagues) if leagues is not None else list(LEAGUES.values())
    for season in SEASONS[-2:]:
        for code in codes:
            try:
                download_season(code, season, force=True)
            except requests.HTTPError as e:
                print(f"Could not update {code} {season}: {e}")


def download_fixtures() -> Path:
    """Download upcoming fixture list (next ~2 weeks) with bookmaker odds."""
    dest = RAW_DIR / "fixtures.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Fixtures downloaded to {dest}")
    return dest


if __name__ == "__main__":
    import sys
    if "--update" in sys.argv:
        update_current_season()
    else:
        download_all()
