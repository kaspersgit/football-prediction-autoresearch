from datetime import date
from pathlib import Path

import requests

from src.config import LEAGUE_NAMES

LEAGUES = {name.lower(): code for code, name in LEAGUE_NAMES.items()}

BASE_URL = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
RAW_DIR = Path("data/raw")


def _current_season_start_year() -> int:
    """Return the year the current football season started (e.g. 2025 for 2025-26)."""
    today = date.today()
    # Season starts in August; before August we're in the season that started last year
    return today.year if today.month >= 8 else today.year - 1


def _all_seasons(from_year: int = 2013) -> list[str]:
    end_year = _current_season_start_year()
    return [f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(from_year, end_year + 1)]


SEASONS = _all_seasons()


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


def download_season(league_code: str, season: str, force: bool = False) -> Path:
    """Download one league-season CSV.

    football-data.co.uk sometimes hasn't published a not-yet-started season's
    file for a given league yet: the URL can 301-redirect to a *different*
    league's file (observed: SP1 2627 -> P1's data), or serve an ambiguous
    "300 Multiple Choices" HTML page saved verbatim as if it were the CSV.
    Both would silently corrupt training data with another league's results
    under the wrong label, so the response's own 'Div' column must match the
    requested league code before it's trusted and written to disk.
    """
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    dest = RAW_DIR / f"{league_code}_{season}.csv"
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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Downloaded {dest.name}")
    return dest


def download_all(force: bool = False) -> list[Path]:
    paths = []
    for country, code in LEAGUES.items():
        for season in SEASONS:
            try:
                p = download_season(code, season, force=force)
                paths.append(p)
            except requests.HTTPError as e:
                print(f"Skipping {code} {season}: {e}")
    return paths


def update_current_season() -> None:
    """Re-download the current (and previous) season to get latest results."""
    for season in SEASONS[-2:]:
        for code in LEAGUES.values():
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
