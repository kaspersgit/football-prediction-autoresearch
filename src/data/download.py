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


def download_season(league_code: str, season: str, force: bool = False) -> Path:
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    dest = RAW_DIR / f"{league_code}_{season}.csv"
    if dest.exists() and not force:
        return dest
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
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
