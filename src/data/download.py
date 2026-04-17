from pathlib import Path

import requests

LEAGUES = {
    "england": "E0",
    "germany": "D1",
    "spain": "SP1",
    "italy": "I1",
}

# Generate season codes from 2013-14 to current
SEASONS = [
    f"{str(y)[2:]}{str(y+1)[2:]}"
    for y in range(2013, 2025)
]  # ['1314', '1415', ..., '2425']

BASE_URL = "https://www.football-data.co.uk/mmz4281"
RAW_DIR = Path("data/raw")


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
    """Re-download the current season to get latest results."""
    current = SEASONS[-1]
    for code in LEAGUES.values():
        try:
            download_season(code, current, force=True)
        except requests.HTTPError as e:
            print(f"Could not update {code} {current}: {e}")


if __name__ == "__main__":
    import sys
    if "--update" in sys.argv:
        update_current_season()
    else:
        download_all()
