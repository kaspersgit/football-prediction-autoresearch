#!/usr/bin/env python3
"""
Probe script for api-sports.io (API-Football).

Answers three questions before we commit to full integration:
  1. Historical season coverage — do they have lineups for 2023/24 and 2024/25?
  2. Team name matching — how well do their names align with football-data.co.uk?
  3. Rate-limit headroom — how many requests remain on the free tier?

Usage:
  export API_FOOTBALL_KEY=your_key_here
  uv run python scripts/probe_apifootball.py

Sign up free at: https://dashboard.api-football.com/register
"""
import json
import os
import sys
import time
from difflib import get_close_matches

import requests

BASE_URL = "https://v3.football.api-sports.io"

# API-Football league IDs for our 4 leagues
LEAGUES = {
    "E0": {"api_id": 39,  "name": "Premier League"},
    "D1": {"api_id": 78,  "name": "Bundesliga"},
    "SP1": {"api_id": 140, "name": "La Liga"},
    "I1":  {"api_id": 135, "name": "Serie A"},
}

# football-data.co.uk season code → API-Football season year (season that starts that year)
SEASON_MAP = {
    "2324": 2023,
    "2425": 2024,
}

# Our team names per league from football-data.co.uk (2023/24 + 2024/25)
OUR_TEAMS = {
    "E0": [
        "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Burnley",
        "Chelsea","Crystal Palace","Everton","Fulham","Ipswich","Leicester",
        "Liverpool","Luton","Man City","Man United","Newcastle","Nott'm Forest",
        "Sheffield United","Southampton","Tottenham","West Ham","Wolves",
    ],
    "D1": [
        "Augsburg","Bayern Munich","Bochum","Darmstadt","Dortmund","Ein Frankfurt",
        "FC Koln","Freiburg","Heidenheim","Hoffenheim","Holstein Kiel","Leverkusen",
        "M'gladbach","Mainz","RB Leipzig","St Pauli","Stuttgart","Union Berlin",
        "Werder Bremen","Wolfsburg",
    ],
    "SP1": [
        "Alaves","Almeria","Ath Bilbao","Ath Madrid","Barcelona","Betis","Cadiz",
        "Celta","Espanol","Getafe","Girona","Granada","Las Palmas","Leganes",
        "Mallorca","Osasuna","Real Madrid","Sevilla","Sociedad","Valencia",
        "Valladolid","Vallecano","Villarreal",
    ],
    "I1": [
        "Atalanta","Bologna","Cagliari","Como","Empoli","Fiorentina","Frosinone",
        "Genoa","Inter","Juventus","Lazio","Lecce","Milan","Monza","Napoli",
        "Parma","Roma","Salernitana","Sassuolo","Torino","Udinese","Venezia","Verona",
    ],
}


_REQUEST_COUNT = 0

def get(session: requests.Session, path: str, params: dict = None) -> requests.Response:
    global _REQUEST_COUNT
    if _REQUEST_COUNT > 0:
        time.sleep(7)  # free tier: 10 req/min → 6s between requests, use 7 to be safe
    url = f"{BASE_URL}{path}"
    resp = session.get(url, params=params or {}, timeout=15)
    _REQUEST_COUNT += 1
    resp.raise_for_status()
    return resp


def print_rate_limit(resp: requests.Response) -> dict:
    remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
    limit      = resp.headers.get("x-ratelimit-requests-limit", "?")
    print(f"  Rate limit: {remaining} / {limit} requests remaining today")
    return {"remaining": remaining, "limit": limit}


def probe_connectivity(session: requests.Session) -> bool:
    print("\n── 1. CONNECTIVITY ─────────────────────────────────")
    try:
        resp = get(session, "/status")
        data = resp.json()
        print(f"  Status: {resp.status_code}")
        print_rate_limit(resp)
        account = data.get("response", {}).get("account", {})
        sub = data.get("response", {}).get("subscription", {})
        print(f"  Account: {account.get('firstname')} {account.get('lastname')} <{account.get('email')}>")
        print(f"  Plan: {sub.get('plan')}  |  Active: {sub.get('active')}")
        requests_info = data.get("response", {}).get("requests", {})
        print(f"  Requests today: {requests_info.get('current')} / {requests_info.get('limit_day')}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def probe_season_coverage(session: requests.Session) -> dict:
    """Check how many fixtures exist per league/season and whether lineup data is present."""
    print("\n── 2. SEASON COVERAGE ──────────────────────────────")
    coverage = {}

    for league_code, league_info in LEAGUES.items():
        api_id = league_info["api_id"]
        print(f"\n  {league_code} — {league_info['name']} (API id {api_id})")
        league_lineup_checked = False

        for our_season, api_season in SEASON_MAP.items():
            resp = get(session, "/fixtures", {"league": api_id, "season": api_season})
            data = resp.json()
            fixtures = data.get("response", [])
            n = len(fixtures)
            print(f"    Season {our_season} ({api_season}): {n} fixtures", end="")

            if fixtures and not league_lineup_checked:
                # Check lineup for one completed fixture per league (saves requests)
                sample = next(
                    (f for f in fixtures if f["fixture"]["status"]["short"] == "FT"),
                    fixtures[0],
                )
                fix_id = sample["fixture"]["id"]
                fix_date = sample["fixture"]["date"][:10]
                home = sample["teams"]["home"]["name"]
                away = sample["teams"]["away"]["name"]

                lineup_resp = get(session, "/fixtures/lineups", {"fixture": fix_id})
                lineups = lineup_resp.json().get("response", [])
                has_lineup = len(lineups) == 2 and all(
                    len(t.get("startXI", [])) == 11 for t in lineups
                )
                print(f"  →  sample {fix_id} ({fix_date}): {home} vs {away}")
                print(f"       Lineup: {'✅ full 11v11' if has_lineup else '⚠️  incomplete/missing'}")
                league_lineup_checked = True
                coverage[(league_code, our_season)] = {
                    "n_fixtures": n,
                    "has_lineup_sample": has_lineup,
                    "sample_fixture_id": fix_id,
                }
            else:
                print()
                coverage[(league_code, our_season)] = {
                    "n_fixtures": n,
                    "has_lineup_sample": league_lineup_checked,
                    "sample_fixture_id": None,
                }

    print_rate_limit(resp)
    return coverage


def probe_team_names(session: requests.Session) -> dict:
    """Fetch team names from API and compare with our football-data.co.uk names."""
    print("\n── 3. TEAM NAME MATCHING ───────────────────────────")
    mapping = {}

    for league_code, league_info in LEAGUES.items():
        api_id = league_info["api_id"]
        resp = get(session, "/teams", {"league": api_id, "season": 2024})
        data = resp.json()
        api_teams = [t["team"]["name"] for t in data.get("response", [])]
        api_ids   = {t["team"]["name"]: t["team"]["id"] for t in data.get("response", [])}

        our_teams = OUR_TEAMS[league_code]
        exact, fuzzy, missing = [], [], []

        print(f"\n  {league_code} — {league_info['name']}")
        print(f"  {'Our name':<25} {'API match':<30} {'API id':>8}  {'type'}")
        print(f"  {'-'*25} {'-'*30} {'-'*8}  {'-'*6}")

        league_map = {}
        for our_name in sorted(our_teams):
            if our_name in api_ids:
                print(f"  {our_name:<25} {our_name:<30} {api_ids[our_name]:>8}  exact")
                exact.append(our_name)
                league_map[our_name] = api_ids[our_name]
            else:
                close = get_close_matches(our_name, api_teams, n=1, cutoff=0.5)
                if close:
                    match = close[0]
                    print(f"  {our_name:<25} {match:<30} {api_ids[match]:>8}  fuzzy")
                    fuzzy.append(our_name)
                    league_map[our_name] = api_ids[match]
                else:
                    print(f"  {our_name:<25} {'— NO MATCH —':<30} {'':>8}  ⚠️ missing")
                    missing.append(our_name)

        print(f"\n  Summary: {len(exact)} exact, {len(fuzzy)} fuzzy, {len(missing)} unmatched")
        mapping[league_code] = {
            "our_to_api_id": league_map,
            "exact": exact,
            "fuzzy": fuzzy,
            "missing": missing,
            "api_teams": api_teams,
        }
        time.sleep(0.25)

    print_rate_limit(resp)
    return mapping


def probe_injury_endpoint(session: requests.Session) -> None:
    """Check if the injury/suspension endpoint is available on the free tier."""
    print("\n── 4. INJURY ENDPOINT ──────────────────────────────")
    # Use Premier League 2024 as test
    resp = get(session, "/injuries", {"league": 39, "season": 2024})
    data = resp.json()
    errors = data.get("errors", {})
    results = data.get("response", [])
    if errors:
        print(f"  ⚠️  Errors: {errors}")
    else:
        print(f"  ✅ Injury endpoint accessible — {len(results)} injury records returned")
        if results:
            sample = results[0]
            print(f"  Sample: {sample['player']['name']} ({sample['team']['name']}) — {sample['player']['reason']}")
    print_rate_limit(resp)


def main():
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        print("ERROR: Set API_FOOTBALL_KEY environment variable.")
        print("  Get a free key at: https://dashboard.api-football.com/register")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "x-apisports-key": key,
        "Accept": "application/json",
    })

    print("=" * 60)
    print("API-FOOTBALL PROBE")
    print("=" * 60)

    ok = probe_connectivity(session)
    if not ok:
        sys.exit(1)

    coverage = probe_season_coverage(session)
    mapping  = probe_team_names(session)
    probe_injury_endpoint(session)

    # ── Summary ──────────────────────────────────────────────
    print("\n── SUMMARY ─────────────────────────────────────────")

    print("\nCoverage:")
    all_have_lineups = True
    for (league, season), info in sorted(coverage.items()):
        has = "✅" if info["has_lineup_sample"] else "⚠️ "
        print(f"  {league} {season}: {info['n_fixtures']} fixtures  lineup={has}")
        if not info["has_lineup_sample"]:
            all_have_lineups = False

    total_exact = sum(len(m["exact"])   for m in mapping.values())
    total_fuzzy = sum(len(m["fuzzy"])   for m in mapping.values())
    total_miss  = sum(len(m["missing"]) for m in mapping.values())
    total       = total_exact + total_fuzzy + total_miss
    print(f"\nName matching: {total_exact}/{total} exact, {total_fuzzy} fuzzy, {total_miss} unmatched")

    if total_miss > 0:
        print("\nTeams needing manual mapping:")
        for lc, m in mapping.items():
            for name in m["missing"]:
                candidates = get_close_matches(name, m["api_teams"], n=3, cutoff=0.3)
                print(f"  {lc} | {name!r:25s} — candidates: {candidates}")

    print("\nVerdict:")
    print(f"  Historical lineup data: {'✅ available' if all_have_lineups else '⚠️  partial'}")
    print(f"  Team name auto-mapping: {total_exact + total_fuzzy}/{total} ({(total_exact+total_fuzzy)/total:.0%})")
    print(f"  Manual mappings needed: {total_miss} teams")
    print()


if __name__ == "__main__":
    main()
