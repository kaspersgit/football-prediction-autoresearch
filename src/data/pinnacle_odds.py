"""Live pre-match Pinnacle 1X2 odds for production leagues, via The Odds API."""

import os

import pandas as pd
import requests

from src.data.team_aliases import ODDS_API_TEAM_ALIASES

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
# Covers every src.config.SUPPORTED_LEAGUES code, not just the current production
# allowlist, so widening PRODUCTION_LEAGUES later needs no change here. All 11 keys
# were verified against a live GET /v4/sports?apiKey=... call on 2026-08-09 — re-verify
# if The Odds API renames/retires a sport key.
_LEAGUE_TO_SPORT_KEY = {
    "E0": "soccer_epl",
    "D1": "soccer_germany_bundesliga",
    "SP1": "soccer_spain_la_liga",
    "I1": "soccer_italy_serie_a",
    "F1": "soccer_france_ligue_one",
    "N1": "soccer_netherlands_eredivisie",
    "P1": "soccer_portugal_primeira_liga",
    "G1": "soccer_greece_super_league",
    "SC0": "soccer_spl",
    "B1": "soccer_belgium_first_div",
    "T1": "soccer_turkey_super_league",
}
_ODDS_COLUMNS = ["league", "HomeTeam", "AwayTeam", "Date", "PSH", "PSD", "PSA"]
# How many days apart a fixtures.csv row's Date and a live event's commence_time may be
# and still be treated as the same match. The Odds API and football-data.co.uk are
# fetched independently and are not always showing the same matchweek at any given
# moment (e.g. fixtures.csv already on next weekend's round while the Odds API is
# still pricing the round after); matching on team names alone would silently attach
# one round's Pinnacle odds to a different round's fixture row.
_MAX_DATE_DRIFT_DAYS = 1


def _empty_odds_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_ODDS_COLUMNS)


def _resolve_team_name(league: str, odds_api_name: str) -> str:
    """Map an Odds API team name to its football-data.co.uk name (identity if unmapped)."""
    return ODDS_API_TEAM_ALIASES.get(league, {}).get(odds_api_name, odds_api_name)


def _parse_event(league: str, event: dict) -> dict | None:
    """Return a normalized odds row for one Odds API event, or None if unparseable."""
    pinnacle = next(
        (bk for bk in event.get("bookmakers", []) if bk.get("key") == "pinnacle"), None
    )
    if pinnacle is None:
        return None
    h2h = next((m for m in pinnacle.get("markets", []) if m.get("key") == "h2h"), None)
    if h2h is None:
        return None

    prices = {o.get("name"): o.get("price") for o in h2h.get("outcomes", [])}
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    if home_name not in prices or away_name not in prices or "Draw" not in prices:
        return None

    try:
        psh, psd, psa = float(prices[home_name]), float(prices["Draw"]), float(prices[away_name])
    except (TypeError, ValueError):
        return None

    commence_time = event.get("commence_time")
    try:
        match_date = pd.Timestamp(commence_time).normalize().tz_localize(None)
    except (TypeError, ValueError):
        return None

    return {
        "league": league,
        "HomeTeam": _resolve_team_name(league, home_name),
        "AwayTeam": _resolve_team_name(league, away_name),
        "Date": match_date,
        "PSH": psh,
        "PSD": psd,
        "PSA": psa,
    }


def fetch_pinnacle_odds(leagues: set[str]) -> pd.DataFrame:
    """Fetch live pre-match Pinnacle 1X2 odds for the given production league codes.

    Never raises: a missing API key, a per-league request failure, or an
    unparseable response degrades to fewer (or zero) rows, never breaks the caller.
    """
    api_key = os.environ.get("THEODDS_API")
    if not api_key:
        print("THEODDS_API not set — skipping live Pinnacle odds")
        return _empty_odds_df()

    rows = []
    for league in leagues:
        sport_key = _LEAGUE_TO_SPORT_KEY.get(league)
        if sport_key is None:
            continue
        try:
            response = requests.get(
                f"{_ODDS_API_BASE}/{sport_key}/odds/",
                params={
                    "apiKey": api_key,
                    "regions": "eu",
                    "markets": "h2h",
                    "bookmakers": "pinnacle",
                    "oddsFormat": "decimal",
                },
                timeout=30,
            )
            response.raise_for_status()
            events = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Skipping live Pinnacle odds for {league}: {e}")
            continue

        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining is not None or used is not None:
            print(f"The Odds API quota after {league} call: used={used}, remaining={remaining}")

        for event in events:
            row = _parse_event(league, event)
            if row is None:
                print(
                    f"Unmatched or unparseable {league} event: "
                    f"{event.get('home_team')} v {event.get('away_team')}"
                )
                continue
            rows.append(row)

    if not rows:
        return _empty_odds_df()
    return pd.DataFrame(rows, columns=_ODDS_COLUMNS)


def attach_pinnacle_odds(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge live Pinnacle odds onto fixtures_df, overwriting NaN PSH/PSD/PSA placeholders.

    Matches on (league, HomeTeam, AwayTeam) as the join key, then requires the two
    sources' match dates to agree within _MAX_DATE_DRIFT_DAYS. football-data.co.uk's
    fixtures.csv and The Odds API are fetched independently and are not always
    showing the same round at the same moment; without this check, a team-name-only
    match could silently attach one round's Pinnacle odds to a different round's
    fixture — same teams, wrong match.
    """
    from src.config import PRODUCTION_LEAGUES

    odds = fetch_pinnacle_odds(set(PRODUCTION_LEAGUES))
    if odds.empty:
        return fixtures_df

    merged = fixtures_df.merge(
        odds, on=["league", "HomeTeam", "AwayTeam"], how="left", suffixes=("", "_live")
    )
    date_drift = (merged["Date"] - merged["Date_live"]).abs()
    same_round = date_drift <= pd.Timedelta(days=_MAX_DATE_DRIFT_DAYS)
    mismatched = merged["Date_live"].notna() & ~same_round
    if mismatched.any():
        for _, row in merged[mismatched].iterrows():
            print(
                f"Skipping live Pinnacle odds for {row['league']} {row['HomeTeam']} v "
                f"{row['AwayTeam']}: fixtures.csv has {row['Date'].date()}, "
                f"live odds are for {row['Date_live'].date()} — different round"
            )

    for col in ["PSH", "PSD", "PSA"]:
        live_col = merged[f"{col}_live"].where(same_round)
        merged[col] = live_col.combine_first(merged[col])
        merged = merged.drop(columns=[f"{col}_live"])
    merged = merged.drop(columns=["Date_live"])
    return merged
