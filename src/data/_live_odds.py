"""Shared The Odds API plumbing for one bookmaker's live pre-match 1X2 odds.

Not a public entry point — src.data.pinnacle_odds and src.data.bet365_odds are the
per-bookmaker modules that wrap this with their own bookmaker key, odds-column
prefix, and region.
"""

import os
from collections.abc import Callable

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
# How many days apart a fixtures.csv row's Date and a live event's commence_time may be
# and still be treated as the same match. The Odds API and football-data.co.uk are
# fetched independently and are not always showing the same matchweek at any given
# moment (e.g. fixtures.csv already on next weekend's round while the Odds API is
# still pricing the round after); matching on team names alone would silently attach
# one round's odds to a different round's fixture — same teams, wrong match.
_MAX_DATE_DRIFT_DAYS = 1


def _odds_columns(col_prefix: str) -> list[str]:
    return ["league", "HomeTeam", "AwayTeam", "Date", f"{col_prefix}H", f"{col_prefix}D", f"{col_prefix}A"]


def _empty_odds_df(col_prefix: str) -> pd.DataFrame:
    return pd.DataFrame(columns=_odds_columns(col_prefix))


def _resolve_team_name(league: str, odds_api_name: str) -> str:
    """Map an Odds API team name to its football-data.co.uk name (identity if unmapped)."""
    return ODDS_API_TEAM_ALIASES.get(league, {}).get(odds_api_name, odds_api_name)


def _parse_event(
    league: str, event: dict, bookmaker_key: str, col_prefix: str, label: str
) -> tuple[dict | None, str | None]:
    """Return (normalized odds row, None) or (None, reason) if the event yields no row.

    The most common reason by far is "no <label> odds yet" — the bookmaker simply
    hasn't priced a fixture further out on the calendar — which is expected and not a
    data problem. Genuine parse failures (malformed prices/dates) are reported
    separately so the two don't get confused in the logs.
    """
    bookmaker = next(
        (bk for bk in event.get("bookmakers", []) if bk.get("key") == bookmaker_key), None
    )
    if bookmaker is None:
        return None, f"no {label} odds yet"
    h2h = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
    if h2h is None:
        return None, f"no {label} odds yet"

    prices = {o.get("name"): o.get("price") for o in h2h.get("outcomes", [])}
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    if home_name not in prices or away_name not in prices or "Draw" not in prices:
        return None, f"no {label} odds yet"

    try:
        home_odds = float(prices[home_name])
        draw_odds = float(prices["Draw"])
        away_odds = float(prices[away_name])
    except (TypeError, ValueError):
        return None, f"unparseable {label} price"

    commence_time = event.get("commence_time")
    try:
        match_date = pd.Timestamp(commence_time).normalize().tz_localize(None)
    except (TypeError, ValueError):
        return None, "unparseable commence_time"

    return {
        "league": league,
        "HomeTeam": _resolve_team_name(league, home_name),
        "AwayTeam": _resolve_team_name(league, away_name),
        "Date": match_date,
        f"{col_prefix}H": home_odds,
        f"{col_prefix}D": draw_odds,
        f"{col_prefix}A": away_odds,
    }, None


def fetch_bookmaker_odds(
    leagues: set[str], bookmaker_key: str, col_prefix: str, label: str, regions: str
) -> pd.DataFrame:
    """Fetch live pre-match 1X2 odds for `bookmaker_key` across the given league codes.

    Never raises: a missing API key, a per-league request failure, or an
    unparseable response degrades to fewer (or zero) rows, never breaks the caller.
    """
    api_key = os.environ.get("THEODDS_API")
    if not api_key:
        print(f"THEODDS_API not set — skipping live {label} odds")
        return _empty_odds_df(col_prefix)

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
                    "regions": regions,
                    "markets": "h2h",
                    "bookmakers": bookmaker_key,
                    "oddsFormat": "decimal",
                },
                timeout=30,
            )
            response.raise_for_status()
            events = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Skipping live {label} odds for {league}: {e}")
            continue

        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining is not None or used is not None:
            print(f"The Odds API quota after {league} call: used={used}, remaining={remaining}")

        skipped_no_odds_yet = 0
        for event in events:
            row, reason = _parse_event(league, event, bookmaker_key, col_prefix, label)
            if row is None:
                if reason == f"no {label} odds yet":
                    skipped_no_odds_yet += 1
                else:
                    print(
                        f"Skipping {league} event ({reason}): "
                        f"{event.get('home_team')} v {event.get('away_team')}"
                    )
                continue
            rows.append(row)
        if skipped_no_odds_yet:
            print(
                f"{league}: {skipped_no_odds_yet} event(s) not yet priced by "
                f"{label} — skipped (expected for fixtures further out)"
            )

    if not rows:
        return _empty_odds_df(col_prefix)
    return pd.DataFrame(rows, columns=_odds_columns(col_prefix))


def attach_bookmaker_odds(
    fixtures_df: pd.DataFrame,
    fetch_fn: Callable[[set[str]], pd.DataFrame],
    leagues: set[str],
    col_prefix: str,
    label: str,
) -> pd.DataFrame:
    """Left-merge live odds onto fixtures_df, overwriting NaN <col_prefix>{H,D,A} placeholders.

    Matches on (league, HomeTeam, AwayTeam) as the join key, then requires the two
    sources' match dates to agree within _MAX_DATE_DRIFT_DAYS. football-data.co.uk's
    fixtures.csv and The Odds API are fetched independently and are not always
    showing the same round at the same moment; without this check, a team-name-only
    match could silently attach one round's odds to a different round's fixture —
    same teams, wrong match.
    """
    odds = fetch_fn(leagues)
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
                f"Skipping live {label} odds for {row['league']} {row['HomeTeam']} v "
                f"{row['AwayTeam']}: fixtures.csv has {row['Date'].date()}, "
                f"live odds are for {row['Date_live'].date()} — different round"
            )

    for suffix in ("H", "D", "A"):
        col = f"{col_prefix}{suffix}"
        live_col = merged[f"{col}_live"].where(same_round)
        merged[col] = live_col.combine_first(merged[col])
        merged = merged.drop(columns=[f"{col}_live"])
    merged = merged.drop(columns=["Date_live"])
    return merged
