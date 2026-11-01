"""OddsPapi (oddspapi.io) live pre-match 1X2 odds — the primary provider.

Not a public entry point — src.data._live_odds.fetch_bookmaker_odds tries this first
and only falls back to The Odds API (the legacy provider, still in this same package)
if OddsPapi is unavailable (no key, or the call itself failed). Which leagues to ask
for is entirely the caller's decision (unchanged); this module only knows how to turn
a set of our league codes into OddsPapi tournament IDs and parse what comes back.

Account/plan details (2026-11-01, free plan, 250 requests/month):
- 1 request = 1 credit, flat, regardless of response size or how many
  bookmakers/markets/tournaments a call returns data for.
- `tournamentIds` on /v4/odds-by-tournaments accepts at most 3 IDs per call, and
  `bookmaker` takes exactly one value — so N leagues costs ceil(N/3) calls per
  bookmaker, not N.
- A failed call (rate-limited, timed out, whatever) still consumes a credit in
  practice — this module paces calls (_REQUEST_PACING_SECONDS) to stay under the
  observed ~0.5-0.7s cooldown rather than retry into more charged failures.
- /v4/participants (team-ID -> name) has no per-ID filter — it returns every
  participant for the sport in one call. Fetched at most once per process and
  cached at module level; the modest extra credit is worth not re-deriving names.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from src.data.oddspapi_team_aliases import ODDSPAPI_TEAM_ALIASES

_BASE = "https://api.oddspapi.io/v4"
_SPORT_ID_SOCCER = 10
_MARKET_ID_1X2 = 101
_OUTCOME_HOME, _OUTCOME_DRAW, _OUTCOME_AWAY = "101", "102", "103"
_MAX_TOURNAMENT_IDS_PER_CALL = 3
_REQUEST_PACING_SECONDS = 1.5
_STATUS_NOT_STARTED = 0

# Verified against a live GET /v4/tournaments?sportId=10 call on 2026-11-01. Re-verify
# if OddsPapi restructures a league's tournament (relegation/rebrand/etc. could, in
# principle, change the ID; in practice these have been stable).
_LEAGUE_TO_TOURNAMENT_ID = {
    "E0": 17,
    "D1": 35,
    "SP1": 8,
    "I1": 23,
    "F1": 34,
    "N1": 37,
    "P1": 238,
    "G1": 185,
    "SC0": 36,
    "B1": 38,
    "T1": 52,
}


class OddsPapiUnavailable(Exception):
    """Raised whenever OddsPapi itself couldn't be used — not when it validly
    returned zero rows for a league/bookmaker (that's a real result, not a failure).
    The caller (src.data._live_odds.fetch_bookmaker_odds) catches this and falls back
    to the legacy The Odds API provider for the whole request.
    """


def _api_key() -> str:
    api_key = os.environ.get("ODDSPAPI_API_KEY")
    if not api_key:
        raise OddsPapiUnavailable("ODDSPAPI_API_KEY not set")
    return api_key


_last_request_at: float | None = None


def _pace_request() -> None:
    """Sleep as needed so this call starts at least _REQUEST_PACING_SECONDS after the
    previous one — across calls, not just within one fetch_oddspapi_bookmaker_odds,
    since e.g. a Pinnacle fetch immediately followed by a Bet365 fetch hit this same
    cooldown in practice (verified live, 2026-11-01)."""
    global _last_request_at
    if _last_request_at is not None:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _REQUEST_PACING_SECONDS:
            time.sleep(_REQUEST_PACING_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def _get(path: str, params: dict) -> dict | list:
    _pace_request()
    try:
        response = requests.get(f"{_BASE}/{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        raise OddsPapiUnavailable(f"{path} call failed: {e}") from e


_participants_cache: dict[int, str] | None = None


def _get_participants(api_key: str) -> dict[int, str]:
    """Team ID -> name, for every soccer participant. Fetched once per process."""
    global _participants_cache
    if _participants_cache is None:
        raw = _get("participants", {"sportId": _SPORT_ID_SOCCER, "apiKey": api_key})
        _participants_cache = {int(k): v for k, v in raw.items()}
    return _participants_cache


def _resolve_team_name(league: str, oddspapi_name: str) -> str:
    """Map an OddsPapi participant name to its football-data.co.uk name (identity if
    unmapped)."""
    return ODDSPAPI_TEAM_ALIASES.get(league, {}).get(oddspapi_name, oddspapi_name)


def _batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_fixture(
    league: str, fixture: dict, bookmaker_key: str, col_prefix: str, participants: dict[int, str]
) -> dict | None:
    if fixture.get("statusId") != _STATUS_NOT_STARTED:
        return None
    market = (
        fixture.get("bookmakerOdds", {}).get(bookmaker_key, {}).get("markets", {}).get(str(_MARKET_ID_1X2))
    )
    if not market:
        return None
    outcomes = market.get("outcomes", {})
    try:
        home_odds = outcomes[_OUTCOME_HOME]["players"]["0"]["price"]
        draw_odds = outcomes[_OUTCOME_DRAW]["players"]["0"]["price"]
        away_odds = outcomes[_OUTCOME_AWAY]["players"]["0"]["price"]
    except (KeyError, TypeError):
        return None
    try:
        home_odds, draw_odds, away_odds = float(home_odds), float(draw_odds), float(away_odds)
    except (TypeError, ValueError):
        return None

    home_name = participants.get(fixture.get("participant1Id"))
    away_name = participants.get(fixture.get("participant2Id"))
    if not home_name or not away_name:
        return None

    try:
        match_date = pd.Timestamp(fixture.get("startTime")).normalize().tz_localize(None)
    except (TypeError, ValueError):
        return None

    return {
        "league": league,
        "HomeTeam": _resolve_team_name(league, home_name),
        "AwayTeam": _resolve_team_name(league, away_name),
        "Date": match_date,
        f"{col_prefix}H": home_odds,
        f"{col_prefix}D": draw_odds,
        f"{col_prefix}A": away_odds,
    }


def fetch_oddspapi_bookmaker_odds(
    leagues: set[str], bookmaker_key: str, col_prefix: str, label: str
) -> pd.DataFrame:
    """Fetch live pre-match 1X2 odds for `bookmaker_key` across the given league codes.

    Raises OddsPapiUnavailable (never any other exception) if the provider can't be
    used at all for this request — missing key, auth failure, network error. A
    league/bookmaker combination that's simply not covered (e.g. Bet365 has no
    Belgian Pro League fixtures) is a valid empty result, not a failure, and does not
    raise.
    """
    api_key = _api_key()
    participants = _get_participants(api_key)

    tournament_ids = [
        _LEAGUE_TO_TOURNAMENT_ID[lg] for lg in leagues if lg in _LEAGUE_TO_TOURNAMENT_ID
    ]
    tid_to_league = {tid: lg for lg, tid in _LEAGUE_TO_TOURNAMENT_ID.items() if tid in tournament_ids}

    rows = []
    for batch in _batched(tournament_ids, _MAX_TOURNAMENT_IDS_PER_CALL):
        fixtures = _get(
            "odds-by-tournaments",
            {
                "bookmaker": bookmaker_key,
                "tournamentIds": ",".join(str(t) for t in batch),
                "oddsFormat": "decimal",
                "apiKey": api_key,
            },
        )
        if isinstance(fixtures, dict):  # an error payload that still returned HTTP 200
            raise OddsPapiUnavailable(f"odds-by-tournaments returned an error: {fixtures}")
        for fixture in fixtures:
            league = tid_to_league.get(fixture.get("tournamentId"))
            if league is None:
                continue
            row = _parse_fixture(league, fixture, bookmaker_key, col_prefix, participants)
            if row is not None:
                rows.append(row)

    columns = ["league", "HomeTeam", "AwayTeam", "Date", f"{col_prefix}H", f"{col_prefix}D", f"{col_prefix}A"]
    if not rows:
        return pd.DataFrame(columns=columns)
    print(f"OddsPapi: fetched {len(rows)} live {label} odds row(s) across {len(tournament_ids)} league(s)")
    return pd.DataFrame(rows, columns=columns)
