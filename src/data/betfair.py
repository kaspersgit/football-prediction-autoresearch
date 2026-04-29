"""
Betfair Exchange API integration.

Fetches live match-odds prices for upcoming fixtures and returns them
as a dict keyed by (home_team, away_team) → {"H": float, "D": float, "A": float}.

Authentication uses the Interactive Login endpoint (username + password + app key).
Credentials are read from config/betfair_credentials.json — never commit that file.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import requests

_LOGIN_URL = "https://identitysso.betfair.com/api/login"
_API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
_CREDS_PATH = Path("config/betfair_credentials.json")

# Betfair event type ID for soccer
_SOCCER_TYPE_ID = "1"

# Selection names Betfair uses for 1X2 markets
_HOME_WIN = "Home"
_DRAW = "The Draw"
_AWAY_WIN = "Away"


# ---------------------------------------------------------------------------
# Credentials & session
# ---------------------------------------------------------------------------

def load_credentials() -> dict:
    if not _CREDS_PATH.exists():
        raise FileNotFoundError(
            f"{_CREDS_PATH} not found. "
            "Copy config/betfair_credentials.json.example, fill in your details."
        )
    return json.loads(_CREDS_PATH.read_text())


def login(creds: dict) -> str:
    """Authenticate and return a session token."""
    resp = requests.post(
        _LOGIN_URL,
        data={"username": creds["username"], "password": creds["password"]},
        headers={
            "X-Application": creds["app_key"],
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "SUCCESS":
        raise RuntimeError(f"Betfair login failed: {body.get('error', body)}")
    return body["token"]


def _rpc(session_token: str, app_key: str, operation: str, params: dict) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "method": f"SportsAPING/v1.0/{operation}",
        "params": params,
        "id": 1,
    }
    resp = requests.post(
        _API_URL,
        json=payload,
        headers={
            "X-Application": app_key,
            "X-Authentication": session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"Betfair API error: {body['error']}")
    return body.get("result", {})


# ---------------------------------------------------------------------------
# Name normalisation & fuzzy matching
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lower-case, strip punctuation, common abbreviation expansions."""
    name = name.lower()
    replacements = {
        "manchester": "man", "united": "utd", "athletic": "ath",
        "atletico": "ath", "atlético": "ath", "real ": "", "fc ": "", " fc": "",
        "borussia ": "", "bayer ": "", "paris saint-germain": "psg",
        "paris sg": "psg", "paris fc": "paris fc",  # keep Paris FC distinct
        "nottm forest": "nott'm forest", "nott'm forest": "nott'm forest",
        "wolverhampton": "wolves", "brighton & hove albion": "brighton",
        "west bromwich": "west brom", "sheffield wednesday": "sheff wed",
        "sheffield united": "sheff utd", "queens park rangers": "qpr",
        "aston": "aston",  # keep aston villa intact
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r"[^a-z0-9 ]", "", name).strip()
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _best_match(name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    scored = [(c, _similarity(name, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, score = scored[0] if scored else (None, 0)
    return best if score >= threshold else None


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fetch_betfair_odds(fixtures: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    """
    Given a list of (home_team, away_team) fixture pairs, return Betfair
    Exchange best-available back prices for the Match Odds (1X2) market.

    Returns: {(home, away): {"H": float, "D": float, "A": float}}
    Missing fixtures are silently omitted from the result dict.
    """
    creds = load_credentials()
    token = login(creds)
    app_key = creds["app_key"]

    # Step 1: list upcoming soccer events
    events = _rpc(token, app_key, "listEvents", {
        "filter": {
            "eventTypeIds": [_SOCCER_TYPE_ID],
            "inPlayOnly": False,
        },
        "maxResults": 1000,
    })

    if not events:
        return {}

    event_ids = [e["event"]["id"] for e in events]
    event_names = {e["event"]["id"]: e["event"]["name"] for e in events}

    # Step 2: list Match Odds markets for those events
    catalogues = _rpc(token, app_key, "listMarketCatalogue", {
        "filter": {
            "eventIds": event_ids,
            "marketTypeCodes": ["MATCH_ODDS"],
        },
        "marketProjection": ["RUNNER_DESCRIPTION", "EVENT"],
        "maxResults": 1000,
    })

    if not catalogues:
        return {}

    # Build index: (home_name, away_name) → market_id
    # Betfair event names are typically "Team A v Team B"
    market_index: dict[tuple[str, str], str] = {}
    for cat in catalogues:
        event_name = cat.get("event", {}).get("name", "")
        parts = re.split(r" v | vs ", event_name, flags=re.IGNORECASE)
        if len(parts) == 2:
            market_index[(parts[0].strip(), parts[1].strip())] = cat["marketId"]

    bf_home_names = [h for h, _ in market_index]
    bf_away_names = [a for _, a in market_index]

    # Match our fixture names to Betfair event names
    fixture_to_market: dict[tuple[str, str], str] = {}
    for home, away in fixtures:
        best_home = _best_match(home, bf_home_names)
        if best_home is None:
            continue
        # Among markets with that home team, find best away match
        candidates_away = [a for (h, a) in market_index if h == best_home]
        best_away = _best_match(away, candidates_away)
        if best_away is None:
            continue
        market_id = market_index.get((best_home, best_away))
        if market_id:
            fixture_to_market[(home, away)] = market_id

    if not fixture_to_market:
        return {}

    # Step 3: fetch prices for matched markets
    market_ids = list(set(fixture_to_market.values()))
    books = _rpc(token, app_key, "listMarketBook", {
        "marketIds": market_ids,
        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        "orderProjection": "EXECUTABLE",
        "currencyCode": "GBP",
    })

    # Build market_id → {H, D, A} prices
    prices_by_market: dict[str, dict[str, float]] = {}
    for book in books:
        market_id = book["marketId"]
        runner_prices: dict[str, float] = {}
        for runner in book.get("runners", []):
            name = runner.get("runnerName", runner.get("selectionId", ""))
            best_back = runner.get("ex", {}).get("availableToBack", [])
            if best_back:
                price = float(best_back[0]["price"])
                if name == _HOME_WIN:
                    runner_prices["H"] = price
                elif name == _DRAW:
                    runner_prices["D"] = price
                elif name == _AWAY_WIN:
                    runner_prices["A"] = price
        if len(runner_prices) == 3:
            prices_by_market[market_id] = runner_prices

    # Assemble result
    result: dict[tuple[str, str], dict[str, float]] = {}
    for (home, away), market_id in fixture_to_market.items():
        if market_id in prices_by_market:
            result[(home, away)] = prices_by_market[market_id]

    return result
