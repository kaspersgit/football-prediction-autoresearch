"""Live pre-match Bet365 1X2 odds for production leagues, via The Odds API.

Bet365 has no public odds API of its own — see src.data.pinnacle_odds for why this
goes through The Odds API aggregator instead, same as Pinnacle. Bet365 is listed
there as a UK-region bookmaker (key "bet365"), unlike Pinnacle's "eu" region.

This refreshes B365H/B365D/B365A specifically because that's the model's primary
edge/training feature (see README.md) — fixtures.csv's own B365 columns are only as
fresh as football-data.co.uk's last publish, not live.

NOT currently wired into _run_predict() (main.py): live-tested against production
on 2026-09-20 across 46 fixtures in 5 leagues (some kicking off same-day) and this
account's plan returned zero Bet365 prices for every one of them — both "bet365"
and "bet365ww" as the bookmaker key — while Pinnacle succeeded for many of the same
fixtures in the same run. That's an account/plan entitlement gap, not "not priced
yet". Wire attach_bet365_odds() back into _run_predict() once the plan covers
Bet365 — until then it would just spend API credits for nothing.
"""

import pandas as pd

from src.data._live_odds import attach_bookmaker_odds, fetch_bookmaker_odds

_BOOKMAKER_KEY = "bet365"
_COL_PREFIX = "B365"
_LABEL = "Bet365"
_REGIONS = "uk"


def fetch_bet365_odds(leagues: set[str]) -> pd.DataFrame:
    """Fetch live pre-match Bet365 1X2 odds for the given league codes."""
    return fetch_bookmaker_odds(leagues, _BOOKMAKER_KEY, _COL_PREFIX, _LABEL, _REGIONS)


def attach_bet365_odds(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge live Bet365 odds onto fixtures_df, overwriting fixtures.csv's B365H/B365D/B365A."""
    from src.config import PRODUCTION_LEAGUES

    return attach_bookmaker_odds(
        fixtures_df, fetch_bet365_odds, set(PRODUCTION_LEAGUES), _COL_PREFIX, _LABEL
    )
