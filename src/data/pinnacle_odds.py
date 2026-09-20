"""Live pre-match Pinnacle 1X2 odds for production leagues, via The Odds API."""

import pandas as pd

from src.data._live_odds import attach_bookmaker_odds, fetch_bookmaker_odds

_BOOKMAKER_KEY = "pinnacle"
_COL_PREFIX = "PS"
_LABEL = "Pinnacle"
_REGIONS = "eu"


def fetch_pinnacle_odds(leagues: set[str]) -> pd.DataFrame:
    """Fetch live pre-match Pinnacle 1X2 odds for the given league codes."""
    return fetch_bookmaker_odds(leagues, _BOOKMAKER_KEY, _COL_PREFIX, _LABEL, _REGIONS)


def attach_pinnacle_odds(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge live Pinnacle odds onto fixtures_df, overwriting NaN PSH/PSD/PSA placeholders."""
    from src.config import PRODUCTION_LEAGUES

    return attach_bookmaker_odds(
        fixtures_df, fetch_pinnacle_odds, set(PRODUCTION_LEAGUES), _COL_PREFIX, _LABEL
    )
