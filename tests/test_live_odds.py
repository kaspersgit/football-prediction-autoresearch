import pandas as pd

from src.config import SUPPORTED_LEAGUES
from src.data._live_odds import _LEAGUE_TO_SPORT_KEY, fetch_bookmaker_odds
from src.data._oddspapi_client import OddsPapiUnavailable


def test_sport_key_map_covers_every_supported_league():
    assert set(_LEAGUE_TO_SPORT_KEY) == set(SUPPORTED_LEAGUES)


def test_fetch_bookmaker_odds_prefers_oddspapi_when_it_succeeds(monkeypatch):
    oddspapi_result = pd.DataFrame([{"league": "E0"}])

    monkeypatch.setattr(
        "src.data._live_odds.fetch_oddspapi_bookmaker_odds",
        lambda leagues, bookmaker_key, col_prefix, label: oddspapi_result,
    )

    def theoddsapi_should_not_be_called(*a, **kw):
        raise AssertionError("fell back to The Odds API even though OddsPapi succeeded")

    monkeypatch.setattr(
        "src.data._live_odds._fetch_via_theoddsapi", theoddsapi_should_not_be_called
    )

    result = fetch_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle", "eu")

    assert result is oddspapi_result


def test_fetch_bookmaker_odds_falls_back_when_oddspapi_unavailable(monkeypatch):
    def oddspapi_unavailable(leagues, bookmaker_key, col_prefix, label):
        raise OddsPapiUnavailable("no key")

    monkeypatch.setattr(
        "src.data._live_odds.fetch_oddspapi_bookmaker_odds", oddspapi_unavailable
    )
    fallback_result = pd.DataFrame([{"league": "E0"}])
    monkeypatch.setattr(
        "src.data._live_odds._fetch_via_theoddsapi",
        lambda leagues, bookmaker_key, col_prefix, label, regions: fallback_result,
    )

    result = fetch_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle", "eu")

    assert result is fallback_result
