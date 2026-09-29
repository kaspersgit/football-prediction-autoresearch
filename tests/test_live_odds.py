from src.config import SUPPORTED_LEAGUES
from src.data._live_odds import _LEAGUE_TO_SPORT_KEY


def test_sport_key_map_covers_every_supported_league():
    assert set(_LEAGUE_TO_SPORT_KEY) == set(SUPPORTED_LEAGUES)
