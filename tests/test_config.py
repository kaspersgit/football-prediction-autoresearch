from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_MAX_EDGE,
    DEFAULT_MAX_ODDS,
    DEFAULT_MAX_OVERROUND,
    EXCLUDED_BETTING_LEAGUES,
    LEAGUE_NAMES,
    PRODUCTION_LEAGUES,
    SUPPORTED_LEAGUES,
    filter_production_fixtures,
)


def test_betting_defaults_match_verified_configuration():
    assert DEFAULT_MAX_ODDS == 5.0
    assert DEFAULT_MAX_EDGE == 0.20
    assert DEFAULT_MAX_OVERROUND == 0.07
    assert EXCLUDED_BETTING_LEAGUES == frozenset(
        {"SP1", "D1", "I1", "SC0", "B1", "T1"}
    )


def test_production_leagues_are_supported():
    assert PRODUCTION_LEAGUES <= set(SUPPORTED_LEAGUES)
    assert EXCLUDED_BETTING_LEAGUES == frozenset(
        set(SUPPORTED_LEAGUES) - PRODUCTION_LEAGUES
    )


def test_all_known_leagues_are_supported():
    assert set(SUPPORTED_LEAGUES) == {
        "E0",
        "D1",
        "SP1",
        "I1",
        "F1",
        "N1",
        "P1",
        "G1",
        "SC0",
        "B1",
        "T1",
    }
    assert set(LEAGUE_NAMES) == set(SUPPORTED_LEAGUES)


def test_main_uses_shared_betting_defaults():
    source = Path("main.py").read_text()

    assert "return DEFAULT_MAX_ODDS" in source
    assert "return DEFAULT_MAX_EDGE" in source
    assert "_PREDICT_MAX_ODDS = DEFAULT_MAX_ODDS" in source
    assert "_PREDICT_MAX_EDGE = DEFAULT_MAX_EDGE" in source
    assert "_PREDICT_MAX_OVERROUND = DEFAULT_MAX_OVERROUND" in source
    assert "_SKIP_LEAGUES" not in source
    assert "max_overround=DEFAULT_MAX_OVERROUND" in source


def test_production_fixture_filter_uses_allowlist():
    fixtures = pd.DataFrame(
        {
            "league": ["E0", "D1", "G1", "B1"],
            "fixture_id": [1, 2, 3, 4],
        }
    )

    result = filter_production_fixtures(fixtures)

    assert result["fixture_id"].tolist() == [1, 3]


def test_main_does_not_claim_removed_pinnacle_filter_is_active():
    source = Path("main.py").read_text().lower()

    assert "pinnacle_note" not in source
    assert "before pinnacle filter" not in source
    assert "with pinnacle filter active" not in source
