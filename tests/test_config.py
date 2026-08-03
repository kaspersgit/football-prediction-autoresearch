from pathlib import Path

from src.config import (
    DEFAULT_MAX_EDGE,
    DEFAULT_MAX_ODDS,
    DEFAULT_MAX_OVERROUND,
    EXCLUDED_BETTING_LEAGUES,
)


def test_betting_defaults_match_verified_configuration():
    assert DEFAULT_MAX_ODDS == 5.0
    assert DEFAULT_MAX_EDGE == 0.20
    assert DEFAULT_MAX_OVERROUND == 0.07
    assert EXCLUDED_BETTING_LEAGUES == frozenset(
        {"F1", "SP1", "D1", "I1", "SC0", "B1", "T1"}
    )


def test_main_uses_shared_betting_defaults():
    source = Path("main.py").read_text()

    assert "return DEFAULT_MAX_ODDS" in source
    assert "return DEFAULT_MAX_EDGE" in source
    assert "_PREDICT_MAX_ODDS = DEFAULT_MAX_ODDS" in source
    assert "_PREDICT_MAX_EDGE = DEFAULT_MAX_EDGE" in source
    assert "_PREDICT_MAX_OVERROUND = DEFAULT_MAX_OVERROUND" in source
    assert 'fixture_features["league"].isin(EXCLUDED_BETTING_LEAGUES)' in source
    assert "_SKIP_LEAGUES" not in source
    assert "skip_leagues=EXCLUDED_BETTING_LEAGUES" in source
    assert "max_overround=DEFAULT_MAX_OVERROUND" in source


def test_main_does_not_claim_removed_pinnacle_filter_is_active():
    source = Path("main.py").read_text().lower()

    assert "pinnacle_note" not in source
    assert "before pinnacle filter" not in source
    assert "with pinnacle filter active" not in source
