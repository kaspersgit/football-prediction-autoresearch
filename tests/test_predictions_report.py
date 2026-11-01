import re
from datetime import datetime

import pandas as pd

from src.config import PRODUCTION_LEAGUES
from src.evaluation.predictions_report import (
    _forecast_card_html,
    _top_bets_html,
    generate_predictions_html,
)


def _pred_rows():
    return [
        {
            "Date": datetime(2026, 5, 10),
            "League": "england",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "ModelH": 0.55,
            "ModelD": 0.25,
            "ModelA": 0.20,
            "B365H": 1.90,
            "B365D": 3.50,
            "B365A": 4.50,
            "CustomMaxH": float("nan"),
            "CustomMaxD": float("nan"),
            "CustomMaxA": float("nan"),
            "CustomMaxBkH": "",
            "CustomMaxBkD": "",
            "CustomMaxBkA": "",
            "ValueBets": [("H", 0.023)],
        }
    ]


def _historical_bets():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=50),
        "league": ["E0"] * 50,
        "stake": [1.0] * 50,
        "profit": ([0.9, -1.0, 0.9, -1.0, 0.9]) * 10,
        "odds": [1.90] * 50,
        "model_prob": [0.55] * 50,
        "implied_prob": [0.526] * 50,
        "y_true": ["H", "A", "H", "A", "H"] * 10,
        "y_pred": ["H", "H", "H", "H", "H"] * 10,
    })


def test_all_bets_includes_model_prob():
    rows = _pred_rows()
    html = generate_predictions_html(
        rows, threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=_historical_bets(),
    )
    assert 'data-model-prob="0.5500"' in html


def test_top_bets_html_includes_model_prob_attr():
    bets = [
        {
            "date": "Sat May 10",
            "league": "england",
            "home": "Arsenal",
            "away": "Chelsea",
            "outcome": "H",
            "edge": 0.023,
            "b365_odds": 1.90,
            "max_odds": float("nan"),
            "max_bk": "",
            "model_prob": 0.55,
        }
    ]
    html = _top_bets_html(bets)
    assert 'data-model-prob="0.5500"' in html


def test_top_bets_html_renders_min_acceptable_odds():
    bets = [
        {
            "date": "Sat May 10",
            "league": "england",
            "home": "Arsenal",
            "away": "Chelsea",
            "outcome": "H",
            "edge": 0.023,
            "b365_odds": 1.90,
            "max_odds": float("nan"),
            "max_bk": "",
            "model_prob": 0.55,
            "min_acceptable_odds": 1.85,
        }
    ]
    html = _top_bets_html(bets)
    assert "1.85" in html


def test_top_bets_html_min_odds_missing_renders_placeholder():
    bets = [
        {
            "date": "Sat May 10",
            "league": "england",
            "home": "Arsenal",
            "away": "Chelsea",
            "outcome": "H",
            "edge": 0.023,
            "b365_odds": 1.90,
            "max_odds": float("nan"),
            "max_bk": "",
            "model_prob": 0.55,
        }
    ]
    html = _top_bets_html(bets)
    assert html.count("—") >= 1


def test_generate_predictions_html_includes_min_acceptable_odds_from_pred_rows():
    rows = _pred_rows()
    rows[0]["MinAcceptableOdds"] = {"H": 1.75}
    html = generate_predictions_html(
        rows, threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=_historical_bets(),
    )
    assert "1.75" in html


def test_forecast_card_html_empty_when_no_historical():
    assert _forecast_card_html(None) == ""
    assert _forecast_card_html(pd.DataFrame()) == ""


def test_forecast_card_html_contains_container_id():
    html = _forecast_card_html(_historical_bets())
    assert "forecast-card-container" in html


def test_generate_predictions_html_includes_forecast_card():
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=_historical_bets(),
    )
    assert "forecast-card-container" in html


def test_generate_predictions_html_no_forecast_card_without_historical():
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=None,
    )
    assert "forecast-card-container" not in html


def test_generate_predictions_html_empty_fixtures_notice():
    html = generate_predictions_html(
        [], threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=None,
    )
    assert "empty-fixtures-notice" in html
    assert "No upcoming production fixtures available" in html
    assert "0 fixtures" in html


def test_generate_predictions_html_no_empty_notice_with_fixtures():
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=None,
    )
    assert "No upcoming production fixtures available" not in html


def test_generate_predictions_html_shows_matchweek_fixture_funnel():
    """Three numbers: total fixtures this matchweek, how many we have data on
    (survived featurisation/filtering), and how many are value bets."""
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=None, total_upcoming_fixtures=12,
    )
    assert "12 fixtures this matchweek" in html
    assert "1 with data" in html  # len(_pred_rows())
    assert "1 value bets" in html  # the one ValueBets entry in _pred_rows()


def test_generate_predictions_html_clamps_matchweek_total_to_at_least_with_data():
    """A caller passing total_upcoming_fixtures=0 (e.g. an old call site) while
    pred_rows is non-empty shouldn't render a nonsensical "0 this matchweek, 1 with
    data"."""
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=None, total_upcoming_fixtures=0,
    )
    assert "1 fixtures this matchweek" in html


def test_top_bets_html_empty_is_valid_table_row():
    html = _top_bets_html([])
    assert html.startswith("<tr>")
    assert "No value bets found at this threshold." in html


def test_top_bets_html_marks_secondary_columns_for_mobile_hiding():
    """League/Best Odds/Min Odds carry col-league/col-max/col-min classes so the
    phone-width media query can drop them -- Bet/Edge/B365-Odds (the columns that
    must stay visible in portrait) carry no such class."""
    bet = {
        "date": "Sat May 10", "league": "england", "home": "Arsenal", "away": "Chelsea",
        "outcome": "H", "edge": 0.05, "b365_odds": 1.9,
        "max_odds": 2.0, "max_bk": "Bet365", "model_prob": 0.55,
        "min_acceptable_odds": 1.8,
    }
    html = _top_bets_html([bet])
    assert 'class="col-league"' in html
    assert 'class="odds-val col-max"' in html
    assert 'class="odds-val col-min"' in html


def test_generate_predictions_html_has_mobile_media_query_for_top_bets_table():
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3), historical_bets=None,
    )
    assert "@media (max-width: 560px)" in html
    assert ".top-bets-card .col-league, .top-bets-card .col-max, .top-bets-card .col-min" in html


def test_monthly_table_js_covers_every_production_league():
    """The client-side monthly-performance table's LG_ORDER/LG_NAMES lookups are
    hardcoded JS, not derived from PRODUCTION_LEAGUES -- a league added to production
    silently drops out of "Historical Performance by Month" without this guard
    (found live for Greece/G1, missed since EXP-20260810-002 added it)."""
    html = generate_predictions_html(
        _pred_rows(), threshold=0.0, fetched_at=datetime(2026, 5, 3),
        historical_bets=_historical_bets(),
    )
    lg_order_match = re.search(r"var LG_ORDER = \[([^\]]*)\];", html)
    assert lg_order_match, "LG_ORDER array not found in generated HTML"
    lg_order = [c.strip("'\" ") for c in lg_order_match.group(1).split(",")]

    lg_names_match = re.search(r"var LG_NAMES = \{([^}]*)\};", html)
    assert lg_names_match, "LG_NAMES object not found in generated HTML"
    lg_names = dict(re.findall(r"'([^']+)':'([^']+)'", lg_names_match.group(1)))

    for league in PRODUCTION_LEAGUES:
        assert league in lg_order, f"{league} missing from monthly table's LG_ORDER"
        assert league in lg_names, f"{league} missing from monthly table's LG_NAMES"


def test_save_predictions_html_empty_writes_file(tmp_path):
    from src.evaluation.predictions_report import save_predictions_html

    out = tmp_path / "predictions_empty.html"
    path = save_predictions_html(
        [], threshold=0.0, fetched_at=datetime(2026, 5, 3), output_path=out,
    )
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "empty-fixtures-notice" in content
