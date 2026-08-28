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


def test_top_bets_html_empty_is_valid_table_row():
    html = _top_bets_html([])
    assert html.startswith("<tr>")
    assert "No value bets found at this threshold." in html


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
