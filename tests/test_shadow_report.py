import pandas as pd
import pytest

from src.evaluation.shadow import PREDICTION_COLUMNS, SETTLEMENT_COLUMNS
from src.evaluation.shadow_report import generate_shadow_report


def _predictions():
    records = [
        {
            "prediction_id": "settled-home",
            "run_id": "run-1",
            "model_commit": "abc123",
            "fetched_at": "2026-08-04T06:00:00Z",
            "fixture_date": "2026-08-08",
            "league": "E0",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "outcome": "H",
            "model_probability": 0.60,
            "fair_probability": 0.50,
            "captured_b365_odds": 1.90,
            "captured_best_odds": 2.00,
            "captured_best_bookmaker": "B365",
            "edge": 0.10,
            "applied_threshold": 0.04,
            "is_value_bet": True,
            "is_production_fixture": True,
        },
        {
            "prediction_id": "settled-draw",
            "run_id": "run-1",
            "model_commit": "abc123",
            "fetched_at": "2026-08-04T06:00:00Z",
            "fixture_date": "2026-08-08",
            "league": "E0",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "outcome": "D",
            "model_probability": 0.20,
            "fair_probability": 0.25,
            "captured_b365_odds": 3.50,
            "captured_best_odds": 3.50,
            "captured_best_bookmaker": "B365",
            "edge": -0.05,
            "applied_threshold": 0.04,
            "is_value_bet": False,
            "is_production_fixture": True,
        },
        {
            "prediction_id": "settled-away",
            "run_id": "run-1",
            "model_commit": "abc123",
            "fetched_at": "2026-08-04T06:00:00Z",
            "fixture_date": "2026-08-08",
            "league": "E0",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "outcome": "A",
            "model_probability": 0.20,
            "fair_probability": 0.25,
            "captured_b365_odds": 4.00,
            "captured_best_odds": 4.00,
            "captured_best_bookmaker": "B365",
            "edge": -0.05,
            "applied_threshold": 0.04,
            "is_value_bet": False,
            "is_production_fixture": True,
        },
        {
            "prediction_id": "pending-home",
            "run_id": "run-2",
            "model_commit": "def456",
            "fetched_at": "2026-08-11T06:00:00Z",
            "fixture_date": "2026-08-15",
            "league": "D1",
            "home_team": "Bayern",
            "away_team": "Dortmund",
            "outcome": "H",
            "model_probability": 0.60,
            "fair_probability": 0.50,
            "captured_b365_odds": 2.00,
            "captured_best_odds": 2.00,
            "captured_best_bookmaker": "B365",
            "edge": 0.10,
            "applied_threshold": 0.04,
            "is_value_bet": True,
            "is_production_fixture": True,
        },
        {
            "prediction_id": "pending-draw",
            "run_id": "run-2",
            "model_commit": "def456",
            "fetched_at": "2026-08-11T06:00:00Z",
            "fixture_date": "2026-08-15",
            "league": "D1",
            "home_team": "Bayern",
            "away_team": "Dortmund",
            "outcome": "D",
            "model_probability": 0.20,
            "fair_probability": 0.25,
            "captured_b365_odds": 3.50,
            "captured_best_odds": 3.50,
            "captured_best_bookmaker": "B365",
            "edge": -0.05,
            "applied_threshold": 0.04,
            "is_value_bet": False,
            "is_production_fixture": True,
        },
        {
            "prediction_id": "pending-away",
            "run_id": "run-2",
            "model_commit": "def456",
            "fetched_at": "2026-08-11T06:00:00Z",
            "fixture_date": "2026-08-15",
            "league": "D1",
            "home_team": "Bayern",
            "away_team": "Dortmund",
            "outcome": "A",
            "model_probability": 0.20,
            "fair_probability": 0.25,
            "captured_b365_odds": 4.00,
            "captured_best_odds": 4.00,
            "captured_best_bookmaker": "B365",
            "edge": -0.05,
            "applied_threshold": 0.04,
            "is_value_bet": False,
            "is_production_fixture": True,
        },
    ]
    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)


def _settlements():
    records = [
        {
            "prediction_id": "settled-home",
            "settled_at": "2026-08-09T06:00:00Z",
            "actual_result": "H",
            "is_winner": True,
            "profit": 1.0,
            "result_file_b365_odds": 1.80,
            "result_file_best_odds": 1.80,
            "closing_line_value": 2.0 / 1.8 - 1.0,
        },
        {
            "prediction_id": "settled-draw",
            "settled_at": "2026-08-09T06:00:00Z",
            "actual_result": "H",
            "is_winner": False,
            "profit": -1.0,
            "result_file_b365_odds": 3.25,
            "result_file_best_odds": 3.25,
            "closing_line_value": 3.5 / 3.25 - 1.0,
        },
        {
            "prediction_id": "settled-away",
            "settled_at": "2026-08-09T06:00:00Z",
            "actual_result": "H",
            "is_winner": False,
            "profit": -1.0,
            "result_file_b365_odds": 4.50,
            "result_file_best_odds": 4.50,
            "closing_line_value": 4.0 / 4.5 - 1.0,
        },
    ]
    return pd.DataFrame(records, columns=SETTLEMENT_COLUMNS)


def test_shadow_report_labels_sparse_results_as_informational(tmp_path):
    """Omitting the monitoring boundary would turn sparse evidence into a release signal."""
    output = generate_shadow_report(_predictions(), _settlements(), tmp_path / "shadow.html")

    html = output.read_text()
    assert "Forward shadow evaluation" in html
    assert "Informational" in html
    assert "monitoring only" in html.lower()
    assert "closing-line value" in html.lower()
    assert "<strong>2</strong>Prediction runs" in html
    assert "<strong>1</strong>Settled fixtures" in html
    assert "<strong>1</strong>Pending fixtures" in html
    assert "Settled prediction records" not in html
    assert "Pending prediction records" not in html
    assert "Captured-price ROI" in html
    assert "+100.00%" in html
    assert "League breakdown" in html
    assert (
        "<tr><td>E0</td><td>1</td><td>+1.00</td>"
        "<td>+100.00%</td><td>+11.11%</td></tr>"
    ) in html
    assert "Model commit breakdown" in html
    assert (
        "<tr><td>abc123</td><td>1</td><td>+1.00</td>"
        "<td>+100.00%</td><td>+11.11%</td></tr>"
    ) in html
    assert "Cumulative profit" in html
    assert (
        "<tr><td>2026-08-08</td><td>+1.00</td><td>+1.00</td></tr>"
    ) in html


def test_shadow_report_renders_weekly_interval_for_two_complete_weeks(tmp_path):
    """Dropping the weekly interval would hide uncertainty once two weeks are available."""
    settlements = pd.concat(
        [
            _settlements(),
            pd.DataFrame(
                [
                    {
                        "prediction_id": "pending-home",
                        "settled_at": "2026-08-16T06:00:00Z",
                        "actual_result": "A",
                        "is_winner": False,
                        "profit": -1.0,
                        "result_file_b365_odds": 2.20,
                        "result_file_best_odds": 2.20,
                        "closing_line_value": 2.0 / 2.2 - 1.0,
                    }
                ],
                columns=SETTLEMENT_COLUMNS,
            ),
        ],
        ignore_index=True,
    )

    output = generate_shadow_report(_predictions(), settlements, tmp_path / "shadow.html")

    html = output.read_text()
    assert "Weekly 95% ROI interval" in html
    assert "-100.00% to +100.00%" in html
    assert (
        "<tr><td>D1</td><td>1</td><td>-1.00</td>"
        "<td>-100.00%</td><td>-9.09%</td></tr>"
    ) in html
    assert (
        "<tr><td>def456</td><td>1</td><td>-1.00</td>"
        "<td>-100.00%</td><td>-9.09%</td></tr>"
    ) in html
    assert (
        "<tr><td>2026-08-15</td><td>-1.00</td><td>+0.00</td></tr>"
    ) in html


def test_shadow_report_rejects_orphan_settlement(tmp_path):
    """Reporting must fail instead of silently dropping a settlement with no prediction."""
    settlements = _settlements().copy()
    settlements.loc[0, "prediction_id"] = "orphan"

    with pytest.raises(ValueError, match="without matching prediction"):
        generate_shadow_report(_predictions(), settlements, tmp_path / "shadow.html")


def test_shadow_report_rejects_reordered_prediction_columns(tmp_path):
    """A report must not normalize a ledger whose persisted schema order changed."""
    predictions = _predictions()[list(reversed(PREDICTION_COLUMNS))]

    with pytest.raises(ValueError, match="columns"):
        generate_shadow_report(predictions, _settlements(), tmp_path / "shadow.html")
