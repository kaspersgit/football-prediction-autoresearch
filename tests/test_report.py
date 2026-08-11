
import pandas as pd

from src.evaluation.report import generate_report


def _make_results():
    return pd.DataFrame({
        "y_true": ["H", "D", "A", "H", "H"] * 10,
        "y_pred": ["H", "H", "A", "A", "H"] * 10,
        "B365H": [2.0] * 50,
        "B365D": [3.5] * 50,
        "B365A": [4.0] * 50,
        "Date": pd.date_range("2024-01-01", periods=50),
        "HomeTeam": ["Arsenal"] * 50,
        "AwayTeam": ["Chelsea"] * 50,
        "league": ["E0"] * 50,
        "season": ["2324"] * 50,
        "profit": ([1.0, -1.0, 3.0, -1.0, 0.8]) * 10,
        "cumulative_profit": list(range(50)),
    })

def test_generate_report_creates_html(tmp_path):
    df = _make_results()
    out = tmp_path / "report.html"
    generate_report(
        results_df=df,
        accuracy=0.52,
        roi=-3.5,
        stability=0.12,
        output_path=out,
    )
    assert out.exists()
    content = out.read_text()
    assert "<html" in content.lower()
    assert "ROI" in content
    assert "Accuracy" in content
    assert "t-stat" in content
    assert "+0.85" in content


def test_report_embeds_production_bets_as_a_separate_toggleable_dataset(tmp_path):
    """The 'Production config' dashboard toggle must read from the actual production
    portfolio (per-league thresholds + Pinnacle filter), not the all-market bets."""
    all_market = _make_results()
    production = _make_results().iloc[:3].copy()
    production["league"] = "N1"
    out = tmp_path / "report.html"

    generate_report(
        results_df=all_market,
        accuracy=0.52,
        roi=-3.5,
        stability=0.12,
        output_path=out,
        production_results=production,
    )

    content = out.read_text()
    assert "var PRODUCTION_BETS=" in content
    assert content.count('"N1"') == 3
    assert 'onclick="setMode(\'production\')"' in content
    assert "var BACKTEST_BETS=" in content


def test_report_without_production_results_still_renders_an_empty_production_dataset(tmp_path):
    out = tmp_path / "report.html"

    generate_report(
        results_df=_make_results(),
        accuracy=0.5,
        roi=0.0,
        stability=0.0,
        output_path=out,
    )

    content = out.read_text()
    assert "var PRODUCTION_BETS=[];" in content


def test_report_enables_all_observed_leagues_and_offers_production_preset(tmp_path):
    df = _make_results().iloc[:2].copy()
    df["league"] = ["E0", "G1"]
    all_predictions = pd.DataFrame({"league": ["D1", "B1"]})
    out = tmp_path / "report.html"

    generate_report(
        results_df=df,
        accuracy=0.5,
        roi=1.0,
        stability=0.1,
        output_path=out,
        all_predictions=all_predictions,
        production_leagues={"E0", "G1"},
    )

    content = out.read_text()
    for league in ["E0", "D1", "G1", "B1"]:
        assert f'id="btn-lg-{league}"' in content
    assert 'var _activeLeagues={"E0":true,"D1":true,"G1":true,"B1":true};' in content
    assert 'var _productionLeagues=["E0","G1"];' in content
    assert "Production markets" in content


def test_report_renders_static_weekly_roi_interval_for_initial_all_market_bets(tmp_path):
    df = _make_results().iloc[:4].copy()
    df["Date"] = ["2026-08-04", "2026-08-05", "2026-08-11", "2026-08-12"]
    df["profit"] = [1.0, -1.0, 2.0, -1.0]
    out = tmp_path / "report.html"

    generate_report(
        results_df=df,
        accuracy=0.5,
        roi=25.0,
        stability=0.1,
        output_path=out,
    )

    content = out.read_text()
    assert "Weekly 95% ROI interval" in content
    assert "+0.00% to +50.00%" in content
    assert 'id="summary-weekly-roi"' in content
    assert "Bootstrap CI over ISO weeks, recomputed from the bets shown." in content


def test_report_labels_weekly_roi_interval_when_there_are_not_enough_completed_weeks(tmp_path):
    df = _make_results().iloc[:1].copy()
    out = tmp_path / "report.html"

    generate_report(
        results_df=df,
        accuracy=0.5,
        roi=100.0,
        stability=0.1,
        output_path=out,
    )

    assert "Not enough completed weeks" in out.read_text()


def test_report_offers_a_pinnacle_confirmation_toggle_that_does_not_touch_calibration(tmp_path):
    """The Pinnacle toggle should let a user reconstruct production's confirmation veto
    from the all-market screen without being locked to production leagues, and it must
    only ever filter placed bets, not the calibration chart's raw predictions."""
    out = tmp_path / "report.html"

    generate_report(
        results_df=_make_results(),
        accuracy=0.5,
        roi=1.0,
        stability=0.1,
        output_path=out,
    )

    content = out.read_text()
    assert 'id="btn-pinnacle"' in content
    assert "function togglePinnacle()" in content
    assert "_requirePinnacle && b.pinnacle_confirmed !== true" in content
    # The calibration dataset (ALL_BETS / filteredAllBets) must not be pinnacle-filtered.
    assert "filteredAllBets = ALL_BETS.filter" in content
    calibration_filter = content.split("filteredAllBets = ALL_BETS.filter")[1].split("});")[0]
    assert "_requirePinnacle" not in calibration_filter


def test_report_accuracy_card_reacts_to_filters(tmp_path):
    out = tmp_path / "report.html"

    generate_report(
        results_df=_make_results(),
        accuracy=0.5,
        roi=1.0,
        stability=0.1,
        output_path=out,
    )

    content = out.read_text()
    assert 'id="summary-accuracy"' in content
    assert "Raw model accuracy" in content
    assert "50.0%" in content
