
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
