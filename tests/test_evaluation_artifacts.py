import pandas as pd

from src.evaluation.artifacts import save_bet_artifacts


def test_bet_artifacts_keep_evaluation_and_production_separate(tmp_path):
    evaluation = pd.DataFrame({"league": ["E0", "D1"], "profit": [1.0, -1.0]})
    production = pd.DataFrame({"league": ["E0"], "profit": [1.0]})

    evaluation_path, production_path = save_bet_artifacts(
        evaluation,
        production,
        reports_dir=tmp_path,
    )

    assert pd.read_csv(evaluation_path)["league"].tolist() == ["E0", "D1"]
    assert pd.read_csv(production_path)["league"].tolist() == ["E0"]
    assert evaluation_path.name == "evaluation_bets.csv"
    assert production_path.name == "backtest_bets.csv"
