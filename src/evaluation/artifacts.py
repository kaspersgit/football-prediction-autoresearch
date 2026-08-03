from pathlib import Path

import pandas as pd


def save_bet_artifacts(
    evaluation_results: pd.DataFrame,
    production_results: pd.DataFrame,
    reports_dir: Path = Path("reports"),
) -> tuple[Path, Path]:
    """Persist research-wide and production-only bet histories separately."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = reports_dir / "evaluation_bets.csv"
    production_path = reports_dir / "backtest_bets.csv"
    evaluation_results.to_csv(evaluation_path, index=False)
    production_results.to_csv(production_path, index=False)
    return evaluation_path, production_path
