#!/usr/bin/env python3
"""
Main pipeline: data → model → evaluation → HTML report.

Usage:
  python main.py              # full pipeline
  python main.py --update     # re-download current season, then full pipeline
"""
import sys
from pathlib import Path

from src.data.loader import load_all_data
from src.evaluation.metrics import compute_betting_results, compute_roi, compute_stability
from src.evaluation.report import generate_report
from src.model.features import build_features_with_odds
from src.model.train import split_by_season, train_model


def run_pipeline():
    if "--update" in sys.argv:
        print("Updating current season data...")
        from src.data.download import update_current_season

        update_current_season()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("Training model...")
    results = train_model(df)

    # Build evaluation DataFrame
    _, test_df = split_by_season(df)
    X_test, y_test, odds_test = build_features_with_odds(test_df)

    eval_df = odds_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = results["y_pred"]

    print("Computing metrics...")
    betting_results = compute_betting_results(eval_df)
    roi = compute_roi(betting_results)
    stability = compute_stability(betting_results)
    accuracy = results["accuracy"]

    print("\n=== RESULTS ===")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"ROI:       {roi:+.2f}%")
    print(f"Stability: {stability:.4f}")
    print(f"Test bets: {len(betting_results)}")

    print("\nGenerating report...")
    generate_report(
        results_df=betting_results,
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        output_path=Path("reports/evaluation_report.html"),
    )
    print("Done. Open reports/evaluation_report.html to view results.")


if __name__ == "__main__":
    run_pipeline()
