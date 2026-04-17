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
from src.evaluation.metrics import compute_roi, compute_stability, compute_value_betting_results
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

    # Multi-outcome value betting
    betting_results = compute_value_betting_results(
        eval_df,
        results["y_proba"],
        results["classes"],
    )
    roi = compute_roi(betting_results)
    stability = compute_stability(betting_results)
    accuracy = results["accuracy"]

    n_matches = len(eval_df)
    n_bets = len(betting_results)
    print("\n=== RESULTS ===")
    print(f"Accuracy:  {accuracy:.3f}  (on all {n_matches} test matches)")
    print(f"Value bets:{n_bets} / {n_matches} matches ({n_bets / n_matches:.1%})")
    print(f"ROI:       {roi:+.2f}%")
    print(f"Stability: {stability:.4f}")
    print(f"Test bets: {n_bets}")

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
