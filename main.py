#!/usr/bin/env python3
"""
Main pipeline: data → model → evaluation → HTML report + profit chart.

Modes:
  python main.py                    # backtest on last 2 seasons, threshold=0.0
  python main.py --threshold 0.05   # backtest with 5% minimum edge filter
  python main.py --predict          # train on all data, predict upcoming fixtures
  python main.py --update           # re-download latest season results, then backtest
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_all_data
from src.evaluation.metrics import compute_roi, compute_stability, compute_value_betting_results
from src.evaluation.report import generate_report
from src.model.features import FEATURE_COLS, build_features_with_odds, build_fixture_features
from src.model.train import split_by_season, train_on_all_data, train_walkforward


def _parse_threshold() -> float:
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return 0.0


def _save_profit_chart(betting_results, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(betting_results["Date"], betting_results["cumulative_profit"],
            color="#1565c0", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(
        betting_results["Date"], betting_results["cumulative_profit"], 0,
        where=betting_results["cumulative_profit"] >= 0, alpha=0.15, color="#43a047",
    )
    ax.fill_between(
        betting_results["Date"], betting_results["cumulative_profit"], 0,
        where=betting_results["cumulative_profit"] < 0, alpha=0.15, color="#e53935",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit Over Time")
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Profit chart saved to {output_path}")


def _run_predict():
    from src.data.download import download_fixtures, update_current_season
    from src.data.loader import load_fixtures

    threshold = _parse_threshold()

    print("Updating latest season results...")
    update_current_season()

    print("Downloading upcoming fixtures...")
    download_fixtures()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("Training on full dataset...")
    model = train_on_all_data(df)

    print("Loading fixtures...")
    fixtures_df = load_fixtures()
    print(f"Found {len(fixtures_df)} upcoming fixtures in tracked leagues")

    print("Building fixture features...")
    fixture_features = build_fixture_features(df, fixtures_df)
    if fixture_features.empty:
        print("No fixtures could be featurised (teams may lack sufficient history).")
        return

    dropped = len(fixtures_df) - len(fixture_features)
    if dropped:
        print(f"  {dropped} fixture(s) dropped — teams with < {5} games history")

    X_fix = fixture_features[FEATURE_COLS]
    y_proba = model.predict_proba(X_fix)
    classes = list(model.classes_)

    _print_predictions(fixture_features, y_proba, classes, threshold)


def _print_predictions(fixture_features, y_proba, classes, threshold: float) -> None:
    outcome_label = {"H": "Home", "D": "Draw", "A": "Away"}
    odds_col = {"H": "B365H", "D": "B365D", "A": "B365A"}

    print(f"\n{'='*90}")
    print(f"UPCOMING FIXTURE PREDICTIONS  (threshold: {threshold:+.2f} edge over fair odds)")
    print(f"{'='*90}")

    # fixture_features is reset_index'd so iloc position == y_proba row
    fixture_features = fixture_features.reset_index(drop=True)

    by_league = fixture_features.groupby("league")
    for league, group in by_league:
        print(f"\n--- {league.upper()} ---")
        print(f"{'Date':<12} {'Home':<22} {'Away':<22} {'H%':>5} {'D%':>5} {'A%':>5}  {'Odds H/D/A':>14}  Value bets")
        print("-" * 90)

        for i, row in group.iterrows():
            probs = {c: float(y_proba[i, j]) for j, c in enumerate(classes)}

            raw_implied = {o: 1.0 / float(row[odds_col[o]]) for o in ["H", "D", "A"]}
            total_implied = sum(raw_implied.values())
            fair = {o: raw_implied[o] / total_implied for o in raw_implied}

            value_bets = []
            for o in ["H", "D", "A"]:
                edge = probs[o] - fair[o]
                if edge > threshold:
                    value_bets.append(f"{outcome_label[o]}(+{edge:.1%})")

            date_str = row["Date"].strftime("%Y-%m-%d")
            odds_str = f"{row['B365H']:.2f}/{row['B365D']:.2f}/{row['B365A']:.2f}"
            value_str = ", ".join(value_bets) if value_bets else "-"

            print(f"{date_str:<12} {row['HomeTeam']:<22} {row['AwayTeam']:<22} "
                  f"{probs['H']:>5.0%} {probs['D']:>5.0%} {probs['A']:>5.0%}  "
                  f"{odds_str:>14}  {value_str}")

    print(f"\n{'='*90}")


def _run_backtest():
    if "--update" in sys.argv:
        print("Updating current season data...")
        from src.data.download import update_current_season
        update_current_season()

    threshold = _parse_threshold()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("Running walk-forward backtest (one model per test season)...")
    results = train_walkforward(df)

    eval_df = results["odds_test"].copy()
    eval_df["y_true"] = results["y_test"].values

    betting_results = compute_value_betting_results(
        eval_df,
        results["y_proba"],
        results["classes"],
        threshold=threshold,
    )
    roi = compute_roi(betting_results)
    stability = compute_stability(betting_results)
    accuracy = results["accuracy"]

    n_matches = len(eval_df)
    n_bets = len(betting_results)
    print("\n=== BACKTEST RESULTS ===")
    print(f"Accuracy:  {accuracy:.3f}  (on all {n_matches} test matches)")
    print(f"Threshold: {threshold:+.3f}  (minimum edge over fair odds)")
    print(f"Bets:      {n_bets} / {n_matches} matches ({n_bets / n_matches:.1%})")
    print(f"ROI:       {roi:+.2f}%")
    print(f"Stability: {stability:.4f}")

    _save_profit_chart(betting_results, Path("reports/profit_curve.png"))

    print("\nGenerating report...")
    generate_report(
        results_df=betting_results,
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        output_path=Path("reports/evaluation_report.html"),
    )
    print("Done. Open reports/evaluation_report.html to view results.")


def run_pipeline():
    if "--predict" in sys.argv:
        _run_predict()
    else:
        _run_backtest()


if __name__ == "__main__":
    run_pipeline()
