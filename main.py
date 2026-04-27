#!/usr/bin/env python3
"""
Main pipeline: data → model → evaluation → HTML report + profit chart.

Leagues: England (E0), Germany (D1), Spain (SP1), Italy (I1), France (F1), Netherlands (N1), Portugal (P1)

Modes:
  python main.py                    # backtest on last 2 seasons, threshold=0.0 (all bets)
  python main.py --per-league       # one model per league (default recommended)
  python main.py --threshold 0.05   # backtest with 5% minimum edge filter
  python main.py --predict          # train on all data, predict upcoming fixtures
  python main.py --update           # re-download latest season results, then backtest
  python main.py --monthly          # monthly walk-forward retrain (experimental)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_all_data
from src.evaluation.metrics import compute_roi, compute_stability, compute_value_betting_results
from src.evaluation.report import generate_report
from src.model.features import FEATURE_COLS, build_features_with_odds, build_fixture_features
from src.model.train import split_by_season, train_on_all_data, train_walkforward, train_walkforward_monthly


def _parse_threshold() -> float:
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return 0.0


def _parse_kelly() -> float:
    for i, arg in enumerate(sys.argv):
        if arg == "--kelly" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return 0.0


def _parse_per_league() -> bool:
    return "--per-league" in sys.argv


def _parse_binary() -> bool:
    return "--binary" in sys.argv


def _parse_monthly() -> bool:
    return "--monthly" in sys.argv


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
    from datetime import datetime
    from src.data.download import download_fixtures, update_current_season
    from src.data.loader import load_fixtures

    threshold = _parse_threshold()
    fetched_at = datetime.now()

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

    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold)
    _print_predictions(fixture_features, y_proba, classes, threshold, fetched_at)
    _save_predictions_csv(fixture_features, y_proba, classes, threshold, fetched_at)
    _save_predictions_html(pred_rows, threshold, fetched_at)


def _pinnacle_fair(row) -> dict | None:
    """Return vig-corrected Pinnacle fair probs from PSH/PSD/PSA, or None if unavailable."""
    try:
        ps_total = 1/float(row["PSH"]) + 1/float(row["PSD"]) + 1/float(row["PSA"])
        return {
            "H": (1/float(row["PSH"])) / ps_total,
            "D": (1/float(row["PSD"])) / ps_total,
            "A": (1/float(row["PSA"])) / ps_total,
        }
    except (TypeError, ValueError, ZeroDivisionError, KeyError):
        return None


def _build_prediction_rows(fixture_features, y_proba, classes, threshold: float) -> list[dict]:
    """Build a list of per-fixture prediction dicts (used by both print and CSV export)."""
    odds_col = {"H": "B365H", "D": "B365D", "A": "B365A"}
    has_pinnacle = all(c in fixture_features.columns for c in ("PSH", "PSD", "PSA"))
    fixture_features = fixture_features.reset_index(drop=True)
    rows = []
    for i, row in fixture_features.iterrows():
        probs = {c: float(y_proba[i, j]) for j, c in enumerate(classes)}
        raw_implied = {o: 1.0 / float(row[odds_col[o]]) for o in ["H", "D", "A"]}
        total_implied = sum(raw_implied.values())
        fair = {o: raw_implied[o] / total_implied for o in raw_implied}
        ps_fair = _pinnacle_fair(row) if has_pinnacle else None

        value_bets = []
        for o in ["H", "D", "A"]:
            edge = probs[o] - fair[o]
            if edge <= threshold:
                continue
            if ps_fair is not None and ps_fair[o] <= fair[o]:
                continue
            value_bets.append((o, edge))

        rows.append({
            "Date": row["Date"],
            "League": row["league"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "B365H": float(row["B365H"]),
            "B365D": float(row["B365D"]),
            "B365A": float(row["B365A"]),
            "ModelH": probs["H"],
            "ModelD": probs["D"],
            "ModelA": probs["A"],
            "FairH": fair["H"],
            "FairD": fair["D"],
            "FairA": fair["A"],
            "ValueBets": value_bets,
        })
    return rows


def _print_predictions(fixture_features, y_proba, classes, threshold: float, fetched_at=None) -> None:
    from datetime import datetime
    outcome_label = {"H": "Home", "D": "Draw", "A": "Away"}

    has_pinnacle = all(c in fixture_features.columns for c in ("PSH", "PSD", "PSA"))
    pinnacle_note = " + Pinnacle filter" if has_pinnacle else ""
    fetch_str = fetched_at.strftime("%Y-%m-%d %H:%M") if fetched_at else "unknown"

    W = 100
    print(f"\n{'='*W}")
    print(f"UPCOMING FIXTURE PREDICTIONS")
    print(f"Odds fetched: {fetch_str}  |  threshold: {threshold:+.2f}{pinnacle_note}")
    print(f"⚠  Verify odds are still current before placing any bet")
    print(f"{'='*W}")

    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold)

    by_league: dict[str, list] = {}
    for r in pred_rows:
        by_league.setdefault(r["League"], []).append(r)

    for league, rows in sorted(by_league.items()):
        print(f"\n--- {league.upper()} ---")
        print(f"{'Date':<12} {'Home':<22} {'Away':<22} {'H%/D%/A%':>11}  {'B365 H / D / A (fetched)':>26}  Value bets")
        print("-" * W)
        for r in rows:
            date_str = r["Date"].strftime("%Y-%m-%d")
            prob_str = f"{r['ModelH']:.0%}/{r['ModelD']:.0%}/{r['ModelA']:.0%}"
            odds_str = f"{r['B365H']:.2f} / {r['B365D']:.2f} / {r['B365A']:.2f}"
            if r["ValueBets"]:
                vb_parts = [f"{outcome_label[o]}(+{e:.1%})" for o, e in r["ValueBets"]]
                value_str = ", ".join(vb_parts)
            else:
                value_str = "-"
            print(f"{date_str:<12} {r['HomeTeam']:<22} {r['AwayTeam']:<22} {prob_str:>11}  {odds_str:>26}  {value_str}")

    # --- Top picks summary ---
    all_bets: list[dict] = []
    for r in pred_rows:
        for outcome, edge in r["ValueBets"]:
            odds_map = {"H": r["B365H"], "D": r["B365D"], "A": r["B365A"]}
            all_bets.append({
                "Date": r["Date"].strftime("%Y-%m-%d"),
                "League": r["League"],
                "Home": r["HomeTeam"],
                "Away": r["AwayTeam"],
                "Bet": outcome_label[outcome],
                "Edge": edge,
                "Odds": odds_map[outcome],
            })
    all_bets.sort(key=lambda x: x["Edge"], reverse=True)
    top = all_bets[:10]

    print(f"\n{'='*W}")
    value_count = sum(1 for r in pred_rows if r["ValueBets"])
    print(f"Value bets found: {value_count} / {len(pred_rows)} fixtures  |  total individual bets: {len(all_bets)}")
    print(f"\n{'TOP 10 VALUE BETS (by edge)'}")
    print(f"{'#':<3} {'Date':<12} {'League':<5} {'Home':<22} {'Away':<22} {'Bet':<5} {'Edge':>6}  {'Odds':>6}")
    print("-" * W)
    for i, b in enumerate(top, 1):
        print(f"{i:<3} {b['Date']:<12} {b['League'].upper():<5} {b['Home']:<22} {b['Away']:<22} {b['Bet']:<5} {b['Edge']:>+5.1%}  {b['Odds']:>6.2f}")
    print(f"{'='*W}\n")


def _save_predictions_csv(fixture_features, y_proba, classes, threshold: float, fetched_at) -> None:
    """Save a timestamped CSV of all fixtures + odds + model probs for pre-bet verification."""
    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold)
    records = []
    for r in pred_rows:
        value_labels = "+".join(o for o, _ in r["ValueBets"]) if r["ValueBets"] else ""
        records.append({
            "fetched_at": fetched_at.strftime("%Y-%m-%d %H:%M"),
            "Date": r["Date"].strftime("%Y-%m-%d"),
            "League": r["League"],
            "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            "B365H": r["B365H"],
            "B365D": r["B365D"],
            "B365A": r["B365A"],
            "ModelH": round(r["ModelH"], 4),
            "ModelD": round(r["ModelD"], 4),
            "ModelA": round(r["ModelA"], 4),
            "FairH": round(r["FairH"], 4),
            "FairD": round(r["FairD"], 4),
            "FairA": round(r["FairA"], 4),
            "ValueBets": value_labels,
        })
    out = pd.DataFrame(records)
    ts = fetched_at.strftime("%Y%m%d_%H%M")
    path = Path(f"reports/predictions_{ts}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Predictions saved to {path}  (compare B365 odds before placing bets)")


def _save_predictions_html(pred_rows: list[dict], threshold: float, fetched_at) -> None:
    from src.evaluation.predictions_report import save_predictions_html
    ts = fetched_at.strftime("%Y%m%d_%H%M")
    path = Path(f"reports/predictions_{ts}.html")
    save_predictions_html(pred_rows, threshold, fetched_at, path)
    print(f"HTML report saved to {path}")


def _print_split_analysis(betting_results: pd.DataFrame, odds_test: pd.DataFrame) -> None:
    """Print ROI breakdowns by league and by tier matchup."""
    bets = betting_results.copy()
    bets["Date"] = pd.to_datetime(bets["Date"])
    odds_test = odds_test.copy()
    odds_test["Date"] = pd.to_datetime(odds_test["Date"])

    bets = bets.merge(
        odds_test[["HomeTeam", "AwayTeam", "Date", "league"]],
        on=["HomeTeam", "AwayTeam", "Date"], how="left",
    )

    # Per-league strength tiers (top/mid/bot = top-third/mid/bottom-third of that league)
    tiers: dict[str, str] = {}
    for lg in odds_test["league"].unique():
        lg_df = odds_test[odds_test["league"] == lg]
        ts: dict[str, list] = {}
        for _, row in lg_df.iterrows():
            total = 1/row["B365H"] + 1/row["B365D"] + 1/row["B365A"]
            ts.setdefault(row["HomeTeam"], []).append((1/row["B365H"]) / total)
            ts.setdefault(row["AwayTeam"], []).append((1/row["B365A"]) / total)
        avg = {t: float(np.mean(v)) for t, v in ts.items()}
        t33, t67 = np.percentile(list(avg.values()), 33), np.percentile(list(avg.values()), 67)
        for team, s in avg.items():
            tiers[team] = "top" if s >= t67 else ("bot" if s < t33 else "mid")

    bets["home_tier"] = bets["HomeTeam"].map(tiers)
    bets["away_tier"] = bets["AwayTeam"].map(tiers)

    LG = {"E0": "England", "D1": "Germany", "SP1": "Spain", "I1": "Italy",
          "F1": "France", "N1": "Netherlands", "P1": "Portugal"}

    def _roi(sub): return sub["profit"].sum() / len(sub) * 100 if len(sub) else float("nan")

    print("\n=== ROI BY LEAGUE ===")
    print(f"  {'League':>10}  {'Bets':>5}  {'ROI':>8}")
    print("  " + "-"*28)
    for lc, ln in LG.items():
        sub = bets[bets["league"] == lc]
        if len(sub) == 0:
            continue
        print(f"  {ln:>10}  {len(sub):>5}  {_roi(sub):>+7.2f}%")

    print("\n=== ROI BY TIER MATCHUP ===")
    print(f"  {'Matchup':>12}  {'Bets':>5}  {'ROI':>8}")
    print("  " + "-"*30)
    for ht in ["top", "mid", "bot"]:
        for at in ["top", "mid", "bot"]:
            sub = bets[(bets["home_tier"] == ht) & (bets["away_tier"] == at)]
            if len(sub) < 5:
                continue
            print(f"  {ht+' vs '+at:>12}  {len(sub):>5}  {_roi(sub):>+7.2f}%")

    print("\n=== ROI BY LEAGUE × TIER MATCHUP ===")
    for lc, ln in LG.items():
        lg_bets = bets[bets["league"] == lc]
        if len(lg_bets) == 0:
            continue
        print(f"\n  {ln}:")
        print(f"  {'Matchup':>12}  {'Bets':>5}  {'ROI':>8}")
        print("  " + "-"*28)
        for ht in ["top", "mid", "bot"]:
            for at in ["top", "mid", "bot"]:
                sub = lg_bets[(lg_bets["home_tier"] == ht) & (lg_bets["away_tier"] == at)]
                if len(sub) < 5:
                    continue
                print(f"  {ht+' vs '+at:>12}  {len(sub):>5}  {_roi(sub):>+7.2f}%")


def _run_backtest():
    if "--update" in sys.argv:
        print("Updating current season data...")
        from src.data.download import update_current_season
        update_current_season()

    threshold = _parse_threshold()
    kelly = _parse_kelly()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    per_league = _parse_per_league()
    binary = _parse_binary()
    monthly = _parse_monthly()

    if monthly:
        print("Running monthly walk-forward backtest (per-league, retrained each month)...")
        results = train_walkforward_monthly(df)
    else:
        if per_league and binary:
            mode = "3 binary models × 4 leagues = 12 models per test season"
        elif per_league:
            mode = "one multi-class model per league per test season (4 models)"
        elif binary:
            mode = "3 binary models per test season (one per outcome)"
        else:
            mode = "one multi-class model per test season"
        print(f"Running walk-forward backtest ({mode})...")
        results = train_walkforward(df, per_league=per_league, binary_outcomes=binary)

    eval_df = results["odds_test"].copy()
    eval_df["y_true"] = results["y_test"].values

    betting_results = compute_value_betting_results(
        eval_df,
        results["y_proba"],
        results["classes"],
        threshold=threshold,
        kelly_fraction=kelly,
    )
    roi = compute_roi(betting_results)
    stability = compute_stability(betting_results)
    accuracy = results["accuracy"]

    n_matches = len(eval_df)
    n_bets = len(betting_results)
    sizing = f"Kelly x{kelly}" if kelly > 0.0 else "flat"
    t_stat = stability * (n_bets ** 0.5)
    print("\n=== BACKTEST RESULTS ===")
    print(f"Accuracy:  {accuracy:.3f}  (on all {n_matches} test matches)")
    print(f"Threshold: {threshold:+.3f}  (minimum edge over fair odds)")
    print(f"Sizing:    {sizing}")
    print(f"Bets:      {n_bets} / {n_matches} matches ({n_bets / n_matches:.1%})")
    print(f"ROI:       {roi:+.2f}%")
    print(f"Stability: {stability:.4f}")
    print(f"t-stat:    {t_stat:+.2f}  (need |t| > 2 for significance; ROI indistinct from 0 until then)")

    _print_split_analysis(betting_results, eval_df)

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


def _run_compare_vig():
    """Run the walk-forward backtest twice and print a side-by-side comparison:
    'fair' baseline (vig-stripped, current default) vs 'raw' baseline (1/odds, EV-correct)."""
    threshold = _parse_threshold()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")
    print("Running walk-forward backtest (one model per test season)...")
    results = train_walkforward(df, per_league=False)

    eval_df = results["odds_test"].copy()
    eval_df["y_true"] = results["y_test"].values

    fair_bets = compute_value_betting_results(
        eval_df, results["y_proba"], results["classes"],
        threshold=threshold, edge_baseline="fair",
    )
    raw_bets = compute_value_betting_results(
        eval_df, results["y_proba"], results["classes"],
        threshold=threshold, edge_baseline="raw",
    )

    n_matches = len(eval_df)
    W = 72

    def _stats(bets):
        if bets.empty:
            return {"n": 0, "roi": float("nan"), "stab": float("nan"), "profit": 0.0}
        return {
            "n": len(bets),
            "roi": compute_roi(bets),
            "stab": compute_stability(bets),
            "profit": bets["profit"].sum(),
        }

    fs, rs = _stats(fair_bets), _stats(raw_bets)

    print(f"\n{'='*W}")
    print(f"VIG BASELINE COMPARISON  |  threshold: {threshold:+.2f}  |  test matches: {n_matches}")
    print(f"{'='*W}")
    print(f"{'Metric':<22}  {'fair (vig-stripped)':>20}  {'raw (1/odds)':>20}")
    print("-" * W)
    print(f"{'Bets placed':<22}  {fs['n']:>20}  {rs['n']:>20}")
    print(f"{'Bet rate':<22}  {fs['n']/n_matches:>19.1%}  {rs['n']/n_matches:>19.1%}")
    print(f"{'Total profit (units)':<22}  {fs['profit']:>+20.2f}  {rs['profit']:>+20.2f}")
    print(f"{'ROI':<22}  {fs['roi']:>+19.2f}%  {rs['roi']:>+19.2f}%")
    print(f"{'Stability':<22}  {fs['stab']:>20.4f}  {rs['stab']:>20.4f}")
    print(f"{'='*W}")

    # Per-threshold sweep to show breakeven point
    print(f"\n{'THRESHOLD SWEEP — ROI at each minimum-edge cutoff'}")
    print(f"{'Threshold':<12}  {'fair bets':>10}  {'fair ROI':>10}  {'raw bets':>10}  {'raw ROI':>10}")
    print("-" * 58)
    for t in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        fb = compute_value_betting_results(
            eval_df, results["y_proba"], results["classes"],
            threshold=t, edge_baseline="fair",
        )
        rb = compute_value_betting_results(
            eval_df, results["y_proba"], results["classes"],
            threshold=t, edge_baseline="raw",
        )
        f_roi = f"{compute_roi(fb):+.2f}%" if not fb.empty else "  n/a"
        r_roi = f"{compute_roi(rb):+.2f}%" if not rb.empty else "  n/a"
        print(f"{t:>+.2f}{'':8}  {len(fb):>10}  {f_roi:>10}  {len(rb):>10}  {r_roi:>10}")

    print(f"\nNote: 'raw' bets ⊆ 'fair' bets — raw is strictly more conservative.\n"
          f"Bets flagged by 'fair' but not 'raw' are negative-EV at these actual odds.")

    # --- Per-league breakdown at base threshold ---
    LG = {"E0": "England", "D1": "Germany", "SP1": "Spain", "I1": "Italy",
          "F1": "France", "N1": "Netherlands", "P1": "Portugal"}

    def _league_roi(bets, eval_df):
        if bets.empty:
            return {}
        merged = bets.merge(
            eval_df[["HomeTeam", "AwayTeam", "Date", "league"]],
            on=["HomeTeam", "AwayTeam", "Date"], how="left",
        )
        out = {}
        for lc, ln in LG.items():
            sub = merged[merged["league"] == lc]
            if len(sub) == 0:
                continue
            out[ln] = (len(sub), sub["profit"].sum() / len(sub) * 100)
        return out

    fair_lg = _league_roi(fair_bets, eval_df)
    raw_lg = _league_roi(raw_bets, eval_df)
    all_leagues = sorted(set(list(fair_lg.keys()) + list(raw_lg.keys())))

    print(f"\nPER-LEAGUE ROI  (threshold: {threshold:+.2f})")
    print(f"{'League':<14}  {'fair bets':>10}  {'fair ROI':>10}  {'raw bets':>10}  {'raw ROI':>10}")
    print("-" * 60)
    for ln in all_leagues:
        fn, fr = fair_lg.get(ln, (0, float("nan")))
        rn, rr = raw_lg.get(ln, (0, float("nan")))
        f_roi = f"{fr:>+.2f}%" if fn else "   n/a"
        r_roi = f"{rr:>+.2f}%" if rn else "   n/a"
        print(f"{ln:<14}  {fn:>10}  {f_roi:>10}  {rn:>10}  {r_roi:>10}")

    # --- Side-by-side profit charts ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
    for ax, bets, label, colour in [
        (axes[0], fair_bets, "fair (vig-stripped)", "#1565c0"),
        (axes[1], raw_bets,  "raw (1/odds)",        "#6a1b9a"),
    ]:
        if bets.empty:
            ax.set_title(f"{label} — no bets")
            continue
        ax.plot(bets["Date"], bets["cumulative_profit"], color=colour, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(bets["Date"], bets["cumulative_profit"], 0,
                        where=bets["cumulative_profit"] >= 0, alpha=0.15, color="#43a047")
        ax.fill_between(bets["Date"], bets["cumulative_profit"], 0,
                        where=bets["cumulative_profit"] < 0, alpha=0.15, color="#e53935")
        roi_val = compute_roi(bets)
        ax.set_title(f"{label}  |  {len(bets)} bets  |  ROI {roi_val:+.2f}%")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative profit (units)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Vig baseline comparison  (threshold {threshold:+.2f})", fontsize=13)
    fig.tight_layout()
    chart_path = Path("reports/vig_comparison.png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nProfit charts saved to {chart_path}")


def run_pipeline():
    if "--predict" in sys.argv:
        _run_predict()
    elif "--compare-vig" in sys.argv:
        _run_compare_vig()
    else:
        _run_backtest()


if __name__ == "__main__":
    run_pipeline()
