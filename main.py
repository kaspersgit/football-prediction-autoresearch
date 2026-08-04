#!/usr/bin/env python3
"""
Main pipeline: data → model → evaluation → HTML report + profit chart.

Evaluation: all supported leagues. Production inference: explicit allowlist in src/config.py.

Modes:
  python main.py                    # backtest on last 2 seasons (default settings)
  python main.py --per-league       # one model per league (default recommended)
  python main.py --threshold 0.05   # override minimum edge filter (default: 0.03)
  python main.py --max-odds 5.0     # override max B365 odds filter (default: 5.0)
  python main.py --predict          # train on all data, predict upcoming fixtures
  python main.py --update           # re-download latest season results, then backtest
  python main.py --monthly          # monthly walk-forward retrain (experimental)

Default staking: flat 1 unit per bet
  Threshold: 0.03  (3% minimum edge over B365 fair price)
  Max odds:  5.0   (B365 odds above 5.0 excluded; model signal degrades above this)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DEFAULT_MAX_EDGE,
    DEFAULT_MAX_ODDS,
    DEFAULT_MAX_OVERROUND,
    EXCLUDED_BETTING_LEAGUES,
    LEAGUE_NAMES,
    PRODUCTION_LEAGUES,
    SUPPORTED_LEAGUES,
    filter_production_fixtures,
)
from src.data.loader import load_all_data
from src.evaluation.artifacts import save_bet_artifacts
from src.evaluation.metrics import compute_roi, compute_stability, compute_value_betting_results
from src.evaluation.report import generate_report
from src.evaluation.threshold_selector import select_league_thresholds
from src.model.features import FEATURE_COLS, build_fixture_features
from src.model.train import _CLASSES as _TRAIN_CLASSES
from src.model.train import (
    train_on_all_data_per_league,
    train_walkforward,
    train_walkforward_monthly,
)

_THRESHOLD_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]


def _parse_threshold() -> float:
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return 0.03  # validated default: 3% minimum edge


def _parse_max_odds() -> float:
    for i, arg in enumerate(sys.argv):
        if arg == "--max-odds" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return DEFAULT_MAX_ODDS


def _parse_max_edge() -> float:
    for i, arg in enumerate(sys.argv):
        if arg == "--max-edge" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return DEFAULT_MAX_EDGE


def _parse_min_season_games() -> int:
    for i, arg in enumerate(sys.argv):
        if arg == "--min-season-games" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 0


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


def _load_historical_bets() -> pd.DataFrame | None:
    """Load production backtest bets for the live prediction report, if present."""
    bets_path = Path("reports/backtest_bets.csv")
    if bets_path.exists():
        return pd.read_csv(bets_path, parse_dates=["Date"])
    return None


def _save_empty_predictions_report(threshold: float, fetched_at, reason: str) -> None:
    """Write empty prediction artifacts so CI Pages staging always has an HTML file.

    Early exits (no fixtures / allowlist / missing models) must still produce
    ``reports/predictions_*.html``; otherwise the GitHub Actions deploy step fails.
    """
    print(reason)
    historical_bets = _load_historical_bets()
    ts = fetched_at.strftime("%Y%m%d_%H%M")
    csv_path = Path(f"reports/predictions_{ts}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "fetched_at", "Date", "League", "HomeTeam", "AwayTeam",
            "B365H", "B365D", "B365A",
            "ModelH", "ModelD", "ModelA",
            "FairH", "FairD", "FairA",
            "ValueBets",
        ]
    ).to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}  (empty — {reason})")
    _save_predictions_html([], threshold, fetched_at, historical_bets=historical_bets)


def _run_predict():
    from datetime import datetime

    from src.data.download import download_fixtures, update_current_season
    from src.data.loader import load_fixtures

    threshold = _parse_threshold()
    fetched_at = datetime.now()

    import json as _json
    _lt_path = Path("models/league_thresholds.json")
    league_thresholds: dict | None = _json.loads(_lt_path.read_text()) if _lt_path.exists() else None
    if league_thresholds:
        print("Using per-league thresholds: " +
              ", ".join(f"{lg}={v:+.2f}" for lg, v in sorted(league_thresholds.items())))
    else:
        print(f"No league_thresholds.json found — using global threshold {threshold:+.2f}")

    print("Updating latest season results...")
    update_current_season()

    print("Downloading upcoming fixtures...")
    download_fixtures()

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("Training per-league models on full dataset...")
    models = train_on_all_data_per_league(df)

    print("Loading fixtures...")
    fixtures_df = load_fixtures()
    print(f"Found {len(fixtures_df)} upcoming fixtures in tracked leagues")

    print("Building fixture features...")
    fixture_features = build_fixture_features(df, fixtures_df)
    if fixture_features.empty:
        _save_empty_predictions_report(
            threshold, fetched_at,
            "No fixtures could be featurised (teams may lack sufficient history).",
        )
        return

    dropped = len(fixtures_df) - len(fixture_features)
    if dropped:
        print(f"  {dropped} fixture(s) dropped — teams with < {5} games history")

    fixture_features = filter_production_fixtures(fixture_features)
    if fixture_features.empty:
        _save_empty_predictions_report(
            threshold, fetched_at,
            "No upcoming fixtures are in the production market allowlist.",
        )
        return

    modelled_leagues = set(models)
    missing_models = sorted(set(fixture_features["league"]) - modelled_leagues)
    if missing_models:
        print("Skipping fixtures without trained models: " + ", ".join(missing_models))
        fixture_features = fixture_features[
            fixture_features["league"].isin(modelled_leagues)
        ].reset_index(drop=True)
    if fixture_features.empty:
        _save_empty_predictions_report(
            threshold, fetched_at,
            "No production fixtures have a trained model.",
        )
        return

    # Per-league inference: use each league's own model
    X_fix = fixture_features[FEATURE_COLS]
    classes = list(_TRAIN_CLASSES)
    y_proba = np.zeros((len(X_fix), len(classes)))
    for league, model in models.items():
        mask = (fixture_features["league"] == league).values
        if mask.sum() == 0:
            continue
        proba = model.predict_proba(X_fix[mask])
        model_classes = list(model.classes_)
        col_order = [model_classes.index(c) for c in classes if c in model_classes]
        y_proba[mask] = proba[:, col_order]

    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold,
                                       league_thresholds=league_thresholds)

    historical_bets = _load_historical_bets()

    _print_predictions(fixture_features, y_proba, classes, threshold, fetched_at,
                       league_thresholds=league_thresholds)
    _save_predictions_csv(fixture_features, y_proba, classes, threshold, fetched_at,
                          league_thresholds=league_thresholds)
    _save_predictions_html(pred_rows, threshold, fetched_at, historical_bets=historical_bets)



_PREDICT_MAX_ODDS = DEFAULT_MAX_ODDS
_PREDICT_MAX_EDGE = DEFAULT_MAX_EDGE
_PREDICT_MAX_OVERROUND = DEFAULT_MAX_OVERROUND


def _build_prediction_rows(
    fixture_features, y_proba, classes, threshold: float,
    league_thresholds: dict | None = None,
) -> list[dict]:
    """Build a list of per-fixture prediction dicts (used by both print and CSV export)."""
    fixture_features = fixture_features.reset_index(drop=True)
    rows = []
    for i, row in fixture_features.iterrows():
        probs = {c: float(y_proba[i, j]) for j, c in enumerate(classes)}
        b365h, b365d, b365a = float(row["B365H"]), float(row["B365D"]), float(row["B365A"])
        raw_implied = {"H": 1.0 / b365h, "D": 1.0 / b365d, "A": 1.0 / b365a}
        total_implied = sum(raw_implied.values())
        fair = {o: raw_implied[o] / total_implied for o in raw_implied}

        # Skip high-vig markets (same filter as backtest)
        overround = total_implied - 1.0
        if overround > _PREDICT_MAX_OVERROUND:
            continue

        league = row.get("league", "")
        t = (league_thresholds or {}).get(league, threshold)

        value_bets = []
        for o in ["H", "D", "A"]:
            edge = probs[o] - fair[o]
            b365_odds = {"H": b365h, "D": b365d, "A": b365a}[o]
            if edge <= t:
                continue
            if edge > _PREDICT_MAX_EDGE:
                continue
            if b365_odds > _PREDICT_MAX_ODDS:
                continue
            value_bets.append((o, edge))

        def _safe_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        rows.append({
            "Date": row["Date"],
            "League": row["league"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "B365H": float(row["B365H"]),
            "B365D": float(row["B365D"]),
            "B365A": float(row["B365A"]),
            "CustomMaxH": _safe_float(row.get("CustomMaxH")),
            "CustomMaxD": _safe_float(row.get("CustomMaxD")),
            "CustomMaxA": _safe_float(row.get("CustomMaxA")),
            "CustomMaxBkH": row.get("CustomMaxBkH", ""),
            "CustomMaxBkD": row.get("CustomMaxBkD", ""),
            "CustomMaxBkA": row.get("CustomMaxBkA", ""),
            "ModelH": probs["H"],
            "ModelD": probs["D"],
            "ModelA": probs["A"],
            "FairH": fair["H"],
            "FairD": fair["D"],
            "FairA": fair["A"],
            "ValueBets": value_bets,
        })
    return rows


def _print_predictions(fixture_features, y_proba, classes, threshold: float, fetched_at=None,
                       league_thresholds: dict | None = None) -> None:
    outcome_label = {"H": "Home", "D": "Draw", "A": "Away"}

    fetch_str = fetched_at.strftime("%Y-%m-%d %H:%M") if fetched_at else "unknown"

    W = 100
    print(f"\n{'='*W}")
    print("UPCOMING FIXTURE PREDICTIONS")
    print(f"Odds fetched: {fetch_str}  |  threshold: {threshold:+.2f}")
    print("⚠  Verify odds are still current before placing any bet")
    print(f"{'='*W}")

    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold,
                                       league_thresholds=league_thresholds)

    by_league: dict[str, list] = {}
    for r in pred_rows:
        by_league.setdefault(r["League"], []).append(r)

    for league, rows in sorted(by_league.items()):
        print(f"\n--- {league.upper()} ---")
        hdr = f"{'Date':<12} {'Home':<22} {'Away':<22} {'H%/D%/A%':>11}  {'B365 H/D/A':>16}  Value bets"
        print(hdr)
        print("-" * W)
        for r in rows:
            date_str = r["Date"].strftime("%Y-%m-%d")
            prob_str = f"{r['ModelH']:.0%}/{r['ModelD']:.0%}/{r['ModelA']:.0%}"
            odds_str = f"{r['B365H']:.2f}/{r['B365D']:.2f}/{r['B365A']:.2f}"
            line = f"{date_str:<12} {r['HomeTeam']:<22} {r['AwayTeam']:<22} {prob_str:>11}  {odds_str:>16}"
            if r["ValueBets"]:
                vb_parts = [f"{outcome_label[o]}(+{e:.1%})" for o, e in r["ValueBets"]]
                line += "  " + ", ".join(vb_parts)
            else:
                line += "  -"
            print(line)

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


def _save_predictions_csv(fixture_features, y_proba, classes, threshold: float, fetched_at,
                          league_thresholds: dict | None = None) -> None:
    """Save a timestamped CSV of all fixtures + odds + model probs for pre-bet verification."""
    pred_rows = _build_prediction_rows(fixture_features, y_proba, classes, threshold,
                                       league_thresholds=league_thresholds)
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


def _save_predictions_html(pred_rows: list[dict], threshold: float, fetched_at,
                            historical_bets=None) -> None:
    from src.evaluation.predictions_report import save_predictions_html
    ts = fetched_at.strftime("%Y%m%d_%H%M")
    path = Path(f"reports/predictions_{ts}.html")
    profit_curve = Path("reports/profit_curve.png")
    save_predictions_html(pred_rows, threshold, fetched_at, path,
                          profit_curve_path=profit_curve if profit_curve.exists() else None,
                          historical_bets=historical_bets)
    print(f"HTML report saved to {path}")


def _print_edge_analysis(eval_df, y_proba, classes, threshold, max_odds) -> None:
    """Show ROI/bets/accuracy per edge bucket and simulate different max_edge caps."""
    CAPS = [0.10, 0.15, 0.20, 0.25, 0.30, float("inf")]
    BASE_KWARGS = dict(
        threshold=threshold,
        inv_odds_factor=20.0,
        min_stake=3.0,
        max_odds=max_odds,
    )

    # Build the uncapped per-bet edge table for bucket analysis.
    outcomes = list(classes)
    outcome_map = {"H": "B365H", "D": "B365D", "A": "B365A"}
    rows = []
    for i, row in eval_df.reset_index(drop=True).iterrows():
        raw = {o: 1.0 / float(row[outcome_map[o]]) for o in outcomes}
        total = sum(raw.values())
        fair = {o: raw[o] / total for o in raw}
        for j, outcome in enumerate(outcomes):
            mp = float(y_proba[i, j])
            edge = mp - fair[outcome]
            if edge > threshold and float(row[outcome_map[outcome]]) <= max_odds:
                rows.append({
                    "edge": edge,
                    "outcome": outcome,
                    "y_true": row["y_true"],
                    "odds": float(row[outcome_map[outcome]]),
                    "month": pd.to_datetime(row["Date"]).month,
                    "league": row.get("league", ""),
                })
    all_bets_df = pd.DataFrame(rows)

    BUCKETS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
               (0.20, 0.25), (0.25, 0.30), (0.30, 1.0)]

    print("\n=== EDGE DISTRIBUTION (all potential bets before max-edge cap) ===")
    print(f"  {'Edge bucket':>14}  {'Bets':>6}  {'Win%':>6}  {'Avg odds':>9}  {'Flat ROI':>9}")
    print("  " + "-" * 50)
    for lo, hi in BUCKETS:
        sub = all_bets_df[(all_bets_df["edge"] >= lo) & (all_bets_df["edge"] < hi)]
        if len(sub) == 0:
            continue
        win_rate = (sub["outcome"] == sub["y_true"]).mean()
        avg_odds = sub["odds"].mean()
        flat_roi = ((sub["outcome"] == sub["y_true"]) * (sub["odds"] - 1) +
                    (sub["outcome"] != sub["y_true"]) * -1).mean() * 100
        print(f"  {lo:.0%} – {hi:.0%}       {len(sub):>6}  {win_rate:>5.1%}  {avg_odds:>9.2f}  {flat_roi:>+8.2f}%")

    # Timing of high-edge bets
    high_edge = all_bets_df[all_bets_df["edge"] >= 0.20]
    if len(high_edge) > 0:
        month_counts = high_edge.groupby("month").size()
        months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        print("\n  High-edge bets (≥20%) by month: "
              + "  ".join(f"{months[m]}:{n}" for m, n in month_counts.items()))

    # Cap sweep: ROI and bet count at each cap
    print("\n=== MAX-EDGE CAP SWEEP ===")
    print(f"  {'Cap':>8}  {'Bets':>6}  {'Excl.':>6}  {'ROI':>9}  {'Stability':>10}  {'t-stat':>8}")
    print("  " + "-" * 56)
    base_bets = compute_value_betting_results(eval_df, y_proba, classes, **BASE_KWARGS)
    base_n = len(base_bets)
    for cap in CAPS:
        bets = compute_value_betting_results(eval_df, y_proba, classes, max_edge=cap, **BASE_KWARGS)
        roi = compute_roi(bets) if not bets.empty else float("nan")
        stab = compute_stability(bets) if not bets.empty else float("nan")
        tstat = stab * (len(bets) ** 0.5) if not bets.empty else float("nan")
        excl = base_n - len(bets)
        cap_label = f"≤{cap:.0%}" if cap < float("inf") else "none"
        print(f"  {cap_label:>8}  {len(bets):>6}  {excl:>6}  {roi:>+8.2f}%  {stab:>10.4f}  {tstat:>+8.2f}")


def _print_split_analysis(betting_results: pd.DataFrame, odds_test: pd.DataFrame) -> None:
    """Print ROI breakdowns by league and by tier matchup."""
    bets = betting_results.copy()
    bets["Date"] = pd.to_datetime(bets["Date"])
    odds_test = odds_test.copy()
    odds_test["Date"] = pd.to_datetime(odds_test["Date"])

    if "league" not in bets.columns:
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

    LG = LEAGUE_NAMES

    def _roi(sub): return sub["profit"].sum() / sub["stake"].sum() * 100 if len(sub) else float("nan")

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

    threshold        = _parse_threshold()
    kelly            = _parse_kelly()
    max_odds         = _parse_max_odds()
    max_edge         = _parse_max_edge()
    min_season_games = _parse_min_season_games()

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
            mode = f"3 binary models × {len(SUPPORTED_LEAGUES)} leagues per test season"
        elif per_league:
            mode = f"one multi-class model per league per test season ({len(SUPPORTED_LEAGUES)} models)"
        elif binary:
            mode = "3 binary models per test season (one per outcome)"
        else:
            mode = "one multi-class model per test season"
        print(f"Running walk-forward backtest ({mode})...")
        results = train_walkforward(df, per_league=per_league, binary_outcomes=binary)

    eval_df = results["odds_test"].copy()
    eval_df["y_true"] = results["y_test"].values

    # Staking: flat 1 unit per bet (Kelly available via --kelly flag).
    inv_odds_factor = 0.0
    min_stake       = 1.0

    if "season_results" in results:
        # Per-season threshold calibration:
        # For each test season, calibrate per-league thresholds from all PRIOR test seasons,
        # then apply them. Season 1 (no prior data) uses the global default threshold.
        prior_season_data: list[dict] = []
        per_season_chunks: list[pd.DataFrame] = []
        final_threshold_map: dict[str, float] = {lg: threshold for lg in SUPPORTED_LEAGUES}

        for sr in results["season_results"]:
            if prior_season_data:
                final_threshold_map = select_league_thresholds(
                    prior_season_data,
                    leagues=SUPPORTED_LEAGUES,
                    grid=_THRESHOLD_GRID,
                    max_odds=max_odds,
                    max_overround=DEFAULT_MAX_OVERROUND,
                    max_edge=max_edge,
                )

            leagues_in_season = set(sr["eval_df"]["league"].unique())
            season_chunks: list[pd.DataFrame] = []
            for league in SUPPORTED_LEAGUES:
                if league not in PRODUCTION_LEAGUES:
                    continue
                if league not in leagues_in_season:
                    continue
                league_threshold = final_threshold_map.get(league, threshold)
                skip_all_but_this = set(SUPPORTED_LEAGUES) - {league}
                lg_bets = compute_value_betting_results(
                    sr["eval_df"],
                    sr["y_proba"],
                    sr["classes"],
                    threshold=league_threshold,
                    kelly_fraction=kelly,
                    inv_odds_factor=inv_odds_factor,
                    min_stake=min_stake,
                    max_odds=max_odds,
                    skip_leagues=skip_all_but_this,
                    max_edge=max_edge,
                    min_season_games=min_season_games,
                    max_overround=DEFAULT_MAX_OVERROUND,
                )
                season_chunks.append(lg_bets)

            prior_season_data.append(sr)
            if season_chunks:
                per_season_chunks.append(
                    pd.concat(season_chunks).sort_values("Date").reset_index(drop=True)
                )

        if per_season_chunks:
            production_results = (
                pd.concat(per_season_chunks).sort_values("Date").reset_index(drop=True)
            )
            production_results["cumulative_profit"] = production_results["profit"].cumsum()
        else:
            production_results = pd.DataFrame(
                columns=["Date", "HomeTeam", "AwayTeam", "league", "profit",
                         "cumulative_profit"]
            )
    else:
        # Monthly mode — no season_results; use global threshold unchanged
        production_results = compute_value_betting_results(
            eval_df,
            results["y_proba"],
            results["classes"],
            threshold=threshold,
            kelly_fraction=kelly,
            inv_odds_factor=inv_odds_factor,
            min_stake=min_stake,
            max_odds=max_odds,
            skip_leagues=EXCLUDED_BETTING_LEAGUES,
            max_edge=max_edge,
            min_season_games=min_season_games,
            max_overround=DEFAULT_MAX_OVERROUND,
        )
        final_threshold_map = {lg: threshold for lg in SUPPORTED_LEAGUES}

    # Research evaluation is deliberately independent from the production portfolio:
    # every observed supported league uses the fixed CLI threshold.
    evaluation_results = compute_value_betting_results(
        eval_df,
        results["y_proba"],
        results["classes"],
        threshold=threshold,
        kelly_fraction=kelly,
        inv_odds_factor=inv_odds_factor,
        min_stake=min_stake,
        max_odds=max_odds,
        max_edge=max_edge,
        min_season_games=min_season_games,
        max_overround=DEFAULT_MAX_OVERROUND,
    )

    roi = compute_roi(evaluation_results) if not evaluation_results.empty else float("nan")
    stability = (
        compute_stability(evaluation_results) if not evaluation_results.empty else float("nan")
    )
    accuracy = results["accuracy"]

    n_matches = len(eval_df)
    n_bets = len(evaluation_results)
    if kelly > 0.0:
        sizing = f"Kelly x{kelly}"
    else:
        sizing = "flat 1 unit"
    t_stat = stability * (n_bets ** 0.5)
    print("\n=== ALL-MARKET EVALUATION ===")
    season_filter_str = f"  |  Min season games: {min_season_games}" if min_season_games > 0 else ""
    print(f"Accuracy:  {accuracy:.3f}  (on all {n_matches} test matches)")
    print(f"Threshold: {threshold:+.3f}  |  Max odds: {max_odds:.1f}  |  Staking: {sizing}{season_filter_str}")
    print(f"Bets:      {n_bets} / {n_matches} matches ({n_bets / n_matches:.1%})")
    print(f"ROI:       {roi:+.2f}%")
    print(f"Stability: {stability:.4f}")
    print(f"t-stat:    {t_stat:+.2f}  (need |t| > 2 for significance; ROI indistinct from 0 until then)")
    print("League thresholds (last calibration): " +
          ", ".join(f"{lg}={v:+.2f}" for lg, v in sorted(final_threshold_map.items())))
    import json as _json
    _thresholds_path = Path("models/league_thresholds.json")
    _thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    _thresholds_path.write_text(_json.dumps(final_threshold_map, indent=2))
    print(f"League thresholds saved to {_thresholds_path}")

    _print_edge_analysis(eval_df, results["y_proba"], results["classes"], threshold, max_odds)
    _print_split_analysis(evaluation_results, eval_df)

    if not production_results.empty:
        production_roi = compute_roi(production_results)
        print(
            f"Production portfolio: {len(production_results)} bets | "
            f"ROI {production_roi:+.2f}% | leagues {', '.join(sorted(PRODUCTION_LEAGUES))}"
        )
        _save_profit_chart(production_results, Path("reports/profit_curve.png"))

    evaluation_path, bets_path = save_bet_artifacts(evaluation_results, production_results)
    print(f"All-market evaluation bets saved to {evaluation_path}")
    print(f"Production backtest bets saved to {bets_path}")

    league_lookup = eval_df[["Date", "HomeTeam", "AwayTeam", "league"]].drop_duplicates()

    # Build all_predictions: one row per (match × outcome) for embedding in the eval report.
    y_proba = results["y_proba"]
    classes = list(results["classes"])
    pred_rows = []
    for i, row in eval_df.iterrows():
        b365h = float(row["B365H"])
        b365d = float(row["B365D"])
        b365a = float(row["B365A"])
        total_implied = 1.0 / b365h + 1.0 / b365d + 1.0 / b365a
        raw_implied = {"H": 1.0 / b365h, "D": 1.0 / b365d, "A": 1.0 / b365a}
        fair = {k: v / total_implied for k, v in raw_implied.items()}
        odds_map = {"H": b365h, "D": b365d, "A": b365a}
        for j, outcome in enumerate(classes):
            model_prob = float(y_proba[i, j])
            implied_prob = fair[outcome]
            pred_rows.append({
                "date": row["Date"],
                "home": row.get("HomeTeam", ""),
                "away": row.get("AwayTeam", ""),
                "outcome": outcome,
                "model_prob": model_prob,
                "implied_prob": implied_prob,
                "odds": odds_map[outcome],
                "y_true": row["y_true"],
                "edge": model_prob - implied_prob,
            })
    all_predictions = pd.DataFrame(pred_rows)
    all_predictions = all_predictions.merge(
        league_lookup, left_on=["date", "home", "away"],
        right_on=["Date", "HomeTeam", "AwayTeam"], how="left"
    ).drop(columns=["Date", "HomeTeam", "AwayTeam"], errors="ignore")

    print("\nGenerating report...")
    generate_report(
        results_df=evaluation_results,
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        output_path=Path("reports/evaluation_report.html"),
        all_predictions=all_predictions,
        production_leagues=PRODUCTION_LEAGUES,
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

    print("\nNote: 'raw' bets ⊆ 'fair' bets — raw is strictly more conservative.\n"
          "Bets flagged by 'fair' but not 'raw' are negative-EV at these actual odds.")

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
