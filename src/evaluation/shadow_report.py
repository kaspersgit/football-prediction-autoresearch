"""HTML monitoring report for the immutable forward shadow ledgers."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from src.evaluation.shadow import validate_shadow_ledgers
from src.evaluation.uncertainty import weekly_roi_interval

_FIXTURE_COLUMNS = ["fixture_date", "league", "home_team", "away_team"]


def _display(value: object) -> str:
    """Return an HTML-safe table value, using a dash for missing data."""
    if value is None or pd.isna(value):
        return "—"
    return escape(str(value))


def _percentage(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{value:+.2f}%"


def _is_value_bet(values: pd.Series) -> pd.Series:
    """Handle boolean values that have been round-tripped through CSV."""
    return values.astype(str).str.lower().eq("true")


def _breakdown_table(bets: pd.DataFrame, column: str, title: str) -> str:
    if bets.empty:
        return f"<h2>{title}</h2><p>No settled qualifying bets yet.</p>"

    rows = []
    for name, group in bets.groupby(column, dropna=False, sort=True):
        bet_count = len(group)
        profit = group["profit"].sum()
        roi = profit / bet_count * 100.0
        mean_clv = group["closing_line_value"].mean() * 100.0
        rows.append(
            "<tr>"
            f"<td>{_display(name)}</td>"
            f"<td>{bet_count}</td>"
            f"<td>{profit:+.2f}</td>"
            f"<td>{_percentage(roi)}</td>"
            f"<td>{_percentage(mean_clv)}</td>"
            "</tr>"
        )
    return f"""
<h2>{title}</h2>
<table>
  <thead><tr><th>{escape(column.replace('_', ' ').title())}</th><th>Bets</th><th>Profit</th><th>Captured-price ROI</th><th>Mean CLV</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def _cumulative_profit_table(bets: pd.DataFrame) -> str:
    if bets.empty:
        return "<h2>Cumulative profit</h2><p>No settled qualifying bets yet.</p>"

    daily_profit = (
        bets.assign(fixture_date=pd.to_datetime(bets["fixture_date"]).dt.date)
        .groupby("fixture_date", sort=True)["profit"]
        .sum()
        .reset_index()
    )
    daily_profit["cumulative_profit"] = daily_profit["profit"].cumsum()
    rows = "".join(
        "<tr>"
        f"<td>{day.fixture_date.isoformat()}</td>"
        f"<td>{day.profit:+.2f}</td>"
        f"<td>{day.cumulative_profit:+.2f}</td>"
        "</tr>"
        for day in daily_profit.itertuples(index=False)
    )
    return f"""
<h2>Cumulative profit</h2>
<p>Flat-stake cumulative profit from settled qualifying bets at captured prices.</p>
<table>
  <thead><tr><th>Fixture date</th><th>Daily profit</th><th>Cumulative profit</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _weekly_interval_metric(bets: pd.DataFrame) -> str:
    """Render the reproducible weekly ROI interval only when it is meaningful."""
    if bets.empty:
        return ""
    interval = weekly_roi_interval(
        bets.rename(columns={"fixture_date": "Date"}).assign(stake=1.0)
    )
    if interval is None:
        return ""
    lower, upper = interval
    return (
        '<div class="metric"><strong>'
        f"{lower:+.2f}% to {upper:+.2f}%"
        "</strong>Weekly 95% ROI interval</div>"
    )


def generate_shadow_report(
    predictions: pd.DataFrame, settlements: pd.DataFrame, output_path: Path
) -> Path:
    """Generate an informational, monitoring-only report from shadow ledgers."""
    validate_shadow_ledgers(predictions, settlements)
    joined = predictions.merge(
        settlements,
        on="prediction_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_settlement"),
    )
    settled_ids = set(settlements["prediction_id"]).intersection(predictions["prediction_id"])
    fixture_status = (
        joined.assign(_is_settled=joined["prediction_id"].isin(settled_ids))
        .groupby(_FIXTURE_COLUMNS, dropna=False)["_is_settled"]
        .all()
    )
    prediction_run_count = predictions["run_id"].nunique()
    settled_fixture_count = int(fixture_status.sum())
    pending_fixture_count = int((~fixture_status).sum())

    # A fixture+outcome can be predicted more than once a day (repeat manual
    # triggers, or a scheduled run alongside one). Only the day's earliest
    # snapshot counts toward the metrics below, so a re-run never double-counts
    # the same real match; every snapshot still stays in the ledger for audit.
    canonical_ids = set(
        predictions.assign(_fetched_at=pd.to_datetime(predictions["fetched_at"], utc=True))
        .sort_values("_fetched_at")
        .drop_duplicates(subset=[*_FIXTURE_COLUMNS, "outcome"], keep="first")["prediction_id"]
    )
    settled_qualifying = joined.loc[
        joined["prediction_id"].isin(settled_ids)
        & joined["prediction_id"].isin(canonical_ids)
        & _is_value_bet(joined["is_value_bet"])
    ].copy()

    bet_count = len(settled_qualifying)
    profit = settled_qualifying["profit"].sum() if bet_count else 0.0
    roi = profit / bet_count * 100.0 if bet_count else None
    win_rate = settled_qualifying["is_winner"].mean() * 100.0 if bet_count else None
    mean_clv = settled_qualifying["closing_line_value"].mean() * 100.0 if bet_count else None
    median_clv = settled_qualifying["closing_line_value"].median() * 100.0 if bet_count else None
    weekly_interval = _weekly_interval_metric(settled_qualifying)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Forward shadow evaluation</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #222; }}
    h1, h2 {{ color: #1a237e; }}
    .notice {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 14px 18px; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }}
    .metric {{ background: #f5f5f5; border-radius: 6px; min-width: 155px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 1.35em; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #e8eaf6; }}
    .back-link {{ display: inline-block; margin-bottom: 16px; color: #1565c0; }}
  </style>
</head>
<body>
  <a class="back-link" href="index.html">&larr; Back to index</a>
  <h1>Forward shadow evaluation</h1>
  <div class="notice"><strong>Informational — monitoring only.</strong> Sparse data is informational and is not a release gate. This report records hypothetical flat-stake execution at captured quotes, not accepted real wagers. Shadow data must not select features, tune thresholds, or choose production leagues.</div>
  <h2>Monitoring summary</h2>
  <div class="metrics">
    <div class="metric"><strong>{prediction_run_count}</strong>Prediction runs</div>
    <div class="metric"><strong>{settled_fixture_count}</strong>Settled fixtures</div>
    <div class="metric"><strong>{pending_fixture_count}</strong>Pending fixtures</div>
    <div class="metric"><strong>{bet_count}</strong>Settled qualifying value bets</div>
    <div class="metric"><strong>{profit:+.2f}</strong>Captured-price profit</div>
    <div class="metric"><strong>{_percentage(roi)}</strong>Captured-price ROI</div>
    <div class="metric"><strong>{_percentage(win_rate)}</strong>Win rate</div>
    <div class="metric"><strong>{_percentage(mean_clv)}</strong>Mean closing-line value</div>
    <div class="metric"><strong>{_percentage(median_clv)}</strong>Median closing-line value</div>
    {weekly_interval}
  </div>
  <p>Closing-line value compares the captured best price with the result-file closing-price proxy; it is unavailable when that proxy is missing.</p>
  {_breakdown_table(settled_qualifying, "league", "League breakdown")}
  {_breakdown_table(settled_qualifying, "model_commit", "Model commit breakdown")}
  {_cumulative_profit_table(settled_qualifying)}
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path
