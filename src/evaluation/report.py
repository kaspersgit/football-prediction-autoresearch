import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Football Prediction Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; background: #f9f9f9; color: #222; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 2px solid #283593; padding-bottom: 6px; }}
  .metrics {{ display: flex; gap: 24px; margin: 24px 0; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 20px 32px; box-shadow: 0 2px 8px rgba(0,0,0,.1); text-align: center; flex: 1; }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; color: {roi_color}; }}
  .metric-card:nth-child(1) .value {{ color: #1565c0; }}
  .metric-card:nth-child(3) .value {{ color: #2e7d32; }}
  .metric-card .label {{ color: #777; font-size: 0.9em; margin-top: 4px; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 16px 0; }}
  .explanation {{ background: #e8eaf6; border-left: 4px solid #3949ab; padding: 12px 18px; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>Football Prediction Evaluation Report</h1>
<h2>Summary Metrics</h2>
<div class="metrics">
  <div class="metric-card"><div class="value">{accuracy:.1%}</div><div class="label">Accuracy</div></div>
  <div class="metric-card"><div class="value" style="color:{roi_color}">{roi:+.2f}%</div><div class="label">ROI (Return on Investment)</div></div>
  <div class="metric-card"><div class="value">{stability:.3f}</div><div class="label">Profit Stability (Sharpe-like)</div></div>
</div>
<div class="explanation">
  <b>ROI</b>: total profit / total staked × 100. Positive means profit over the test period.<br>
  <b>Stability</b>: mean profit per bet / std(profit per bet). Higher = more consistent returns. Above 0.05 is good.
</div>
<h2>Profit Distribution (sorted by profit)</h2>
<img src="data:image/png;base64,{bar_chart_b64}" alt="Profit distribution bar chart">
<h2>Cumulative Profit Over Time</h2>
<img src="data:image/png;base64,{cumulative_chart_b64}" alt="Cumulative profit chart">
<h2>Bet Details</h2>
<p>Total bets: {n_bets} &nbsp;|&nbsp; Correct: {n_correct} &nbsp;|&nbsp; Wrong: {n_wrong}</p>
</body>
</html>"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _bar_chart(profits: pd.Series) -> str:
    sorted_profits = profits.sort_values().values
    colors = ["#e53935" if p < 0 else "#43a047" for p in sorted_profits]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(sorted_profits)), sorted_profits, color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Bet (sorted by profit)")
    ax.set_ylabel("Profit (units)")
    ax.set_title("Per-Bet Profit Distribution (sorted)")
    green_patch = mpatches.Patch(color="#43a047", label="Win")
    red_patch = mpatches.Patch(color="#e53935", label="Loss")
    ax.legend(handles=[green_patch, red_patch])
    return _fig_to_b64(fig)


def _cumulative_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["cumulative_profit"], color="#1565c0", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(
        df["Date"], df["cumulative_profit"], 0,
        where=df["cumulative_profit"] >= 0, alpha=0.15, color="#43a047"
    )
    ax.fill_between(
        df["Date"], df["cumulative_profit"], 0,
        where=df["cumulative_profit"] < 0, alpha=0.15, color="#e53935"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit (units)")
    ax.set_title("Cumulative Profit Over Time")
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _all_predictions_script(all_predictions: "pd.DataFrame | None") -> str:
    if all_predictions is None or all_predictions.empty:
        return "<script>var ALL_BETS=[];</script>"
    needed = ["date", "league", "home", "away", "outcome", "model_prob",
              "implied_prob", "odds", "y_true", "edge"]
    cols = [c for c in needed if c in all_predictions.columns]
    df = all_predictions[cols].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    records = df.to_dict("records")
    return f"<script>\nvar ALL_BETS={json.dumps(records, separators=(',', ':'))};\n</script>"


def generate_report(
    results_df: pd.DataFrame,
    accuracy: float,
    roi: float,
    stability: float,
    output_path: Path,
    all_predictions: "pd.DataFrame | None" = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bar_b64 = _bar_chart(results_df["profit"])
    cum_b64 = _cumulative_chart(results_df)

    n_bets = len(results_df)
    n_correct = (results_df["y_true"] == results_df["y_pred"]).sum()
    n_wrong = n_bets - n_correct
    roi_color = "#2e7d32" if roi >= 0 else "#c62828"

    html = _HTML_TEMPLATE.format(
        accuracy=accuracy,
        roi=roi,
        stability=stability,
        roi_color=roi_color,
        bar_chart_b64=bar_b64,
        cumulative_chart_b64=cum_b64,
        n_bets=n_bets,
        n_correct=n_correct,
        n_wrong=n_wrong,
    )
    all_bets_script = _all_predictions_script(all_predictions)
    html = html.replace("</head>", f"{all_bets_script}\n</head>", 1)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {output_path}")
