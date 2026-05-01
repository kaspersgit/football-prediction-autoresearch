import base64
import math
from datetime import datetime
from pathlib import Path


_LEAGUE_NAMES = {
    # short codes (historical data)
    "E0": "England", "D1": "Germany", "SP1": "Spain",
    "I1": "Italy", "F1": "France", "N1": "Netherlands", "P1": "Portugal",
    # full-name keys from load_fixtures() _LEAGUE_MAP
    "england": "England", "germany": "Germany", "spain": "Spain",
    "italy": "Italy", "france": "France", "netherlands": "Netherlands", "portugal": "Portugal",
}

_LEAGUE_ORDER = ["england", "germany", "spain", "italy", "france", "netherlands", "portugal"]

_OUTCOME_LABEL = {"H": "Home", "D": "Draw", "A": "Away"}
_OUTCOME_COLOR = {"H": "#1565c0", "D": "#616161", "A": "#e65100"}


def _prob_bar(h: float, d: float, a: float) -> str:
    hp, dp, ap = int(h * 100), int(d * 100), int(a * 100)
    return (
        f'<div class="prob-bar" title="Home {hp}% / Draw {dp}% / Away {ap}%">'
        f'<div style="width:{hp}%;background:#1565c0"></div>'
        f'<div style="width:{dp}%;background:#9e9e9e"></div>'
        f'<div style="width:{ap}%;background:#e65100"></div>'
        f'</div>'
        f'<div class="prob-labels">'
        f'<span style="color:#1565c0">{hp}%</span>'
        f'<span style="color:#9e9e9e">{dp}%</span>'
        f'<span style="color:#e65100">{ap}%</span>'
        f'</div>'
    )


def _odds_cell(b365: float, cmax: float, bk: str, outcome: str, edge: float | None) -> str:
    color = _OUTCOME_COLOR[outcome]
    has_max = not math.isnan(cmax) and bk
    max_better = has_max and cmax > b365 + 0.005

    if edge is not None:
        b365_html = f'<span class="b365-val value-price" style="color:{color}">{b365:.2f}</span>'
        edge_html = f'<span class="edge-chip" style="background:{color}">+{edge:.1%}</span>'
    else:
        b365_html = f'<span class="b365-val">{b365:.2f}</span>'
        edge_html = ""

    if max_better:
        max_html = f'<span class="max-odds">↑ {cmax:.2f} <em>{bk}</em></span>'
    else:
        max_html = ""

    css_extra = f' value-odds" style="border-color:{color}' if edge is not None else ""
    return (
        f'<td class="odds-cell{css_extra}">'
        f'{b365_html}{edge_html}{max_html}'
        f'</td>'
    )


def _top_bets_html(all_bets: list[dict]) -> str:
    if not all_bets:
        return "<p>No value bets found at this threshold.</p>"
    rows = []
    for i, b in enumerate(all_bets, 1):
        color = _OUTCOME_COLOR[b["outcome"]]
        edge_pct = f'+{b["edge"]:.1%}'
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
        cmax = b.get("max_odds", float("nan"))
        bk = b.get("max_bk", "")
        has_max = not math.isnan(cmax) and bk
        max_cell = (
            f'<td class="odds-val">{cmax:.2f} <span class="bk-tag">{bk}</span></td>'
            if has_max else '<td class="odds-val">—</td>'
        )
        rows.append(
            f'<tr>'
            f'<td class="rank">{medal}</td>'
            f'<td>{b["date"]}</td>'
            f'<td><span class="league-tag">{_LEAGUE_NAMES.get(b["league"], b["league"])}</span></td>'
            f'<td><strong>{b["home"]}</strong> vs {b["away"]}</td>'
            f'<td><span class="bet-outcome" style="color:{color};border-color:{color}">'
            f'{_OUTCOME_LABEL[b["outcome"]]}</span></td>'
            f'<td><span class="edge-val" style="color:{color}">{edge_pct}</span></td>'
            f'<td class="odds-val">{b["b365_odds"]:.2f}</td>'
            f'{max_cell}'
            f'</tr>'
        )
    return "\n".join(rows)


def _league_section_html(league: str, rows: list[dict]) -> str:
    league_name = _LEAGUE_NAMES.get(league, league)
    n_value = sum(1 for r in rows if r["ValueBets"])
    fixture_rows = []
    for r in rows:
        date_str = r["Date"].strftime("%a %b %d")
        has_value = bool(r["ValueBets"])
        row_class = "value-row" if has_value else ""
        edge_map = {o: e for o, e in r["ValueBets"]}

        odds_cells = ""
        for outcome in ["H", "D", "A"]:
            b365 = r[f"B365{outcome}"]
            cmax = r.get(f"CustomMax{outcome}", float("nan"))
            bk = r.get(f"CustomMaxBk{outcome}", "")
            edge = edge_map.get(outcome)
            odds_cells += _odds_cell(b365, cmax, bk, outcome, edge)

        fixture_rows.append(
            f'<tr class="{row_class}">'
            f'<td class="date-cell">{date_str}</td>'
            f'<td class="team-cell"><strong>{r["HomeTeam"]}</strong></td>'
            f'<td class="vs-cell">vs</td>'
            f'<td class="team-cell">{r["AwayTeam"]}</td>'
            f'<td class="prob-cell">{_prob_bar(r["ModelH"], r["ModelD"], r["ModelA"])}</td>'
            f'{odds_cells}'
            f'</tr>'
        )

    badge = f'<span class="value-count">{n_value} value bet{"s" if n_value != 1 else ""}</span>' if n_value else ""
    return f"""
<div class="league-section">
  <div class="league-header">
    <span class="league-name">{league_name}</span>
    {badge}
  </div>
  <table class="fixture-table">
    <thead>
      <tr>
        <th>Date</th><th colspan="3">Fixture</th>
        <th>Model probs (H/D/A)</th>
        <th>B365 H</th><th>B365 D</th><th>B365 A</th>
      </tr>
      <tr class="subheader">
        <th colspan="5"></th>
        <th colspan="3" style="color:#888;font-size:.7em;font-weight:500">↑ best available odds + bookmaker</th>
      </tr>
    </thead>
    <tbody>
      {"".join(fixture_rows)}
    </tbody>
  </table>
</div>"""


def _profit_curve_html(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
<div class="top-bets-card" style="margin-bottom:24px">
  <div class="card-header">
    Backtest Profit Curve
    <span style="font-size:.8em;font-weight:400;color:#888;margin-left:8px">last 2 seasons walk-forward</span>
  </div>
  <div style="padding:16px 24px">
    <img src="data:image/png;base64,{data}" style="width:100%;max-width:900px;display:block;margin:0 auto" alt="Profit curve">
  </div>
</div>"""


def generate_predictions_html(pred_rows: list[dict], threshold: float, fetched_at: datetime,
                               profit_curve_path: Path | None = None) -> str:
    all_bets = []
    for r in pred_rows:
        for outcome, edge in r["ValueBets"]:
            cmax_key = f"CustomMax{outcome}"
            bk_key = f"CustomMaxBk{outcome}"
            cmax = r.get(cmax_key, float("nan"))
            bk = r.get(bk_key, "")
            all_bets.append({
                "date": r["Date"].strftime("%a %b %d"),
                "league": r["League"],
                "home": r["HomeTeam"],
                "away": r["AwayTeam"],
                "outcome": outcome,
                "edge": edge,
                "b365_odds": r[f"B365{outcome}"],
                "max_odds": cmax if not math.isnan(cmax) else float("nan"),
                "max_bk": bk,
            })
    all_bets.sort(key=lambda x: x["edge"], reverse=True)

    by_league: dict[str, list] = {}
    for r in pred_rows:
        by_league.setdefault(r["League"], []).append(r)

    league_html = "\n".join(
        _league_section_html(lg, by_league[lg])
        for lg in _LEAGUE_ORDER
        if lg in by_league
    )

    fetch_str = fetched_at.strftime("%d %b %Y, %H:%M")
    total_fixtures = len(pred_rows)
    total_value = len(all_bets)
    profit_html = _profit_curve_html(profit_curve_path)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictions — {fetched_at.strftime('%d %b %Y')}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    margin: 0;
    padding: 24px 16px 48px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}

  /* Header */
  .page-header {{
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    color: white;
    border-radius: 12px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(26,35,126,.25);
  }}
  .page-header h1 {{ margin: 0 0 6px; font-size: 1.7em; letter-spacing: -.3px; }}
  .page-header .meta {{ opacity: .8; font-size: .9em; margin-bottom: 14px; }}
  .warning-banner {{
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: .85em;
    display: inline-block;
  }}
  .stats-row {{ display: flex; gap: 16px; margin-top: 20px; }}
  .stat-pill {{
    background: rgba(255,255,255,.18);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: .9em;
    font-weight: 600;
  }}

  /* Cards */
  .top-bets-card {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 24px;
    overflow: hidden;
  }}
  .card-header {{
    padding: 16px 24px;
    border-bottom: 1px solid #eee;
    font-weight: 700;
    font-size: 1.05em;
    color: #1a237e;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .card-header .count-badge {{
    background: #1a237e;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: .8em;
  }}
  .top-bets-card table {{
    width: 100%;
    border-collapse: collapse;
    font-size: .9em;
  }}
  .top-bets-card th {{
    background: #f5f6fa;
    color: #666;
    font-weight: 600;
    text-transform: uppercase;
    font-size: .75em;
    letter-spacing: .5px;
    padding: 10px 16px;
    text-align: left;
  }}
  .top-bets-card td {{ padding: 12px 16px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
  .top-bets-card tr:last-child td {{ border-bottom: none; }}
  .top-bets-card tr:hover td {{ background: #fafbff; }}
  .rank {{ font-size: 1.1em; text-align: center !important; width: 40px; }}
  .league-tag {{
    background: #e8eaf6; color: #3949ab;
    border-radius: 4px; padding: 2px 8px;
    font-size: .8em; font-weight: 600;
  }}
  .bet-outcome {{
    border: 1.5px solid; border-radius: 4px;
    padding: 2px 8px; font-size: .82em; font-weight: 700;
  }}
  .edge-val {{ font-weight: 700; font-size: 1em; }}
  .odds-val {{ font-weight: 600; color: #333; }}
  .bk-tag {{ font-size:.78em; color:#666; font-weight:400; }}

  /* League sections */
  .league-section {{
    background: white; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 20px; overflow: hidden;
  }}
  .league-header {{
    padding: 14px 24px; border-bottom: 1px solid #eee;
    display: flex; align-items: center; gap: 12px;
  }}
  .league-name {{ font-weight: 700; font-size: 1.05em; color: #1a237e; }}
  .value-count {{
    background: #e8f5e9; color: #2e7d32;
    border-radius: 12px; padding: 2px 10px;
    font-size: .8em; font-weight: 600;
  }}
  .fixture-table {{ width: 100%; border-collapse: collapse; font-size: .88em; }}
  .fixture-table th {{
    background: #f5f6fa; color: #888; font-weight: 600;
    text-transform: uppercase; font-size: .72em; letter-spacing: .4px;
    padding: 8px 12px; text-align: left;
  }}
  .fixture-table .subheader th {{
    background: #fafafa; padding: 2px 12px; border-bottom: 1px solid #eee;
  }}
  .fixture-table td {{ padding: 10px 12px; border-bottom: 1px solid #f5f5f5; vertical-align: middle; }}
  .fixture-table tr:last-child td {{ border-bottom: none; }}
  .fixture-table tr:hover td {{ background: #fafbff; }}
  .value-row td {{ background: #f1f8e9 !important; }}
  .value-row:hover td {{ background: #e8f5e9 !important; }}

  .date-cell {{ color: #666; white-space: nowrap; width: 100px; }}
  .team-cell {{ width: 180px; }}
  .vs-cell {{ color: #bbb; text-align: center; width: 28px; padding: 0 !important; }}

  /* Prob bar */
  .prob-cell {{ width: 200px; }}
  .prob-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; margin-bottom: 4px; }}
  .prob-bar div {{ border-radius: 4px; }}
  .prob-labels {{ display: flex; justify-content: space-between; font-size: .75em; font-weight: 600; }}

  /* Odds cells */
  .odds-cell {{ text-align: center; width: 82px; font-weight: 600; color: #444; vertical-align: middle; }}
  .value-odds {{ border-left: 3px solid; }}
  .b365-val {{ display: block; font-size: .92em; }}
  .value-price {{ font-weight: 700; }}
  .edge-chip {{
    display: inline-block; color: white; border-radius: 3px;
    padding: 1px 5px; font-size: .72em; font-weight: 700; margin-top: 2px;
  }}
  .max-odds {{
    display: block; font-size: .75em; color: #2e7d32; font-weight: 500; margin-top: 3px;
  }}
  .max-odds em {{ font-style: normal; color: #555; }}

  /* Footer */
  .page-footer {{ text-align: center; color: #aaa; font-size: .82em; margin-top: 32px; }}
</style>
</head>
<body>
<div class="container">

  <div class="page-header">
    <h1>⚽ Match Week Predictions</h1>
    <div class="meta">Odds fetched {fetch_str} &nbsp;·&nbsp; Edge threshold {threshold:+.0%}</div>
    <div class="warning-banner">⚠ Verify odds are still current before placing any bet</div>
    <div class="stats-row">
      <div class="stat-pill">{total_fixtures} fixtures</div>
      <div class="stat-pill">{total_value} value bet{"s" if total_value != 1 else ""}</div>
    </div>
  </div>

  <div class="top-bets-card">
    <div class="card-header">
      Top Value Bets
      <span class="count-badge">{len(all_bets)}</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Date</th><th>League</th><th>Fixture</th>
          <th>Bet</th><th>Edge</th><th>B365 Odds</th><th>Best Odds</th>
        </tr>
      </thead>
      <tbody>
        {_top_bets_html(all_bets)}
      </tbody>
    </table>
  </div>

  {profit_html}

  {league_html}

  <div class="page-footer">
    Generated {fetch_str} · Football Prediction Model
  </div>
</div>
</body>
</html>"""


def save_predictions_html(pred_rows: list[dict], threshold: float, fetched_at: datetime,
                           output_path: Path, profit_curve_path: Path | None = None) -> Path:
    html = generate_predictions_html(pred_rows, threshold, fetched_at,
                                      profit_curve_path=profit_curve_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
