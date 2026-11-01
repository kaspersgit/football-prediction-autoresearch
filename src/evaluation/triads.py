"""Standalone diagnostic for the non-transitive-triad anomaly.

Not a model feature. van Ours (2025), "Non-transitive patterns in sports match
outcomes: a profitable anomaly" documents that, across 25 EPL seasons, teams form
rock-paper-scissors triads (A>B, B>C, C>A) more often than chance, and that
bookmakers ignore this because they price for consistency (see autoresearch/current.md
active hypothesis 13). This is one operationalisation of a betting test: flag matches
whose two teams sit in an intransitive triad on trailing head-to-head record, then
check whether backing the market underdog pays better in those matches than elsewhere.

The operationalisation is ours, not the paper's (the paper documents the pattern more
than a bet rule), so a null result here is soft evidence, not a refutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.hot_hand import _fair_probs, _roi, _settle_odds, _tstat

MIN_MEETINGS = 2  # per pair, before the pair can be part of a judged triad


def _dominance_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def run_triads_diagnostic(
    df: pd.DataFrame, min_meetings: int = MIN_MEETINGS,
    leagues: set[str] | None = None, production_leagues: set[str] | None = None,
) -> dict:
    if leagues is not None:
        df = df[df["league"].isin(leagues)]
    df = df.sort_values("Date").reset_index(drop=True)

    # Running head-to-head points per unordered pair, updated as we sweep in time.
    # pair_pts[(x,y)] = [points x has taken vs y, points y has taken vs x], x <= y.
    pair_pts: dict[tuple[str, str], list[int]] = {}
    # Teams seen per league, to scope the third-team search.
    league_teams: dict[str, set[str]] = {}

    def dominates(x: str, y: str) -> int | None:
        """+1 if x has the better H2H record vs y, -1 if worse, 0 tie, None if < min."""
        k = _dominance_key(x, y)
        rec = pair_pts.get(k)
        if rec is None:
            return None
        # meetings ~ (points are 3/1 per game); require enough games, approximate by
        # total points >= 3 * min_meetings is too strict, so track games separately.
        games = pair_meetings.get(k, 0)
        if games < min_meetings:
            return None
        px, py = rec if k == (x, y) else rec[::-1]
        return int(np.sign(px - py))

    pair_meetings: dict[tuple[str, str], int] = {}

    rows = []
    for _, row in df.iterrows():
        h, a, lg = row["HomeTeam"], row["AwayTeam"], row["league"]
        fair = _fair_probs(row, strip_vig=True)

        # --- classify this match BEFORE updating the H2H ledger with its result ---
        if fair is not None:
            teams = league_teams.get(lg, set())
            intransitive = False
            d_ha = dominates(h, a)
            if d_ha is not None and d_ha != 0:
                for c in teams:
                    if c in (h, a):
                        continue
                    d_hc, d_ca = dominates(h, c), dominates(c, a)
                    if d_hc is None or d_ca is None or d_hc == 0 or d_ca == 0:
                        continue
                    # cycle: h>a, a>c, c>h  OR  a>h, h>c, c>a
                    if d_ha > 0 and d_ca > 0 and d_hc < 0:
                        intransitive = True
                        break
                    if d_ha < 0 and d_ca < 0 and d_hc > 0:
                        intransitive = True
                        break

            fav_side = "H" if fair["H"] >= fair["A"] else "A"
            dog_side = "A" if fav_side == "H" else "H"
            dog_odds = _settle_odds(row, dog_side)
            ftr = row["FTR"]
            rows.append({
                "Date": row["Date"], "league": lg, "season": row["season"],
                "in_triad": intransitive, "stake": 1.0,
                "dog_profit": (dog_odds - 1.0) if ftr == dog_side else -1.0,
            })

        # --- now fold this match's result into the ledger ---
        league_teams.setdefault(lg, set()).update((h, a))
        k = _dominance_key(h, a)
        rec = pair_pts.setdefault(k, [0, 0])
        pair_meetings[k] = pair_meetings.get(k, 0) + 1
        hp = 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
        ap = 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)
        if k == (h, a):
            rec[0] += hp
            rec[1] += ap
        else:
            rec[0] += ap
            rec[1] += hp

    bets = pd.DataFrame(rows)
    out: dict = {}
    print("\n=== NON-TRANSITIVE TRIAD DIAGNOSTIC ===")
    print(f"Matches classified: {len(bets)}  "
          f"({bets['Date'].min().date()} to {bets['Date'].max().date()})")
    tri = bets[bets["in_triad"]]
    non = bets[~bets["in_triad"]]
    print(f"In an intransitive triad: {len(tri)} ({len(tri) / len(bets):.1%})\n")

    for label, g in (("in intransitive triad", tri), ("not in one", non)):
        if g.empty:
            continue
        roi = _roi(g.rename(columns={"dog_profit": "profit"}))
        t = _tstat(g["dog_profit"])
        out[label] = {"n": len(g), "dog_roi": roi, "tstat": t}
        print(f"  back market underdog, {label:<22}  n={len(g):>6}  ROI {roi:+7.2f}%  t={t:+5.2f}")

    if not tri.empty and not non.empty:
        x, y = tri["dog_profit"].to_numpy(), non["dog_profit"].to_numpy()
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        welch_t = float((x.mean() - y.mean()) / se) if se > 0 else 0.0
        gap = out["in intransitive triad"]["dog_roi"] - out["not in one"]["dog_roi"]
        out["contrast"] = {"roi_gap_pp": gap, "welch_t": welch_t}
        print(f"\n  ROI gap (triad - non-triad): {gap:+.2f} pp   Welch t = {welch_t:+.2f}")

    if production_leagues and not tri.empty:
        prod = tri[tri["league"].isin(production_leagues)]
        if not prod.empty:
            out["production_triad"] = {
                "n": len(prod),
                "dog_roi": _roi(prod.rename(columns={"dog_profit": "profit"})),
                "tstat": _tstat(prod["dog_profit"])}
            print(f"  production leagues, in-triad: n={len(prod)}  "
                  f"ROI {out['production_triad']['dog_roi']:+.2f}%  "
                  f"t={out['production_triad']['tstat']:+.2f}")

    print("\n  per league (in-triad only):")
    for lg, g in tri.groupby("league"):
        print(f"    {lg:<4} n={len(g):>5}  dog ROI "
              f"{_roi(g.rename(columns={'dog_profit': 'profit'})):+7.2f}%")
    return out
