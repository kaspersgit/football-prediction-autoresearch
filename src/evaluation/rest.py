"""Standalone diagnostic for an asymmetric rest-disadvantage edge.

Not a model feature. Raw "days rest" / "days since last match" were reverted twice as
*symmetric* team features (EXP-20260419-S023, EXP-20260501-D088). The untested angle
(autoresearch/current.md active hypothesis 13) is the *asymmetry*: a short-rested
favourite against a well-rested underdog may be systematically overpriced because the
market underweights congestion. This backs the well-rested underdog in exactly those
spots and checks it against the market and against the symmetric baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.hot_hand import _fair_probs, _roi, _settle_odds, _tstat


def _days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """home_rest / away_rest = days since each team's previous match (any competition
    present in the data). NaN for a team's first observed match."""
    df = df.sort_values("Date").reset_index(drop=True)
    last_seen: dict[str, pd.Timestamp] = {}
    home_rest = np.full(len(df), np.nan)
    away_rest = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        d = row["Date"]
        for team, arr in ((row["HomeTeam"], home_rest), (row["AwayTeam"], away_rest)):
            prev = last_seen.get(team)
            if prev is not None:
                arr[i] = (d - prev).days
        last_seen[row["HomeTeam"]] = d
        last_seen[row["AwayTeam"]] = d
    df = df.copy()
    df["home_rest"] = home_rest
    df["away_rest"] = away_rest
    return df


def run_rest_diagnostic(
    df: pd.DataFrame,
    short_rest: int = 3,
    long_rest: int = 6,
    fav_max_odds: float = 1.80,
    leagues: set[str] | None = None,
    production_leagues: set[str] | None = None,
) -> dict:
    if leagues is not None:
        df = df[df["league"].isin(leagues)]
    df = _days_rest(df)
    df = df.dropna(subset=["home_rest", "away_rest"]).reset_index(drop=True)

    print("\n=== ASYMMETRIC REST DIAGNOSTIC ===")
    print(f"Matches with rest known: {len(df)}  "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    print(f"'congested favourite' = fair odds <= {fav_max_odds}, <= {short_rest}d rest, "
          f"opponent >= {long_rest}d rest. Bet the rested underdog, flat 1 unit.\n")

    rows = []
    for _, row in df.iterrows():
        fair = _fair_probs(row, strip_vig=True)
        if fair is None:
            continue
        h_odds = 1.0 / fair["H"] if fair["H"] > 0 else np.inf
        a_odds = 1.0 / fair["A"] if fair["A"] > 0 else np.inf
        hr, ar = row["home_rest"], row["away_rest"]
        rest_gap = ar - hr  # >0: away more rested

        fav_side = dog_side = None
        if h_odds <= fav_max_odds and hr <= short_rest and ar >= long_rest:
            fav_side, dog_side = "H", "A"
        elif a_odds <= fav_max_odds and ar <= short_rest and hr >= long_rest:
            fav_side, dog_side = "A", "H"
        if dog_side is None:
            continue

        ftr = row["FTR"]
        dog_odds = _settle_odds(row, dog_side)
        fav_odds = _settle_odds(row, fav_side)
        rows.append({
            "Date": row["Date"], "league": row["league"], "season": row["season"],
            "rest_gap": abs(rest_gap), "stake": 1.0,
            "dog_profit": (dog_odds - 1.0) if ftr == dog_side else -1.0,
            "fav_profit": (fav_odds - 1.0) if ftr == fav_side else -1.0,
        })
    bets = pd.DataFrame(rows)

    out: dict = {}
    if bets.empty:
        print("No qualifying spots. Loosen the thresholds.")
        return out

    dog_roi = _roi(bets.rename(columns={"dog_profit": "profit"}))
    fav_roi = _roi(bets.rename(columns={"fav_profit": "profit"}))
    dog_t = _tstat(bets["dog_profit"])
    paired = _tstat(bets["dog_profit"] - bets["fav_profit"])
    out["all"] = {"n": len(bets), "dog_roi": dog_roi, "fav_roi": fav_roi,
                  "dog_tstat": dog_t, "paired_tstat": paired}
    print(f"Qualifying spots: {len(bets)}")
    print(f"  back rested underdog:  ROI {dog_roi:+7.2f}%  t={dog_t:+5.2f}")
    print(f"  back congested fav:    ROI {fav_roi:+7.2f}%")
    print(f"  paired t (dog - fav):  {paired:+5.2f}")

    print("  per league:")
    for lg, g in bets.groupby("league"):
        out.setdefault("by_league", {})[lg] = {
            "n": len(g), "dog_roi": _roi(g.rename(columns={"dog_profit": "profit"}))}
        print(f"    {lg:<4} n={len(g):>4}  dog ROI "
              f"{_roi(g.rename(columns={'dog_profit': 'profit'})):+7.2f}%")

    if production_leagues:
        prod = bets[bets["league"].isin(production_leagues)]
        if not prod.empty:
            out["production"] = {
                "n": len(prod),
                "dog_roi": _roi(prod.rename(columns={"dog_profit": "profit"})),
                "dog_tstat": _tstat(prod["dog_profit"])}
            print(f"  production leagues: n={len(prod)}  dog ROI "
                  f"{out['production']['dog_roi']:+.2f}%  t={out['production']['dog_tstat']:+.2f}")
    return out
