"""Standalone diagnostic for the COD (Combined Odds Distribution) overreaction claim.

Not a model feature and does not touch training. Replicates Wheatcroft, "Profiting
from overreaction in soccer betting odds" (JQAS 2020) -- see autoresearch/current.md
active hypothesis 11. Distinct from the rejected hot-hand item 9: this is a
season-cumulative points quantile benchmarked against a Monte-Carlo simulation of the
odds themselves, not a trailing-window probability residual.

Method:
  - For each team, within each season, simulate every completed prior match this
    season from its vig-stripped odds-implied (win/draw/loss) probabilities, m times,
    and sum simulated points -> a distribution of season-to-date point totals.
  - COD = the quantile of the team's ACTUAL season-to-date points within that
    distribution. Low COD => the team banked fewer points than the market implied
    (unlucky / market over-adjusted). Require >= 6 completed prior matches.
  - Betting test: back teams with a low COD, flat 1 unit, best available price. Report
    ROI by COD decile and low-vs-high-COD contrast, per league / per season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.hot_hand import _fair_probs, _roi, _settle_odds, _tstat

MIN_PRIOR_MATCHES = 6
DEFAULT_SIMS = 512


def _team_season_cod(
    prob_win: np.ndarray, prob_draw: np.ndarray, actual_points: np.ndarray,
    rng: np.random.Generator, sims: int,
) -> np.ndarray:
    """COD per match for one team-season. Match k uses matches 0..k-1 (strictly prior).

    prob_win/prob_draw: (T,) vig-stripped pre-match probabilities for this team.
    actual_points: (T,) points the team actually took (3/1/0).
    Returns (T,) COD in [0,1]; NaN where fewer than MIN_PRIOR_MATCHES priors exist.
    """
    t = len(prob_win)
    cod = np.full(t, np.nan)
    if t <= MIN_PRIOR_MATCHES:
        return cod
    u = rng.random((t, sims))
    sim_points = np.where(u < prob_win[:, None], 3,
                          np.where(u < (prob_win + prob_draw)[:, None], 1, 0))
    # Cumulative sums of matches strictly before k: prefix[k] = sum(rows 0..k-1).
    sim_prefix = np.vstack([np.zeros((1, sims)), np.cumsum(sim_points, axis=0)[:-1]])
    act_prefix = np.concatenate([[0.0], np.cumsum(actual_points)[:-1]])
    for k in range(MIN_PRIOR_MATCHES + 1, t):
        cod[k] = float(np.mean(sim_prefix[k] <= act_prefix[k]))
    return cod


def _points_for(ftr: str, is_home: bool) -> int:
    if ftr == "D":
        return 1
    win = (ftr == "H") if is_home else (ftr == "A")
    return 3 if win else 0


def compute_cod(df: pd.DataFrame, sims: int = DEFAULT_SIMS, seed: int = 0) -> pd.DataFrame:
    """Attach home_cod / away_cod to each match (NaN until a team has >=6 season games)."""
    df = df.sort_values("Date").reset_index(drop=True)
    rng = np.random.default_rng(seed)

    # Long per-team-match table with the team's own win/draw prob and actual points.
    recs = []
    for i, row in df.iterrows():
        fair = _fair_probs(row, strip_vig=True)
        if fair is None:
            continue
        for team, is_home in ((row["HomeTeam"], True), (row["AwayTeam"], False)):
            recs.append({
                "match_idx": i, "season": row["season"], "league": row["league"],
                "Date": row["Date"], "team": team, "is_home": is_home,
                "p_win": fair["H"] if is_home else fair["A"],
                "p_draw": fair["D"],
                "points": _points_for(row["FTR"], is_home),
            })
    long = pd.DataFrame(recs)

    long["cod"] = np.nan
    for _, g in long.groupby(["season", "league", "team"], sort=False):
        g = g.sort_values(["Date", "match_idx"])
        cod = _team_season_cod(
            g["p_win"].to_numpy(), g["p_draw"].to_numpy(),
            g["points"].to_numpy(float), rng, sims,
        )
        long.loc[g.index, "cod"] = cod

    home = long[long["is_home"]].set_index("match_idx")["cod"]
    away = long[~long["is_home"]].set_index("match_idx")["cod"]
    df["home_cod"] = df.index.map(home)
    df["away_cod"] = df.index.map(away)
    return df


def _side_bets(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, side) where that side's COD is defined: bet that side flat."""
    rows = []
    for i, row in df.iterrows():
        ftr = row["FTR"]
        for side, cod in (("H", row["home_cod"]), ("A", row["away_cod"])):
            if pd.isna(cod):
                continue
            odds = _settle_odds(row, side)
            rows.append({
                "Date": row["Date"], "league": row["league"], "season": row["season"],
                "side": side, "cod": float(cod), "odds": odds, "stake": 1.0,
                "profit": (odds - 1.0) if ftr == side else -1.0,
            })
    return pd.DataFrame(rows)


def _contrast_bets(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match with both CODs: back the lower-COD side, record the other too."""
    rows = []
    for i, row in df.iterrows():
        hc, ac = row["home_cod"], row["away_cod"]
        if pd.isna(hc) or pd.isna(ac) or hc == ac:
            continue
        ftr = row["FTR"]
        low_side = "H" if hc < ac else "A"
        high_side = "A" if low_side == "H" else "H"
        low_odds, high_odds = _settle_odds(row, low_side), _settle_odds(row, high_side)
        rows.append({
            "Date": row["Date"], "league": row["league"], "season": row["season"],
            "cod_gap": abs(hc - ac), "stake": 1.0,
            "profit": (low_odds - 1.0) if ftr == low_side else -1.0,
            "high_profit": (high_odds - 1.0) if ftr == high_side else -1.0,
        })
    return pd.DataFrame(rows)


def run_cod_diagnostic(
    df: pd.DataFrame, sims: int = DEFAULT_SIMS,
    leagues: set[str] | None = None, production_leagues: set[str] | None = None,
) -> dict:
    if leagues is not None:
        df = df[df["league"].isin(leagues)]
    df = compute_cod(df.copy(), sims=sims)

    side = _side_bets(df)
    contrast = _contrast_bets(df)

    print("\n=== COD (season-luck) DIAGNOSTIC ===")
    print(f"Matches: {len(df)}  ({df['Date'].min().date()} to {df['Date'].max().date()})  "
          f"Leagues: {', '.join(sorted(df['league'].unique()))}")
    print(f"Monte-Carlo sims per team-season: {sims}. Backing low-COD teams should, per "
          f"Wheatcroft, offer positive value.\n")

    out: dict = {"deciles": {}, "contrast": {}, "by_league": {}, "by_season": {}}

    print("--- ROI of backing a side, bucketed by that side's COD decile ---")
    side = side.copy()
    side["decile"] = pd.qcut(side["cod"], 10, labels=False, duplicates="drop")
    for d, g in side.groupby("decile"):
        roi, t = _roi(g), _tstat(g["profit"])
        out["deciles"][int(d)] = {"n": len(g), "roi": roi, "tstat": t,
                                  "cod_lo": g["cod"].min(), "cod_hi": g["cod"].max()}
        print(f"  decile {int(d):>2} (COD {g['cod'].min():.2f}-{g['cod'].max():.2f})  "
              f"n={len(g):>6}  ROI {roi:+7.2f}%  t={t:+5.2f}")

    print("\n--- back the lower-COD side in each match, vs backing the higher-COD side ---")
    if not contrast.empty:
        paired = _tstat(contrast["profit"] - contrast["high_profit"])
        out["contrast"]["all"] = {
            "n": len(contrast), "low_roi": _roi(contrast),
            "high_roi": _roi(contrast, "high_profit"),
            "low_tstat": _tstat(contrast["profit"]), "paired_tstat": paired,
        }
        print(f"  n={len(contrast)}  low-COD ROI {_roi(contrast):+.2f}%  "
              f"high-COD ROI {_roi(contrast, 'high_profit'):+.2f}%  "
              f"low t={_tstat(contrast['profit']):+.2f}  paired t={paired:+.2f}")
        for min_gap in (0.0, 0.25, 0.5):
            sub = contrast[contrast["cod_gap"] >= min_gap]
            if sub.empty:
                continue
            print(f"    cod_gap >= {min_gap:<4} n={len(sub):>6}  low-COD ROI {_roi(sub):+7.2f}%  "
                  f"paired t={_tstat(sub['profit'] - sub['high_profit']):+5.2f}")

    # Wheatcroft's headline: lowest-COD-decile side-bets, per league / season.
    lowest = side[side["decile"] == side["decile"].min()]
    print("\n--- lowest-COD decile only, per league ---")
    for lg, g in lowest.groupby("league"):
        roi, t = _roi(g), _tstat(g["profit"])
        out["by_league"][lg] = {"n": len(g), "roi": roi, "tstat": t}
        print(f"    {lg:<4} n={len(g):>5}  ROI {roi:+7.2f}%  t={t:+5.2f}")
    print("--- lowest-COD decile only, per season ---")
    for sea, g in lowest.groupby("season"):
        roi, t = _roi(g), _tstat(g["profit"])
        out["by_season"][sea] = {"n": len(g), "roi": roi, "tstat": t}
        print(f"    {sea}  n={len(g):>5}  ROI {roi:+7.2f}%  t={t:+5.2f}")

    if production_leagues:
        prod = lowest[lowest["league"].isin(production_leagues)]
        out["production_lowest_decile"] = {"n": len(prod), "roi": _roi(prod),
                                           "tstat": _tstat(prod["profit"])}
        print(f"\n  production leagues, lowest-COD decile: n={len(prod)}  "
              f"ROI {_roi(prod):+.2f}%  t={_tstat(prod['profit']):+.2f}")
    return out
