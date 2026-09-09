"""Standalone diagnostic for the contrarian "hot-hand fallacy" momentum hypothesis.

This is **not** a model feature and does not touch training. It replicates the claim
from football-data.co.uk's "A Profitable Betting System?" (see autoresearch/current.md
active hypothesis 9): bettors overweight recent winning/losing streaks, so the market
systematically underprices recently "cold" teams and overprices "hot" ones. The test
backs the colder team in every match at level stakes and checks whether backing colder
beats backing hotter, per league and overall, on our own data and leagues.

Method (mirrors the article):
  - Per team per match, an odds-adjusted score against the vig-stripped fair win prob p:
        score = (1 - p) if the team won, else -p   (a draw counts as "did not win")
  - A team's "heat rating" is that score averaged over a trailing window of completed
    matches (article used 6, "purely arbitrary"; we also try our own WINDOW = 5),
    shifted one match so it is strictly pre-match.
  - match_rating = home_heat_rating - away_heat_rating; negative => home is colder.
  - Bet the colder side, flat 1 unit, settled at the best available market price.

Fair probabilities are vig-stripped from Pinnacle closing odds (PSCH/PSCD/PSCA) where
all three are present, else CustomMax{H,D,A}, else B365{H,D,A} -- the same
best-to-fallback order the rest of the pipeline uses for execution prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.features import WINDOW

_FAIR_SOURCES = (("PSCH", "PSCD", "PSCA"), ("CustomMaxH", "CustomMaxD", "CustomMaxA"),
                 ("B365H", "B365D", "B365A"))
_SETTLE_SOURCES = (("CustomMaxH", "CustomMaxD", "CustomMaxA"), ("B365H", "B365D", "B365A"))
DEFAULT_WINDOWS = (6, WINDOW)
DEFAULT_RATING_FILTERS = (0.0, 0.25, 0.5)


def _valid_triple(row: pd.Series, cols: tuple[str, str, str]) -> bool:
    for c in cols:
        v = row.get(c)
        if v is None or pd.isna(v):
            return False
        try:
            if float(v) <= 1.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _fair_probs(row: pd.Series, strip_vig: bool = True) -> dict[str, float] | None:
    """Fair {H,D,A} probabilities from the best available odds source.

    strip_vig=True normalises the three implied probabilities to sum to 1 (what the
    rest of the pipeline uses). strip_vig=False keeps raw 1/odds -- the article is
    ambiguous on whether its per-match score used stripped or raw prices, and its
    rating spread is far wider than the stripped version produces, so both are tested.
    """
    for h_col, d_col, a_col in _FAIR_SOURCES:
        if _valid_triple(row, (h_col, d_col, a_col)):
            raw = {"H": 1.0 / float(row[h_col]), "D": 1.0 / float(row[d_col]),
                   "A": 1.0 / float(row[a_col])}
            if not strip_vig:
                return raw
            tot = sum(raw.values())
            return {k: v / tot for k, v in raw.items()}
    return None


def _settle_odds(row: pd.Series, outcome: str) -> float:
    """Best available decimal price for `outcome`, CustomMax -> B365."""
    for h_col, d_col, a_col in _SETTLE_SOURCES:
        col = {"H": h_col, "D": d_col, "A": a_col}[outcome]
        v = row.get(col)
        if v is not None and not pd.isna(v):
            try:
                f = float(v)
                if f > 1.0:
                    return f
            except (TypeError, ValueError):
                pass
    return float(row[{"H": "B365H", "D": "B365D", "A": "B365A"}[outcome]])


def _team_match_scores(df: pd.DataFrame, strip_vig: bool = True) -> pd.DataFrame:
    """One row per (match, team): odds-adjusted score for that team in that match."""
    df = df.sort_values("Date").reset_index(drop=True)
    records = []
    for i, row in df.iterrows():
        fair = _fair_probs(row, strip_vig=strip_vig)
        if fair is None:
            continue
        ftr = row["FTR"]
        for team, is_home in ((row["HomeTeam"], True), (row["AwayTeam"], False)):
            p = fair["H"] if is_home else fair["A"]
            won = (ftr == "H") if is_home else (ftr == "A")
            records.append({
                "match_idx": i,
                "Date": row["Date"],
                "team": team,
                "is_home": is_home,
                "score": (1.0 - p) if won else (-p),
            })
    return pd.DataFrame(records)


def _heat_ratings(scores: pd.DataFrame, window: int, cumulative: bool = False) -> pd.DataFrame:
    """Per-team heat rating, strictly pre-match (score shifted one game).

    window > 0 and cumulative=False: trailing mean over the previous `window` matches.
    cumulative=True: expanding mean over all previous matches (the article's
    "cumulative-season ratings" variant); `window` is then the minimum game count.
    """
    scores = scores.sort_values(["team", "Date", "match_idx"]).reset_index(drop=True)
    if cumulative:
        scores["heat"] = (
            scores.groupby("team", group_keys=False)["score"]
            .apply(lambda s: s.shift(1).expanding(min_periods=window).mean())
        )
    else:
        scores["heat"] = (
            scores.groupby("team", group_keys=False)["score"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
        )
    return scores


def _bets_for_window(df: pd.DataFrame, scores: pd.DataFrame, window: int,
                     cumulative: bool = False) -> pd.DataFrame:
    """Build the colder-side / hotter-side bet table for one window length."""
    rated = _heat_ratings(scores, window, cumulative=cumulative)
    home = rated[rated["is_home"]].set_index("match_idx")["heat"]
    away = rated[~rated["is_home"]].set_index("match_idx")["heat"]

    rows = []
    for i, row in df.iterrows():
        h_heat, a_heat = home.get(i), away.get(i)
        if h_heat is None or a_heat is None or pd.isna(h_heat) or pd.isna(a_heat):
            continue
        match_rating = float(h_heat) - float(a_heat)
        if match_rating == 0.0:
            continue
        ftr = row["FTR"]
        colder = "H" if match_rating < 0 else "A"
        hotter = "A" if colder == "H" else "H"
        colder_odds = _settle_odds(row, colder)
        hotter_odds = _settle_odds(row, hotter)
        rows.append({
            "Date": row["Date"],
            "league": row["league"],
            "season": row["season"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "match_rating": match_rating,
            "abs_rating": abs(match_rating),
            "colder_side": colder,
            "colder_odds": colder_odds,
            "hotter_odds": hotter_odds,
            "stake": 1.0,
            "profit": (colder_odds - 1.0) if ftr == colder else -1.0,
            "hotter_profit": (hotter_odds - 1.0) if ftr == hotter else -1.0,
        })
    return pd.DataFrame(rows)


def _roi(bets: pd.DataFrame, profit_col: str = "profit") -> float:
    if bets.empty:
        return float("nan")
    return bets[profit_col].sum() / bets["stake"].sum() * 100.0


def _tstat(series: pd.Series) -> float:
    """One-sample t-stat of per-bet profit vs 0 (repo convention: stability * sqrt(n))."""
    s = series.std()
    if not np.isfinite(s) or s == 0 or len(series) == 0:
        return 0.0
    return float(series.mean() / s * np.sqrt(len(series)))


def _paired_tstat(bets: pd.DataFrame) -> float:
    """Paired t-stat for (colder profit - hotter profit) across the same matches."""
    diff = bets["profit"] - bets["hotter_profit"]
    return _tstat(diff)


def _summise(bets: pd.DataFrame) -> dict:
    return {
        "n": int(len(bets)),
        "colder_roi": _roi(bets, "profit"),
        "hotter_roi": _roi(bets, "hotter_profit"),
        "colder_tstat": _tstat(bets["profit"]) if not bets.empty else 0.0,
        "paired_tstat": _paired_tstat(bets) if not bets.empty else 0.0,
    }


def run_hot_hand_diagnostic(
    df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    rating_filters: tuple[float, ...] = DEFAULT_RATING_FILTERS,
    leagues: set[str] | None = None,
    production_leagues: set[str] | None = None,
    strip_vig: bool = True,
    cumulative: bool = False,
) -> dict:
    """Run the diagnostic and return a nested results dict. Also prints a report."""
    if leagues is not None:
        df = df[df["league"].isin(leagues)]
    df = df.sort_values("Date").reset_index(drop=True)
    scores = _team_match_scores(df, strip_vig=strip_vig)

    print("\n=== HOT-HAND FALLACY DIAGNOSTIC ===")
    print(f"Matches: {len(df)}  ({df['Date'].min().date()} to {df['Date'].max().date()})  "
          f"Leagues: {', '.join(sorted(df['league'].unique()))}")
    print(f"Rating: {'expanding (cumulative)' if cumulative else 'trailing window'}, "
          f"vig-{'stripped' if strip_vig else 'raw (1/odds)'} score.")
    print("Bet the colder team (lower trailing odds-adjusted score), flat 1 unit.")
    print("Replication check: does colder-team ROI exceed hotter-team ROI, per the article?\n")

    out: dict = {}
    for window in windows:
        bets = _bets_for_window(df, scores, window, cumulative=cumulative)
        out[window] = {"all": {}, "signed": {}, "by_league": {}, "by_season": {}}
        if bets.empty:
            print(f"[window={window}] no bets (insufficient history)")
            continue

        print(f"--- window = {window} ---")
        for min_abs in rating_filters:
            sub = bets[bets["abs_rating"] >= min_abs]
            s = _summise(sub)
            out[window]["all"][min_abs] = s
            print(f"|match_rating| >= {min_abs:<4}  n={s['n']:>6}  "
                  f"colder ROI {s['colder_roi']:+7.2f}%  hotter ROI {s['hotter_roi']:+7.2f}%  "
                  f"colder t={s['colder_tstat']:+5.2f}  paired t={s['paired_tstat']:+5.2f}")

        # Signed cut: the article's actual form -- back home when strongly colder
        # (match_rating <= -X) or away when strongly colder (match_rating >= +X).
        for x in [f for f in rating_filters if f > 0]:
            home_cold = bets[bets["match_rating"] <= -x]
            away_cold = bets[bets["match_rating"] >= x]
            hs, as_ = _summise(home_cold), _summise(away_cold)
            out[window]["signed"][x] = {"home_colder": hs, "away_colder": as_}
            print(f"match_rating <= -{x:<4} (home colder) n={hs['n']:>5}  "
                  f"colder ROI {hs['colder_roi']:+7.2f}%  t={hs['colder_tstat']:+5.2f}   | "
                  f"  >= +{x:<4} (away colder) n={as_['n']:>5}  colder ROI "
                  f"{as_['colder_roi']:+7.2f}%  t={as_['colder_tstat']:+5.2f}")

        base = bets  # per-league / per-season reported at the unfiltered cut
        print("  per league (|match_rating| >= 0):")
        for lg, g in base.groupby("league"):
            s = _summise(g)
            out[window]["by_league"][lg] = s
            print(f"    {lg:<4} n={s['n']:>5}  colder {s['colder_roi']:+7.2f}%  "
                  f"hotter {s['hotter_roi']:+7.2f}%  paired t={s['paired_tstat']:+5.2f}")
        print("  per season (|match_rating| >= 0):")
        for sea, g in base.groupby("season"):
            s = _summise(g)
            out[window]["by_season"][sea] = s
            print(f"    {sea}  n={s['n']:>5}  colder {s['colder_roi']:+7.2f}%  "
                  f"hotter {s['hotter_roi']:+7.2f}%  paired t={s['paired_tstat']:+5.2f}")

        if production_leagues:
            prod = base[base["league"].isin(production_leagues)]
            s = _summise(prod)
            out[window]["production"] = s
            print(f"  production leagues ({','.join(sorted(production_leagues))}): "
                  f"n={s['n']}  colder {s['colder_roi']:+.2f}%  hotter {s['hotter_roi']:+.2f}%  "
                  f"paired t={s['paired_tstat']:+.2f}")
        print()

    return out
