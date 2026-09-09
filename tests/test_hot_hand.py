import numpy as np
import pandas as pd

from src.evaluation.hot_hand import (
    _bets_for_window,
    _fair_probs,
    _heat_ratings,
    _team_match_scores,
    run_hot_hand_diagnostic,
)


def _match(date, home, away, ftr, oh, od, oa, league="E0", season="2324"):
    return {
        "Date": pd.Timestamp(date), "HomeTeam": home, "AwayTeam": away, "FTR": ftr,
        "B365H": oh, "B365D": od, "B365A": oa,
        "CustomMaxH": oh, "CustomMaxD": od, "CustomMaxA": oa,
        "PSCH": oh, "PSCD": od, "PSCA": oa,
        "league": league, "season": season,
    }


def _frame(rows):
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def test_fair_probs_strip_vig_sums_to_one():
    row = pd.Series(_match("2024-01-01", "A", "B", "H", 2.0, 3.5, 4.0))
    fair = _fair_probs(row, strip_vig=True)
    assert abs(sum(fair.values()) - 1.0) < 1e-9
    raw = _fair_probs(row, strip_vig=False)
    assert sum(raw.values()) > 1.0  # overround kept


def test_fair_probs_source_priority_pinnacle_first():
    row = pd.Series({
        **_match("2024-01-01", "A", "B", "H", 2.0, 3.5, 4.0),
        "PSCH": 10.0, "PSCD": 10.0, "PSCA": 10.0,  # degenerate but valid → equal probs
    })
    fair = _fair_probs(row, strip_vig=True)
    assert abs(fair["H"] - 1 / 3) < 1e-9


def test_team_match_score_formula():
    # Home won at fair p; score should be (1 - p_home_fair) for home, -p_away_fair for away.
    rows = [_match("2024-01-01", "A", "B", "H", 2.0, 4.0, 4.0)]
    scores = _team_match_scores(_frame(rows), strip_vig=True)
    raw = {"H": 0.5, "D": 0.25, "A": 0.25}
    tot = sum(raw.values())
    p_home, p_away = raw["H"] / tot, raw["A"] / tot
    home_row = scores[scores["is_home"]].iloc[0]
    away_row = scores[~scores["is_home"]].iloc[0]
    assert abs(home_row["score"] - (1.0 - p_home)) < 1e-9
    assert abs(away_row["score"] - (-p_away)) < 1e-9


def test_draw_counts_as_not_won_for_both_sides():
    rows = [_match("2024-01-01", "A", "B", "D", 2.5, 3.0, 3.0)]
    scores = _team_match_scores(_frame(rows), strip_vig=True)
    assert (scores["score"] < 0).all()  # both sides "failed to win"


def test_heat_rating_has_no_lookahead():
    # A team's rating for match k must use only matches < k (shift(1) before rolling).
    rows = [
        _match("2024-01-01", "A", "X1", "H", 2.0, 3.5, 4.0),
        _match("2024-01-08", "A", "X2", "H", 2.0, 3.5, 4.0),
        _match("2024-01-15", "X3", "A", "H", 2.0, 3.5, 4.0),
    ]
    scores = _team_match_scores(_frame(rows), strip_vig=True)
    rated = _heat_ratings(scores, window=2)
    a_rows = rated[rated["team"] == "A"].sort_values("Date")
    # First two matches: fewer than `window` prior games → rating undefined.
    assert pd.isna(a_rows.iloc[0]["heat"])
    assert pd.isna(a_rows.iloc[1]["heat"])
    # Third match: mean of the first two scores, none of the third.
    assert abs(a_rows.iloc[2]["heat"] - a_rows.iloc[:2]["score"].mean()) < 1e-9


def test_bets_pick_the_colder_side_and_sign_profit():
    # Build history so team H_COLD has low scores (keeps losing) and H_HOT high.
    rows = []
    for i in range(6):
        d = f"2024-01-{i + 1:02d}"
        rows.append(_match(d, "COLD", f"Opp{i}", "A", 2.0, 3.5, 3.5))   # COLD loses at home
        rows.append(_match(d, f"Opp{i}b", "HOT", "A", 3.5, 3.5, 2.0))   # HOT wins away
    rows.append(_match("2024-02-01", "COLD", "HOT", "H", 2.5, 3.3, 2.8))
    df = _frame(rows)
    scores = _team_match_scores(df, strip_vig=True)
    bets = _bets_for_window(df, scores, window=6)
    final = bets[(bets["HomeTeam"] == "COLD") & (bets["AwayTeam"] == "HOT")]
    assert len(final) == 1
    b = final.iloc[0]
    assert b["match_rating"] < 0  # home (COLD) is colder
    assert b["colder_side"] == "H"
    # FTR was "H" → colder side won → positive profit at colder_odds - 1.
    assert b["profit"] > 0
    assert abs(b["profit"] - (b["colder_odds"] - 1.0)) < 1e-9
    assert abs(b["hotter_profit"] + 1.0) < 1e-9


def test_run_diagnostic_smoke():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for w in range(30):
        rng.shuffle(teams)
        for k in range(0, 8, 2):
            rows.append(_match(
                pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=w),
                teams[k], teams[k + 1], rng.choice(["H", "D", "A"]),
                2.4, 3.3, 2.9,
            ))
    out = run_hot_hand_diagnostic(_frame(rows), windows=(5,), rating_filters=(0.0,))
    assert 5 in out
    assert "colder_roi" in out[5]["all"][0.0]
    assert out[5]["all"][0.0]["n"] > 0
