import pandas as pd
import pytest

import main
import src.data.pinnacle_odds as pinnacle_odds


def _fake_odds():
    return pd.DataFrame(
        [
            {
                "league": "E0",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": pd.Timestamp("2026-08-21"),
                "PSH": 2.00,
                "PSD": 3.50,
                "PSA": 4.00,
            }
        ]
    )


def test_track_pinnacle_vig_appends_snapshot_with_overround(monkeypatch, tmp_path):
    monkeypatch.setattr(pinnacle_odds, "fetch_pinnacle_odds", lambda leagues: _fake_odds())
    monkeypatch.chdir(tmp_path)

    main._run_track_pinnacle_vig()

    history_path = tmp_path / "data" / "diagnostics" / "pinnacle_vig_history.csv"
    assert history_path.exists()
    recorded = pd.read_csv(history_path)
    assert len(recorded) == 1
    row = recorded.iloc[0]
    assert row["league"] == "E0"
    assert row["home_team"] == "Arsenal"
    expected_overround = (1 / 2.00 + 1 / 3.50 + 1 / 4.00) - 1.0
    assert row["overround"] == pytest.approx(expected_overround)


def test_track_pinnacle_vig_appends_to_existing_history(monkeypatch, tmp_path):
    monkeypatch.setattr(pinnacle_odds, "fetch_pinnacle_odds", lambda leagues: _fake_odds())
    monkeypatch.chdir(tmp_path)

    main._run_track_pinnacle_vig()
    main._run_track_pinnacle_vig()

    recorded = pd.read_csv(tmp_path / "data" / "diagnostics" / "pinnacle_vig_history.csv")
    assert len(recorded) == 2


def test_track_pinnacle_vig_skips_write_when_no_odds(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        pinnacle_odds, "fetch_pinnacle_odds", lambda leagues: pd.DataFrame(columns=["league"])
    )
    monkeypatch.chdir(tmp_path)

    main._run_track_pinnacle_vig()

    assert not (tmp_path / "data" / "diagnostics").exists()
    assert "No live Pinnacle odds available" in capsys.readouterr().out
