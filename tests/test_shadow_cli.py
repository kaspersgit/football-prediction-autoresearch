import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main


def test_shadow_model_commit_uses_unknown_when_local_git_metadata_is_unavailable(monkeypatch):
    """Local recovery must still persist a snapshot outside a Git checkout."""
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def unavailable_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(main.subprocess, "run", unavailable_git)

    assert main._shadow_model_commit() == "unknown"


def _commit_all(repository: Path, message: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "add", "."], cwd=repository, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repository,
        check=True,
        capture_output=True,
        env=env,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_shadow_model_commit_ignores_consecutive_ledger_only_commits(
    tmp_path, monkeypatch
):
    """Scheduled ledger persistence must not create a new model-version group."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "main.py").write_text("MODEL = 1\n")
    model_revision = _commit_all(tmp_path, "add model")

    ledger_dir = tmp_path / "data" / "shadow"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "predictions.csv").write_text("prediction_id\nfirst\n")
    _commit_all(tmp_path, "first ledger update")
    (ledger_dir / "settlements.csv").write_text("prediction_id\nfirst\n")
    _commit_all(tmp_path, "second ledger update")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "stale-event-sha")
    assert main._shadow_model_commit() == model_revision

    (tmp_path / "main.py").write_text("MODEL = 2\n")
    changed_revision = _commit_all(tmp_path, "change model")

    assert main._shadow_model_commit() == changed_revision


def test_settle_shadow_route_settles_without_prediction_inference(monkeypatch):
    """The recovery CLI must not train or create a new prediction snapshot."""
    calls = []

    def settlement_route():
        calls.append("settle")

    def prediction_or_backtest():
        raise AssertionError("--settle-shadow must not run inference")

    monkeypatch.setattr(main, "_run_settle_shadow", settlement_route, raising=False)
    monkeypatch.setattr(main, "_run_predict", prediction_or_backtest)
    monkeypatch.setattr(main, "_run_backtest", prediction_or_backtest)
    monkeypatch.setattr(sys, "argv", ["main.py", "--settle-shadow"])

    main.run_pipeline()

    assert calls == ["settle"]


def test_manual_settlement_refreshes_results_then_settles_and_reports(monkeypatch):
    """The recovery job rebuilds monitoring output from refreshed completed results."""
    results = pd.DataFrame({"FTR": ["H"]})
    calls = []

    def refresh_results():
        calls.append("refresh")

    def load_results():
        calls.append("load")
        return results

    def settle_and_report(loaded_results, settled_at):
        calls.append(("settle_and_report", loaded_results, settled_at))

    monkeypatch.setattr(main, "update_current_season", refresh_results, raising=False)
    monkeypatch.setattr(main, "load_all_data", load_results)
    monkeypatch.setattr(main, "_settle_and_report_shadow", settle_and_report, raising=False)
    monkeypatch.setattr(main.pd.Timestamp, "now", lambda tz: pd.Timestamp("2026-08-05T12:00:00Z"))

    main._run_settle_shadow()

    assert calls[:2] == ["refresh", "load"]
    assert calls[2][0] == "settle_and_report"
    assert calls[2][1] is results
    assert calls[2][2] == pd.Timestamp("2026-08-05T12:00:00Z")


def test_shadow_snapshot_records_three_non_value_outcomes_for_high_vig_fixture(
    monkeypatch, tmp_path
):
    """The audit ledger must retain the fixture even when the live report omits high vig."""
    fixture_features = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-08-09")],
            "league": ["E0"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "B365H": [2.0],
            "B365D": [2.0],
            "B365A": [2.0],
        }
    )
    monkeypatch.setattr(main, "PREDICTIONS_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(main, "_shadow_run_id", lambda fetched_at: "test-run", raising=False)
    monkeypatch.setattr(main, "_shadow_model_commit", lambda: "test-commit", raising=False)

    appended = main._record_shadow_predictions(
        fixture_features=fixture_features,
        y_proba=np.array([[0.45, 0.30, 0.25]]),
        classes=["H", "D", "A"],
        fetched_at=pd.Timestamp("2026-08-05T12:00:00Z"),
        threshold=0.05,
        league_thresholds={"E0": 0.05},
    )

    records = pd.read_csv(tmp_path / "predictions.csv")
    assert appended == 3
    assert set(records["outcome"]) == {"H", "D", "A"}
    assert records["is_value_bet"].eq(False).all()
    assert main._build_prediction_rows(
        fixture_features,
        np.array([[0.45, 0.30, 0.25]]),
        ["H", "D", "A"],
        threshold=0.05,
        league_thresholds={"E0": 0.05},
    ) == []


@pytest.mark.parametrize(
    "odds, probabilities",
    [
        ((5.5, 3.2, 3.5), (0.38, 0.35, 0.27)),
        ((2.0, 3.5, 4.0), (0.75, 0.15, 0.10)),
    ],
    ids=["excessive-odds", "excessive-edge"],
)
def test_shadow_and_live_rows_reject_the_same_non_executable_edge(
    monkeypatch, tmp_path, odds, probabilities
):
    """The monitor must not count an edge that the live report rejects."""
    fixture_features = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-08-09")],
            "league": ["E0"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "B365H": [odds[0]],
            "B365D": [odds[1]],
            "B365A": [odds[2]],
        }
    )
    probability_array = np.array([probabilities])
    monkeypatch.setattr(main, "PREDICTIONS_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(main, "_shadow_run_id", lambda fetched_at: "test-run")
    monkeypatch.setattr(main, "_shadow_model_commit", lambda: "test-commit")

    main._record_shadow_predictions(
        fixture_features=fixture_features,
        y_proba=probability_array,
        classes=["H", "D", "A"],
        fetched_at=pd.Timestamp("2026-08-05T12:00:00Z"),
        threshold=0.05,
        league_thresholds={"E0": 0.05},
    )

    shadow_home = pd.read_csv(tmp_path / "predictions.csv").query("outcome == 'H'").iloc[0]
    live_rows = main._build_prediction_rows(
        fixture_features,
        probability_array,
        ["H", "D", "A"],
        threshold=0.05,
        league_thresholds={"E0": 0.05},
    )
    assert bool(shadow_home.is_value_bet) is False
    assert live_rows[0]["ValueBets"] == []


def _make_pinnacle_fixture(psh, psd, psa):
    return pd.DataFrame({
        "Date": [pd.Timestamp("2026-08-09")],
        "league": ["E0"],
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "B365H": [2.0],
        "B365D": [4.0],
        "B365A": [4.0],
        "PSH": [psh],
        "PSD": [psd],
        "PSA": [psa],
    })


def test_build_prediction_rows_vetoes_bet_when_pinnacle_disagrees():
    fixture_features = _make_pinnacle_fixture(3.0, 3.0, 3.0)  # pinnacle_fair[H] = 0.333

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
        pinnacle_confirmation_margin=0.015,
    )

    assert rows[0]["ValueBets"] == []


def test_build_prediction_rows_keeps_bet_when_pinnacle_agrees():
    fixture_features = _make_pinnacle_fixture(1.8, 6.0, 6.0)  # pinnacle_fair[H] = 0.625

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
        pinnacle_confirmation_margin=0.015,
    )

    assert [o for o, _ in rows[0]["ValueBets"]] == ["H"]


def test_build_prediction_rows_vetoes_bet_when_pinnacle_odds_are_null():
    fixture_features = _make_pinnacle_fixture(float("nan"), float("nan"), float("nan"))

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
        pinnacle_confirmation_margin=0.015,
    )

    assert rows[0]["ValueBets"] == []


def test_build_prediction_rows_pinnacle_check_is_off_by_default():
    fixture_features = _make_pinnacle_fixture(3.0, 3.0, 3.0)  # would veto if active

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
    )

    assert [o for o, _ in rows[0]["ValueBets"]] == ["H"]
