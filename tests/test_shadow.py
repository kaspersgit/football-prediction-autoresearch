import numpy as np
import pandas as pd
import pytest

from src.evaluation.shadow import (
    PREDICTION_COLUMNS,
    SETTLEMENT_COLUMNS,
    append_ledger,
    build_prediction_records,
    build_settlement_records,
    read_shadow_ledgers,
    settle_shadow_predictions,
)


def _fixture_features():
    return pd.DataFrame(
        {
            "Date": ["2026-08-08"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "league": ["E0"],
            "B365H": [2.0],
            "B365D": [3.5],
            "B365A": [4.0],
            "CustomMaxH": [float("nan")],
            "CustomMaxD": [3.6],
            "CustomMaxA": [4.2],
            "CustomMaxBkH": [""],
            "CustomMaxBkD": ["Bookmaker D"],
            "CustomMaxBkA": ["Bookmaker A"],
        }
    )


def test_build_prediction_records_captures_all_outcomes_and_prices():
    """Missing canonical outcome rows would make this audit snapshot incomplete."""
    records = build_prediction_records(
        fixture_features=_fixture_features(),
        y_proba=np.array([[0.20, 0.25, 0.55]]),
        classes=["A", "D", "H"],
        fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
        threshold=0.03,
        league_thresholds={"E0": 0.04},
        run_id="run-1",
        model_commit="abc123",
    )

    assert records["outcome"].tolist() == ["A", "D", "H"]
    assert records["prediction_id"].nunique() == 3
    assert records.loc[records.outcome == "H", "captured_best_odds"].item() == 2.0
    assert records.loc[records.outcome == "H", "applied_threshold"].item() == 0.04


def test_build_prediction_records_accepts_the_exact_forward_boundary():
    """Treating the inclusive boundary as legacy would lose the first valid snapshot."""
    records = build_prediction_records(
        fixture_features=_fixture_features(),
        y_proba=np.array([[0.20, 0.25, 0.55]]),
        classes=["A", "D", "H"],
        fetched_at=pd.Timestamp("2026-08-04T00:00:00Z"),
        threshold=0.03,
        league_thresholds={},
        run_id="run-1",
        model_commit="abc123",
    )

    assert len(records) == 3


@pytest.mark.parametrize(
    "fetched_at, message",
    [
        (pd.Timestamp("2026-08-04T00:00:00"), "timezone-aware"),
        (pd.Timestamp("2026-08-03T23:59:59Z"), "forward shadow evaluation boundary"),
    ],
    ids=["naive", "pre-boundary"],
)
def test_build_prediction_records_rejects_non_forward_timestamps(fetched_at, message):
    """Naive or historical timestamps would make the forward-only boundary ambiguous."""
    with pytest.raises(ValueError, match=message):
        build_prediction_records(
            fixture_features=_fixture_features(),
            y_proba=np.array([[0.20, 0.25, 0.55]]),
            classes=["A", "D", "H"],
            fetched_at=fetched_at,
            threshold=0.03,
            league_thresholds={},
            run_id="run-1",
            model_commit="abc123",
        )


def test_build_prediction_records_maps_reordered_classes_to_canonical_outcomes():
    """Indexing probabilities by canonical position would attach them to the wrong outcome."""
    records = build_prediction_records(
        fixture_features=_fixture_features(),
        y_proba=np.array([[0.55, 0.20, 0.25]]),
        classes=["H", "A", "D"],
        fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
        threshold=0.03,
        league_thresholds={},
        run_id="run-1",
        model_commit="abc123",
    )

    assert records.set_index("outcome")["model_probability"].to_dict() == {
        "A": 0.20,
        "D": 0.25,
        "H": 0.55,
    }


def test_build_prediction_records_excludes_non_production_leagues():
    """Persisting an excluded league would expand the monitor beyond production scope."""
    fixtures = _fixture_features().assign(league="D1")

    records = build_prediction_records(
        fixture_features=fixtures,
        y_proba=np.array([[0.20, 0.25, 0.55]]),
        classes=["A", "D", "H"],
        fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
        threshold=0.03,
        league_thresholds={},
        run_id="run-1",
        model_commit="abc123",
    )

    assert records.empty
    assert records.columns.tolist() == PREDICTION_COLUMNS


def test_build_prediction_records_uses_deterministic_identity_inputs():
    """Unstable IDs would duplicate reruns, while missing identity fields would alias records."""
    kwargs = {
        "fixture_features": _fixture_features(),
        "y_proba": np.array([[0.20, 0.25, 0.55]]),
        "classes": ["A", "D", "H"],
        "fetched_at": pd.Timestamp("2026-08-04T06:00:00Z"),
        "threshold": 0.03,
        "league_thresholds": {},
        "run_id": "run-1",
        "model_commit": "abc123",
    }

    first = build_prediction_records(**kwargs)
    identical = build_prediction_records(**kwargs)
    different_run = build_prediction_records(**{**kwargs, "run_id": "run-2"})

    assert first["prediction_id"].tolist() == identical["prediction_id"].tolist()
    assert set(first["prediction_id"]).isdisjoint(different_run["prediction_id"])
    assert first["prediction_id"].nunique() == 3


def test_build_prediction_records_rejects_duplicate_class_labels():
    """A duplicate label would silently select one arbitrary probability column."""
    with pytest.raises(ValueError, match="unique"):
        build_prediction_records(
            fixture_features=_fixture_features(),
            y_proba=np.array([[0.20, 0.25, 0.30, 0.25]]),
            classes=["A", "D", "H", "H"],
            fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
            threshold=0.03,
            league_thresholds={},
            run_id="run-1",
            model_commit="abc123",
        )


@pytest.mark.parametrize(
    "y_proba",
    [
        np.array([0.20, 0.25, 0.55]),
        np.array([[0.20, 0.25]]),
        np.array([[0.20, 0.25, 0.55, 0.0]]),
        np.array([[0.20, 0.25, 0.55], [0.30, 0.30, 0.40]]),
    ],
    ids=["one-dimensional", "too-few-columns", "too-many-columns", "too-many-rows"],
)
def test_build_prediction_records_rejects_probability_shape_mismatch(y_proba):
    """An unexpected probability shape would misalign fixtures or outcome columns."""
    with pytest.raises(ValueError, match="shape"):
        build_prediction_records(
            fixture_features=_fixture_features(),
            y_proba=y_proba,
            classes=["A", "D", "H"],
            fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
            threshold=0.03,
            league_thresholds={},
            run_id="run-1",
            model_commit="abc123",
        )


def _prediction_records():
    return build_prediction_records(
        fixture_features=_fixture_features(),
        y_proba=np.array([[0.20, 0.25, 0.55]]),
        classes=["A", "D", "H"],
        fetched_at=pd.Timestamp("2026-08-04T06:00:00Z"),
        threshold=0.03,
        league_thresholds={"E0": 0.04},
        run_id="run-1",
        model_commit="abc123",
    )


def _prediction_rows():
    return _prediction_records()


def _completed_result(ftr="H", custom_max_h=1.80):
    return pd.DataFrame(
        {
            "Date": ["2026-08-08"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "league": ["E0"],
            "FTR": [ftr],
            "B365H": [1.75],
            "B365D": [3.25],
            "B365A": [4.50],
            "CustomMaxH": [custom_max_h],
            "CustomMaxD": [3.50],
            "CustomMaxA": [4.75],
        }
    )


def test_append_ledger_writes_new_records_with_a_single_header(tmp_path):
    """Duplicating the CSV header would corrupt later ledger reads."""
    path = tmp_path / "predictions.csv"

    appended = append_ledger(_prediction_records(), path, "prediction_id")

    assert appended == 3
    assert path.read_text().count("prediction_id") == 1
    assert len(pd.read_csv(path)) == 3


def test_append_ledger_makes_an_identical_rerun_a_noop(tmp_path):
    """Replaying a scheduled prediction run must not duplicate its audit records."""
    path = tmp_path / "predictions.csv"
    records = _prediction_records()
    append_ledger(records, path, "prediction_id")

    appended = append_ledger(records, path, "prediction_id")

    assert appended == 0
    assert len(pd.read_csv(path)) == 3


def test_append_ledger_rejects_conflicting_existing_prediction_id(tmp_path):
    """Changing a stored prediction behind the same ID would violate immutability."""
    path = tmp_path / "predictions.csv"
    original = _prediction_records()
    append_ledger(original, path, "prediction_id")
    conflicting = original.copy()
    conflicting.loc[0, "model_probability"] = 0.99

    with pytest.raises(ValueError, match="conflicting"):
        append_ledger(conflicting, path, "prediction_id")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.drop(columns="model_commit"),
        lambda frame: frame.assign(unexpected="value"),
        lambda frame: frame[list(reversed(PREDICTION_COLUMNS))],
        lambda frame: frame.set_axis(
            [*PREDICTION_COLUMNS[:-1], "prediction_id"], axis="columns"
        ),
    ],
    ids=["missing", "extra", "reordered", "duplicate"],
)
def test_append_empty_rejects_malformed_stored_prediction_columns(tmp_path, mutate):
    """An empty append must not bypass validation of an already malformed ledger."""
    path = tmp_path / "predictions.csv"
    mutate(_prediction_rows().iloc[:1].copy()).to_csv(path, index=False)

    with pytest.raises(ValueError, match="columns"):
        append_ledger(
            pd.DataFrame(columns=PREDICTION_COLUMNS), path, "prediction_id"
        )


def test_append_empty_rejects_identical_duplicate_stored_ids(tmp_path):
    """Repeated physical rows would make the append-only ledger ID non-unique."""
    path = tmp_path / "predictions.csv"
    row = _prediction_rows().iloc[[0]]
    pd.concat([row, row], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate prediction_id"):
        append_ledger(
            pd.DataFrame(columns=PREDICTION_COLUMNS), path, "prediction_id"
        )


def test_append_empty_rejects_conflicting_duplicate_stored_ids(tmp_path):
    """Two payloads behind one stored ID must fail before any new append is considered."""
    path = tmp_path / "predictions.csv"
    row = _prediction_rows().iloc[[0]]
    conflicting = row.copy()
    conflicting.loc[:, "model_probability"] = 0.99
    pd.concat([row, conflicting], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match="conflicting duplicate prediction_id"):
        append_ledger(
            pd.DataFrame(columns=PREDICTION_COLUMNS), path, "prediction_id"
        )


@pytest.mark.parametrize(
    "column, invalid_value, message",
    [
        ("prediction_id", "", "non-empty"),
        ("outcome", "X", "outcome"),
        ("is_value_bet", "sometimes", "is_value_bet"),
        ("model_probability", 1.1, "model_probability"),
        ("captured_best_odds", 0.0, "captured_best_odds"),
    ],
)
def test_append_empty_rejects_invalid_stored_prediction_values(
    tmp_path, column, invalid_value, message
):
    """Invalid persisted domains must fail at read time instead of entering reports."""
    path = tmp_path / "predictions.csv"
    records = _prediction_rows().iloc[[0]].copy()
    records[column] = [invalid_value]
    records.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        append_ledger(
            pd.DataFrame(columns=PREDICTION_COLUMNS), path, "prediction_id"
        )


def test_read_shadow_ledgers_rejects_orphan_settlement(tmp_path):
    """A settlement without a prediction must not be silently discarded by the join."""
    predictions_path = tmp_path / "predictions.csv"
    settlements_path = tmp_path / "settlements.csv"
    append_ledger(_prediction_rows(), predictions_path, "prediction_id")
    settlement = build_settlement_records(
        predictions=_prediction_rows(),
        settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
        results=_completed_result(),
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )[SETTLEMENT_COLUMNS].iloc[[0]]
    settlement.loc[:, "prediction_id"] = "orphan"
    settlement.to_csv(settlements_path, index=False)

    with pytest.raises(ValueError, match="without matching prediction"):
        read_shadow_ledgers(predictions_path, settlements_path)


def test_read_shadow_ledgers_rejects_settlements_without_prediction_file(tmp_path):
    """A settlement-only ledger is malformed, not an empty monitor."""
    settlements_path = tmp_path / "settlements.csv"
    settlement = build_settlement_records(
        predictions=_prediction_rows(),
        settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
        results=_completed_result(),
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )[SETTLEMENT_COLUMNS].iloc[[0]]
    settlement.to_csv(settlements_path, index=False)

    with pytest.raises(ValueError, match="without matching prediction"):
        read_shadow_ledgers(tmp_path / "missing-predictions.csv", settlements_path)


def test_build_settlements_uses_captured_profit_and_result_file_closing_proxy():
    """Using refreshed odds for profit would misstate hypothetical execution."""
    settlements = build_settlement_records(
        predictions=_prediction_rows(),
        settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
        results=_completed_result(ftr="H", custom_max_h=1.80),
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )

    home = settlements[settlements.outcome == "H"].iloc[0]
    assert home.profit == pytest.approx(1.0)
    assert home.closing_line_value == pytest.approx(2.0 / 1.8 - 1.0)


def test_build_settlements_leaves_predictions_without_completed_results_pending():
    """Settling a fixture without a matching result would create a false outcome."""
    results = _completed_result()
    results.loc[0, "HomeTeam"] = "Liverpool"

    settlements = build_settlement_records(
        predictions=_prediction_rows(),
        settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
        results=results,
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )

    assert settlements.empty
    assert settlements.columns.tolist() == [*SETTLEMENT_COLUMNS, "outcome"]


def test_build_settlements_rebuilds_existing_predictions_with_original_settlement_time():
    """An idempotent rerun must compare the original immutable settlement row."""
    predictions = _prediction_rows()
    existing = build_settlement_records(
        predictions=predictions,
        settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
        results=_completed_result(),
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )

    settlements = build_settlement_records(
        predictions=predictions,
        settlements=existing[SETTLEMENT_COLUMNS],
        results=_completed_result(),
        settled_at=pd.Timestamp("2026-08-13T06:00:00Z"),
    )

    assert settlements["settled_at"].tolist() == existing["settled_at"].tolist()


def test_build_settlements_rejects_ambiguous_completed_fixture_matches():
    """Choosing one of two completed results could settle a fixture incorrectly."""
    results = pd.concat([_completed_result(), _completed_result()], ignore_index=True)

    with pytest.raises(ValueError, match="ambiguous"):
        build_settlement_records(
            predictions=_prediction_rows(),
            settlements=pd.DataFrame(columns=SETTLEMENT_COLUMNS),
            results=results,
            settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
        )


def test_settle_shadow_predictions_appends_new_settlements_once(tmp_path):
    """Repeated settlement jobs must append each immutable outcome at most once."""
    predictions_path = tmp_path / "predictions.csv"
    settlements_path = tmp_path / "settlements.csv"
    append_ledger(_prediction_rows(), predictions_path, "prediction_id")

    first = settle_shadow_predictions(
        _completed_result(),
        predictions_path=predictions_path,
        settlements_path=settlements_path,
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )
    second = settle_shadow_predictions(
        _completed_result(),
        predictions_path=predictions_path,
        settlements_path=settlements_path,
        settled_at=pd.Timestamp("2026-08-13T06:00:00Z"),
    )

    assert first == 3
    assert second == 0
    assert len(pd.read_csv(settlements_path)) == 3


def test_settle_shadow_predictions_rejects_changed_result_file_price_for_existing_id(tmp_path):
    """A corrected closing price must not silently overwrite an immutable settlement."""
    predictions_path = tmp_path / "predictions.csv"
    settlements_path = tmp_path / "settlements.csv"
    append_ledger(_prediction_rows(), predictions_path, "prediction_id")
    settle_shadow_predictions(
        _completed_result(),
        predictions_path=predictions_path,
        settlements_path=settlements_path,
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )
    corrected_results = _completed_result(custom_max_h=1.70)

    with pytest.raises(ValueError, match="conflicting"):
        settle_shadow_predictions(
            corrected_results,
            predictions_path=predictions_path,
            settlements_path=settlements_path,
            settled_at=pd.Timestamp("2026-08-13T06:00:00Z"),
        )


def test_settle_shadow_predictions_rejects_legacy_predictions_before_shadow_boundary(tmp_path):
    """Settling an imported pre-shadow prediction would fabricate forward evidence."""
    predictions_path = tmp_path / "predictions.csv"
    settlements_path = tmp_path / "settlements.csv"
    legacy_predictions = _prediction_rows()
    legacy_predictions["fetched_at"] = "2026-08-03T23:59:59Z"
    legacy_predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="forward shadow evaluation boundary"):
        settle_shadow_predictions(
            _completed_result(),
            predictions_path=predictions_path,
            settlements_path=settlements_path,
            settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
        )


def test_settle_shadow_predictions_persists_missing_closing_price_as_empty_clv(tmp_path):
    """A missing result-file price must retain the settlement without inventing CLV."""
    predictions_path = tmp_path / "predictions.csv"
    settlements_path = tmp_path / "settlements.csv"
    append_ledger(_prediction_rows(), predictions_path, "prediction_id")
    results = _completed_result()
    results.loc[0, ["B365H", "CustomMaxH"]] = float("nan")

    settled = settle_shadow_predictions(
        results,
        predictions_path=predictions_path,
        settlements_path=settlements_path,
        settled_at=pd.Timestamp("2026-08-12T06:00:00Z"),
    )
    stored = pd.read_csv(settlements_path)
    home = stored[stored.prediction_id == _prediction_rows().query("outcome == 'H'").iloc[0].prediction_id]

    assert settled == 3
    assert pd.isna(home.iloc[0].result_file_best_odds)
    assert pd.isna(home.iloc[0].closing_line_value)
