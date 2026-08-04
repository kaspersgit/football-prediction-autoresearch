"""Append-only storage primitives for forward shadow evaluation."""

from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PRODUCTION_LEAGUES
from src.evaluation.eligibility import is_execution_eligible

SHADOW_START_UTC = pd.Timestamp("2026-08-04T00:00:00Z")
PREDICTIONS_PATH = Path("data/shadow/predictions.csv")
SETTLEMENTS_PATH = Path("data/shadow/settlements.csv")

PREDICTION_COLUMNS = [
    "prediction_id",
    "run_id",
    "model_commit",
    "fetched_at",
    "fixture_date",
    "league",
    "home_team",
    "away_team",
    "outcome",
    "model_probability",
    "fair_probability",
    "captured_b365_odds",
    "captured_best_odds",
    "captured_best_bookmaker",
    "edge",
    "applied_threshold",
    "is_value_bet",
    "is_production_fixture",
]

SETTLEMENT_COLUMNS = [
    "prediction_id",
    "settled_at",
    "actual_result",
    "is_winner",
    "profit",
    "result_file_b365_odds",
    "result_file_best_odds",
    "closing_line_value",
]

_OUTCOMES = ("A", "D", "H")
_SETTLEMENT_RECORD_COLUMNS = [*SETTLEMENT_COLUMNS, "outcome"]


def _prediction_id(
    run_id: str,
    fixture_date: pd.Timestamp,
    league: str,
    home_team: str,
    away_team: str,
    outcome: str,
) -> str:
    fixture_key = "|".join(
        (run_id, fixture_date.date().isoformat(), league, home_team, away_team, outcome)
    )
    return hashlib.sha256(fixture_key.encode("utf-8")).hexdigest()


def _captured_best_price(row: pd.Series, outcome: str, b365_odds: float) -> tuple[float, str]:
    custom_odds = row.get(f"CustomMax{outcome}")
    try:
        if custom_odds is not None and not pd.isna(custom_odds):
            return float(custom_odds), str(row.get(f"CustomMaxBk{outcome}", ""))
    except (TypeError, ValueError):
        pass
    return b365_odds, "B365"


def build_prediction_records(
    fixture_features: pd.DataFrame,
    y_proba: np.ndarray,
    classes,
    fetched_at,
    threshold: float,
    league_thresholds: dict[str, float],
    run_id: str,
    model_commit: str,
) -> pd.DataFrame:
    """Build one immutable prediction-time record for each fixture and outcome."""
    fetched_at = pd.Timestamp(fetched_at)
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be a timezone-aware UTC timestamp")
    fetched_at = fetched_at.tz_convert("UTC")
    if fetched_at < SHADOW_START_UTC:
        raise ValueError("fetched_at precedes the forward shadow evaluation boundary")
    classes = list(classes)
    if len(classes) != len(_OUTCOMES) or len(set(classes)) != len(classes):
        raise ValueError("classes must contain exactly three unique labels: A, D, and H")
    if set(classes) != set(_OUTCOMES):
        raise ValueError("classes must contain exactly A, D, and H")
    if np.shape(y_proba) != (len(fixture_features), len(_OUTCOMES)):
        raise ValueError(
            "y_proba shape must equal (len(fixture_features), 3)"
        )

    class_positions = {outcome: position for position, outcome in enumerate(classes)}

    records = []
    for fixture_position, (_, fixture) in enumerate(fixture_features.iterrows()):
        league = str(fixture.get("league", ""))
        if league not in PRODUCTION_LEAGUES:
            continue

        fixture_date = pd.Timestamp(fixture["Date"])
        home_team = str(fixture["HomeTeam"])
        away_team = str(fixture["AwayTeam"])
        b365_odds = {outcome: float(fixture[f"B365{outcome}"]) for outcome in _OUTCOMES}
        total_implied = sum(1.0 / odds for odds in b365_odds.values())
        fixture_overround = total_implied - 1.0
        applied_threshold = float(league_thresholds.get(league, threshold))

        for outcome in _OUTCOMES:
            model_probability = float(y_proba[fixture_position, class_positions[outcome]])
            fair_probability = (1.0 / b365_odds[outcome]) / total_implied
            edge = model_probability - fair_probability
            captured_best_odds, captured_best_bookmaker = _captured_best_price(
                fixture, outcome, b365_odds[outcome]
            )
            records.append(
                {
                    "prediction_id": _prediction_id(
                        run_id, fixture_date, league, home_team, away_team, outcome
                    ),
                    "run_id": run_id,
                    "model_commit": model_commit,
                    "fetched_at": fetched_at,
                    "fixture_date": fixture_date,
                    "league": league,
                    "home_team": home_team,
                    "away_team": away_team,
                    "outcome": outcome,
                    "model_probability": model_probability,
                    "fair_probability": fair_probability,
                    "captured_b365_odds": b365_odds[outcome],
                    "captured_best_odds": captured_best_odds,
                    "captured_best_bookmaker": captured_best_bookmaker,
                    "edge": edge,
                    "applied_threshold": applied_threshold,
                    "is_value_bet": is_execution_eligible(
                        edge=edge,
                        threshold=applied_threshold,
                        b365_odds=b365_odds[outcome],
                        fixture_overround=fixture_overround,
                    ),
                    "is_production_fixture": True,
                }
            )

    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)


def _normalise_csv_records(records: pd.DataFrame) -> pd.DataFrame:
    """Apply CSV's own stable representation before comparing ledger rows."""
    return pd.read_csv(StringIO(records.to_csv(index=False)))


def _canonical_columns(records: pd.DataFrame, id_column: str) -> list[str]:
    """Return the schema selected by a frame, rejecting all non-canonical headers."""
    if records.columns.has_duplicates:
        raise ValueError("ledger columns must not contain duplicate names")
    if id_column not in records:
        raise ValueError(f"ledger records must include {id_column!r}")
    columns = list(records.columns)
    if columns == PREDICTION_COLUMNS:
        return PREDICTION_COLUMNS
    if columns == SETTLEMENT_COLUMNS:
        return SETTLEMENT_COLUMNS
    raise ValueError("ledger columns must use the exact ordered prediction or settlement schema")


def _validate_non_empty_ids(records: pd.DataFrame, id_column: str) -> None:
    if records[id_column].isna().any() or (
        records[id_column].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"ledger records must have non-empty {id_column!r} values")

    duplicated = records[id_column].duplicated(keep=False)
    if not duplicated.any():
        return
    for record_id, group in records.loc[duplicated].groupby(id_column, sort=False):
        first = group.iloc[0]
        if not all(first.equals(row) for _, row in group.iloc[1:].iterrows()):
            raise ValueError(f"conflicting duplicate {id_column}: {record_id}")
    duplicate_id = records.loc[duplicated, id_column].iloc[0]
    raise ValueError(f"duplicate {id_column}: {duplicate_id}")


def _validate_boolean_domain(records: pd.DataFrame, column: str) -> None:
    def is_boolean(value: object) -> bool:
        return isinstance(value, (bool, np.bool_)) or (
            isinstance(value, str) and value.strip().lower() in {"true", "false"}
        )

    if not records[column].map(is_boolean).all():
        raise ValueError(f"ledger {column!r} values must be true or false")


def _numeric_values(
    records: pd.DataFrame,
    column: str,
    *,
    allow_missing: bool = False,
) -> pd.Series:
    values = pd.to_numeric(records[column], errors="coerce")
    invalid_missing = values.isna() & (not allow_missing or records[column].notna())
    if invalid_missing.any() or np.isinf(values.dropna()).any():
        qualifier = "numeric or empty" if allow_missing else "finite numeric"
        raise ValueError(f"ledger {column!r} values must be {qualifier}")
    return values


def _validate_prediction_domains(records: pd.DataFrame) -> None:
    if not records["outcome"].isin(_OUTCOMES).all():
        raise ValueError("ledger 'outcome' values must be A, D, or H")
    for column in ("is_value_bet", "is_production_fixture"):
        _validate_boolean_domain(records, column)
    for column in ("model_probability", "fair_probability"):
        values = _numeric_values(records, column)
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"ledger {column!r} values must be between 0 and 1")
    for column in ("captured_b365_odds", "captured_best_odds"):
        values = _numeric_values(records, column)
        if not values.gt(0.0).all():
            raise ValueError(f"ledger {column!r} values must be greater than zero")
    _numeric_values(records, "edge")
    _numeric_values(records, "applied_threshold")

    for column in ("run_id", "model_commit", "league", "home_team", "away_team"):
        if records[column].isna().any() or (
            records[column].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"ledger {column!r} values must be non-empty")
    try:
        pd.to_datetime(records["fixture_date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("ledger 'fixture_date' values must be valid timestamps") from error
    for value in records["fetched_at"]:
        fetched_at = _utc_timestamp(value, "prediction fetched_at")
        if fetched_at < SHADOW_START_UTC:
            raise ValueError("prediction fetched_at precedes the forward shadow evaluation boundary")


def _validate_settlement_domains(records: pd.DataFrame) -> None:
    if not records["actual_result"].isin(_OUTCOMES).all():
        raise ValueError("ledger 'actual_result' values must be A, D, or H")
    _validate_boolean_domain(records, "is_winner")
    _numeric_values(records, "profit")
    for column in ("result_file_b365_odds", "result_file_best_odds"):
        values = _numeric_values(records, column, allow_missing=True)
        if not values.dropna().gt(0.0).all():
            raise ValueError(f"ledger {column!r} values must be greater than zero or empty")
    _numeric_values(records, "closing_line_value", allow_missing=True)
    for value in records["settled_at"]:
        _utc_timestamp(value, "settled_at")


def _validate_ledger(records: pd.DataFrame, columns: list[str], id_column: str) -> None:
    actual_columns = _canonical_columns(records, id_column)
    if actual_columns != columns:
        raise ValueError("ledger columns do not match the expected exact ordered schema")
    _validate_non_empty_ids(records, id_column)
    if columns == PREDICTION_COLUMNS:
        _validate_prediction_domains(records)
    else:
        _validate_settlement_domains(records)


def _read_ledger(path: Path, columns: list[str], id_column: str) -> pd.DataFrame:
    """Read and validate one canonical ledger, treating a missing file as empty."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        records = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"ledger {path} has no canonical header") from error
    _validate_ledger(records, columns, id_column)
    return records


def validate_shadow_ledgers(
    predictions: pd.DataFrame, settlements: pd.DataFrame
) -> None:
    """Validate both canonical schemas and settlement foreign keys."""
    _validate_ledger(predictions, PREDICTION_COLUMNS, "prediction_id")
    _validate_ledger(settlements, SETTLEMENT_COLUMNS, "prediction_id")
    prediction_ids = set(predictions["prediction_id"].astype(str))
    settlement_ids = set(settlements["prediction_id"].astype(str))
    orphan_ids = sorted(settlement_ids - prediction_ids)
    if orphan_ids:
        raise ValueError(
            "settlement prediction_id values without matching prediction: "
            + ", ".join(orphan_ids)
        )


def read_shadow_ledgers(
    predictions_path: Path = PREDICTIONS_PATH,
    settlements_path: Path = SETTLEMENTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both canonical ledgers and enforce settlement foreign-key integrity."""
    predictions = _read_ledger(predictions_path, PREDICTION_COLUMNS, "prediction_id")
    settlements = _read_ledger(settlements_path, SETTLEMENT_COLUMNS, "prediction_id")
    validate_shadow_ledgers(predictions, settlements)
    return predictions, settlements


def append_ledger(records: pd.DataFrame, path: Path, id_column: str) -> int:
    """Append new ledger rows while rejecting attempts to mutate existing IDs."""
    columns = _canonical_columns(records, id_column)
    _validate_ledger(records, columns, id_column)
    incoming = _normalise_csv_records(records)
    _validate_ledger(incoming, columns, id_column)
    existing = _read_ledger(path, columns, id_column)
    if incoming.empty:
        return 0

    known_rows: dict[object, pd.Series] = {}
    for _, row in existing.iterrows():
        record_id = row[id_column]
        previous = known_rows.get(record_id)
        if previous is not None and not previous.equals(row):
            raise ValueError(f"conflicting duplicate {id_column}: {record_id}")
        known_rows[record_id] = row

    rows_to_append = []
    for _, row in incoming.iterrows():
        record_id = row[id_column]
        previous = known_rows.get(record_id)
        if previous is None:
            rows_to_append.append(row)
            known_rows[record_id] = row
        elif not previous.equals(row):
            raise ValueError(f"conflicting duplicate {id_column}: {record_id}")

    if not rows_to_append:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_to_append, columns=records.columns).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )
    return len(rows_to_append)


def _fixture_key(row: pd.Series, date_column: str) -> tuple[str, str, str, str]:
    fixture_date = pd.Timestamp(row[date_column]).date().isoformat()
    return (
        fixture_date,
        str(row["league"]),
        str(row["HomeTeam"] if "HomeTeam" in row else row["home_team"]),
        str(row["AwayTeam"] if "AwayTeam" in row else row["away_team"]),
    )


def _utc_timestamp(value, field_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware UTC timestamp")
    return timestamp.tz_convert("UTC")


def _result_file_price(result: pd.Series, outcome: str) -> float:
    custom_odds = result.get(f"CustomMax{outcome}")
    if custom_odds is not None and not pd.isna(custom_odds):
        try:
            custom_price = float(custom_odds)
        except (TypeError, ValueError):
            custom_price = float("nan")
        if custom_price > 0:
            return custom_price

    b365_odds = result.get(f"B365{outcome}")
    if b365_odds is not None and not pd.isna(b365_odds):
        try:
            b365_price = float(b365_odds)
        except (TypeError, ValueError):
            b365_price = float("nan")
        if b365_price > 0:
            return b365_price
    return float("nan")


def build_settlement_records(
    predictions: pd.DataFrame,
    settlements: pd.DataFrame,
    results: pd.DataFrame,
    settled_at,
) -> pd.DataFrame:
    """Build new immutable outcomes for predictions with one completed matching result."""
    settled_at = _utc_timestamp(settled_at, "settled_at")

    completed_results: dict[tuple[str, str, str, str], pd.Series] = {}
    for _, result in results.iterrows():
        actual_result = result.get("FTR")
        if actual_result not in _OUTCOMES:
            continue
        result_key = _fixture_key(result, "Date")
        if result_key in completed_results:
            raise ValueError(f"ambiguous completed result for fixture: {result_key}")
        completed_results[result_key] = result

    existing_settled_at = {
        str(row["prediction_id"]): _utc_timestamp(row["settled_at"], "settled_at")
        for _, row in settlements.iterrows()
    }
    records = []
    for _, prediction in predictions.iterrows():
        prediction_id = str(prediction["prediction_id"])
        fetched_at = _utc_timestamp(prediction["fetched_at"], "prediction fetched_at")
        if fetched_at < SHADOW_START_UTC:
            raise ValueError("prediction fetched_at precedes the forward shadow evaluation boundary")

        result = completed_results.get(_fixture_key(prediction, "fixture_date"))
        if result is None:
            continue

        outcome = str(prediction["outcome"])
        actual_result = str(result["FTR"])
        captured_best_odds = float(prediction["captured_best_odds"])
        result_file_b365_odds = result.get(f"B365{outcome}")
        try:
            result_file_b365_odds = float(result_file_b365_odds)
        except (TypeError, ValueError):
            result_file_b365_odds = float("nan")
        result_file_best_odds = _result_file_price(result, outcome)
        closing_line_value = float("nan")
        if not pd.isna(result_file_best_odds):
            closing_line_value = captured_best_odds / result_file_best_odds - 1.0

        is_winner = outcome == actual_result
        records.append(
            {
                "prediction_id": prediction_id,
                "settled_at": existing_settled_at.get(prediction_id, settled_at),
                "actual_result": actual_result,
                "is_winner": is_winner,
                "profit": captured_best_odds - 1.0 if is_winner else -1.0,
                "result_file_b365_odds": result_file_b365_odds,
                "result_file_best_odds": result_file_best_odds,
                "closing_line_value": closing_line_value,
                "outcome": outcome,
            }
        )

    return pd.DataFrame(records, columns=_SETTLEMENT_RECORD_COLUMNS)


def settle_shadow_predictions(
    results: pd.DataFrame,
    predictions_path: Path = PREDICTIONS_PATH,
    settlements_path: Path = SETTLEMENTS_PATH,
    settled_at=None,
) -> int:
    """Append outcomes for completed shadow predictions and return their count."""
    predictions, settlements = read_shadow_ledgers(predictions_path, settlements_path)
    if settled_at is None:
        settled_at = pd.Timestamp.now(tz="UTC")
    records = build_settlement_records(predictions, settlements, results, settled_at)
    return append_ledger(records[SETTLEMENT_COLUMNS], settlements_path, "prediction_id")
