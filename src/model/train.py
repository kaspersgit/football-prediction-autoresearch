from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src.model.features import build_features_with_odds

MODEL_PATH = Path("models/baseline.joblib")
TEST_SEASONS = 2

_MODEL_CFG = dict(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    l2_regularization=0.05,
    random_state=42,
)

_LGBM_CFG = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    reg_lambda=0.05,
    random_state=42,
    verbosity=-1,
)


def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season"].unique())
    test_seasons = set(seasons[-TEST_SEASONS:])
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test


_LEAGUES = ["E0", "D1", "SP1", "I1", "F1", "N1", "P1"]
_CLASSES = np.array(["A", "D", "H"])  # canonical alphabetical order


def _train_binary_models(X_train, y_train) -> dict[str, object]:
    """Train one binary HistGBM per outcome (H/D/A). Returns {outcome: model}."""
    models = {}
    for outcome in _CLASSES:
        y_bin = (y_train == outcome).astype(int)
        m = HistGradientBoostingClassifier(**_MODEL_CFG)
        m.fit(X_train, y_bin)
        models[outcome] = m
    return models


def _binary_proba_matrix(models: dict, X) -> np.ndarray:
    """Assemble (n, 3) proba matrix in [A, D, H] column order from binary models."""
    cols = []
    for outcome in _CLASSES:
        p = models[outcome].predict_proba(X)
        pos_idx = list(models[outcome].classes_).index(1)
        cols.append(p[:, pos_idx])
    return np.column_stack(cols)


def _predict_per_league(full_X, full_y, full_odds, train_mask, test_mask):
    """Train one HistGBM per league; return predictions aligned to test_mask row order."""
    test_positions = np.where(test_mask)[0]
    pos_to_seq = {int(p): seq for seq, p in enumerate(test_positions)}

    n_test = test_mask.sum()
    y_pred_out = np.empty(n_test, dtype=object)
    y_proba_out = np.zeros((n_test, 3))
    y_true_out = full_y[test_mask].values

    for league in _LEAGUES:
        l_train = train_mask & (full_odds["league"] == league)
        l_test = test_mask & (full_odds["league"] == league)
        if l_test.sum() == 0:
            continue

        model = HistGradientBoostingClassifier(**_MODEL_CFG)
        model.fit(full_X[l_train], full_y[l_train])

        classes = list(model.classes_)
        y_proba_l = model.predict_proba(full_X[l_test])
        y_pred_l = model.predict(full_X[l_test])

        # Reorder proba columns to canonical [A, D, H] order
        col_order = [classes.index(c) for c in _CLASSES if c in classes]
        y_proba_l = y_proba_l[:, col_order]

        for i, pos in enumerate(np.where(l_test)[0]):
            seq = pos_to_seq[int(pos)]
            y_pred_out[seq] = y_pred_l[i]
            y_proba_out[seq] = y_proba_l[i]

    return y_pred_out, y_proba_out, y_true_out


def _predict_per_league_binary(full_X, full_y, full_odds, train_mask, test_mask):
    """Train 3 binary models per league (12 total); return predictions aligned to test_mask row order."""
    test_positions = np.where(test_mask)[0]
    pos_to_seq = {int(p): seq for seq, p in enumerate(test_positions)}

    n_test = test_mask.sum()
    y_pred_out = np.empty(n_test, dtype=object)
    y_proba_out = np.zeros((n_test, 3))
    y_true_out = full_y[test_mask].values

    for league in _LEAGUES:
        l_train = train_mask & (full_odds["league"] == league)
        l_test = test_mask & (full_odds["league"] == league)
        if l_test.sum() == 0:
            continue

        models = _train_binary_models(full_X[l_train], full_y[l_train])
        y_proba_l = _binary_proba_matrix(models, full_X[l_test])
        y_pred_l = _CLASSES[np.argmax(y_proba_l, axis=1)]

        for i, pos in enumerate(np.where(l_test)[0]):
            seq = pos_to_seq[int(pos)]
            y_pred_out[seq] = y_pred_l[i]
            y_proba_out[seq] = y_proba_l[i]

    return y_pred_out, y_proba_out, y_true_out


def train_walkforward(df: pd.DataFrame, n_test_seasons: int = TEST_SEASONS,
                      per_league: bool = False, binary_outcomes: bool = False) -> dict:
    """
    Season-level walk-forward backtest.

    Features are computed on the full dataset so Elo ratings carry forward
    from training seasons into each test season (no cold-start reset).

    per_league=True:      one model per league per test season (4 models)
    binary_outcomes=True: one binary classifier per outcome per test season (3 models)
    both=True:            one binary classifier per outcome per league (12 models)
    """
    full_X, full_y, full_odds = build_features_with_odds(df)
    full_X = full_X.reset_index(drop=True)
    full_y = full_y.reset_index(drop=True)
    full_odds = full_odds.reset_index(drop=True)

    seasons = sorted(full_odds["season"].unique())
    test_seasons = seasons[-n_test_seasons:]

    season_results = []
    model = None

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        train_mask = full_odds["season"].isin(train_seasons)
        test_mask = full_odds["season"] == test_season

        if per_league and binary_outcomes:
            y_pred, y_proba, y_true = _predict_per_league_binary(
                full_X, full_y, full_odds, train_mask, test_mask
            )
            odds_test = full_odds[test_mask].reset_index(drop=True)
            accuracy = (y_pred == y_true).mean()
            print(f"  Season {test_season}: {train_mask.sum()} train / {test_mask.sum()} test — accuracy {accuracy:.3f}")
            season_results.append({
                "y_pred": y_pred, "y_proba": y_proba, "y_true": y_true,
                "odds_test": odds_test, "classes": _CLASSES,
            })
        elif per_league:
            y_pred, y_proba, y_true = _predict_per_league(
                full_X, full_y, full_odds, train_mask, test_mask
            )
            odds_test = full_odds[test_mask].reset_index(drop=True)
            accuracy = (y_pred == y_true).mean()
            print(f"  Season {test_season}: {train_mask.sum()} train / {test_mask.sum()} test — accuracy {accuracy:.3f}")
            season_results.append({
                "y_pred": y_pred,
                "y_proba": y_proba,
                "y_true": y_true,
                "odds_test": odds_test,
                "classes": _CLASSES,
            })
        elif binary_outcomes:
            X_train = full_X[train_mask]
            y_train = full_y[train_mask]
            X_test = full_X[test_mask]
            y_test = full_y[test_mask]
            odds_test = full_odds[test_mask]

            models = _train_binary_models(X_train, y_train)
            y_proba = _binary_proba_matrix(models, X_test)
            y_pred = _CLASSES[np.argmax(y_proba, axis=1)]
            accuracy = (y_pred == y_test.values).mean()
            print(f"  Season {test_season}: {train_mask.sum()} train / {test_mask.sum()} test — accuracy {accuracy:.3f}")
            season_results.append({
                "y_pred": y_pred, "y_proba": y_proba, "y_true": y_test.values,
                "odds_test": odds_test, "classes": _CLASSES,
            })
        else:
            X_train = full_X[train_mask]
            y_train = full_y[train_mask]
            X_test = full_X[test_mask]
            y_test = full_y[test_mask]
            odds_test = full_odds[test_mask]

            model = LGBMClassifier(**_LGBM_CFG)
            model.fit(X_train, y_train)
            classes = np.array(sorted(model.classes_))
            col_order = [list(model.classes_).index(c) for c in classes]
            y_proba = model.predict_proba(X_test)[:, col_order]
            y_pred = classes[np.argmax(y_proba, axis=1)]
            accuracy = (y_pred == y_test.values).mean()
            print(f"  Season {test_season}: {train_mask.sum()} train / {test_mask.sum()} test — accuracy {accuracy:.3f}")

            season_results.append({
                "y_pred": y_pred,
                "y_proba": y_proba,
                "y_true": y_test.values,
                "odds_test": odds_test,
                "classes": classes,
            })

    if not per_league:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

    y_pred_all = np.concatenate([r["y_pred"] for r in season_results])
    y_proba_all = np.vstack([r["y_proba"] for r in season_results])
    y_true_all = np.concatenate([r["y_true"] for r in season_results])
    odds_all = pd.concat([r["odds_test"] for r in season_results]).reset_index(drop=True)

    accuracy = (y_pred_all == y_true_all).mean()
    print(f"Combined test accuracy: {accuracy:.3f}")

    return {
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
        "y_test": pd.Series(y_true_all),
        "classes": season_results[-1]["classes"],
        "odds_test": odds_all,
        "accuracy": accuracy,
    }


def train_walkforward_monthly(df: pd.DataFrame, n_test_seasons: int = TEST_SEASONS) -> dict:
    """
    Monthly walk-forward backtest (per-league).

    Retrains one model per league at the start of every calendar month, using all
    data strictly before that month. Each model then predicts only that month's matches.
    This keeps the model calibrated to in-season form changes rather than going stale
    after a single pre-season fit.
    """
    full_X, full_y, full_odds = build_features_with_odds(df)
    full_X = full_X.reset_index(drop=True)
    full_y = full_y.reset_index(drop=True)
    full_odds = full_odds.reset_index(drop=True)

    dates = pd.to_datetime(full_odds["Date"])
    seasons = sorted(full_odds["season"].unique())
    test_seasons = set(seasons[-n_test_seasons:])

    # All months that fall in the test period
    test_mask_all = full_odds["season"].isin(test_seasons)
    test_months = sorted(dates[test_mask_all].dt.to_period("M").unique())

    month_results = []
    for month in test_months:
        month_start = month.to_timestamp()
        train_mask = dates < month_start
        test_mask  = (dates.dt.to_period("M") == month) & test_mask_all

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        y_pred_m, y_proba_m, y_true_m = _predict_per_league(
            full_X, full_y, full_odds, train_mask, test_mask
        )
        odds_m = full_odds[test_mask].reset_index(drop=True)
        accuracy_m = (y_pred_m == y_true_m).mean()
        print(f"  {month}: {train_mask.sum():>5} train / {test_mask.sum():>4} test — accuracy {accuracy_m:.3f}")
        month_results.append({
            "y_pred": y_pred_m,
            "y_proba": y_proba_m,
            "y_true": y_true_m,
            "odds_test": odds_m,
            "classes": _CLASSES,
        })

    y_pred_all  = np.concatenate([r["y_pred"]  for r in month_results])
    y_proba_all = np.vstack(      [r["y_proba"] for r in month_results])
    y_true_all  = np.concatenate([r["y_true"]  for r in month_results])
    odds_all    = pd.concat(      [r["odds_test"] for r in month_results]).reset_index(drop=True)
    accuracy    = (y_pred_all == y_true_all).mean()
    print(f"Combined test accuracy: {accuracy:.3f}")

    return {
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
        "y_test": pd.Series(y_true_all),
        "classes": _CLASSES,
        "odds_test": odds_all,
        "accuracy": accuracy,
    }


def train_on_all_data(df: pd.DataFrame):
    """Train on the full dataset (no holdout) for live fixture prediction."""
    X, y, _ = build_features_with_odds(df)
    model = HistGradientBoostingClassifier(**_MODEL_CFG)
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained on {len(X)} matches, saved to {MODEL_PATH}")
    return model


def load_model():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    from src.data.loader import load_all_data
    df = load_all_data()
    results = train_model(df)
    print(f"Accuracy: {results['accuracy']:.3f}")
