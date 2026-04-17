from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.model.features import FEATURE_COLS, build_features_with_odds

MODEL_PATH = Path("models/baseline.joblib")
TEST_SEASONS = 2

# Index of league_code in FEATURE_COLS (HistGBM needs categorical_features indices)
_CATEGORICAL_FEATURES = [FEATURE_COLS.index("league_code")]


def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season"].unique())
    test_seasons = set(seasons[-TEST_SEASONS:])
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test


def train_model(df: pd.DataFrame) -> dict:
    train_df, test_df = split_by_season(df)

    X_train, y_train, _ = build_features_with_odds(train_df)
    X_test, y_test, odds_test = build_features_with_odds(test_df)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        categorical_features=_CATEGORICAL_FEATURES,
        random_state=42,
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    accuracy = (y_pred == y_test.values).mean()
    print(f"Test accuracy: {accuracy:.3f}")

    return {
        "pipeline": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": classes,
        "odds_test": odds_test,
        "accuracy": accuracy,
    }


def load_model():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    from src.data.loader import load_all_data
    df = load_all_data()
    results = train_model(df)
    print(f"Accuracy: {results['accuracy']:.3f}")
