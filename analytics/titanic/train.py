from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import build_preprocessor, prepare_features, load_dataset


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42


def build_models(preprocessor):
    """Create the three baseline classification models."""

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def main():

    print("=" * 60)
    print("TITANIC ML MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    df = load_dataset()

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 2. Prepare X and y
    # ---------------------------------------------------------
    X, y = prepare_features(df)

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    # ---------------------------------------------------------
    # 3. Train/test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/Test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # ---------------------------------------------------------
    # 4. Build models
    # ---------------------------------------------------------
    preprocessor = build_preprocessor()

    models = build_models(preprocessor)

    # ---------------------------------------------------------
    # 5. Train models
    # ---------------------------------------------------------
    for name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training: {name}")

        model.fit(X_train, y_train)

        print(f"{name} training completed.")

        # Save trained pipeline
        model_path = MODELS_DIR / f"{name}.joblib"

        joblib.dump(model, model_path)

        print(f"Saved model: {model_path}")

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()