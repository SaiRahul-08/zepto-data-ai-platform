from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import (
    build_preprocessor,
    load_dataset,
    prepare_features,
)


BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def build_models():
    """Build the three ML pipelines."""

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
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
                ("preprocessor", build_preprocessor()),
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
                ("preprocessor", build_preprocessor()),
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
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_features(df)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    results = []

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Evaluating: {name}")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        result = {
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
        }

        results.append(result)

        print(
            f"Accuracy : "
            f"{result['accuracy_mean']:.4f} "
            f"+/- {result['accuracy_std']:.4f}"
        )

        print(
            f"Precision: "
            f"{result['precision_mean']:.4f}"
        )

        print(
            f"Recall   : "
            f"{result['recall_mean']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{result['f1_mean']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{result['roc_auc_mean']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_file = (
        OUTPUTS_DIR /
        "cross_validation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved to:")
    print(output_file)


if __name__ == "__main__":
    main()