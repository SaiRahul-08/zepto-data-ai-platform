from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
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
    """Create baseline and class-weighted models."""

    models = {
        "logistic_regression_baseline": Pipeline(
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

        "logistic_regression_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "decision_tree_baseline": Pipeline(
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

        "decision_tree_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest_baseline": Pipeline(
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

        "random_forest_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def main():

    print("=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_features(df)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget percentages:")
    print(
        (y.value_counts(normalize=True) * 100).round(2)
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
    }

    results = []

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
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
        }

        results.append(result)

        print(
            f"Accuracy : {result['accuracy']:.4f}"
        )
        print(
            f"Precision: {result['precision']:.4f}"
        )
        print(
            f"Recall   : {result['recall']:.4f}"
        )
        print(
            f"F1 Score : {result['f1']:.4f}"
        )
        print(
            f"ROC-AUC  : {result['roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_file = (
        OUTPUTS_DIR /
        "imbalance_analysis.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE COMPARISON")
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