from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from preprocess import (
    build_preprocessor,
    load_dataset,
    prepare_features,
)


BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def build_model():
    """Build the Random Forest pipeline."""

    return Pipeline(
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
    )


def main():

    print("=" * 60)
    print("RANDOM FOREST — CORRECTED THRESHOLD TUNING")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_features(df)

    # ---------------------------------------------------------
    # 1. Create the final train/test split FIRST
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining samples:")
    print(len(X_train))

    print("Test samples:")
    print(len(X_test))

    # ---------------------------------------------------------
    # 2. Perform 5-fold CV ONLY on training data
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = build_model()

    print("\nGenerating training-only out-of-fold probabilities...")

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # ---------------------------------------------------------
    # 3. Test different thresholds
    # ---------------------------------------------------------

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = []

    print("\nThreshold results:")
    print("-" * 60)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_train,
            predictions,
        )

        precision = precision_score(
            y_train,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_train,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_train,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # 4. Select threshold using training OOF predictions
    # ---------------------------------------------------------

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    selected_threshold = float(
        best_row["threshold"]
    )

    print("\n" + "=" * 60)
    print("SELECTED THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold: {selected_threshold:.2f}"
    )

    print(
        f"OOF Accuracy : "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"OOF Precision: "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"OOF Recall   : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"OOF F1       : "
        f"{best_row['f1']:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Save threshold analysis
    # ---------------------------------------------------------

    output_file = (
        OUTPUTS_DIR /
        "threshold_tuning_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Plot threshold analysis
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        results_df["threshold"],
        results_df["f1"],
        marker="o",
        label="F1",
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title(
        "Random Forest — Training OOF Threshold Analysis"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = (
        PLOTS_DIR /
        "threshold_tuning.png"
    )

    plt.savefig(plot_file)
    plt.close()

    print("\nResults saved to:")
    print(output_file)

    print("\nPlot saved to:")
    print(plot_file)


if __name__ == "__main__":
    main()