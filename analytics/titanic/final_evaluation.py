from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocess import (
    load_dataset,
    prepare_features,
)


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TUNED_THRESHOLD = 0.30


def main():

    print("=" * 60)
    print("FINAL RANDOM FOREST EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = load_dataset()

    X, y = prepare_features(df)

    # ---------------------------------------------------------
    # 2. Recreate the untouched test set
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

    print("\nTest samples:")
    print(len(X_test))

    # ---------------------------------------------------------
    # 3. Load trained Random Forest
    # ---------------------------------------------------------

    model_path = (
        MODELS_DIR /
        "random_forest.joblib"
    )

    model = joblib.load(model_path)

    # ---------------------------------------------------------
    # 4. Generate probabilities
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # 5. Evaluate default threshold = 0.50
    # ---------------------------------------------------------

    default_predictions = (
        probabilities >= 0.50
    ).astype(int)

    # ---------------------------------------------------------
    # 6. Evaluate tuned threshold = 0.45
    # ---------------------------------------------------------

    tuned_predictions = (
        probabilities >= TUNED_THRESHOLD
    ).astype(int)

    # ---------------------------------------------------------
    # 7. Calculate metrics
    # ---------------------------------------------------------

    default_metrics = {
        "threshold": 0.50,
        "accuracy": accuracy_score(
            y_test,
            default_predictions,
        ),
        "precision": precision_score(
            y_test,
            default_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            default_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            default_predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    tuned_metrics = {
        "threshold": TUNED_THRESHOLD,
        "accuracy": accuracy_score(
            y_test,
            tuned_predictions,
        ),
        "precision": precision_score(
            y_test,
            tuned_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            tuned_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            tuned_predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    # ---------------------------------------------------------
    # 8. Print comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DEFAULT THRESHOLD — 0.50")
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{default_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{default_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{default_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{default_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{default_metrics['roc_auc']:.4f}"
    )

    print("\n" + "=" * 60)
    print("TUNED THRESHOLD — 0.45")
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{tuned_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{tuned_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{tuned_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{tuned_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{tuned_metrics['roc_auc']:.4f}"
    )

    # ---------------------------------------------------------
    # 9. Confusion matrix
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX — TUNED THRESHOLD")
    print("=" * 60)

    cm = confusion_matrix(
        y_test,
        tuned_predictions,
    )

    print(cm)

    # ---------------------------------------------------------
    # 10. Classification report
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT — TUNED THRESHOLD")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            tuned_predictions,
            target_names=[
                "Not Survived",
                "Survived",
            ],
        )
    )

    # ---------------------------------------------------------
    # 11. Save final comparison
    # ---------------------------------------------------------

    results = pd.DataFrame(
        [
            default_metrics,
            tuned_metrics,
        ]
    )

    output_file = (
        OUTPUTS_DIR /
        "final_threshold_comparison.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print("\nResults saved to:")
    print(output_file)

    print("\nFinal evaluation completed successfully.")


if __name__ == "__main__":
    main()  