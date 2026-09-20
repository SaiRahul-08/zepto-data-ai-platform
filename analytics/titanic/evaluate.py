from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from preprocess import load_dataset, prepare_features


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
]


def load_test_data():
    """Create the same stratified test split used during training."""

    df = load_dataset()

    X, y = prepare_features(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


def evaluate_model(model_name, X_test, y_test):
    """Evaluate one trained model."""

    model_path = MODELS_DIR / f"{model_name}.joblib"

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not Survived", "Survived"],
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )

    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    )

    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
    }


def main():

    print("=" * 60)
    print("TITANIC MODEL EVALUATION")
    print("=" * 60)

    X_test, y_test = load_test_data()

    results = []

    plt.figure(figsize=(8, 6))

    for model_name in MODEL_NAMES:

        result = evaluate_model(
            model_name,
            X_test,
            y_test,
        )

        results.append(result)

        plt.plot(
            result["fpr"],
            result["tpr"],
            label=(
                f"{model_name} "
                f"(AUC = {result['roc_auc']:.3f})"
            ),
        )

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.title("ROC Curves — Titanic Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "roc_curves_comparison.png"
    )

    plt.close()

    # Remove arrays before saving the summary
    summary = []

    for result in results:
        summary.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
            }
        )

    results_df = pd.DataFrame(summary)

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    )

    results_path = OUTPUTS_DIR / "model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nResults saved to:")
    print(results_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()