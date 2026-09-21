"""
Train and evaluate the Support Assistance intent classifier.

Pipeline:
    Support queries
        -> TF-IDF features
        -> Logistic Regression
        -> Intent prediction
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "support_assistance" / "data" / "support_faq.csv"
MODEL_DIR = BASE_DIR / "support_assistance" / "models"
OUTPUT_DIR = BASE_DIR / "support_assistance" / "outputs"

RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Normalize support query text."""
    return str(text).lower().strip()


def load_dataset():
    """Load the support dataset."""
    df = pd.read_csv(DATA_PATH)

    required_columns = {"query", "intent", "response"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def main():
    print("=" * 60)
    print("SUPPORT ASSISTANCE - INTENT CLASSIFIER")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    X_text = df["query"].apply(clean_text)
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF training shape: {X_train_tfidf.shape}")
    print(f"TF-IDF testing shape: {X_test_tfidf.shape}")

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    print(report)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    labels = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    joblib.dump(
        model,
        MODEL_DIR / "intent_classifier.joblib",
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.joblib",
    )

    print("\nSaved files:")

    print(
        MODEL_DIR / "intent_classifier.joblib"
    )

    print(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    print(
        OUTPUT_DIR / "classification_report.csv"
    )

    print(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()