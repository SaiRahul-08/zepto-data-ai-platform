"""
NLP preprocessing pipeline for the Support Assistance module.

Loads support queries, cleans the text, creates TF-IDF features,
and creates train/test datasets for intent classification.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "support_assistance" / "data" / "support_faq.csv"
OUTPUT_DIR = BASE_DIR / "support_assistance" / "outputs"

RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Basic text normalization."""
    text = str(text).lower().strip()
    return text


def load_data() -> pd.DataFrame:
    """Load the support dataset."""
    df = pd.read_csv(DATA_PATH)

    required_columns = {"query", "intent", "response"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def create_features(df: pd.DataFrame):
    """Create TF-IDF features from support queries."""

    queries = df["query"].apply(clean_text)
    labels = df["intent"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    X = vectorizer.fit_transform(queries)

    return X, labels, vectorizer


def main():
    """Run preprocessing pipeline."""

    print("=" * 60)
    print("SUPPORT ASSISTANCE - NLP PREPROCESSING")
    print("=" * 60)

    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Number of intents: {df['intent'].nunique()}")

    X, y, vectorizer = create_features(df)

    print(f"TF-IDF matrix shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    print("\nIntent distribution:")
    print(y.value_counts())

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()