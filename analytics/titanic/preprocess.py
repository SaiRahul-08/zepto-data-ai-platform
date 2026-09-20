from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "titanic.csv"


# Features selected for the baseline ML models
NUMERICAL_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
]


def load_dataset():
    """Load the Titanic dataset."""

    return pd.read_csv(DATA_FILE)


def prepare_features(df):
    """
    Separate input features X from target y.

    Columns that are identifiers, high-cardinality text,
    or have excessive missing values are removed.
    """

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    X = df[features].copy()
    y = df["Survived"].copy()

    return X, y


def build_preprocessor():
    """
    Build the preprocessing pipeline.

    Numerical features:
        - Fill missing values using median
        - Standardize using StandardScaler

    Categorical features:
        - Fill missing values using most frequent value
        - One-hot encode categories
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def main():
    """Test the preprocessing pipeline."""

    print("=" * 60)
    print("TITANIC PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_dataset()

    print("\nOriginal dataset shape:")
    print(df.shape)

    X, y = prepare_features(df)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nFeature matrix shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    # Build the preprocessing pipeline
    preprocessor = build_preprocessor()

    # Fit and transform only for pipeline verification.
    X_processed = preprocessor.fit_transform(X)

    print("\nProcessed feature matrix shape:")
    print(X_processed.shape)

    print("\nMissing values after preprocessing:")
    print(pd.DataFrame(X_processed).isnull().sum().sum())

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()