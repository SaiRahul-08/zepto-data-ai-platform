"""
Titanic Advanced Machine Learning Analysis

This module adds advanced analysis to the Titanic ML pipeline:

1. SMOTE class balancing
2. GridSearchCV hyperparameter tuning
3. Random Forest Out-of-Bag (OOB) evaluation
4. Linear Regression for Fare prediction
5. R-squared and Adjusted R-squared
6. Residual analysis
7. Breusch-Pagan heteroscedasticity test
8. Result CSV files
9. Diagnostic plots
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "titanic.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """Load the Titanic dataset."""

    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("TITANIC ADVANCED MACHINE LEARNING ANALYSIS")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    return df


# ============================================================
# CLASSIFICATION FEATURES
# ============================================================

CLASSIFICATION_NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

CLASSIFICATION_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
]


def build_classification_preprocessor():
    """Build preprocessing for classification models."""

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
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
                "num",
                numerical_pipeline,
                CLASSIFICATION_NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CLASSIFICATION_CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor

# ============================================================
# 1. SMOTE + GRID SEARCH
# ============================================================

def run_smote_grid_search(df):
    """
    Apply SMOTE inside an imbalanced-learn Pipeline and
    perform GridSearchCV for Logistic Regression.
    """

    print("\n" + "=" * 60)
    print("SMOTE + GRID SEARCH")
    print("=" * 60)

    features = (
        CLASSIFICATION_NUMERIC_FEATURES
        + CLASSIFICATION_CATEGORICAL_FEATURES
    )

    X = df[features].copy()
    y = df["Survived"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_classification_preprocessor()

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        "classifier__solver": [
            "liblinear",
            "lbfgs",
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    print("\nRunning GridSearchCV...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nSMOTE + GridSearchCV results:")
    print(f"Best parameters : {grid_search.best_params_}")
    print(f"Best CV F1      : {grid_search.best_score_:.4f}")
    print(f"Test Accuracy   : {accuracy:.4f}")
    print(f"Test Precision  : {precision:.4f}")
    print(f"Test Recall     : {recall:.4f}")
    print(f"Test F1         : {f1:.4f}")

    results = pd.DataFrame(
        grid_search.cv_results_
    )

    selected_columns = [
        "param_classifier__C",
        "param_classifier__solver",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score",
    ]

    results = results[selected_columns]

    output_file = OUTPUTS_DIR / "smote_grid_search_results.csv"

    results.to_csv(
        output_file,
        index=False,
    )

    joblib.dump(
        best_model,
        MODELS_DIR / "smote_logistic_regression.joblib",
    )

    print(f"\nGrid search results saved to:")
    print(output_file)

    print("\nSMOTE + GridSearchCV completed successfully.")


# ============================================================
# 2. RANDOM FOREST OOB ANALYSIS
# ============================================================

def run_random_forest_oob(df):
    """Train Random Forest with OOB evaluation."""

    print("\n" + "=" * 60)
    print("RANDOM FOREST OOB ANALYSIS")
    print("=" * 60)

    features = (
        CLASSIFICATION_NUMERIC_FEATURES
        + CLASSIFICATION_CATEGORICAL_FEATURES
    )

    X = df[features].copy()
    y = df["Survived"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_classification_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    predictions = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    print("\nRandom Forest OOB results:")
    print(f"OOB Score      : {model.oob_score_:.4f}")
    print(f"Test Accuracy  : {accuracy:.4f}")
    print(f"Test Precision : {precision:.4f}")
    print(f"Test Recall    : {recall:.4f}")
    print(f"Test F1        : {f1:.4f}")

    results = pd.DataFrame(
        [
            {
                "model": "random_forest_oob",
                "oob_score": model.oob_score_,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        ]
    )

    output_file = OUTPUTS_DIR / "random_forest_oob_results.csv"

    results.to_csv(
        output_file,
        index=False,
    )

    joblib.dump(
        model,
        MODELS_DIR / "random_forest_oob.joblib",
    )

    joblib.dump(
        preprocessor,
        MODELS_DIR / "random_forest_oob_preprocessor.joblib",
    )

    print("\nOOB results saved to:")
    print(output_file)


# ============================================================
# 3. LINEAR REGRESSION FOR FARE
# ============================================================

REGRESSION_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
]


def run_fare_regression(df):
    """
    Predict Fare using Linear Regression.

    Calculates:
    - R-squared
    - Adjusted R-squared
    - RMSE
    - residual statistics
    - Breusch-Pagan heteroscedasticity test
    """

    print("\n" + "=" * 60)
    print("FARE LINEAR REGRESSION")
    print("=" * 60)

    regression_df = df[
        REGRESSION_FEATURES + ["Fare"]
    ].copy()

    regression_df = regression_df.dropna()

    X = regression_df[
        REGRESSION_FEATURES
    ]

    y = regression_df["Fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    residuals = y_test - predictions

    r2 = r2_score(
        y_test,
        predictions,
    )

    n = len(y_test)
    p = X_test.shape[1]

    adjusted_r2 = (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    print("\nRegression results:")
    print(f"R-squared       : {r2:.4f}")
    print(f"Adjusted R²     : {adjusted_r2:.4f}")
    print(f"RMSE            : {rmse:.4f}")

    print("\nRegression coefficients:")

    coefficients = pd.DataFrame(
        {
            "feature": REGRESSION_FEATURES,
            "coefficient": model.coef_,
        }
    )

    print(coefficients.to_string(index=False))

    # --------------------------------------------------------
    # Breusch-Pagan heteroscedasticity test
    # --------------------------------------------------------

    X_test_with_constant = sm.add_constant(
        X_test
    )

    bp_test = het_breuschpagan(
        residuals,
        X_test_with_constant,
    )

    bp_lm_statistic = bp_test[0]
    bp_lm_pvalue = bp_test[1]
    bp_f_statistic = bp_test[2]
    bp_f_pvalue = bp_test[3]

    print("\nBreusch-Pagan test:")
    print(
        f"LM Statistic : {bp_lm_statistic:.4f}"
    )
    print(
        f"LM p-value   : {bp_lm_pvalue:.4f}"
    )
    print(
        f"F Statistic  : {bp_f_statistic:.4f}"
    )
    print(
        f"F p-value    : {bp_f_pvalue:.4f}"
    )

    if bp_lm_pvalue < 0.05:
        heteroscedasticity_result = (
            "Evidence of heteroscedasticity"
        )
    else:
        heteroscedasticity_result = (
            "No statistically significant evidence "
            "of heteroscedasticity"
        )

    print(
        f"\nHeteroscedasticity interpretation:"
    )
    print(
        heteroscedasticity_result
    )

    # --------------------------------------------------------
    # Regression results CSV
    # --------------------------------------------------------

    regression_results = pd.DataFrame(
        [
            {
                "r_squared": r2,
                "adjusted_r_squared": adjusted_r2,
                "rmse": rmse,
                "bp_lm_statistic": bp_lm_statistic,
                "bp_lm_pvalue": bp_lm_pvalue,
                "bp_f_statistic": bp_f_statistic,
                "bp_f_pvalue": bp_f_pvalue,
                "heteroscedasticity_result":
                    heteroscedasticity_result,
            }
        ]
    )

    output_file = (
        OUTPUTS_DIR
        / "fare_regression_results.csv"
    )

    regression_results.to_csv(
        output_file,
        index=False,
    )

    coefficients_file = (
        OUTPUTS_DIR
        / "fare_regression_coefficients.csv"
    )

    coefficients.to_csv(
        coefficients_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save regression model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODELS_DIR / "fare_linear_regression.joblib",
    )

    # --------------------------------------------------------
    # Residual data
    # --------------------------------------------------------

    residual_df = pd.DataFrame(
        {
            "actual_fare": y_test.values,
            "predicted_fare": predictions,
            "residual": residuals.values,
            "absolute_residual":
                np.abs(residuals.values),
        }
    )

    residual_file = (
        OUTPUTS_DIR
        / "fare_residuals.csv"
    )

    residual_df.to_csv(
        residual_file,
        index=False,
    )

    # --------------------------------------------------------
    # Plot 1: Actual vs Predicted
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6,
    )

    minimum = min(
        y_test.min(),
        predictions.min(),
    )

    maximum = max(
        y_test.max(),
        predictions.max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Actual Fare"
    )

    plt.ylabel(
        "Predicted Fare"
    )

    plt.title(
        "Actual vs Predicted Fare"
    )

    plt.tight_layout()

    actual_predicted_plot = (
        PLOTS_DIR
        / "fare_actual_vs_predicted.png"
    )

    plt.savefig(
        actual_predicted_plot,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: Residuals vs Fitted
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        predictions,
        residuals,
        alpha=0.6,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Fitted Values"
    )

    plt.ylabel(
        "Residuals"
    )

    plt.title(
        "Residuals vs Fitted Values"
    )

    plt.tight_layout()

    residual_plot = (
        PLOTS_DIR
        / "fare_residuals_vs_fitted.png"
    )

    plt.savefig(
        residual_plot,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: Residual distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    sns.histplot(
        residuals,
        kde=True,
    )

    plt.xlabel(
        "Residual"
    )

    plt.title(
        "Residual Distribution"
    )

    plt.tight_layout()

    residual_distribution_plot = (
        PLOTS_DIR
        / "fare_residual_distribution.png"
    )

    plt.savefig(
        residual_distribution_plot,
        dpi=150,
    )

    plt.close()

    print("\nRegression results saved to:")
    print(output_file)

    print("\nResidual data saved to:")
    print(residual_file)

    print("\nRegression plots saved to:")
    print(actual_predicted_plot)
    print(residual_plot)
    print(residual_distribution_plot)

    print(
        "\nFare regression analysis completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    run_smote_grid_search(df)

    run_random_forest_oob(df)

    run_fare_regression(df)

    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()