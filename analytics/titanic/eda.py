from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "titanic.csv"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load the Titanic dataset."""
    return pd.read_csv(DATA_FILE)


def inspect_dataset(df):
    """Display the basic dataset structure."""

    print("=" * 60)
    print("TITANIC DATASET — INITIAL INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMissing-value percentage:")
    print((df.isnull().sum() / len(df) * 100).round(2))

    print("\nTarget distribution:")
    print(df["Survived"].value_counts())

    print("\nTarget distribution percentage:")
    print((df["Survived"].value_counts(normalize=True) * 100).round(2))

    print("\nSummary statistics:")
    print(df.describe(include="all"))


def create_plots(df):
    """Create exploratory data analysis plots."""

    # 1. Survival distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Survived")
    plt.title("Titanic Survival Distribution")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_distribution.png")
    plt.close()

    # 2. Survival by gender
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Sex", hue="Survived")
    plt.title("Survival by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_gender.png")
    plt.close()

    # 3. Survival by passenger class
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Survival by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_class.png")
    plt.close()

    # 4. Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "age_distribution.png")
    plt.close()

    # 5. Fare distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Fare", bins=30, kde=True)
    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fare_distribution.png")
    plt.close()

    # 6. Correlation heatmap
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    print("\nEDA plots created successfully.")
    print(f"Plots saved in: {PLOTS_DIR}")


def main():

    df = load_dataset()

    inspect_dataset(df)

    create_plots(df)


if __name__ == "__main__":
    main()