from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "outputs" / "raw_books.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "clean_books.csv"

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ---------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------

def clean_price(value):
    """
    Convert a scraped GBP price into a numeric float.

    Handles normal and malformed currency representations,
    such as £45.17 and Â£45.17.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Remove common currency representations.
    value = (
        value
        .replace("Â£", "")
        .replace("£", "")
        .replace("GBP", "")
        .strip()
    )

    try:
        return float(value)
    except ValueError:
        return None


def clean_rating(value):
    """
    Convert textual star ratings into integers 1–5.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    return RATING_MAP.get(value)


def clean_availability(value):
    """
    Convert availability text into a boolean.

    True  -> product is in stock
    False -> product is not in stock
    """

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return "in stock" in value


# ---------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------

def clean_data(df):
    """
    Clean the raw scraped DataFrame.
    """

    df = df.copy()

    print("\nStarting data cleaning...")
    print(f"Raw rows: {len(df)}")

    # -----------------------------------------------------
    # Price
    # -----------------------------------------------------

    df["price_gbp"] = df["price"].apply(clean_price)

    invalid_prices = df["price_gbp"].isna().sum()

    print(f"Invalid price values: {invalid_prices}")

    if invalid_prices > 0:
        median_price = df["price_gbp"].median()

        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

        print(
            f"Invalid numeric prices replaced with "
            f"median: {median_price:.2f}"
        )

    # -----------------------------------------------------
    # Rating
    # -----------------------------------------------------

    df["rating"] = df["star_rating"].apply(clean_rating)

    invalid_ratings = df["rating"].isna().sum()

    print(f"Invalid rating values: {invalid_ratings}")

    if invalid_ratings > 0:
        median_rating = int(
            round(df["rating"].median())
        )

        df["rating"] = df["rating"].fillna(
            median_rating
        )

        print(
            f"Invalid numeric ratings replaced with "
            f"median: {median_rating}"
        )

    df["rating"] = df["rating"].astype(int)

    # -----------------------------------------------------
    # Availability
    # -----------------------------------------------------

    df["in_stock"] = df["availability"].apply(
        clean_availability
    )

    # Explicit boolean type
    df["in_stock"] = df["in_stock"].astype(bool)

    # -----------------------------------------------------
    # Currency conversion
    # -----------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    df["price_inr"] = df["price_inr"].round(2)

    # -----------------------------------------------------
    # Validate required textual fields
    # -----------------------------------------------------

    required_text_columns = [
        "title",
        "category"
    ]

    before_drop = len(df)

    df = df.dropna(
        subset=required_text_columns
    )

    dropped_rows = before_drop - len(df)

    if dropped_rows > 0:
        print(
            f"Dropped {dropped_rows} rows because "
            f"required text fields were missing."
        )

    # -----------------------------------------------------
    # Keep only the required clean columns
    # -----------------------------------------------------

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    return df


def main():

    print("=" * 60)
    print("ZEPTO DATA & AI PLATFORM")
    print("MODULE 1 — DATA CLEANING")
    print("=" * 60)

    # -----------------------------------------------------
    # Load raw data
    # -----------------------------------------------------

    print(f"\nReading raw data from:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    # -----------------------------------------------------
    # Clean data
    # -----------------------------------------------------

    clean_df = clean_data(df)

    # -----------------------------------------------------
    # Save clean dataset
    # -----------------------------------------------------

    clean_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("CLEANING COMPLETED")
    print("=" * 60)

    print(f"Rows after cleaning: {len(clean_df)}")
    print(f"Columns: {list(clean_df.columns)}")

    print("\nData types:")
    print(clean_df.dtypes)

    print("\nFirst 5 cleaned rows:")
    print(clean_df.head())

    print("\nStock distribution:")
    print(clean_df["in_stock"].value_counts())

    print("\nRating distribution:")
    print(clean_df["rating"].value_counts().sort_index())

    print("\nGBP → INR conversion:")
    print(
        clean_df[
            ["price_gbp", "price_inr"]
        ].head()
    )

    print(f"\nSaved clean dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()