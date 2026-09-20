from pathlib import Path
import sqlite3

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATABASE_FILE = BASE_DIR / "outputs" / "zepto_books.db"


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------

def get_connection():
    """
    Connect to the SQLite database.
    """

    connection = sqlite3.connect(DATABASE_FILE)

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ---------------------------------------------------------
# Query 1
# SELECT + ORDER BY + LIMIT
# ---------------------------------------------------------

def query_top_expensive_books(connection):

    query = """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ---------------------------------------------------------
# Query 2
# WHERE + ORDER BY
# ---------------------------------------------------------

def query_high_rated_books(connection):

    query = """
        SELECT
            title,
            rating,
            price_gbp,
            price_inr
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, price_gbp DESC;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ---------------------------------------------------------
# Query 3
# DISTINCT
# ---------------------------------------------------------

def query_categories(connection):

    query = """
        SELECT DISTINCT
            category_name
        FROM categories
        ORDER BY category_name;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ---------------------------------------------------------
# Query 4
# BETWEEN
# ---------------------------------------------------------

def query_mid_range_books(connection):

    query = """
        SELECT
            title,
            price_gbp,
            price_inr,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp ASC;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ---------------------------------------------------------
# Query 5
# IN
# ---------------------------------------------------------

def query_selected_ratings(connection):

    query = """
        SELECT
            title,
            rating,
            price_gbp
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title ASC;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ---------------------------------------------------------
# Query 6
# JOIN
# ---------------------------------------------------------

def query_books_with_categories(connection):

    query = """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY c.category_name, b.title;
    """

    return pd.read_sql_query(
        query,
        connection
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def compare_sql_and_pandas_join(connection):
    """
    Reproduce the SQL JOIN using pandas.merge()
    and compare both outputs.
    """

    print("\n" + "=" * 60)
    print("SQL JOIN vs PANDAS JOIN")
    print("=" * 60)

    # -----------------------------------------------------
    # Load both tables using pd.read_sql()
    # -----------------------------------------------------

    books_df = pd.read_sql(
        "SELECT * FROM books",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        connection
    )

    print(f"\nBooks loaded with pd.read_sql(): {len(books_df)}")
    print(
        f"Categories loaded with pd.read_sql(): "
        f"{len(categories_df)}"
    )

    # -----------------------------------------------------
    # Reproduce SQL JOIN using pandas.merge()
    # -----------------------------------------------------

    pandas_join = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    # -----------------------------------------------------
    # Select equivalent columns
    # -----------------------------------------------------

    pandas_join = pandas_join[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ].sort_values(
        by=["category_name", "title"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Get the SQL JOIN again
    # -----------------------------------------------------

    sql_join = query_books_with_categories(
        connection
    )

    sql_join = sql_join[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ].sort_values(
        by=["category_name", "title"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Normalize data types for comparison
    # -----------------------------------------------------

    pandas_join["in_stock"] = (
        pandas_join["in_stock"]
        .astype(int)
    )

    sql_join["in_stock"] = (
        sql_join["in_stock"]
        .astype(int)
    )

    # -----------------------------------------------------
    # Compare
    # -----------------------------------------------------

    are_equal = pandas_join.equals(
        sql_join
    )

    print("\nSQL JOIN rows:")
    print(len(sql_join))

    print("\nPandas JOIN rows:")
    print(len(pandas_join))

    print("\nJOIN outputs identical:")
    print(are_equal)

    # -----------------------------------------------------
    # Save Pandas JOIN output
    # -----------------------------------------------------

    pandas_output = (
        BASE_DIR
        / "outputs"
        / "pandas_join_result.csv"
    )

    pandas_join.to_csv(
        pandas_output,
        index=False
    )

    print("\nPandas JOIN result saved to:")
    print(pandas_output)

    # -----------------------------------------------------
    # Show sample
    # -----------------------------------------------------

    print("\nPandas JOIN sample:")
    print(
        pandas_join.head(10).to_string(
            index=False
        )
    )

    return are_equal

def main():

    print("=" * 60)
    print("ZEPTO DATA & AI PLATFORM")
    print("MODULE 1 — SQL ANALYSIS")
    print("=" * 60)

    connection = get_connection()

    try:

        # -------------------------------------------------
        # Query 1
        # -------------------------------------------------

        print("\nQUERY 1 — TOP 10 MOST EXPENSIVE BOOKS")
        print("-" * 60)

        result_1 = query_top_expensive_books(
            connection
        )

        print(result_1.to_string(index=False))

        # -------------------------------------------------
        # Query 2
        # -------------------------------------------------

        print("\nQUERY 2 — HIGH-RATED BOOKS")
        print("-" * 60)

        result_2 = query_high_rated_books(
            connection
        )

        print(result_2.head(10).to_string(index=False))

        # -------------------------------------------------
        # Query 3
        # -------------------------------------------------

        print("\nQUERY 3 — DISTINCT CATEGORIES")
        print("-" * 60)

        result_3 = query_categories(
            connection
        )

        print(result_3.to_string(index=False))

        # -------------------------------------------------
        # Query 4
        # -------------------------------------------------

        print("\nQUERY 4 — BOOKS BETWEEN £20 AND £40")
        print("-" * 60)

        result_4 = query_mid_range_books(
            connection
        )

        print(result_4.head(10).to_string(index=False))

        # -------------------------------------------------
        # Query 5
        # -------------------------------------------------

        print("\nQUERY 5 — BOOKS WITH RATING 4 OR 5")
        print("-" * 60)

        result_5 = query_selected_ratings(
            connection
        )

        print(result_5.head(10).to_string(index=False))

        # -------------------------------------------------
        # Query 6
        # -------------------------------------------------

        print("\nQUERY 6 — BOOKS WITH CATEGORY JOIN")
        print("-" * 60)

        result_6 = query_books_with_categories(
            connection
        )

        print(result_6.head(10).to_string(index=False))

        # -------------------------------------------------
        # Save JOIN output
        # -------------------------------------------------

        join_output = (
            BASE_DIR
            / "outputs"
            / "sql_join_result.csv"
        )

        result_6.to_csv(
            join_output,
            index=False
        )

        print("\nJOIN result saved to:")
        print(join_output)

                # -------------------------------------------------
        # SQL vs Pandas JOIN comparison
        # -------------------------------------------------

        join_match = compare_sql_and_pandas_join(
            connection
        )

        if join_match:
            print(
                "\nSUCCESS: SQL JOIN and Pandas JOIN "
                "produce identical results."
            )
        else:
            print(
                "\nWARNING: SQL JOIN and Pandas JOIN "
                "do not match."
            )

    finally:

        connection.close()


if __name__ == "__main__":
    
    main()