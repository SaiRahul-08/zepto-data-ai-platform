from pathlib import Path
import sqlite3

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "outputs" / "clean_books.csv"
DATABASE_FILE = BASE_DIR / "outputs" / "zepto_books.db"


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------

def create_connection():
    """
    Create a connection to the SQLite database.
    """

    connection = sqlite3.connect(DATABASE_FILE)

    # Enable foreign-key enforcement.
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# ---------------------------------------------------------
# Create database schema
# ---------------------------------------------------------

def create_tables(connection):
    """
    Create the normalized categories and books tables.
    """

    cursor = connection.cursor()

    # Categories table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # Books table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    connection.commit()


# ---------------------------------------------------------
# Insert categories
# ---------------------------------------------------------

def insert_categories(connection, df):
    """
    Insert unique categories and return their IDs.
    """

    cursor = connection.cursor()

    categories = sorted(
        df["category"].dropna().unique()
    )

    for category in categories:
        cursor.execute(
            """
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    connection.commit()

    # Read category IDs into a dictionary.
    cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    )

    category_map = {
        category_name: category_id
        for category_id, category_name in cursor.fetchall()
    }

    return category_map


# ---------------------------------------------------------
# Insert books
# ---------------------------------------------------------

def insert_books(connection, df, category_map):
    """
    Insert cleaned book records into the books table.
    """

    cursor = connection.cursor()

    for _, row in df.iterrows():

        category_id = category_map[row["category"]]

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                int(category_id),
            )
        )

    connection.commit()


# ---------------------------------------------------------
# Database verification
# ---------------------------------------------------------

def verify_database(connection):
    """
    Print basic database statistics.
    """

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM categories"
    )

    category_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM books"
    )

    book_count = cursor.fetchone()[0]

    print("\nDatabase verification")
    print("-" * 40)

    print(f"Categories: {category_count}")
    print(f"Books: {book_count}")

    print("\nCategories table:")

    cursor.execute(
        """
        SELECT *
        FROM categories
        ORDER BY category_id
        """
    )

    for row in cursor.fetchall():
        print(row)

    print("\nSample books:")

    cursor.execute(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        LIMIT 5
        """
    )

    for row in cursor.fetchall():
        print(row)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("ZEPTO DATA & AI PLATFORM")
    print("MODULE 1 — SQLITE DATABASE")
    print("=" * 60)

    # Make sure output directory exists.
    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load cleaned data.
    print("\nLoading cleaned dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")

    # Create connection.
    connection = create_connection()

    try:

        # Create normalized schema.
        create_tables(connection)

        # Insert categories.
        category_map = insert_categories(
            connection,
            df
        )

        # Insert books.
        insert_books(
            connection,
            df,
            category_map
        )

        # Verify.
        verify_database(connection)

    finally:

        connection.close()

    print("\nDatabase created successfully:")
    print(DATABASE_FILE)


if __name__ == "__main__":
    main()