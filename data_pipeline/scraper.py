import csv
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "raw_books.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZeptoDataPipeline/1.0)"
}


def get_soup(url):
    """
    Download a webpage and return its BeautifulSoup object.
    """
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=15
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    """
    Get the first three book categories from the website.
    """
    soup = get_soup(BASE_URL)

    category_links = soup.select(
        ".side_categories ul li ul li a"
    )

    categories = []

    for link in category_links[:3]:
        category_name = link.get_text(strip=True)
        category_url = BASE_URL + link["href"]

        categories.append(
            {
                "name": category_name,
                "url": category_url
            }
        )

    return categories


def scrape_category(category_name, category_url):
    """
    Scrape every book from one category, including pagination.
    """
    books = []

    current_url = category_url

    while current_url:

        print(f"Scraping: {current_url}")

        soup = get_soup(current_url)

        products = soup.select("article.product_pod")

        for product in products:

            title = product.h3.a.get("title", "").strip()

            price = product.select_one(
                ".price_color"
            ).get_text(strip=True)

            rating_element = product.select_one(
                "p.star-rating"
            )

            star_rating = ""

            if rating_element:
                classes = rating_element.get("class", [])

                for rating in ["One", "Two", "Three", "Four", "Five"]:
                    if rating in classes:
                        star_rating = rating
                        break

            availability_element = product.select_one(
                ".availability"
            )

            availability = ""

            if availability_element:
                availability = availability_element.get_text(
                    " ",
                    strip=True
                )

            books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name
                }
            )

        next_button = soup.select_one(
            "li.next a"
        )

        if next_button:

            next_href = next_button.get("href")

            current_url = (
                current_url.rsplit("/", 1)[0]
                + "/"
                + next_href
            )

        else:
            current_url = None

        time.sleep(0.2)

    return books


def save_books(books):
    """
    Save scraped books to a CSV file.
    """
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "title",
        "price",
        "star_rating",
        "availability",
        "category"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(books)


def main():

    print("=" * 60)
    print("ZEPTO DATA & AI PLATFORM")
    print("MODULE 1 — BOOK SCRAPER")
    print("=" * 60)

    print("\nFinding categories...")

    categories = get_categories()

    print(f"Found {len(categories)} categories to scrape:")

    for category in categories:
        print(f"  - {category['name']}")

    all_books = []

    for category in categories:

        print(
            f"\nStarting category: {category['name']}"
        )

        category_books = scrape_category(
            category["name"],
            category["url"]
        )

        print(
            f"Books collected from "
            f"{category['name']}: {len(category_books)}"
        )

        all_books.extend(category_books)

    save_books(all_books)

    unique_categories = sorted(
        set(book["category"] for book in all_books)
    )

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETED")
    print("=" * 60)

    print(f"Total books: {len(all_books)}")
    print(f"Total categories: {len(unique_categories)}")

    print("\nCategories:")
    for category in unique_categories:
        count = sum(
            1
            for book in all_books
            if book["category"] == category
        )

        print(f"  {category}: {count} books")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()