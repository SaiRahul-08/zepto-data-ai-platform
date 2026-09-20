# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end data pipeline that collects book data from Books to Scrape, cleans the scraped data, converts GBP prices to INR, stores the data in a normalized SQLite database, and performs SQL and Pandas analysis.

## Pipeline Architecture

```text
Books to Scrape
       |
       v
requests + BeautifulSoup
       |
       v
raw_books.csv
       |
       v
Data Cleaning
       |
       +---- price_gbp
       +---- rating
       +---- in_stock
       +---- price_inr
       |
       v
clean_books.csv
       |
       v
SQLite Database
       |
       +---- categories
       |
       +---- books
       |
       v
SQL Analysis
       |
       v
Pandas JOIN Comparison
```
