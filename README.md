# Zepto Data AI Platform

An end-to-end data and AI engineering platform built with Python, SQL, Machine Learning, NLP, and FastAPI.

This project is developed as a modular engineering portfolio that demonstrates the complete workflow from data collection and database analysis to machine learning and AI-powered support assistance.

---

## Project Overview

The platform is organized into three major modules:

| Module   | Focus                         | Technologies                                      |
| -------- | ----------------------------- | ------------------------------------------------- |
| Module 1 | Data Pipeline & SQL Analytics | Python, Pandas, BeautifulSoup, SQLite, SQL        |
| Module 2 | Titanic Machine Learning      | Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib |
| Module 3 | Support Assistance NLP API    | TF-IDF, Machine Learning, FastAPI, Pytest         |

---

# Architecture

```text
                         Zepto Data AI Platform
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
       Data Pipeline        Machine Learning      AI Support
              |                   |                   |
       Web Scraping          Titanic Dataset      FAQ Dataset
              |                   |                   |
       Data Cleaning             EDA              NLP Preprocessing
              |                   |                   |
          SQLite DB          ML Training         Intent Classification
              |                   |                   |
       SQL + Pandas          Evaluation          FastAPI REST API
              |                   |                   |
              +-------------------+-------------------+
                                  |
                                  v
                         Analytics & AI Services
```
