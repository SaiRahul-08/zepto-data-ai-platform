# Zepto Support Assistance

An NLP-based customer support intent classification system with a FastAPI inference layer.

## Overview

The Support Assistance module classifies customer support queries into predefined intents and returns an appropriate support response.

The module combines:

- Text preprocessing
- TF-IDF feature extraction
- Machine learning classification
- Confidence-based prediction
- Saved ML models
- Automated tests
- FastAPI REST API
- Swagger/OpenAPI documentation

## Architecture

Customer Query
|
v
FastAPI `/predict`
|
v
Prediction Layer
|
+---- TF-IDF Vectorizer
|
+---- Intent Classifier
|
v
Confidence Check
|
+---- Confident prediction
|
+---- `unknown` for low confidence
|
v
Support Response

## Dataset

Dataset:

`support_assistance/data/support_faq.csv`

The dataset contains 50 customer support examples distributed across 10 intents.

### Supported Intents

- order_status
- order_cancel
- refund
- payment_issue
- delivery_delay
- product_availability
- product_info
- account_issue
- return_request
- coupon_issue

Each record contains:

- `query`
- `intent`
- `response`

## NLP Preprocessing

The preprocessing pipeline:

1. Loads the support FAQ dataset.
2. Separates customer queries and intent labels.
3. Converts text into TF-IDF features.
4. Splits the dataset into training and testing sets.
5. Reports dataset and vocabulary statistics.

Example result:

```text
Dataset shape: (50, 3)
Number of intents: 10
TF-IDF matrix shape: (50, 115)
Vocabulary size: 115
Training samples: 40
Testing samples: 10
```
