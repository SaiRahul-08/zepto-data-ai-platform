"""
Support Assistance prediction system.

Loads the trained TF-IDF vectorizer and intent classifier,
predicts the intent of a new customer query, and retrieves
the corresponding support response.
"""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "support_assistance" / "data" / "support_faq.csv"
MODEL_PATH = (
    BASE_DIR
    / "support_assistance"
    / "models"
    / "intent_classifier.joblib"
)
VECTORIZER_PATH = (
    BASE_DIR
    / "support_assistance"
    / "models"
    / "tfidf_vectorizer.joblib"
)


def clean_text(text: str) -> str:
    """Normalize a customer query."""
    return str(text).lower().strip()


def load_system():
    """Load model, vectorizer, and support responses."""

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    df = pd.read_csv(DATA_PATH)

    responses = (
        df[["intent", "response"]]
        .drop_duplicates("intent")
        .set_index("intent")["response"]
        .to_dict()
    )

    return model, vectorizer, responses

def predict_query(query: str, threshold: float = 0.35):
    """Predict intent and return a response only when confidence is sufficient."""

    model, vectorizer, responses = load_system()

    cleaned_query = clean_text(query)

    if not cleaned_query:
        return (
            "unknown",
            0.0,
            "Please enter a support question so I can assist you.",
        )

    features = vectorizer.transform([cleaned_query])

    predicted_intent = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = probabilities.max()

    if confidence < threshold:
        return (
            "unknown",
            confidence,
            "I'm not confident enough to identify your issue. "
            "Could you please provide more details about your problem?",
        )

    response = responses.get(
        predicted_intent,
        "Sorry, I could not find a suitable support response.",
    )

    return predicted_intent, confidence, response


def main():
    print("=" * 60)
    print("SUPPORT ASSISTANCE - PREDICTION")
    print("=" * 60)

    test_queries = [
        "I want to track my package",
        "My payment is not working",
        "I forgot my password",
        "Can I return something I purchased?",
        "Why is my discount code not working?",
    ]

    for query in test_queries:

        intent, confidence, response = predict_query(query)

        print("\nCustomer Query:")
        print(query)

        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f}")

        print("Support Response:")
        print(response)

        print("-" * 60)


if __name__ == "__main__":
    main()