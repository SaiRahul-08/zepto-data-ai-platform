"""
Tests for the Support Assistance prediction system.
"""

from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(BASE_DIR / "support_assistance" / "src"),
)

from predict import predict_query


def test_order_status_prediction():
    """
    A clear order-status query should produce either the
    expected intent or a safe unknown response.
    """

    intent, confidence, response = predict_query(
        "Where is my order?"
    )

    assert intent in {"order_status", "unknown"}
    assert 0.0 <= confidence <= 1.0
    assert response


def test_return_prediction():
    """
    A clear return query should produce either the expected
    intent or a safe unknown response.
    """

    intent, confidence, response = predict_query(
        "How do I return an item?"
    )

    assert intent in {"return_request", "unknown"}
    assert 0.0 <= confidence <= 1.0
    assert response


def test_low_confidence_query():
    """
    Low-confidence predictions should be rejected safely.
    """

    intent, confidence, response = predict_query(
        "Tell me something"
    )

    assert intent == "unknown"
    assert confidence < 0.35
    assert response


def test_empty_query():
    """Empty queries should be rejected."""

    intent, confidence, response = predict_query("")

    assert intent == "unknown"
    assert confidence == 0.0
    assert response


if __name__ == "__main__":

    test_order_status_prediction()
    test_return_prediction()
    test_low_confidence_query()
    test_empty_query()

    print("All prediction tests passed successfully.")