"""
FastAPI backend for the Support Assistance system.

Exposes the trained intent-classification model
through a REST API.
"""

from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(BASE_DIR / "support_assistance" / "src"),
)

from predict import predict_query


app = FastAPI(
    title="Zepto Support Assistance API",
    description="AI-powered customer support intent classification API",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    """Request schema for support prediction."""

    query: str


class SupportResponse(BaseModel):
    """Response schema for support prediction."""

    intent: str
    confidence: float
    response: str


@app.get("/")
def root():
    """API health endpoint."""

    return {
        "service": "Zepto Support Assistance API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    """Health check endpoint."""

    return {
        "status": "healthy",
    }


@app.post("/predict", response_model=SupportResponse)
def predict(request: SupportRequest):
    """Predict the support intent for a customer query."""

    intent, confidence, response = predict_query(
        request.query
    )

    return SupportResponse(
        intent=intent,
        confidence=confidence,
        response=response,
    )