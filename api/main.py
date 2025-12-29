"""
api.main

Notes:
- FastAPI inference service for Heart Disease prediction.
- Loads:
  - models/best_model.joblib (selected automatically during training)
  - models/scaler.joblib (saved preprocessing transformer)
- Endpoints:
  - GET  /health
  - POST /predict -> returns prediction + confidence (probability of disease)

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from typing import Dict  # Type hints

import joblib  # Artifact loading
import pandas as pd  # Build inference DataFrame
from fastapi import FastAPI, HTTPException  # API framework
from pydantic import BaseModel, Field  # Validation

from src.heartml.config import (
    PROJECT_ROOT,
    FEATURE_NAMES,
    NUMERICAL_FEATURES,
    TARGET_COL,
    BEST_MODEL_PATH,
)  # Shared schema


SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"  # Saved scaler


class PredictRequest(BaseModel):
    """Request schema: feature mapping."""
    features: Dict[str, float] = Field(..., description="Feature mapping {feature_name: value}")


class PredictResponse(BaseModel):
    """Response schema: prediction and confidence."""
    prediction: int
    confidence: float
    model_artifact: str


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")  # App instance

model = None  # Loaded model
scaler = None  # Loaded scaler


@app.on_event("startup")
def _startup() -> None:
    """Load artifacts on startup."""
    global model, scaler  # Use module globals

    if not BEST_MODEL_PATH.exists():  # Validate model exists
        raise RuntimeError(f"Best model not found: {BEST_MODEL_PATH}")  # Fail fast

    if not SCALER_PATH.exists():  # Validate scaler exists
        raise RuntimeError(f"Scaler not found: {SCALER_PATH}")  # Fail fast

    model = joblib.load(BEST_MODEL_PATH)  # Load best model
    scaler = joblib.load(SCALER_PATH)  # Load scaler


@app.get("/health")
def health() -> Dict[str, str]:
    """Health endpoint."""
    return {"status": "ok"}  # Liveness check


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict disease risk and confidence score."""
    if model is None or scaler is None:  # Defensive
        raise HTTPException(status_code=500, detail="Artifacts not loaded")  # Server error

    feature_cols = [c for c in FEATURE_NAMES if c != TARGET_COL]  # Exclude target
    missing = [c for c in feature_cols if c not in req.features]  # Missing inputs

    if missing:  # Validate request completeness
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")  # Client error

    row = {c: req.features[c] for c in feature_cols}  # Ordered row
    X = pd.DataFrame([row], columns=feature_cols)  # One-row DF

    # Apply scaling only to the numeric subset (same as training).
    X_scaled = X.copy()
    X_scaled[NUMERICAL_FEATURES] = scaler.transform(X[NUMERICAL_FEATURES])

    proba = float(model.predict_proba(X_scaled)[0, 1])  # P(disease)
    pred = int(proba >= 0.5)  # Default threshold

    return PredictResponse(
        prediction=pred,
        confidence=proba,
        model_artifact=BEST_MODEL_PATH.name,
    )
