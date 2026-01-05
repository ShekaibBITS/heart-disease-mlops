"""
api.main

FastAPI inference service for Heart Disease prediction.

Key behavior:
- Service MUST start even if MLflow model is unavailable.
- On startup: attempt to load model from MLflow.
- If load fails: log full error details; keep service running.
- /health reports model_loaded status.
- /predict returns 503 if model not loaded.

Endpoints:
  - GET  /health
  - GET  /metrics
  - POST /predict
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import Dict, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, generate_latest

from src.heartml.config import FEATURE_NAMES, TARGET_COL


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api-logger")


# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter("http_requests_total", "Total number of HTTP requests")


# -----------------------------
# Pydantic schemas
# -----------------------------
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature mapping {feature_name: value}")


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_artifact: str


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None
MODEL_URI: Optional[str] = None
MODEL_LOAD_ERROR: Optional[str] = None
MODEL_LOADED_AT: Optional[float] = None


# -----------------------------
# Middleware
# -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request started: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Request completed: status={response.status_code}")
    return response


@app.middleware("http")
async def count_requests(request: Request, call_next):
    REQUEST_COUNT.inc()
    return await call_next(request)


# -----------------------------
# Metrics
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


# -----------------------------
# MLflow model resolution + loading
# -----------------------------
def _resolve_mlflow_model_uri() -> str:
    """
    Resolve MLflow model URI from env.
    Supported:
      A) runs:/<RUN_ID>/<artifact_path>
      B) models:/<MODEL_NAME>/<STAGE>
    """
    tracking_uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id = (os.getenv("MLFLOW_RUN_ID") or "").strip()
    artifact_path = (os.getenv("MLFLOW_MODEL_ARTIFACT_PATH") or "model").strip()

    model_name = (os.getenv("MLFLOW_MODEL_NAME") or "").strip()
    model_stage = (os.getenv("MLFLOW_MODEL_STAGE") or "Production").strip()

    if run_id:
        return f"runs:/{run_id}/{artifact_path}"
    if model_name:
        return f"models:/{model_name}/{model_stage}"

    raise ValueError(
        "No MLflow model reference configured. Set either:\n"
        "- MLFLOW_RUN_ID (+ optional MLFLOW_MODEL_ARTIFACT_PATH)\n"
        "or\n"
        "- MLFLOW_MODEL_NAME (+ optional MLFLOW_MODEL_STAGE)\n"
        "Optionally set MLFLOW_TRACKING_URI (recommended in Docker)."
    )


def _try_load_model_once() -> None:
    """
    Try to load the model once. Never raises to caller.
    On failure: sets MODEL_LOAD_ERROR and logs details.
    """
    global MODEL, MODEL_URI, MODEL_LOAD_ERROR, MODEL_LOADED_AT

    try:
        uri = _resolve_mlflow_model_uri()
        MODEL_URI = uri
        logger.info(f"Attempting MLflow model load from: {uri}")

        m = mlflow.pyfunc.load_model(uri)

        MODEL = m
        MODEL_LOADED_AT = time.time()
        MODEL_LOAD_ERROR = None

        logger.info("MLflow model loaded successfully.")

    except Exception as e:
        # Keep service alive; store detailed error for /health and logs
        MODEL = None
        MODEL_LOADED_AT = None
        MODEL_LOAD_ERROR = f"{type(e).__name__}: {str(e)}"

        logger.error("MLflow model load failed. Service will continue without model.")
        logger.error(f"Model load error: {MODEL_LOAD_ERROR}")
        logger.error("Traceback:\n" + traceback.format_exc())


def _background_model_loader(retry_seconds: int = 30) -> None:
    """
    Background retry loop: keeps trying to load model until success.
    Useful for docker-compose startup ordering (mlflow may come up later).
    """
    while True:
        if MODEL is None:
            _try_load_model_once()
        time.sleep(max(5, retry_seconds))


@app.on_event("startup")
def _startup() -> None:
    """
    Non-failing startup:
    - Try load once
    - Optionally start retry loop in background
    """
    _try_load_model_once()

    enable_retry = (os.getenv("MODEL_LOAD_RETRY", "true").strip().lower() in {"1", "true", "yes"})
    retry_seconds = int(os.getenv("MODEL_LOAD_RETRY_SECONDS", "30").strip() or "30")

    if enable_retry:
        t = threading.Thread(target=_background_model_loader, args=(retry_seconds,), daemon=True)
        t.start()
        logger.info(f"Background model retry enabled (every {retry_seconds}s).")


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health() -> Dict[str, object]:
    """
    Report liveness + model readiness.
    """
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_uri": MODEL_URI,
        "model_loaded_at": MODEL_LOADED_AT,
        "model_load_error": MODEL_LOAD_ERROR,
    }


# -----------------------------
# Predict
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None or MODEL_URI is None:
        # 503 is correct: service is up but dependency (model) not ready
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model not loaded yet. Check /health for details.",
                "model_uri": MODEL_URI,
                "error": MODEL_LOAD_ERROR,
            },
        )

    feature_cols = [c for c in FEATURE_NAMES if c != TARGET_COL]
    missing = [c for c in feature_cols if c not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    row = {c: req.features[c] for c in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)

    # Preferred: MLflow-logged sklearn Pipeline classifier with predict_proba available.
    # Fallback: use predict output if proba not accessible.
    try:
        proba = MODEL._model_impl.python_model.predict_proba(X)  # type: ignore[attr-defined]
        p1 = float(proba[0][1])
    except Exception:
        y = MODEL.predict(X)
        try:
            p1 = float(y[0])
            p1 = min(1.0, max(0.0, p1))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")

    pred = int(p1 >= 0.5)

    return PredictResponse(
        prediction=pred,
        confidence=p1,
        model_artifact=MODEL_URI,
    )
