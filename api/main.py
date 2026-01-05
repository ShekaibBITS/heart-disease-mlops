"""
api.main

FastAPI inference service for Heart Disease prediction.

Key behavior:
- Service MUST start even if MLflow model is unavailable.
- On startup: attempt to load model from MLflow.
- If load fails: log full error details; keep service running.
- /health reports model_loaded status.
- /predict returns 503 if model not loaded.

Monitoring:
- HTTP metrics (requests, latency, status codes) via prometheus-fastapi-instrumentator at /metrics
- Custom metrics:
  - model_loaded (gauge)
  - predictions_total (counter)
  - prediction_errors_total (counter)
  - inference_latency_seconds (histogram)

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
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from api.routes.admin_pipeline import router as admin_router


# Prefer non-src import for Docker/packaging correctness, but keep fallback for your current layout.
try:
    from heartml.config import FEATURE_NAMES, TARGET_COL  # type: ignore
except Exception:  # pragma: no cover
    from src.heartml.config import FEATURE_NAMES, TARGET_COL  # type: ignore


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api-logger")


# -----------------------------
# Custom Prometheus metrics
# -----------------------------
MODEL_LOADED = Gauge("model_loaded", "1 if ML model is loaded, else 0")

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of successful predictions served",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of prediction failures (exceptions during inference)",
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds (from request payload parsing to model output)",
)


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
# App + model state
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
app.include_router(admin_router) 

MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None
MODEL_URI: Optional[str] = None
MODEL_LOAD_ERROR: Optional[str] = None
MODEL_LOADED_AT: Optional[float] = None


# -----------------------------
# HTTP metrics
# -----------------------------
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# -----------------------------
# Middleware: request logging
# -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request started: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Request completed: status={response.status_code}")
    return response


# -----------------------------
# MLflow model resolution + loading
# -----------------------------
def _configure_mlflow_from_env() -> None:
    """
    Configure MLflow tracking + (optional) MinIO/S3 environment via env vars.
    Non-breaking: if vars are missing, MLflow uses defaults.
    """
    tracking_uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Important for MinIO-backed artifacts: these are read by boto3/MLflow internally.
    # If not provided, no error is raised here.
    # MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
    # are already set via docker-compose env; we do not need to set them again in code.


def _resolve_mlflow_model_uri() -> str:
    """
    Resolve MLflow model URI from env.
    Supported:
      A) runs:/<RUN_ID>/<artifact_path>  (your current deployment contract)
      B) models:/<MODEL_NAME>/<STAGE>    (optional; does not break A)
    """
    _configure_mlflow_from_env()

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
        "Optionally set MLFLOW_TRACKING_URI (recommended in Docker/K8s)."
    )


def _set_model_loaded_metric() -> None:
    MODEL_LOADED.set(1 if MODEL is not None else 0)


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

        # Load model
        m = mlflow.pyfunc.load_model(uri)

        MODEL = m
        MODEL_LOADED_AT = time.time()
        MODEL_LOAD_ERROR = None
        logger.info("MLflow model loaded successfully.")

    except Exception as e:
        MODEL = None
        MODEL_LOADED_AT = None
        MODEL_LOAD_ERROR = f"{type(e).__name__}: {str(e)}"

        logger.error("MLflow model load failed. Service will continue without model.")
        logger.error(f"Model load error: {MODEL_LOAD_ERROR}")
        logger.error("Traceback:\n" + traceback.format_exc())

    finally:
        _set_model_loaded_metric()


def _background_model_loader(retry_seconds: int = 30) -> None:
    """
    Background retry loop: keeps trying to load model until success.
    Useful for docker-compose startup ordering.
    """
    while True:
        if MODEL is None:
            _try_load_model_once()
        time.sleep(max(5, retry_seconds))


def _extract_probability(pred_output: Any) -> float:
    """
    Robustly extract probability-of-positive-class from MLflow pyfunc output.

    Accepts common formats:
    - list/np array: [0.83]
    - pandas DataFrame with column 'proba'
    - list of dicts: [{'proba': 0.83, ...}]
    - dict-like with 'proba'
    - scalar numeric
    """
    # DataFrame with 'proba'
    try:
        if hasattr(pred_output, "columns") and "proba" in pred_output.columns:
            return float(pred_output["proba"].iloc[0])
    except Exception:
        pass

    # dict-like
    try:
        if isinstance(pred_output, dict) and "proba" in pred_output:
            return float(pred_output["proba"])
    except Exception:
        pass

    # list of dicts
    try:
        if isinstance(pred_output, list) and pred_output and isinstance(pred_output[0], dict):
            if "proba" in pred_output[0]:
                return float(pred_output[0]["proba"])
    except Exception:
        pass

    # list/array/series
    try:
        return float(pred_output[0])
    except Exception:
        pass

    # scalar
    try:
        return float(pred_output)
    except Exception:
        raise ValueError(
            "Unsupported MLflow prediction output format. "
            "Standardize your logged model to return probability (recommended: array/list scalar or DF/dict with 'proba')."
        )


# -----------------------------
# Startup: model load only (safe)
# -----------------------------
@app.on_event("startup")
def _startup() -> None:
    _try_load_model_once()

    enable_retry = (os.getenv("MODEL_LOAD_RETRY", "true").strip().lower() in {"1", "true", "yes"})
    retry_seconds_raw = (os.getenv("MODEL_LOAD_RETRY_SECONDS") or "30").strip()
    try:
        retry_seconds = int(retry_seconds_raw)
    except Exception:
        retry_seconds = 30

    if enable_retry:
        t = threading.Thread(target=_background_model_loader, args=(retry_seconds,), daemon=True)
        t.start()
        logger.info(f"Background model retry enabled (every {retry_seconds}s).")


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health() -> Dict[str, object]:
    _set_model_loaded_metric()

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
        _set_model_loaded_metric()
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model not loaded yet. Check /health for details.",
                "model_uri": MODEL_URI,
                "error": MODEL_LOAD_ERROR,
            },
        )

    _set_model_loaded_metric()

    # Ensure request has all expected features (excluding target)
    feature_cols = [c for c in FEATURE_NAMES if c != TARGET_COL]
    missing = [c for c in feature_cols if c not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Build ordered dataframe for model
    X = pd.DataFrame([{c: req.features[c] for c in feature_cols}], columns=feature_cols)

    start = time.time()
    try:
        out = MODEL.predict(X)
        p1 = _extract_probability(out)
        p1 = float(min(1.0, max(0.0, p1)))

        PREDICTIONS_TOTAL.inc()
    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {str(e)}")
    finally:
        INFERENCE_LATENCY_SECONDS.observe(time.time() - start)

    pred = int(p1 >= 0.5)
   
    return PredictResponse(
        prediction=pred,
        confidence=p1,
        model_artifact=MODEL_URI,
    )
