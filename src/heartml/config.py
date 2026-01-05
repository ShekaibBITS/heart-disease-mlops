"""heartml.config

Central configuration constants used across ingestion, preprocessing, training,
evaluation, and serving.

Key updates:
- Default MLflow tracking is a RUNNING SERVER (http://localhost:5000).
- Always allow env var override (Docker: http://mlflow:5000, K8s: http://mlflow:5000).
- Keep a file-based URI available as a fallback option (useful for CI/offline).
"""

from __future__ import annotations

import os
from pathlib import Path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _env_bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip()


# -----------------------------------------------------------------------------
# Project root
# -----------------------------------------------------------------------------
# src/heartml/config.py -> parents[0]=heartml, [1]=src, [2]=repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


# -----------------------------------------------------------------------------
# Train/test split configuration
# -----------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Model training configuration
# -----------------------------------------------------------------------------
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

CV_FOLDS = 5

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [8, 10, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


# -----------------------------------------------------------------------------
# Paths for artifacts and outputs
# -----------------------------------------------------------------------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
LR_MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
RF_TUNED_MODEL_PATH = MODEL_DIR / "random_forest_tuned_model.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
METRICS_JSON_PATH = METRICS_DIR / "metrics.json"


# -----------------------------------------------------------------------------
# MLflow configuration (server-first, env-overridable)
# -----------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = _env_str("MLFLOW_EXPERIMENT_NAME", "heart-disease-uci")

# File-based tracking (useful fallback for CI / offline mode)
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_FILE_TRACKING_URI = f"file:{MLFLOW_TRACKING_DIR.resolve()}"

# Default to a local running MLflow server.
# Override via env:
#   - Local:   MLFLOW_TRACKING_URI=http://localhost:5000
#   - Docker:  MLFLOW_TRACKING_URI=http://mlflow:5000
#   - K8s:     MLFLOW_TRACKING_URI=http://mlflow:5000
#
# Optional offline/CI switch:
#   - MLFLOW_USE_FILE_STORE=true  -> uses file:... even if MLFLOW_TRACKING_URI not set
MLFLOW_USE_FILE_STORE = _env_bool("MLFLOW_USE_FILE_STORE", "false")

_default_tracking = "http://localhost:5000"
MLFLOW_TRACKING_URI = _env_str("MLFLOW_TRACKING_URI", _default_tracking)

# If user wants explicit file store mode (CI/offline), override tracking URI.
if MLFLOW_USE_FILE_STORE:
    MLFLOW_TRACKING_URI = MLFLOW_FILE_TRACKING_URI

# Standardize model artifact path (API expects runs:/<RUN_ID>/<artifact_path>)
MLFLOW_MODEL_ARTIFACT_PATH = _env_str("MLFLOW_MODEL_ARTIFACT_PATH", "model")

# MLflow Registry (optional / non-breaking)
MLFLOW_REGISTER_MODEL = _env_bool("MLFLOW_REGISTER_MODEL", "false")
MLFLOW_MODEL_NAME = _env_str("MLFLOW_MODEL_NAME", "heart_disease_classifier")

# Allow disabling stage transition by setting empty string:
#   MLFLOW_MODEL_STAGE=""  -> no transition call will be made
MLFLOW_MODEL_STAGE = _env_str("MLFLOW_MODEL_STAGE", "Staging")
if MLFLOW_MODEL_STAGE.strip() == "":
    MLFLOW_MODEL_STAGE = ""
