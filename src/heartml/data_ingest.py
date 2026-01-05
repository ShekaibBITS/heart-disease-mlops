# src/heartml/data_ingest.py
"""heartml.data_ingest

Notes (what this script does)
- Downloads the Heart Disease (processed Cleveland) dataset from UCI, as done in the notebook.
- Saves the raw dataset to data/raw for reproducibility and offline runs.
- Provides a load function used by training scripts.

Run (from project root):
    python -m src.heartml.data_ingest
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict

import pandas as pd
import requests

from .config import DATASET_URL, FEATURE_NAMES, RAW_DATA_PATH
from .utils import ensure_dir, sha256_file


def _read_csv_text(text: str) -> pd.DataFrame:
    """Parse UCI text into DataFrame with canonical schema."""
    return pd.read_csv(
        StringIO(text),
        header=None,
        names=FEATURE_NAMES,
        na_values="?",
    )


def _read_cached_csv(path) -> pd.DataFrame:
    """Read cached raw CSV from disk with consistent NA handling."""
    return pd.read_csv(path, na_values="?")


def download_dataset() -> pd.DataFrame:
    """Download the dataset from UCI and return as a pandas DataFrame."""
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    return _read_csv_text(response.text)


def save_raw_dataset(df: pd.DataFrame) -> None:
    """Save the raw dataset to disk for reproducibility."""
    ensure_dir(RAW_DATA_PATH.parent)
    df.to_csv(RAW_DATA_PATH, index=False)


def load_or_download() -> pd.DataFrame:
    """Load raw dataset from disk if present; otherwise download and save it."""
    if RAW_DATA_PATH.exists():
        return _read_cached_csv(RAW_DATA_PATH)

    df = download_dataset()
    save_raw_dataset(df)
    return df


def _dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Small dataset profile for tracking.
    - schema (dtype)
    - missingness summary
    - basic shape
    """
    schema = {c: str(df[c].dtype) for c in df.columns}
    missing_pct = (df.isna().mean() * 100.0).round(3).to_dict()

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "schema": schema,
        "missing_pct": missing_pct,
    }


def _try_log_to_mlflow(df: pd.DataFrame) -> None:
    """
    Optional: log dataset + feature metadata to MLflow.

    MinIO-safe approach:
      - Uses tags/params + artifacts (CSV snapshot + JSON metadata)
      - Does NOT use mlflow.log_input / mlflow.data.* dataset APIs

    Non-breaking: if MLflow not configured or errors occur, it silently skips.
    """
    import os

    tracking_uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if not tracking_uri:
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)

        # Deterministic dataset version
        dataset_sha = sha256_file(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else "missing_raw_data_file"
        profile = _dataset_profile(df)

        def _log_common() -> None:
            mlflow.log_param("dataset_url", DATASET_URL)
            mlflow.log_param("dataset_rows", profile["rows"])
            mlflow.log_param("dataset_cols", profile["cols"])

            mlflow.set_tag("dataset_path", str(RAW_DATA_PATH))
            mlflow.set_tag("dataset_sha256", dataset_sha)
            mlflow.set_tag("feature_names_csv", ",".join(profile["columns"]))

            # JSON metadata artifacts (easy to review)
            mlflow.log_dict(profile["schema"], "data/schema.json")
            mlflow.log_dict(profile["missing_pct"], "data/missing_pct.json")
            mlflow.log_dict(profile, "data/dataset_profile.json")

            # Raw snapshot artifact (stored in MinIO)
            if RAW_DATA_PATH.exists():
                mlflow.log_artifact(str(RAW_DATA_PATH), artifact_path="data/raw")

        active = mlflow.active_run()
        if active is None:
            mlflow.set_experiment("heartml-data-ingest")
            with mlflow.start_run(run_name="data_ingest"):
                _log_common()
        else:
            _log_common()

    except Exception:
        return


if __name__ == "__main__":
    df_raw = load_or_download()

    # Optional MLflow + MinIO artifact logging
    _try_log_to_mlflow(df_raw)

    print("Dataset ready.")
    print(f"Path: {RAW_DATA_PATH}")
    print(f"Shape: {df_raw.shape}")
    print(df_raw.head())
