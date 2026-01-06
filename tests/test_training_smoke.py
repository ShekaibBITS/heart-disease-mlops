# tests/test_training_smoke.py

import os
import pandas as pd

from src.heartml import config, data_ingest
from src.heartml.train import main


def test_training_runs_and_writes_artifacts(monkeypatch, tmp_path):
    """Training should run and generate core metric artifacts (CI-safe)."""

    # 0) Force MLflow to use local file store (NO server required)
    # Put mlruns in tmp_path to avoid repo pollution.
    mlruns_dir = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{mlruns_dir.as_posix()}")

    # Optional: reduce noise and make runs deterministic in CI
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

    # 1) Ensure training writes to a temp dir (no repo pollution)
    metrics_path = tmp_path / "metrics.json"
    comparison_path = tmp_path / "model_comparison.csv"

    monkeypatch.setattr(config, "METRICS_JSON_PATH", str(metrics_path))
    monkeypatch.setattr(config, "MODEL_COMPARISON_PATH", str(comparison_path))

    # 2) Avoid network ingestion (recommended for CI stability)
    fake_df = pd.DataFrame(
        {
            "age": [63, 67, 37, 41],
            "sex": [1, 1, 1, 0],
            "trestbps": [145, 160, 120, 140],
            "chol": [233, 286, 250, 204],
            config.TARGET_COL: [1, 0, 0, 1],
        }
    )
    monkeypatch.setattr(data_ingest, "load_or_download", lambda: fake_df)

    # 3) Run from a controlled working directory
    monkeypatch.chdir(tmp_path)

    # 4) Execute training
    main()

    # 5) Assert artifacts exist
    assert metrics_path.exists(), "metrics.json was not created"
    assert comparison_path.exists(), "model_comparison.csv was not created"
    assert metrics_path.stat().st_size > 0, "metrics.json is empty"
    assert comparison_path.stat().st_size > 0, "model_comparison.csv is empty"
