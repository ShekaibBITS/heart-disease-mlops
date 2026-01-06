# Notes:
# - Smoke test: ensures training executes end-to-end.
# - Validates that training produces expected artifacts.
# - CI-safe: isolates outputs into tmp_path and avoids path/CWD issues.


import pandas as pd

from src.heartml import config
from src.heartml import data_ingest
from src.heartml.train import main


def test_training_runs_and_writes_artifacts(monkeypatch, tmp_path):
    """Training should run and generate core metric artifacts."""

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

    main()

    assert metrics_path.exists()
    assert comparison_path.exists()

    # Optional: basic sanity checks (non-empty files)
    assert metrics_path.stat().st_size > 0
    assert comparison_path.stat().st_size > 0
