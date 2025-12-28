# Notes:
# - Smoke test: ensures training executes end-to-end.
# - Validates that training produces expected artifacts.
# - Intended to catch broken imports, path issues, and runtime regressions.

from pathlib import Path  # Path assertions

from src.heartml.train import main  # Training entry point
from src.heartml.config import METRICS_JSON_PATH, MODEL_COMPARISON_PATH  # Expected outputs


def test_training_runs_and_writes_artifacts():
    """Training should run and generate core metric artifacts."""

    main()  # Run full training

    assert Path(METRICS_JSON_PATH).exists()  # Metrics JSON must be created
    assert Path(MODEL_COMPARISON_PATH).exists()  # Comparison CSV must be created
