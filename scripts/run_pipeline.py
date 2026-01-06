"""
scripts.run_pipeline

Notes (what this script does)
- Provides a single, top-level entrypoint to run the end-to-end Phase-1 & Phase-2 workflow
  (Data Acquisition -> EDA -> Model Training/Evaluation) from a clean VS Code environment.
- This script orchestrates the existing module CLIs (data_ingest.py, eda.py, train.py)
  to ensure the behavior matches the notebook-derived scripts.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a command, stream logs, and fail fast on errors."""
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline control."""
    parser = argparse.ArgumentParser(description="Run the Heart Disease MLOps pipeline (Phase 1 & 2).")

    parser.add_argument(
        "--only",
        choices=["ingest", "eda", "train"],
        default=None,
        help="Run only a single stage (ingest | eda | train).",
    )

    parser.add_argument("--skip-eda", action="store_true", help="Skip the EDA stage.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training stage.")
    return parser.parse_args()


def main() -> None:
    """Main orchestration entrypoint."""
    t0 = time.time()

    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    py = sys.executable

    # Correct module paths for standard src-layout packaging (src/heartml/...)
    ingest_cmd = [py, "-m", "heartml.data_ingest"]
    eda_cmd = [py, "-m", "heartml.eda"]
    train_cmd = [py, "-m", "heartml.train"]

    if args.only == "ingest":
        _run(ingest_cmd, project_root)
        return

    if args.only == "eda":
        _run(eda_cmd, project_root)
        return

    if args.only == "train":
        _run(train_cmd, project_root)
        return

    _run(ingest_cmd, project_root)

    if not args.skip_eda:
        _run(eda_cmd, project_root)

    if not args.skip_train:
        _run(train_cmd, project_root)

    dt = time.time() - t0
    print(f"\nPipeline completed successfully in {dt:.2f} seconds.")
    print("Outputs:")
    print("  data/raw/ and data/processed/")
    print("  artifacts/plots/ and artifacts/metrics/")
    print("  models/")


if __name__ == "__main__":
    main()
