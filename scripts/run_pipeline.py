"""scripts.run_pipeline

Notes (what this script does)
- Provides a single, top-level entrypoint to run the end-to-end Phase-1 & Phase-2 workflow
  (Data Acquisition -> EDA -> Model Training/Evaluation) from a clean VS Code environment.
- This script intentionally orchestrates the existing module CLIs (data_ingest.py, eda.py, train.py)
  to ensure the behavior matches the notebook-derived scripts exactly.
- Outputs (datasets, plots, metrics, models) are written to the same project folders used elsewhere:
    data/raw, data/processed, artifacts/plots, artifacts/metrics, models

How to run (from project root)
    python scripts/run_pipeline.py

Optional flags
    python scripts/run_pipeline.py --skip-eda
    python scripts/run_pipeline.py --skip-train
    python scripts/run_pipeline.py --only ingest
"""

# Import argparse to parse CLI flags in a standard way
import argparse  # Command-line interface

# Import subprocess to call the existing module entrypoints reliably
import subprocess  # Run child processes

# Import sys to forward the current interpreter and propagate exit codes
import sys  # Python interpreter + exit

# Import time to capture basic runtime durations for logs
import time  # Simple timing

# Import Path to reliably build OS-independent paths
from pathlib import Path  # File system paths


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a command, stream logs, and fail fast on errors."""

    # Print the command as a reproducibility aid for logs and debugging
    print(f"\n[RUN] {' '.join(cmd)}")  # Human-readable command

    # Run the command as a child process, inheriting stdout/stderr
    completed = subprocess.run(cmd, cwd=str(cwd))  # Execute

    # If the subprocess failed, stop the pipeline immediately
    if completed.returncode != 0:  # Non-zero means failure
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd}")  # Fail fast


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline control."""

    # Create an argument parser with a short description
    parser = argparse.ArgumentParser(description="Run the Heart Disease MLOps pipeline (Phase 1 & 2).")  # CLI

    # Allow running only one stage when debugging
    parser.add_argument(
        "--only",
        choices=["ingest", "eda", "train"],
        default=None,
        help="Run only a single stage (ingest | eda | train).",
    )  # Stage selector

    # Provide convenience switches to skip stages
    parser.add_argument("--skip-eda", action="store_true", help="Skip the EDA stage.")  # Skip EDA
    parser.add_argument("--skip-train", action="store_true", help="Skip the training stage.")  # Skip train

    # Return parsed arguments
    return parser.parse_args()  # Namespace


def main() -> None:
    """Main orchestration entrypoint."""

    # Capture script start time for a simple end-to-end duration
    t0 = time.time()  # Start timer

    # Resolve the project root as the parent folder of scripts/
    project_root = Path(__file__).resolve().parents[1]  # Project root

    # Parse CLI args
    args = parse_args()  # User options

    # Build commands using the current interpreter to guarantee venv correctness
    py = sys.executable  # Current python interpreter (should be .venv)

    # Define stage commands using -m flag to run modules as packages (enables relative imports)
    ingest_cmd = [py, "-m", "src.heartml.data_ingest"]  # Data acquisition
    eda_cmd = [py, "-m", "src.heartml.eda"]  # Exploratory analysis
    train_cmd = [py, "-m", "src.heartml.train"]  # Modeling + evaluation

    # If user requested a single stage, run only that stage and exit
    if args.only == "ingest":  # Only ingest
        _run(ingest_cmd, project_root)  # Execute
        return  # Done

    if args.only == "eda":  # Only EDA
        _run(eda_cmd, project_root)  # Execute
        return  # Done

    if args.only == "train":  # Only train
        _run(train_cmd, project_root)  # Execute
        return  # Done

    # Otherwise run the full pipeline in order: ingest -> eda -> train
    _run(ingest_cmd, project_root)  # Stage 1: acquire/cache dataset

    # Run EDA unless explicitly skipped
    if not args.skip_eda:  # Guard
        _run(eda_cmd, project_root)  # Stage 2: generate EDA plots

    # Run training unless explicitly skipped
    if not args.skip_train:  # Guard
        _run(train_cmd, project_root)  # Stage 3: train + evaluate + save artifacts

    # Print end-to-end duration for a quick success summary
    dt = time.time() - t0  # Elapsed time
    print(f"\nPipeline completed successfully in {dt:.2f} seconds.")  # Success message

    # Print key output locations to help the user find artifacts quickly
    print("Outputs:")  # Header
    print("  data/raw/ and data/processed/")  # Data folders
    print("  artifacts/plots/ and artifacts/metrics/")  # Artifact folders
    print("  models/")  # Model folder


if __name__ == "__main__":
    # Execute the orchestrator when called as a script
    main()  # Run pipeline
