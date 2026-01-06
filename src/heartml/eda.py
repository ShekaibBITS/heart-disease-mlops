"""heartml.eda

Notes (what this script does)
- Reproduces the notebook's EDA outputs in a script form for VS Code execution.
- Generates and saves:
  * Class distribution (bar + pie)
  * Numerical feature histograms
  * Correlation heatmap
- Saves plots into artifacts/plots for inclusion in the final report.

Run (from project root):
    python src/heartml/eda.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import NUMERICAL_FEATURES, TARGET_COL, PLOTS_DIR
from .utils import ensure_dir
from .preprocess import clean_dataset
from .data_ingest import load_or_download


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot and save class distribution (bar + pie), mirroring the notebook."""
    ensure_dir(PLOTS_DIR)

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found in dataframe.")

    target_counts = df[TARGET_COL].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar plot
    labels = [str(i) for i in target_counts.index.tolist()]
    axes[0].bar(labels, target_counts.values, edgecolor="black")
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")

    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + max(1, int(0.02 * v)), str(int(v)), ha="center")

    # Pie chart
    axes[1].pie(
        target_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Class Balance")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Plot and save numerical feature distributions with mean/median overlays (notebook-style)."""
    ensure_dir(PLOTS_DIR)

    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if not numeric_cols:
        raise ValueError("No numerical features found in dataframe for plotting.")

    # Create a 2x3 grid (up to 6). If more features, we still plot first 6 to keep layout stable.
    cols_to_plot = numeric_cols[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Distribution of Numerical Features", fontsize=16, fontweight="bold")
    axes = axes.ravel()

    for idx, col in enumerate(cols_to_plot):
        series = df[col].dropna()
        axes[idx].hist(series, bins=30, edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"{col.upper()} Distribution", fontweight="bold")
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")

        # Mean/Median
        axes[idx].axvline(series.mean(), linestyle="--", label="Mean")
        axes[idx].axvline(series.median(), linestyle="--", label="Median")

        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    # Remove unused axes
    for j in range(len(cols_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "numerical_distributions.png", dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot and save annotated feature correlation heatmap (notebook-style)."""
    ensure_dir(PLOTS_DIR)

    corr = df.corr(numeric_only=True)
    if corr.empty:
        raise ValueError("Correlation matrix is empty (no numeric columns).")

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    if TARGET_COL in corr.columns:
        print("Top correlations with target:")
        print(corr[TARGET_COL].sort_values(ascending=False).iloc[1:6])


def save_tabular_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Persist tabular EDA outputs (head, describe, missing values)
    so they are reproducible and available for reports and MLflow.
    """
    ensure_dir(output_dir)

    df.head().to_csv(output_dir / "data_head.csv", index=False)

    desc = df.describe().round(2)
    desc.to_csv(output_dir / "data_describe.csv")
    desc.to_markdown(output_dir / "data_describe.md")

    missing_summary = df.isna().sum().reset_index()
    missing_summary.columns = ["feature", "missing_count"]
    missing_summary.to_csv(output_dir / "missing_values.csv", index=False)


def _try_log_eda_to_mlflow(plots_dir: Path, tabular_dir: Path) -> None:
    """
    Optional: log EDA artifacts to MLflow (-> MinIO).
    Non-breaking: if MLflow not configured, it skips.
    """
    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)

        active = mlflow.active_run()
        if active is None:
            mlflow.set_experiment("heartml-eda")
            with mlflow.start_run():
                if plots_dir.exists():
                    mlflow.log_artifacts(str(plots_dir), artifact_path="eda/plots")
                if tabular_dir.exists():
                    mlflow.log_artifacts(str(tabular_dir), artifact_path="eda/tabular")
        else:
            if plots_dir.exists():
                mlflow.log_artifacts(str(plots_dir), artifact_path="eda/plots")
            if tabular_dir.exists():
                mlflow.log_artifacts(str(tabular_dir), artifact_path="eda/tabular")
    except Exception:
        return


if __name__ == "__main__":
    sns.set_style("darkgrid")

    df_raw = load_or_download()
    df_clean = clean_dataset(df_raw)

    plot_class_distribution(df_clean)
    plot_numerical_distributions(df_clean)
    plot_correlation_heatmap(df_clean)

    tabular_dir = PLOTS_DIR.parent / "metrics"
    save_tabular_eda(df_clean, tabular_dir)

    # Optional MLflow logging
    _try_log_eda_to_mlflow(PLOTS_DIR, tabular_dir)

    print("EDA plots saved to:", PLOTS_DIR)
    print("EDA tables saved to:", tabular_dir)
