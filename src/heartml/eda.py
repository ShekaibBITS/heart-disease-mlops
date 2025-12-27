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

# Import numpy for numeric utilities
import numpy as np  # Numerical computing

# Import pandas for DataFrame operations
import pandas as pd  # Data manipulation

# Import matplotlib for plotting
import matplotlib.pyplot as plt  # Plotting library

# Import seaborn for heatmaps and style
import seaborn as sns  # Statistical plots

# Import config for feature groupings and paths
from .config import NUMERICAL_FEATURES, TARGET_COL, PLOTS_DIR  # Central constants

# Import helper to ensure plot directory exists
from .utils import ensure_dir  # Directory creation

# Import preprocessing function so EDA operates on cleaned data
from .preprocess import clean_dataset  # Cleaning consistent with training pipeline

# Import loader to acquire raw data
from .data_ingest import load_or_download  # Download/cache raw dataset


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot and save class distribution (bar + pie), mirroring the notebook."""

    # Ensure output directory exists
    ensure_dir(PLOTS_DIR)  # Create artifacts/plots if required

    # Count class occurrences
    target_counts = df[TARGET_COL].value_counts().sort_index()  # Ensure order [0,1]

    # Create a side-by-side subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Two plots in one row

    # ----- Bar plot -----
    axes[0].bar(["No Disease", "Disease"], target_counts.values, edgecolor="black")  # Bar chart
    axes[0].set_title("Class Distribution")  # Title
    axes[0].set_ylabel("Count")  # Y-axis label

    # Annotate bars with counts
    for i, v in enumerate(target_counts.values):  # Iterate bars
        axes[0].text(i, v + 3, str(int(v)), ha="center")  # Add text labels

    # ----- Pie chart -----
    axes[1].pie(
        target_counts.values,  # Values for slices
        labels=["No Disease", "Disease"],  # Slice labels
        autopct="%1.1f%%",  # Percentage labels
        startangle=90,  # Rotate for readability
    )
    axes[1].set_title("Class Balance")  # Title

    # Improve layout
    fig.tight_layout()  # Avoid overlap

    # Save to disk
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=200)  # Persist artifact

    # Close figure
    plt.close(fig)  # Cleanup

def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Plot and save numerical feature distributions with mean/median overlays (notebook-style)."""

    # Ensure output directory exists
    ensure_dir(PLOTS_DIR)  # Create artifacts/plots if required

    # Create a subplot grid (2x3 supports up to 6 numerical features)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Large figure for report-quality plots
    fig.suptitle("Distribution of Numerical Features", fontsize=16, fontweight="bold")  # Global title
    axes = axes.ravel()  # Flatten axes for simple iteration

    # Plot each numerical feature
    for idx, col in enumerate(NUMERICAL_FEATURES):  # Loop over numeric columns
        axes[idx].hist(df[col], bins=30, edgecolor="black", alpha=0.7)  # Histogram
        axes[idx].set_title(f"{col.upper()} Distribution", fontweight="bold")  # Title
        axes[idx].set_xlabel("Value")  # X-axis label
        axes[idx].set_ylabel("Frequency")  # Y-axis label

        # Mean line
        axes[idx].axvline(
            df[col].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )

        # Median line
        axes[idx].axvline(
            df[col].median(),
            color="green",
            linestyle="--",
            label="Median",
        )

        axes[idx].legend()  # Show mean/median legend
        axes[idx].grid(True, alpha=0.3)  # Light grid

    # Remove unused subplots
    if len(NUMERICAL_FEATURES) < len(axes):
        for i in range(len(NUMERICAL_FEATURES), len(axes)):
            fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()

    # SAVE WITH ORIGINAL FILENAME (unchanged)
    fig.savefig(PLOTS_DIR / "numerical_distributions.png", dpi=200)

    # Close figure
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot and save annotated feature correlation heatmap (notebook-style)."""

    # Ensure output directory exists
    ensure_dir(PLOTS_DIR)

    # Compute correlation matrix
    corr = df.corr(numeric_only=True)

    # Upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create figure
    plt.figure(figsize=(14, 10))

    # Annotated heatmap (numbers inside cells)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,          # SHOW numeric values
        fmt=".2f",           # Two-decimal formatting
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
    )

    # Title
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)

    # Layout
    plt.tight_layout()

    # SAVE WITH ORIGINAL FILENAME (unchanged)
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)

    # Close figure
    plt.close()

    # Console summary (as in notebook)
    if TARGET_COL in corr.columns:
        print("Top correlations with target:")
        print(corr[TARGET_COL].sort_values(ascending=False)[1:6])

def save_tabular_eda(df: pd.DataFrame, output_dir) -> None:
    """
    Notes:
    Persist tabular EDA outputs (head, describe, missing values)
    so they are reproducible and available for reports and MLflow.
    """

    # Ensure directory exists
    ensure_dir(output_dir)

    # Save first 5 rows
    df.head().to_csv(output_dir / "data_head.csv", index=False)

    # Save statistical summary
    df.describe().round(2).to_csv(output_dir / "data_describe.csv")
    df.describe().round(2).to_markdown(output_dir / "data_describe.md")


    # Save missing value summary
    assert df.isna().sum().sum() == 0, "df_clean contains unexpected missing values"
    missing_summary = df.isna().sum().reset_index()
    missing_summary.columns = ["feature", "missing_count"]
    missing_summary.to_csv(output_dir / "missing_values.csv", index=False)

if __name__ == "__main__":
    # Apply seaborn styling similar to notebook (optional)
    sns.set_style("darkgrid")  # Styling

    # Load raw dataset
    df_raw = load_or_download()  # Acquire data

    # Clean dataset (same cleaning as training)
    df_clean = clean_dataset(df_raw)  # Clean + binarize target


    # Generate EDA plots
    plot_class_distribution(df_clean)  # Save class distribution
    plot_numerical_distributions(df_clean)  # Save histograms
    plot_correlation_heatmap(df_clean)  # Save correlation heatmap

    # Save tabular EDA outputs
    save_tabular_eda(df_clean, PLOTS_DIR.parent / "metrics")


    # Print output paths for convenience
    print("EDA plots saved to:", PLOTS_DIR)  # Confirmation
