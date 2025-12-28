"""heartml.evaluate

Notes (what this module does)
- Computes standard classification metrics used in the notebook:
  * Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Generates evaluation artifacts for reporting:
  * Confusion matrix plots
  * ROC curve plots
- Designed to be called from train.py after model fitting.
"""

# Import typing for clear function signatures
from typing import Dict #,Tuple # Type hints for maintainability

# Import numpy for numeric operations
import numpy as np  # Array math

# Import pandas for tabular metric reporting
# import pandas as pd  # DataFrames for summaries

# Import matplotlib for plotting (saved to files)
import matplotlib.pyplot as plt  # Plotting library

# Import seaborn for nicer heatmaps (confusion matrix)
import seaborn as sns  # Statistical plotting

# Import sklearn metrics functions (same family as notebook)
from sklearn.metrics import (  # Metric utilities
    accuracy_score,  # Accuracy
    precision_score,  # Precision
    recall_score,  # Recall
    f1_score,  # F1-score
    roc_auc_score,  # ROC-AUC
    roc_curve,  # ROC curve points
    confusion_matrix,  # Confusion matrix values
    classification_report,  # Text report (optional)
)  # Metrics used across models

# Import project paths for saving plots
from .config import PLOTS_DIR  # Where plots are written

# Import helper to ensure plot directory exists
from .utils import ensure_dir  # Directory creation


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Compute core binary classification metrics.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_proba: Predicted probability for class 1.

    Returns:
        Dictionary of metrics.
    """

    # Compute each metric explicitly for clarity and reporting
    metrics = {  # Collect metrics in a single dict
        "accuracy": float(accuracy_score(y_true, y_pred)),  # Overall correctness
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # Positive predictive value
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # Sensitivity
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),  # Harmonic mean of precision/recall
        "roc_auc": float(roc_auc_score(y_true, y_proba)),  # Threshold-independent ranking metric
    }

    # Return computed metrics
    return metrics  # For logging/comparison


def plot_confusion_matrix(y_true, y_pred, title: str, out_path) -> None:
    """Create and save a confusion matrix plot."""

    # Ensure the plots directory exists
    ensure_dir(PLOTS_DIR)  # Create artifacts/plots if needed

    # Compute confusion matrix values
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]]

    # Create a new figure for this plot
    plt.figure(figsize=(6, 5))  # Consistent sizing for reports

    # Plot the matrix as a heatmap
    sns.heatmap(
        cm,  # Data to plot
        annot=True,  # Show values in cells
        fmt="d",  # Integer formatting
        cmap="Blues",  # Simple readable palette
        cbar=False,  # Hide color bar for compactness
        xticklabels=["No", "Yes"],  # Predicted labels
        yticklabels=["No", "Yes"],  # True labels
    )

    # Add plot labels and title
    plt.xlabel("Predicted")  # X-axis label
    plt.ylabel("Actual")  # Y-axis label
    plt.title(title)  # Plot title

    # Tighten layout so labels fit
    plt.tight_layout()  # Avoid clipping

    # Save plot to disk
    plt.savefig(out_path, dpi=200)  # Persist artifact

    # Close figure to free memory (important in repeated runs)
    plt.close()  # Prevent figure accumulation


def plot_roc_curve(y_true, y_proba, label: str, ax) -> float:
    """Add a ROC curve to a provided matplotlib axis and return AUC."""

    # Compute ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_proba)  # False positive rate, true positive rate

    # Compute AUC value
    auc_value = roc_auc_score(y_true, y_proba)  # Area under ROC curve

    # Plot ROC line
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.4f})")  # Add labeled curve

    # Return the AUC for convenience
    return float(auc_value)  # Useful for logs


def save_roc_plot(roc_items: Dict[str, np.ndarray], y_true, out_path) -> None:
    """Create a combined ROC plot for multiple models and save it."""

    # Ensure output directory exists
    ensure_dir(PLOTS_DIR)  # Ensure artifacts/plots exists

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))  # Standard report size

    # Plot each model's ROC curve
    for model_name, y_proba in roc_items.items():  # Iterate models
        plot_roc_curve(y_true, y_proba, model_name, ax)  # Add curve to axis

    # Plot a diagonal baseline for reference (random classifier)
    ax.plot([0, 1], [0, 1], linestyle="--")  # Baseline

    # Add axis labels and title
    ax.set_xlabel("False Positive Rate")  # X-axis label
    ax.set_ylabel("True Positive Rate")  # Y-axis label
    ax.set_title("ROC Curves")  # Chart title

    # Add legend
    ax.legend(loc="lower right")  # Keep legend readable

    # Tight layout to avoid clipping
    fig.tight_layout()  # Improve spacing

    # Save figure to disk
    fig.savefig(out_path, dpi=200)  # Persist artifact

    # Close figure
    plt.close(fig)  # Cleanup


def print_classification_report(y_true, y_pred, title: str) -> None:
    """Print a classification report (useful for console logs)."""

    # Print section title
    print(f"\n{title}")  # Header
    print("-" * len(title))  # Underline

    # Print sklearn's formatted classification report
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))  # Detailed per-class metrics
