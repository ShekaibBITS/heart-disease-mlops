"""heartml.evaluate

Notes (what this module does)
- Computes standard classification metrics used in the notebook:
  * Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Generates evaluation artifacts for reporting:
  * Confusion matrix plots
  * ROC curve plots
- Designed to be called from train.py after model fitting.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from .config import PLOTS_DIR
from .utils import ensure_dir


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Compute core binary classification metrics."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC requires probabilities and both classes present
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str, out_path) -> None:
    """Create and save a confusion matrix plot."""
    ensure_dir(PLOTS_DIR)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true, y_proba, label: str, ax) -> float:
    """Add a ROC curve to a provided matplotlib axis and return AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.4f})")
    return float(auc_value)


def save_roc_plot(roc_items: Dict[str, np.ndarray], y_true, out_path) -> None:
    """Create a combined ROC plot for multiple models and save it."""
    ensure_dir(PLOTS_DIR)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_proba in roc_items.items():
        # Skip invalid probabilities
        if y_proba is None:
            continue
        plot_roc_curve(y_true, y_proba, model_name, ax)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_classification_report(y_true, y_pred, title: str) -> None:
    """Print a classification report."""
    print(f"\n{title}")
    print("-" * len(title))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
