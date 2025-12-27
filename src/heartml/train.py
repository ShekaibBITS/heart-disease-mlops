"""heartml.train

Notes (what this script does)
- Implements Phase-2 (Feature Engineering & Model Development) from the notebook in VS Code script form.
- Workflow:
  1) Load raw data (download if needed)
  2) Clean data (missing handling + binary target)
  3) Train/test split + scale numerical features
  4) Train Logistic Regression baseline
  5) Train Random Forest baseline
  6) Tune Random Forest using GridSearchCV
  7) Evaluate all models (train + test) using: Accuracy, Precision, Recall, F1, ROC-AUC
  8) Save models, scaler, plots, and metrics artifacts to disk

Run (from project root):
    python src/heartml/train.py
"""

# Import json for writing structured metrics (optional; we use helper save_json)
import json  # Standard library JSON

# Import pandas for creating comparison tables
import pandas as pd  # DataFrame utilities

# Import numpy for numeric operations
import numpy as np  # Numerical utilities

# Import joblib for serializing trained models
import joblib  # Serialization (same as notebook)

# Import sklearn models and CV utilities (as per notebook)
from sklearn.linear_model import LogisticRegression  # Baseline model
from sklearn.ensemble import RandomForestClassifier  # Tree-based model
from sklearn.model_selection import (  # CV and tuning
    StratifiedKFold,  # Stratified folds for classification
    cross_val_score,  # Cross-validation scoring
    GridSearchCV,  # Hyperparameter search
)

# Import project modules for data acquisition and preprocessing
from .data_ingest import load_or_download  # Download/cache dataset
from .preprocess import clean_dataset, split_and_scale, save_processed_artifacts  # Cleaning + scaling

# Import evaluation helpers for metrics and plots
from .evaluate import (  # Evaluation utilities
    compute_metrics,  # Metric computation
    plot_confusion_matrix,  # Confusion matrix plot
    save_roc_plot,  # Combined ROC plot
    print_classification_report,  # Optional detailed report
)

# Import configuration (paths, hyperparams)
from .config import (  # Central constants
    RANDOM_STATE,  # Reproducibility seed
    LR_MAX_ITER,  # LR iterations
    LR_SOLVER,  # LR solver
    RF_N_ESTIMATORS,  # RF trees
    RF_MAX_DEPTH,  # RF depth
    RF_MIN_SAMPLES_SPLIT,  # RF split threshold
    RF_MIN_SAMPLES_LEAF,  # RF leaf threshold
    RF_N_JOBS,  # Parallelism
    CV_FOLDS,  # CV folds
    RF_PARAM_GRID,  # Grid-search space
    MODEL_DIR,  # Models folder
    LR_MODEL_PATH,  # LR model path
    RF_MODEL_PATH,  # RF model path
    RF_TUNED_MODEL_PATH,  # Tuned RF model path
    METRICS_DIR,  # Metrics folder
    MODEL_COMPARISON_PATH,  # Comparison CSV path
    METRICS_JSON_PATH,  # Metrics JSON path
    PLOTS_DIR,  # Plots folder
)  # Imports from config

# Import helpers for directory creation and JSON saving
from .utils import ensure_dir, save_json  # Utilities


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Train/test evaluation for a fitted model; returns a metrics dict."""

    # Predict labels on training set
    y_train_pred = model.predict(X_train)  # Predicted classes (train)

    # Predict class probabilities on training set (probability of class 1)
    y_train_proba = model.predict_proba(X_train)[:, 1]  # Probabilities (train)

    # Predict labels on test set
    y_test_pred = model.predict(X_test)  # Predicted classes (test)

    # Predict class probabilities on test set
    y_test_proba = model.predict_proba(X_test)[:, 1]  # Probabilities (test)

    # Compute train metrics
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)  # Train performance

    # Compute test metrics
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)  # Test performance

    # Package results in a single structure
    result = {  # Consolidated result dict
        "model": name,  # Model name
        "train": train_metrics,  # Train metrics
        "test": test_metrics,  # Test metrics
        "y_test_pred": y_test_pred,  # Store for confusion matrix
        "y_test_proba": y_test_proba,  # Store for ROC curves
    }

    # Return results
    return result  # Used for reporting and artifact creation


def main() -> None:
    """Main training entry point."""

    # Ensure output directories exist (models, plots, metrics)
    ensure_dir(MODEL_DIR)  # Ensure models directory
    ensure_dir(PLOTS_DIR)  # Ensure plots directory
    ensure_dir(METRICS_DIR)  # Ensure metrics directory

    # -----------------------------
    # Step 1: Data acquisition
    # -----------------------------

    # Load raw dataset (download from UCI if not present locally)
    df_raw = load_or_download()  # Data acquisition step

    # -----------------------------
    # Step 2: Data cleaning
    # -----------------------------

    # Clean dataset (impute missing values + binarize target)
    df_clean = clean_dataset(df_raw)  # Cleaned dataset

    # -----------------------------
    # Step 3: Split + scale
    # -----------------------------

    # Split into train/test and scale numerical features
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)  # Prepared features

    # Save cleaned data and scaler for reproducibility (Task 4 foundation)
    save_processed_artifacts(df_clean, scaler)  # Persist artifacts

    # -----------------------------
    # Step 4: Train Logistic Regression
    # -----------------------------

    # Initialize Logistic Regression model (matches notebook configuration)
    lr_model = LogisticRegression(  # Baseline model
        random_state=RANDOM_STATE,  # Reproducible initialization
        max_iter=LR_MAX_ITER,  # Ensure convergence
        solver=LR_SOLVER,  # Solver choice
    )

    # Fit model on training data
    lr_model.fit(X_train, y_train)  # Train LR

    # Cross-validation on training data (accuracy scoring, stratified folds)
    cv_lr = cross_val_score(  # Compute CV scores
        lr_model,  # Estimator
        X_train,  # Features
        y_train,  # Labels
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),  # Stratified CV
        scoring="accuracy",  # Metric used in notebook
    )

    # -----------------------------
    # Step 5: Train Random Forest
    # -----------------------------

    # Initialize Random Forest model (matches notebook configuration)
    rf_model = RandomForestClassifier(  # Tree-based classifier
        n_estimators=RF_N_ESTIMATORS,  # Number of trees
        max_depth=RF_MAX_DEPTH,  # Tree depth limit
        min_samples_split=RF_MIN_SAMPLES_SPLIT,  # Split threshold
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,  # Leaf threshold
        random_state=RANDOM_STATE,  # Reproducibility
        n_jobs=RF_N_JOBS,  # Parallelism
    )

    # Fit model on training data
    rf_model.fit(X_train, y_train)  # Train RF

    # Cross-validation for RF on training data
    cv_rf = cross_val_score(  # Compute CV scores
        rf_model,  # Estimator
        X_train,  # Features
        y_train,  # Labels
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),  # Stratified CV
        scoring="accuracy",  # Accuracy scoring
    )

    # -----------------------------
    # Step 6: Tune Random Forest (GridSearchCV)
    # -----------------------------

    # Define the grid search object
    grid_search = GridSearchCV(  # Hyperparameter tuning
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=RF_N_JOBS),  # Base estimator
        param_grid=RF_PARAM_GRID,  # Parameter grid (from config)
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),  # Stratified CV
        scoring="accuracy",  # Metric consistent with notebook
        n_jobs=RF_N_JOBS,  # Parallelize grid search
        verbose=0,  # Keep console output clean
    )

    # Fit the grid search on training data
    grid_search.fit(X_train, y_train)  # Run tuning

    # Extract the best estimator (tuned Random Forest)
    rf_tuned = grid_search.best_estimator_  # Best model from search

    # -----------------------------
    # Step 7: Evaluate all models
    # -----------------------------

    # Evaluate each fitted model (train + test)
    lr_result = evaluate_model("Logistic Regression", lr_model, X_train, y_train, X_test, y_test)  # LR metrics
    rf_result = evaluate_model("Random Forest", rf_model, X_train, y_train, X_test, y_test)  # RF metrics
    rft_result = evaluate_model("Tuned Random Forest", rf_tuned, X_train, y_train, X_test, y_test)  # Tuned RF metrics

    # Print quick console summary (useful during development)
    print("\nModel evaluation summary (Test ROC-AUC):")  # Header
    print(f"  LR  : {lr_result['test']['roc_auc']:.4f}")  # LR ROC-AUC
    print(f"  RF  : {rf_result['test']['roc_auc']:.4f}")  # RF ROC-AUC
    print(f"  RF* : {rft_result['test']['roc_auc']:.4f} (tuned)")  # Tuned RF ROC-AUC

    # Optional: detailed classification reports
    print_classification_report(y_test, lr_result["y_test_pred"], "Logistic Regression - Classification Report")  # LR report
    print_classification_report(y_test, rf_result["y_test_pred"], "Random Forest - Classification Report")  # RF report
    print_classification_report(y_test, rft_result["y_test_pred"], "Tuned Random Forest - Classification Report")  # Tuned RF report

    # -----------------------------
    # Step 8: Save plots (confusion matrices + ROC curves)
    # -----------------------------

    # Save confusion matrix plots for each model
    plot_confusion_matrix(  # LR confusion matrix
        y_test,  # True labels
        lr_result["y_test_pred"],  # Predicted labels
        "Confusion Matrix - Logistic Regression",  # Title
        PLOTS_DIR / "cm_logistic_regression.png",  # Output file
    )

    plot_confusion_matrix(  # RF confusion matrix
        y_test,  # True labels
        rf_result["y_test_pred"],  # Predicted labels
        "Confusion Matrix - Random Forest",  # Title
        PLOTS_DIR / "cm_random_forest.png",  # Output file
    )

    plot_confusion_matrix(  # Tuned RF confusion matrix
        y_test,  # True labels
        rft_result["y_test_pred"],  # Predicted labels
        "Confusion Matrix - Tuned Random Forest",  # Title
        PLOTS_DIR / "cm_random_forest_tuned.png",  # Output file
    )

    # Save a combined ROC curve plot across models
    save_roc_plot(  # ROC curves
        roc_items={  # Map model names to probabilities
            "Logistic Regression": lr_result["y_test_proba"],  # LR proba
            "Random Forest": rf_result["y_test_proba"],  # RF proba
            "Tuned Random Forest": rft_result["y_test_proba"],  # Tuned RF proba
        },
        y_true=y_test,  # True labels
        out_path=PLOTS_DIR / "roc_curves.png",  # Output file
    )

    # -----------------------------
    # Step 9: Save trained models
    # -----------------------------

    # Persist models using joblib (same approach as notebook)
    joblib.dump(lr_model, LR_MODEL_PATH)  # Save Logistic Regression model
    joblib.dump(rf_model, RF_MODEL_PATH)  # Save Random Forest model
    joblib.dump(rf_tuned, RF_TUNED_MODEL_PATH)  # Save Tuned Random Forest model

    # -----------------------------
    # Step 10: Save metrics artifacts
    # -----------------------------

    # Create a model comparison table (similar to notebook model comparison)
    comparison_df = pd.DataFrame(  # Build summary table
        [
            {
                "model": lr_result["model"],  # Model name
                "cv_accuracy_mean": float(cv_lr.mean()),  # CV mean
                "cv_accuracy_std": float(cv_lr.std()),  # CV std
                **{f"train_{k}": v for k, v in lr_result["train"].items()},  # Train metrics
                **{f"test_{k}": v for k, v in lr_result["test"].items()},  # Test metrics
            },
            {
                "model": rf_result["model"],  # Model name
                "cv_accuracy_mean": float(cv_rf.mean()),  # CV mean
                "cv_accuracy_std": float(cv_rf.std()),  # CV std
                **{f"train_{k}": v for k, v in rf_result["train"].items()},  # Train metrics
                **{f"test_{k}": v for k, v in rf_result["test"].items()},  # Test metrics
            },
            {
                "model": rft_result["model"],  # Model name
                "cv_accuracy_mean": float(grid_search.best_score_),  # Best CV score from tuning
                "cv_accuracy_std": float(0.0),  # Not directly available from GridSearchCV
                **{f"train_{k}": v for k, v in rft_result["train"].items()},  # Train metrics
                **{f"test_{k}": v for k, v in rft_result["test"].items()},  # Test metrics
            },
        ]
    )

    # Save comparison table to CSV for reporting
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)  # Persist as artifact

    # Save a richer JSON metrics object
    metrics_payload = {  # Full metrics dump
        "logistic_regression": {  # LR section
            "cv_scores": [float(x) for x in cv_lr],  # CV list
            "cv_mean": float(cv_lr.mean()),  # CV mean
            "cv_std": float(cv_lr.std()),  # CV std
            "train": lr_result["train"],  # Train metrics
            "test": lr_result["test"],  # Test metrics
        },
        "random_forest": {  # RF section
            "cv_scores": [float(x) for x in cv_rf],  # CV list
            "cv_mean": float(cv_rf.mean()),  # CV mean
            "cv_std": float(cv_rf.std()),  # CV std
            "train": rf_result["train"],  # Train metrics
            "test": rf_result["test"],  # Test metrics
        },
        "tuned_random_forest": {  # Tuned RF section
            "best_params": grid_search.best_params_,  # Best hyperparameters
            "best_cv_score": float(grid_search.best_score_),  # Best CV score
            "train": rft_result["train"],  # Train metrics
            "test": rft_result["test"],  # Test metrics
        },
    }

    # Save metrics JSON using helper
    save_json(metrics_payload, METRICS_JSON_PATH)  # Persist metrics payload

    # Print output locations for convenience
    print("\nArtifacts written:")  # Header
    print(f"  Models:  {MODEL_DIR}")  # Model dir
    print(f"  Plots:   {PLOTS_DIR}")  # Plot dir
    print(f"  Metrics: {METRICS_DIR}")  # Metrics dir


if __name__ == "__main__":
    main()  # Execute training when run as a script
