# src/heartml/train.py
"""heartml.train

Run:
    python -m src.heartml.train
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient  # registry

from .data_ingest import load_or_download
from .evaluate import compute_metrics, plot_confusion_matrix, print_classification_report, save_roc_plot
from .preprocess import clean_dataset, save_processed_artifacts, split_and_scale
from .config import (
    CV_FOLDS,
    LR_MAX_ITER,
    LR_SOLVER,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_COMPARISON_PATH,
    MODEL_DIR,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    RF_TUNED_MODEL_PATH,
    BEST_MODEL_PATH,
    METRICS_DIR,
    METRICS_JSON_PATH,
    PLOTS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
    RF_N_JOBS,
    RF_PARAM_GRID,
    SCALER_PATH,
    # registry toggles
    MLFLOW_REGISTER_MODEL,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_STAGE,
    # data ingest config for lineage (optional)
    DATASET_URL,  # ensure this exists in your config (same used in data_ingest)
)
from .utils import ensure_dir, save_json, sha256_file


# -----------------------------
# Helpers (friendly names + dataset lineage)
# -----------------------------
def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def _dataset_lineage_tags(df_raw: pd.DataFrame) -> Dict[str, Any]:
    dataset_sha = sha256_file(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else "missing_raw_data_file"
    return {
        "dataset_url": DATASET_URL,
        "dataset_path": str(RAW_DATA_PATH),
        "dataset_sha256": dataset_sha,
        "dataset_rows": int(df_raw.shape[0]),
        "dataset_cols": int(df_raw.shape[1]),
        "feature_names_csv": ",".join([str(c) for c in df_raw.columns]),
    }


def _safe_mlflow_set_tags(tags: Dict[str, Any]) -> None:
    for k, v in tags.items():
        try:
            mlflow.set_tag(k, str(v))
        except Exception:
            pass


# -----------------------------
# MLflow metric logging (safe)
# -----------------------------
def _log_metric_safe_batch(metrics: Dict[str, float], step: int = 0) -> None:
    """
    Logs metrics with unique timestamps per key to avoid backend-store
    duplicate-key collisions (metric_pk) in Postgres.

    Behavior: metric values unchanged; only insertion metadata differs.
    """
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    i = 0
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step, timestamp=base_ts + i)
        i += 1


# -----------------------------
# Metrics
# -----------------------------
def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    return {
        "model": name,
        "train": train_metrics,
        "test": test_metrics,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
    }


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# -----------------------------
# Registry (optional)
# -----------------------------
def maybe_register_model(run_id: str, artifact_path: str = "model") -> Optional[Tuple[str, str]]:
    """
    Optional: Register a run artifact as a version in MLflow Model Registry,
    and (optionally) move it to a stage.

    Returns (model_name, version) if registered, else None.

    Non-breaking:
    - Controlled by env MLFLOW_REGISTER_MODEL=true/false
    - Keeps your existing run-id based serving unchanged
    """
    if not MLFLOW_REGISTER_MODEL:
        return None

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)

    if MLFLOW_MODEL_STAGE:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=str(mv.version),
            stage=MLFLOW_MODEL_STAGE,
            archive_existing_versions=False,
        )

    return (MLFLOW_MODEL_NAME, str(mv.version))


# -----------------------------
# Child-run logger (nested)
# -----------------------------
def log_model_run(
    run_name: str,
    model_label: str,
    model_key: str,
    model_obj,
    train_metrics: dict,
    test_metrics: dict,
    cv_mean: float,
    cv_std: float,
    dataset_tags: Dict[str, Any],
    selection_metric: str,
    selection_threshold: float,
    input_example: Optional[pd.DataFrame] = None,
    signature=None,
    extra_params: Optional[dict] = None,
) -> str:
    """
    Logs a single model run. Returns run_id (needed for your API deployment).
    IMPORTANT: model artifact_path must be 'model' because API loads from:
      runs:/<RUN_ID>/model
    """
    display_model_name = f"HeartDisease::{model_key}"
    display_run_name = f"{display_model_name}::{_utc_now_compact()}"

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Friendly UI tags
        mlflow.set_tag("display_model_name", display_model_name)
        mlflow.set_tag("display_run_name", display_run_name)

        # Stable identifiers
        mlflow.set_tag("model_key", model_key)
        mlflow.set_tag("model_label", model_label)
        mlflow.set_tag("model_flavor", "sklearn")

        # Dataset lineage tags
        _safe_mlflow_set_tags(
            {
                "dataset_sha256": dataset_tags.get("dataset_sha256"),
                "dataset_path": dataset_tags.get("dataset_path"),
                "dataset_url": dataset_tags.get("dataset_url"),
                "feature_names_csv": dataset_tags.get("feature_names_csv", ""),
            }
        )
        mlflow.log_param("dataset_rows", int(dataset_tags.get("dataset_rows", 0)))
        mlflow.log_param("dataset_cols", int(dataset_tags.get("dataset_cols", 0)))

        # Selection policy tags
        mlflow.set_tag("selection_metric", selection_metric)
        mlflow.set_tag("selection_threshold", str(selection_threshold))

        # Core params
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)

        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

        # Metrics (safe batch insert; unique timestamps)
        _log_metric_safe_batch(
            {
                # CV summary
                "cv_accuracy_mean": float(cv_mean),
                "cv_accuracy_std": float(cv_std),

                # Train metrics
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_precision": float(train_metrics["precision"]),
                "train_recall": float(train_metrics["recall"]),
                "train_f1": float(train_metrics["f1"]),
                "train_roc_auc": float(train_metrics.get("roc_auc", float("nan"))),

                # Test metrics
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_f1": float(test_metrics["f1"]),
                "test_roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
            },
            step=0,
        )

        # Artifacts (plots/metrics)
        if PLOTS_DIR.exists():
            mlflow.log_artifacts(str(PLOTS_DIR), artifact_path="plots")
        if METRICS_DIR.exists():
            mlflow.log_artifacts(str(METRICS_DIR), artifact_path="metrics")

        # Data lineage artifacts
        if RAW_DATA_PATH.exists():
            mlflow.log_artifact(str(RAW_DATA_PATH), artifact_path="data/raw")
        if PROCESSED_DATA_PATH.exists():
            mlflow.log_artifact(str(PROCESSED_DATA_PATH), artifact_path="data/processed")
        if SCALER_PATH.exists():
            mlflow.log_artifact(str(SCALER_PATH), artifact_path="models")

        # Log model under artifact_path="model" (required by your API)
        mlflow.sklearn.log_model(
            sk_model=model_obj,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        return run.info.run_id


def main() -> None:
    ensure_dir(MODEL_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(METRICS_DIR)

    configure_mlflow()

    # Parent run: pipeline-level lineage + decision
    with mlflow.start_run(run_name="train_pipeline"):
        mlflow.set_tag("stage", "training_pipeline")

        # Load + lineage
        df_raw = load_or_download()
        dataset_tags = _dataset_lineage_tags(df_raw)

        # Put dataset lineage onto parent run
        _safe_mlflow_set_tags(
            {
                "dataset_sha256": dataset_tags["dataset_sha256"],
                "dataset_path": dataset_tags["dataset_path"],
                "dataset_url": dataset_tags["dataset_url"],
                "feature_names_csv": dataset_tags["feature_names_csv"],
            }
        )
        mlflow.log_param("dataset_rows", dataset_tags["dataset_rows"])
        mlflow.log_param("dataset_cols", dataset_tags["dataset_cols"])

        # Clean + split
        df_clean = clean_dataset(df_raw)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)

        input_example = X_train.head(1).astype(float)

        # Persist processed artifacts (kept)
        save_processed_artifacts(df_clean, scaler)

        # -----------------------------
        # Train candidates
        # -----------------------------
        lr_model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
        )
        lr_model.fit(X_train, y_train)
        lr_signature = infer_signature(input_example, lr_model.predict_proba(input_example)[:, 1])

        cv_lr = cross_val_score(
            lr_model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
        )

        rf_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS,
        )
        rf_model.fit(X_train, y_train)
        rf_signature = infer_signature(input_example, rf_model.predict_proba(input_example)[:, 1])

        cv_rf = cross_val_score(
            rf_model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
        )

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=RF_N_JOBS),
            param_grid=RF_PARAM_GRID,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="accuracy",
            n_jobs=RF_N_JOBS,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        rf_tuned = grid_search.best_estimator_
        rft_signature = infer_signature(input_example, rf_tuned.predict_proba(input_example)[:, 1])

        # Evaluate
        lr_result = evaluate_model("Logistic Regression", lr_model, X_train, y_train, X_test, y_test)
        rf_result = evaluate_model("Random Forest", rf_model, X_train, y_train, X_test, y_test)
        rft_result = evaluate_model("Tuned Random Forest", rf_tuned, X_train, y_train, X_test, y_test)

        print("\nModel evaluation summary (Test ROC-AUC):")
        print(f"  LR  : {lr_result['test'].get('roc_auc', float('nan')):.4f}")
        print(f"  RF  : {rf_result['test'].get('roc_auc', float('nan')):.4f}")
        print(f"  RF* : {rft_result['test'].get('roc_auc', float('nan')):.4f} (tuned)")

        print_classification_report(y_test, lr_result["y_test_pred"], "Logistic Regression - Classification Report")
        print_classification_report(y_test, rf_result["y_test_pred"], "Random Forest - Classification Report")
        print_classification_report(y_test, rft_result["y_test_pred"], "Tuned Random Forest - Classification Report")

        # -----------------------------
        # Decision policy
        # -----------------------------
        selection_metric = "test_roc_auc"
        selection_threshold = 0.5  # your inference uses >= 0.5
        mlflow.set_tag("selection_metric", selection_metric)
        mlflow.set_tag("selection_threshold", str(selection_threshold))

        def score(v):
            try:
                return float(v)
            except Exception:
                return float("-inf")

        candidates = [
            ("logistic_regression", lr_model, score(lr_result["test"].get("roc_auc", float("nan")))),
            ("random_forest", rf_model, score(rf_result["test"].get("roc_auc", float("nan")))),
            ("random_forest_tuned", rf_tuned, score(rft_result["test"].get("roc_auc", float("nan")))),
        ]
        best_key, best_model, best_score = max(candidates, key=lambda x: x[2])
        print(f"\nBest model selected: {best_key} (test ROC-AUC = {best_score:.4f})")

        # -----------------------------
        # Plots
        # -----------------------------
        plot_confusion_matrix(
            y_test,
            lr_result["y_test_pred"],
            "Confusion Matrix - Logistic Regression",
            PLOTS_DIR / "cm_logistic_regression.png",
        )
        plot_confusion_matrix(
            y_test,
            rf_result["y_test_pred"],
            "Confusion Matrix - Random Forest",
            PLOTS_DIR / "cm_random_forest.png",
        )
        plot_confusion_matrix(
            y_test,
            rft_result["y_test_pred"],
            "Confusion Matrix - Tuned Random Forest",
            PLOTS_DIR / "cm_random_forest_tuned.png",
        )

        save_roc_plot(
            roc_items={
                "Logistic Regression": lr_result["y_test_proba"],
                "Random Forest": rf_result["y_test_proba"],
                "Tuned Random Forest": rft_result["y_test_proba"],
            },
            y_true=y_test,
            out_path=PLOTS_DIR / "roc_curves.png",
        )

        # Save local models (kept)
        joblib.dump(lr_model, LR_MODEL_PATH)
        joblib.dump(rf_model, RF_MODEL_PATH)
        joblib.dump(rf_tuned, RF_TUNED_MODEL_PATH)
        joblib.dump(best_model, BEST_MODEL_PATH)

        # Metrics files (kept)
        comparison_df = pd.DataFrame(
            [
                {
                    "model": lr_result["model"],
                    "cv_accuracy_mean": float(cv_lr.mean()),
                    "cv_accuracy_std": float(cv_lr.std()),
                    **{f"train_{k}": v for k, v in lr_result["train"].items()},
                    **{f"test_{k}": v for k, v in lr_result["test"].items()},
                },
                {
                    "model": rf_result["model"],
                    "cv_accuracy_mean": float(cv_rf.mean()),
                    "cv_accuracy_std": float(cv_rf.std()),
                    **{f"train_{k}": v for k, v in rf_result["train"].items()},
                    **{f"test_{k}": v for k, v in rf_result["test"].items()},
                },
                {
                    "model": rft_result["model"],
                    "cv_accuracy_mean": float(grid_search.best_score_),
                    "cv_accuracy_std": float(0.0),
                    **{f"train_{k}": v for k, v in rft_result["train"].items()},
                    **{f"test_{k}": v for k, v in rft_result["test"].items()},
                },
            ]
        )
        comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

        metrics_payload = {
            "logistic_regression": {
                "cv_scores": [float(x) for x in cv_lr],
                "cv_mean": float(cv_lr.mean()),
                "cv_std": float(cv_lr.std()),
                "train": lr_result["train"],
                "test": lr_result["test"],
            },
            "random_forest": {
                "cv_scores": [float(x) for x in cv_rf],
                "cv_mean": float(cv_rf.mean()),
                "cv_std": float(cv_rf.std()),
                "train": rf_result["train"],
                "test": rf_result["test"],
            },
            "tuned_random_forest": {
                "best_params": grid_search.best_params_,
                "best_cv_score": float(grid_search.best_score_),
                "train": rft_result["train"],
                "test": rft_result["test"],
            },
            "decision": {
                "selection_metric": selection_metric,
                "selection_threshold": selection_threshold,
                "best_model_key": best_key,
                "best_model_score": float(best_score),
            },
            "dataset": {
                "dataset_sha256": dataset_tags["dataset_sha256"],
                "dataset_path": dataset_tags["dataset_path"],
                "dataset_url": dataset_tags["dataset_url"],
            },
        }
        save_json(metrics_payload, METRICS_JSON_PATH)

        # -----------------------------
        # Log each model run (nested)
        # -----------------------------
        lr_run_id = log_model_run(
            run_name="LR::baseline",
            model_label="Logistic Regression",
            model_key="logistic_regression",
            model_obj=lr_model,
            train_metrics=lr_result["train"],
            test_metrics=lr_result["test"],
            cv_mean=float(cv_lr.mean()),
            cv_std=float(cv_lr.std()),
            dataset_tags=dataset_tags,
            selection_metric=selection_metric,
            selection_threshold=selection_threshold,
            input_example=input_example,
            signature=lr_signature,
            extra_params={"model_type": "logistic_regression", "max_iter": LR_MAX_ITER, "solver": LR_SOLVER},
        )
        lr_reg = maybe_register_model(lr_run_id, artifact_path="model")

        rf_run_id = log_model_run(
            run_name="RF::baseline",
            model_label="Random Forest",
            model_key="random_forest",
            model_obj=rf_model,
            train_metrics=rf_result["train"],
            test_metrics=rf_result["test"],
            cv_mean=float(cv_rf.mean()),
            cv_std=float(cv_rf.std()),
            dataset_tags=dataset_tags,
            selection_metric=selection_metric,
            selection_threshold=selection_threshold,
            input_example=input_example,
            signature=rf_signature,
            extra_params={
                "model_type": "random_forest",
                "n_estimators": RF_N_ESTIMATORS,
                "max_depth": RF_MAX_DEPTH,
                "min_samples_split": RF_MIN_SAMPLES_SPLIT,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "n_jobs": RF_N_JOBS,
            },
        )
        rf_reg = maybe_register_model(rf_run_id, artifact_path="model")

        rft_run_id = log_model_run(
            run_name="RF::tuned",
            model_label="Tuned Random Forest",
            model_key="random_forest_tuned",
            model_obj=rf_tuned,
            train_metrics=rft_result["train"],
            test_metrics=rft_result["test"],
            cv_mean=float(grid_search.best_score_),
            cv_std=0.0,
            dataset_tags=dataset_tags,
            selection_metric=selection_metric,
            selection_threshold=selection_threshold,
            input_example=input_example,
            signature=rft_signature,
            extra_params={
                "model_type": "random_forest_gridsearch",
                "param_grid": str(RF_PARAM_GRID),
                **{f"best_{k}": v for k, v in grid_search.best_params_.items()},
            },
        )
        rft_reg = maybe_register_model(rft_run_id, artifact_path="model")

        # -----------------------------
        # Parent decision artifact: which run won
        # -----------------------------
        run_id_map = {
            "logistic_regression": lr_run_id,
            "random_forest": rf_run_id,
            "random_forest_tuned": rft_run_id,
        }
        best_run_id = run_id_map.get(best_key)

        mlflow.set_tag("best_model_key", best_key)
        mlflow.set_tag("best_model_run_id", str(best_run_id) if best_run_id else "unknown")
        mlflow.log_metric("best_model_score", float(best_score))

        decision_payload = {
            "selection_metric": selection_metric,
            "selection_threshold": selection_threshold,
            "best_model_key": best_key,
            "best_model_score": float(best_score),
            "best_model_run_id": best_run_id,
            "candidate_run_ids": run_id_map,
            "dataset_sha256": dataset_tags["dataset_sha256"],
            "dataset_path": dataset_tags["dataset_path"],
            "dataset_url": dataset_tags["dataset_url"],
        }
        mlflow.log_dict(decision_payload, "decision.json")

        # Print run ids for run-id based API usage
        print("\nMLflow Run IDs (use in docker-compose for API if using runs:/):")
        print("  LR   :", lr_run_id)
        print("  RF   :", rf_run_id)
        print("  RF*  :", rft_run_id)
        print("  BEST :", best_key, "->", best_run_id)

        # Print registry info if enabled
        if MLFLOW_REGISTER_MODEL:
            print("\nMLflow Registry (use in API if using models:/):")
            print(f"  Model name : {MLFLOW_MODEL_NAME}")
            print(f"  Stage      : {MLFLOW_MODEL_STAGE}")
            for label, reg in [("LR", lr_reg), ("RF", rf_reg), ("RF*", rft_reg)]:
                if reg:
                    print(f"  {label} registered: name={reg[0]}, version={reg[1]}")

        print("\nArtifacts written:")
        print(f"  Models:  {MODEL_DIR}")
        print(f"  Plots:   {PLOTS_DIR}")
        print(f"  Metrics: {METRICS_DIR}")


if __name__ == "__main__":
    main()
