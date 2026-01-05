"""heartml.preprocess

Notes (what this module does)
- Cleans the raw dataset as per notebook intent:
  * Convert '?' to NaN (already handled at read-time)
  * Impute missing values (mode for 'ca' and 'thal', median for others)
  * Binarize the target (0 = no disease, 1 = disease)
- Splits the dataset into train/test sets with stratification.
- Scales numerical features using StandardScaler (fit on train, transform train/test).
- Saves the cleaned dataset to data/processed and the scaler to models/.

Run:
    python src/heartml/preprocess.py
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .config import (
    TARGET_COL,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_PATH,
    SCALER_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)
from .utils import ensure_dir


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset and return a cleaned DataFrame."""
    df_clean = df.copy()

    # Convert all columns to numeric (coerce invalid to NaN)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Impute missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if col in ["ca", "thal"]:
                # mode can be empty if column all NaN; guard it
                mode_vals = df_clean[col].mode(dropna=True)
                if len(mode_vals) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_vals.iloc[0])
                else:
                    df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Target must exist
    if TARGET_COL not in df_clean.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found in dataframe.")

    # Binarize target: 0 stays 0; >0 becomes 1
    df_clean[TARGET_COL] = (df_clean[TARGET_COL] > 0).astype(int)

    return df_clean


def split_and_scale(
    df_clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split cleaned data into train/test and scale numerical features."""
    if TARGET_COL not in df_clean.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found in dataframe.")

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Scale only numerical columns that exist
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    if num_cols:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_artifacts(df_clean: pd.DataFrame, scaler: StandardScaler) -> None:
    """Save cleaned dataset and scaler to disk."""
    ensure_dir(PROCESSED_DATA_PATH.parent)
    ensure_dir(SCALER_PATH.parent)

    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)


def _try_log_preprocess_to_mlflow() -> None:
    """
    Optional: log processed dataset + scaler to MLflow artifacts (-> MinIO).
    Non-breaking: does nothing if MLflow not configured.
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
            mlflow.set_experiment("heartml-preprocess")
            with mlflow.start_run():
                if PROCESSED_DATA_PATH.exists():
                    mlflow.log_artifact(str(PROCESSED_DATA_PATH), artifact_path="data/processed")
                if SCALER_PATH.exists():
                    mlflow.log_artifact(str(SCALER_PATH), artifact_path="models")
        else:
            if PROCESSED_DATA_PATH.exists():
                mlflow.log_artifact(str(PROCESSED_DATA_PATH), artifact_path="data/processed")
            if SCALER_PATH.exists():
                mlflow.log_artifact(str(SCALER_PATH), artifact_path="models")
    except Exception:
        return


if __name__ == "__main__":
    from .data_ingest import load_or_download

    df_raw = load_or_download()
    df_cleaned = clean_dataset(df_raw)
    Xtr, Xte, ytr, yte, fitted_scaler = split_and_scale(df_cleaned)

    save_processed_artifacts(df_cleaned, fitted_scaler)
    _try_log_preprocess_to_mlflow()

    print("Preprocessing completed.")
    print(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Train shape: {Xtr.shape}, Test shape: {Xte.shape}")
