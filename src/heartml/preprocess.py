"""heartml.preprocess

Notes (what this module does)
- Cleans the raw dataset as per notebook intent:
  * Convert '?' to NaN (already handled at read-time)
  * Impute missing values (mode for 'ca' and 'thal', median for others)
  * Binarize the target (0 = no disease, 1 = disease)
- Splits the dataset into train/test sets with stratification.
- Scales numerical features using StandardScaler (fit on train, transform train/test).
- Saves the cleaned dataset to data/processed and the scaler to models/.

This module is designed to be imported by train.py, but it can also be run directly:
    python src/heartml/preprocess.py
"""

# Import typing for explicit return types
from typing import Tuple  # Improves readability and IDE support

# Import pandas for DataFrame operations
import pandas as pd  # Data manipulation

# Import numpy for numeric utilities
# import numpy as np  # Numerical computing

# Import scikit-learn utilities for splitting and scaling
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Import joblib for persisting sklearn objects
import joblib  # Model/artifact serialization

# Import project configuration constants
from .config import (
    TARGET_COL,  # Name of the target column
    NUMERICAL_FEATURES,  # Numerical features to scale
    PROCESSED_DATA_PATH,  # Where to save cleaned data
    SCALER_PATH,  # Where to save the scaler
    TEST_SIZE,  # Test split fraction
    RANDOM_STATE,  # Reproducible seed
)  # Central config

# Import helpers for filesystem hygiene
from .utils import ensure_dir  # Directory creation helper


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset and return a cleaned DataFrame.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        Cleaned dataset DataFrame.
    """

    # Make a defensive copy to avoid mutating caller's DataFrame
    df_clean = df.copy()  # Preserve original raw data

    # Ensure all columns are numeric where possible (non-numeric becomes NaN)
    for col in df_clean.columns:  # Iterate over every column
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")  # Safe conversion

    # Handle missing values column-by-column (as per notebook approach)
    for col in df_clean.columns:  # Loop through all columns
        if df_clean[col].isnull().any():  # Only act if column contains NaNs
            if col in ["ca", "thal"]:  # These are categorical-like in the dataset
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])  # Impute with mode
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())  # Impute with median

    # Convert target to binary: 0 stays 0; 1-4 become 1
    df_clean[TARGET_COL] = (df_clean[TARGET_COL] > 0).astype(int)  # Binary label

    # Return the cleaned DataFrame
    return df_clean  # Ready for EDA/modeling


def split_and_scale(
    df_clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split cleaned data into train/test and scale numerical features.

    Args:
        df_clean: Cleaned dataset.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, fitted scaler
    """

    # Separate features (X) from target (y)
    X = df_clean.drop(columns=[TARGET_COL])  # Feature matrix
    y = df_clean[TARGET_COL]  # Target vector

    # Perform a stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # Features
        y,  # Labels
        test_size=TEST_SIZE,  # Hold-out fraction
        random_state=RANDOM_STATE,  # Seed for reproducibility
        stratify=y,  # Maintain class balance across splits
    )

    # Initialize the StandardScaler (fit on training data only)
    scaler = StandardScaler()  # Scaler instance

    # Copy X to avoid SettingWithCopy warnings and keep original data intact
    X_train_scaled = X_train.copy()  # Train features copy
    X_test_scaled = X_test.copy()  # Test features copy

    # Fit scaler on training numerical features and transform
    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )  # Fit+transform on train

    # Transform test numerical features using the scaler fit on training data
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(
        X_test[NUMERICAL_FEATURES]
    )  # Transform on test

    # Return scaled splits and the fitted scaler
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler  # Training-ready outputs


def save_processed_artifacts(df_clean: pd.DataFrame, scaler: StandardScaler) -> None:
    """Save cleaned dataset and scaler to disk."""

    # Ensure output directories exist
    ensure_dir(PROCESSED_DATA_PATH.parent)  # Ensure data/processed exists
    ensure_dir(SCALER_PATH.parent)  # Ensure models/ exists

    # Save cleaned dataset as CSV for downstream reproducibility
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)  # Persist cleaned data

    # Save scaler so inference uses identical preprocessing
    joblib.dump(scaler, SCALER_PATH)  # Persist scaler


if __name__ == "__main__":
    # Import loader locally to avoid circular imports at module import time
    from .data_ingest import load_or_download  # Lazy import for CLI execution

    # Acquire raw dataset (download or load cached)
    df_raw = load_or_download()  # Data acquisition

    # Clean the dataset
    df_cleaned = clean_dataset(df_raw)  # Missing handling + target binarization

    # Split and scale features
    Xtr, Xte, ytr, yte, fitted_scaler = split_and_scale(df_cleaned)  # Prepare modeling inputs

    # Save processed artifacts for reproducibility
    save_processed_artifacts(df_cleaned, fitted_scaler)  # Save cleaned data and scaler

    # Print confirmations for logs
    print("Preprocessing completed.")  # High-level confirmation
    print(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")  # Path confirmation
    print(f"Scaler saved to: {SCALER_PATH}")  # Path confirmation
    print(f"Train shape: {Xtr.shape}, Test shape: {Xte.shape}")  # Split confirmation
