# Notes:
# - Ensures preprocessing is reproducible and stable.
# - Confirms missing values are handled and target is binary.
# - Validates split_and_scale output integrity.

from src.heartml.data_ingest import load_or_download  # Data loader
from src.heartml.preprocess import clean_dataset, split_and_scale  # Preprocessing helpers
from src.heartml.config import TARGET_COL  # Target column


def test_clean_dataset_no_missing_values():
    """Cleaned dataset should not contain missing values."""

    df_raw = load_or_download()  # Load raw dataset
    df_clean = clean_dataset(df_raw)  # Clean dataset
    assert df_clean.isna().sum().sum() == 0  # No NaNs expected


def test_target_is_binary():
    """Target column should be 0/1 after cleaning."""

    df_raw = load_or_download()  # Load raw dataset
    df_clean = clean_dataset(df_raw)  # Clean dataset
    assert set(df_clean[TARGET_COL].unique()).issubset({0, 1})  # Must be binary


def test_split_and_scale_outputs_consistent_shapes():
    """split_and_scale should return consistent shapes and a scaler."""

    df_raw = load_or_download()  # Load raw dataset
    df_clean = clean_dataset(df_raw)  # Clean dataset

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)  # Split + scale

    assert len(X_train) > 0  # Non-empty train
    assert len(X_test) > 0  # Non-empty test
    assert len(X_train) == len(y_train)  # Matching lengths
    assert len(X_test) == len(y_test)  # Matching lengths
    assert scaler is not None  # Scaler should exist
