"""
tests/test_preprocess.py

Notes:
- Ensures preprocessing is reproducible and stable.
- Confirms missing values are handled and target is binary.
- Validates split_and_scale output integrity.
- CI-safe: does NOT depend on external network calls.
"""

import numpy as np
import pandas as pd

from src.heartml.config import TARGET_COL
from src.heartml.preprocess import clean_dataset, split_and_scale


def _sample_raw_df() -> pd.DataFrame:
    """
    Deterministic dataset large enough for stable train/test splitting.
    Using >= 10 rows avoids common stratify/test_size edge cases.
    Includes missing values to validate cleaning logic.
    """
    return pd.DataFrame(
        {
            "age":      [63, 67, np.nan, 37, 41, 56, 52, 58, 44, 60, 49, 54],
            "sex":      [1,  1,  0,      1,  0,  1,  0,  1,  0,  1,  0,  1],
            "trestbps": [145, np.nan,130, 120, 140, 132, 128, np.nan,118, 150, 125, 135],
            "chol":     [233, 286, 250,  np.nan,204, 240, np.nan, 215, 190, 270, 205, 260],
            # Balanced target distribution helps stratified splits
            TARGET_COL: [1,   0,   1,    0,     1,   0,   1,     0,   1,   0,   1,   0],
        }
    )


def test_clean_dataset_no_missing_values():
    """Cleaned dataset should not contain missing values."""
    df_raw = _sample_raw_df()
    df_clean = clean_dataset(df_raw)

    # Robust NA check (works across mixed dtypes)
    assert not df_clean.isna().to_numpy().any()


def test_target_is_binary():
    """Target column should be 0/1 after cleaning."""
    df_raw = _sample_raw_df()
    df_clean = clean_dataset(df_raw)

    # Robust conversion in case target is float/object after cleaning
    unique_vals = set(pd.Series(df_clean[TARGET_COL]).dropna().astype(int).unique())
    assert unique_vals.issubset({0, 1})


def test_split_and_scale_outputs_consistent_shapes():
    """split_and_scale should return consistent shapes and (optionally) a scaler."""
    df_raw = _sample_raw_df()
    df_clean = clean_dataset(df_raw)

    out = split_and_scale(df_clean)

    # Some implementations return 4 items, others return 5 (including scaler).
    assert isinstance(out, tuple)
    assert len(out) in (4, 5)

    if len(out) == 5:
        X_train, X_test, y_train, y_test, scaler = out
        assert scaler is not None
        assert hasattr(scaler, "transform")
    else:
        X_train, X_test, y_train, y_test = out

    # Support numpy arrays or pandas objects
    n_train = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
    n_test = X_test.shape[0] if hasattr(X_test, "shape") else len(X_test)

    assert n_train > 0
    assert n_test > 0
    assert n_train == len(y_train)
    assert n_test == len(y_test)
