# Notes:
# - Validates data ingestion returns a DataFrame with expected structure.
# - Protects CI from ingestion regressions (URL changes, parsing changes).

import pandas as pd  # DataFrame type checks

from src.heartml.data_ingest import load_or_download  # Data loader
from src.heartml.config import TARGET_COL  # Target column name


def test_load_or_download_returns_dataframe():
    """Data ingestion should return a non-empty DataFrame with target column."""

    df = load_or_download()  # Load dataset
    assert isinstance(df, pd.DataFrame)  # Must be a DataFrame
    assert len(df) > 0  # Must not be empty
    assert TARGET_COL in df.columns  # Target must exist
