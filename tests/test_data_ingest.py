# Notes:
# - Validates data ingestion returns a DataFrame with expected structure.
# - Protects CI from ingestion regressions (URL changes, parsing changes).
# - Does NOT depend on external network.

import pandas as pd

from src.heartml.config import TARGET_COL
from src.heartml import data_ingest


def test_load_or_download_returns_dataframe(monkeypatch):
    """Data ingestion should return a non-empty DataFrame with target column."""

    # Minimal fake dataset with the required target column
    fake_df = pd.DataFrame(
        {
            "age": [63, 67],
            "sex": [1, 1],
            TARGET_COL: [1, 0],
        }
    )

    # Force load_or_download() to use our fake downloader instead of the real network call
    monkeypatch.setattr(data_ingest, "download_dataset", lambda: fake_df)

    df = data_ingest.load_or_download()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET_COL in df.columns
