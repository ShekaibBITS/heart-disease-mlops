"""heartml.data_ingest

Notes (what this script does)
- Downloads the Heart Disease (processed Cleveland) dataset from UCI, as done in the notebook.
- Saves the raw dataset to data/raw for reproducibility and offline runs.
- Provides a load function used by training scripts.

Run (from project root):
    python src/heartml/data_ingest.py
"""

# Import StringIO to treat downloaded text as a file-like object for pandas
from io import StringIO  # Allows pd.read_csv on in-memory strings

# Import requests to download the dataset over HTTP
import requests  # HTTP client used in the notebook

# Import pandas for CSV parsing into a DataFrame
import pandas as pd  # Data manipulation library

# Import project configuration (URL, feature names, output paths)
from .config import DATASET_URL, FEATURE_NAMES, RAW_DATA_PATH  # Centralized constants

# Import helper to ensure folders exist
from .utils import ensure_dir  # Directory creation helper


def download_dataset() -> pd.DataFrame:
    """Download the dataset from UCI and return as a pandas DataFrame."""

    # Make an HTTP GET request to download the dataset (timeout avoids hanging indefinitely)
    response = requests.get(DATASET_URL, timeout=30)  # Fetch dataset text

    # Raise an exception for HTTP errors (e.g., 404/500) so failures are visible early
    response.raise_for_status()  # Fail fast on network/HTTP issues

    # Parse the CSV-like content using the same column names as the notebook
    df = pd.read_csv(  # Create DataFrame
        StringIO(response.text),  # Provide the content as a file-like buffer
        header=None,  # The file has no header row
        names=FEATURE_NAMES,  # Assign documented column names
        na_values="?",  # Treat '?' as missing values
    )

    # Return the parsed DataFrame to the caller
    return df  # DataFrame with 14 columns


def save_raw_dataset(df: pd.DataFrame) -> None:
    """Save the raw dataset to disk for reproducibility."""

    # Ensure the parent folder (data/raw) exists before writing
    ensure_dir(RAW_DATA_PATH.parent)  # Create directory tree if needed

    # Save the DataFrame as CSV (index=False prevents adding an extra column)
    df.to_csv(RAW_DATA_PATH, index=False)  # Persist raw data for later runs


def load_or_download() -> pd.DataFrame:
    """Load raw dataset from disk if present; otherwise download and save it."""

    # If the raw dataset already exists locally, load it from disk
    if RAW_DATA_PATH.exists():  # Check local cache
        return pd.read_csv(RAW_DATA_PATH)  # Return cached raw dataset

    # Otherwise, download from UCI
    df = download_dataset()  # Fetch from the internet

    # Save the raw dataset to disk for reproducibility
    save_raw_dataset(df)  # Cache locally

    # Return the downloaded DataFrame
    return df  # DataFrame ready for preprocessing


if __name__ == "__main__":
    # Download (or load) the dataset
    df_raw = load_or_download()  # Acquire dataset

    # Print basic confirmation details (useful for logs and debugging)
    print("Dataset ready.")  # High-level confirmation
    print(f"Path: {RAW_DATA_PATH}")  # Where it is stored
    print(f"Shape: {df_raw.shape}")  # Expected ~ (303, 14)
    print(df_raw.head())  # Preview first 5 rows
