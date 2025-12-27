"""heartml.utils

Notes (what this module does)
- Provides small, reusable helpers used across scripts (directory creation, JSON saving).
- Keeps filesystem logic consistent and reduces duplication.
"""

# Import json to write metrics/artifacts in a structured format
import json  # Standard library JSON utilities

# Import Path for filesystem path handling
from pathlib import Path  # OS-independent path utility


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """

    # Create the directory (and parents) if missing; do nothing if it exists
    path.mkdir(parents=True, exist_ok=True)  # Robust directory creation


def save_json(obj: dict, path: Path) -> None:
    """Save a Python dictionary as pretty-printed JSON.

    Args:
        obj: Dictionary to save.
        path: File path where JSON will be written.
    """

    # Ensure the parent directory exists before writing the file
    ensure_dir(path.parent)  # Prevents 'No such file or directory' errors

    # Open the output path for writing (UTF-8 ensures cross-platform readability)
    with path.open("w", encoding="utf-8") as f:  # Context manager safely closes the file
        json.dump(obj, f, indent=2, sort_keys=True)  # Pretty print for easy reporting
