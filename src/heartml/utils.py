# src/heartml/utils.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists (mkdir -p behavior). Returns Path.
    Accepts str or Path for convenience.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path) -> str:
    """
    Compute SHA256 hash of a file. Accepts str or Path.
    Used for deterministic dataset "versioning" and traceability.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(a: Any, b: Any, *, indent: int = 2) -> Path:
    """
    Save JSON to disk.

    Supports both call styles (for backward compatibility):
      1) save_json(path, data)
      2) save_json(data, path)
    """
    # detect which is path
    if isinstance(a, (str, Path)):
        path, data = a, b
    elif isinstance(b, (str, Path)):
        path, data = b, a
    else:
        raise TypeError(
            "save_json expects one argument to be a path (str/Path) and the other to be JSON-serializable data"
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return p
