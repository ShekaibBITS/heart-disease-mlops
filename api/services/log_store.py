from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

LOG_DIR = Path(os.getenv("ADMIN_CI_LOG_DIR", "logs/admin_ci"))

def log_path(run_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{run_id}.log"

def append(run_id: str, text: str) -> None:
    with log_path(run_id).open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def read(run_id: str, tail: Optional[int] = None) -> str:
    p = log_path(run_id)
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if tail is not None and tail > 0:
        lines = lines[-tail:]
    return "\n".join(lines)
