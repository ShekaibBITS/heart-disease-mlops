from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import time

@dataclass
class RunInfo:
    run_id: str
    status: str
    started_at: float
    ended_at: Optional[float] = None
    error: Optional[str] = None
    meta: Optional[dict] = None

    def model_dump(self):
        return asdict(self)

class RunRegistry:
    def __init__(self) -> None:
        self._runs: Dict[str, RunInfo] = {}

    def create(self, run_id: str, meta: dict) -> RunInfo:
        info = RunInfo(run_id=run_id, status="running", started_at=time.time(), meta=meta)
        self._runs[run_id] = info
        return info

    def succeed(self, run_id: str) -> None:
        if run_id in self._runs:
            self._runs[run_id].status = "success"
            self._runs[run_id].ended_at = time.time()

    def fail(self, run_id: str, error: str) -> None:
        if run_id in self._runs:
            self._runs[run_id].status = "failed"
            self._runs[run_id].error = error
            self._runs[run_id].ended_at = time.time()

    def get(self, run_id: str) -> Optional[RunInfo]:
        return self._runs.get(run_id)
