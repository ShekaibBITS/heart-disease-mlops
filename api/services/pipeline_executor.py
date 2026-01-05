from __future__ import annotations

import sys
import uuid
import time
import traceback
import subprocess
from threading import Thread
from pathlib import Path
from typing import Optional

from api.services.run_registry import RunRegistry
from api.services import log_store
from api.services.admin_metrics import (
    CI_PIPELINE_RUNS_TOTAL,
    CI_PIPELINE_FAILURES_TOTAL,
    CI_PIPELINE_DURATION_SECONDS,
    CI_PIPELINE_IN_PROGRESS,
)

class CICmdExecutor:
    def __init__(self, registry: RunRegistry) -> None:
        self.registry = registry

    def trigger(
        self,
        *,
        only: Optional[str],
        skip_eda: bool,
        skip_train: bool,
        async_mode: bool,
        run_id: Optional[str] = None,
    ) -> str:
        rid = run_id or str(uuid.uuid4())

        meta = {"only": only, "skip_eda": skip_eda, "skip_train": skip_train, "async_mode": async_mode}
        self.registry.create(rid, meta=meta)
        CI_PIPELINE_RUNS_TOTAL.labels(mode="admin_api").inc()

        if async_mode:
            Thread(target=self._run, args=(rid, only, skip_eda, skip_train), daemon=True).start()
        else:
            self._run(rid, only, skip_eda, skip_train)

        return rid

    def _run(self, run_id: str, only: Optional[str], skip_eda: bool, skip_train: bool) -> None:
        CI_PIPELINE_IN_PROGRESS.set(1)
        t0 = time.time()

        try:
            # Run from project root (parent of api/ and scripts/)
            project_root = Path(__file__).resolve().parents[2]

            cmd = [sys.executable, "scripts/run_pipeline.py"]

            # Map API request to EXACT existing CLI flags
            if only:
                cmd += ["--only", only]
            else:
                if skip_eda:
                    cmd += ["--skip-eda"]
                if skip_train:
                    cmd += ["--skip-train"]

            log_store.append(run_id, f"CI PIPELINE START run_id={run_id}")
            log_store.append(run_id, f"CWD: {project_root}")
            log_store.append(run_id, f"CMD: {' '.join(cmd)}")

            # Capture all stdout/stderr for later retrieval
            with CI_PIPELINE_DURATION_SECONDS.time():
                p = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    text=True,
                    capture_output=True,
                )

            if p.stdout:
                log_store.append(run_id, "\n[STDOUT]\n" + p.stdout)
            if p.stderr:
                log_store.append(run_id, "\n[STDERR]\n" + p.stderr)

            if p.returncode != 0:
                CI_PIPELINE_FAILURES_TOTAL.inc()
                raise RuntimeError(f"run_pipeline.py failed with return code {p.returncode}")

            self.registry.succeed(run_id)
            log_store.append(run_id, f"CI PIPELINE SUCCESS run_id={run_id} elapsed_sec={time.time()-t0:.3f}")

        except Exception as e:
            CI_PIPELINE_FAILURES_TOTAL.inc()
            tb = traceback.format_exc()
            self.registry.fail(run_id, f"{type(e).__name__}: {str(e)}")

            log_store.append(run_id, f"CI PIPELINE FAILED run_id={run_id} error={type(e).__name__}: {str(e)}")
            log_store.append(run_id, "TRACEBACK:\n" + tb)

        finally:
            log_store.append(run_id, f"CI PIPELINE END run_id={run_id}")
            CI_PIPELINE_IN_PROGRESS.set(0)
