from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal

from api.security import require_admin
from api.services.pipeline_executor import CICmdExecutor
from api.services.run_registry import RunRegistry
from api.services import log_store

router = APIRouter(prefix="/admin", tags=["admin"])

_registry = RunRegistry()
_executor = CICmdExecutor(_registry)

OnlyStage = Literal["ingest", "eda", "train"]

class TriggerCIPipelineRequest(BaseModel):
    # Mirrors scripts/run_pipeline.py flags exactly
    only: Optional[OnlyStage] = Field(default=None, description="Equivalent to --only")
    skip_eda: bool = Field(default=False, description="Equivalent to --skip-eda")
    skip_train: bool = Field(default=False, description="Equivalent to --skip-train")

    async_mode: bool = Field(default=True, description="Run in background thread")
    run_id: Optional[str] = Field(default=None, description="Optional run_id (else generated)")

@router.post("/ci/pipeline/runs", dependencies=[Depends(require_admin)])
def trigger_ci_pipeline(req: TriggerCIPipelineRequest):
    # Guard: user should not request conflicting flags
    if req.only is not None and (req.skip_eda or req.skip_train):
        raise HTTPException(status_code=400, detail="Do not combine 'only' with skip flags.")

    run_id = _executor.trigger(
        only=req.only,
        skip_eda=req.skip_eda,
        skip_train=req.skip_train,
        async_mode=req.async_mode,
        run_id=req.run_id,
    )
    info = _registry.get(run_id)
    return {"run_id": run_id, "status": info.status if info else "unknown"}

@router.get("/ci/pipeline/runs/{run_id}", dependencies=[Depends(require_admin)])
def get_run_status(run_id: str):
    info = _registry.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="run_id not found")
    return info.model_dump()

@router.get("/ci/pipeline/runs/{run_id}/logs", dependencies=[Depends(require_admin)])
def get_run_logs(run_id: str, tail: int = 300):
    content = log_store.read(run_id, tail=tail)
    if not content:
        raise HTTPException(status_code=404, detail="No logs found for this run_id")
    return {"run_id": run_id, "tail": tail, "logs": content}
