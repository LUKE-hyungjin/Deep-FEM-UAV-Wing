from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class WingParams(BaseModel):
    span_m: float = Field(..., gt=0)
    chord_m: float = Field(..., gt=0)
    sweep_deg: float = Field(..., ge=0)
    thickness_ratio: float = Field(..., gt=0)


class ParamsFile(BaseModel):
    case_id: str
    span_m: float
    chord_m: float
    sweep_deg: float
    thickness_ratio: float
    created_at: str
    pipeline_version: str

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


ReportStatus = Literal["success", "failed"]


class StepReport(BaseModel):
    status: ReportStatus
    failure_reason: str | None = None
    elapsed_ms: int
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


