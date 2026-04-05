"""Task-formalization API routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import DesignTask, TaskCompileResponse, ValidationStatus
from libs.tasking.compiler import compile_design_task
from libs.tasking.validation import validate_design_task

router = APIRouter(prefix="/tasking", tags=["tasking"])


class CompileTaskRequest(BaseModel):
    """Request payload for DesignTask compilation."""

    model_config = ConfigDict(extra="forbid")

    design_spec: DesignSpec
    task_type_hint: Literal["sizing", "topology_sizing", "calibration"] | None = None


class ValidateTaskRequest(BaseModel):
    """Request payload for explicit DesignTask validation."""

    model_config = ConfigDict(extra="forbid")

    design_spec: DesignSpec
    design_task: DesignTask


@router.post("/compile", response_model=TaskCompileResponse)
def compile_task(request: CompileTaskRequest) -> TaskCompileResponse:
    """Compile a DesignSpec into a formal DesignTask."""

    return compile_design_task(request.design_spec, task_type_hint=request.task_type_hint)


@router.post("/validate", response_model=ValidationStatus)
def validate_task(request: ValidateTaskRequest) -> ValidationStatus:
    """Validate a supplied DesignTask against deterministic second-layer rules."""

    return validate_design_task(request.design_task, source_spec=request.design_spec)
