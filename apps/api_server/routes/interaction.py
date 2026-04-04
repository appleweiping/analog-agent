"""Interaction-layer API routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from libs.interaction.spec_compiler import compile_spec
from libs.interaction.validator import validate_design_spec
from libs.schema.design_spec import CompileResponse, DesignSpec, ValidationReport

router = APIRouter(prefix="/interaction", tags=["interaction"])


class CompileRequest(BaseModel):
    """Request payload for compiling a natural-language design spec."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    mode: Literal["strict", "interactive"] = "strict"
    max_repair_rounds: int = Field(default=3, ge=0, le=10)


class ValidateRequest(BaseModel):
    """Request payload for explicit DesignSpec validation."""

    model_config = ConfigDict(extra="forbid")

    design_spec: DesignSpec


@router.post("/compile", response_model=CompileResponse)
def compile_interaction(request: CompileRequest) -> CompileResponse:
    """Compile natural-language requirements into a DesignSpec or clarification request."""

    return compile_spec(
        request.text,
        mode=request.mode,
        max_repair_rounds=request.max_repair_rounds,
    )


@router.post("/validate", response_model=ValidationReport)
def validate_interaction_spec(request: ValidateRequest) -> ValidationReport:
    """Validate a supplied DesignSpec using deterministic interaction-layer rules."""

    return validate_design_spec(request.design_spec)
