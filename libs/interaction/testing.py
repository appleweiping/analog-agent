"""Acceptance-style testing helpers for the interaction layer."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from libs.interaction.spec_compiler import compile_spec
from libs.schema.design_spec import CompileResponse


class AcceptanceCase(BaseModel):
    """Serializable acceptance-case definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    prompt: str
    mode: str = "strict"


class AcceptanceCaseResult(BaseModel):
    """Serializable per-case acceptance result."""

    model_config = ConfigDict(extra="forbid")

    case: AcceptanceCase
    raw_input: str
    raw_output: CompileResponse
    repaired_output: dict | None = None
    validation_report: dict = Field(default_factory=dict)
    result: str
    error_types: list[str] = Field(default_factory=list)


class AcceptanceSummary(BaseModel):
    """Aggregated acceptance metrics."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed_cases: int
    schema_validity_rate: float
    field_completeness_rate: float
    semantic_consistency_rate: float
    actionability_rate: float
    repair_success_rate: float
    error_type_distribution: dict[str, int] = Field(default_factory=dict)


def evaluate_case(case: AcceptanceCase) -> AcceptanceCaseResult:
    """Compile one acceptance case and serialize its outcome."""

    output = compile_spec(case.prompt, mode=case.mode)
    validation_report = {
        "valid": output.status == "compiled",
        "issues": [issue.model_dump() for issue in output.report.validation_issues],
    }
    return AcceptanceCaseResult(
        case=case,
        raw_input=case.prompt,
        raw_output=output,
        repaired_output=output.design_spec.model_dump() if output.design_spec else None,
        validation_report=validation_report,
        result="pass" if output.status in {"compiled", "clarification_required"} else "fail",
        error_types=[issue.code for issue in output.report.validation_issues],
    )


def build_acceptance_summary(results: list[AcceptanceCaseResult]) -> AcceptanceSummary:
    """Build aggregated acceptance metrics from case results."""

    total = len(results)
    passed = sum(1 for result in results if result.result == "pass")
    validity = sum(1 for result in results if result.raw_output.status == "compiled")
    completeness = sum(1 for result in results if result.raw_output.design_spec is not None)
    semantic = sum(1 for result in results if not result.error_types)
    actionability = sum(
        1
        for result in results
        if result.raw_output.design_spec is not None
        and "op" in result.raw_output.design_spec.testbench_plan
    )
    repairs = sum(1 for result in results if result.raw_output.report.repair_attempts > 0)
    repaired_success = sum(
        1
        for result in results
        if result.raw_output.report.repair_attempts > 0 and result.raw_output.status == "compiled"
    )
    counter = Counter(error for result in results for error in result.error_types)

    return AcceptanceSummary(
        total_cases=total,
        passed_cases=passed,
        schema_validity_rate=validity / total if total else 0.0,
        field_completeness_rate=completeness / total if total else 0.0,
        semantic_consistency_rate=semantic / total if total else 0.0,
        actionability_rate=actionability / total if total else 0.0,
        repair_success_rate=(repaired_success / repairs) if repairs else 0.0,
        error_type_distribution=dict(counter),
    )
