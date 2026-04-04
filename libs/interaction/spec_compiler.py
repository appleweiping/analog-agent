"""Compile raw user intent into an interaction-layer DesignSpec."""

from __future__ import annotations

from uuid import uuid4

from libs.interaction.normalizer import normalize_parsed_spec
from libs.interaction.parser import ParsedSpecification, parse_specification
from libs.interaction.repair_loop import run_repair_loop
from libs.interaction.validator import validate_design_spec
from libs.schema.design_spec import (
    CompilationReport,
    ClarificationRequest,
    CompileResponse,
    DesignSpec,
    DESIGN_VARIABLES_BY_FAMILY,
    MetricRange,
    Objectives,
    ValidationIssue,
)
from libs.tasking.default_testbench_planner import plan_testbenches

QUESTION_TEMPLATES = {
    "circuit_family": "請指定電路家族，例如 two_stage_ota、folded_cascode_ota、ldo 或 bandgap。",
    "process_node": "請提供工藝節點，例如 65nm 或 180nm。",
    "load_cap_f": "請提供負載電容，例如 2pF。",
    "input_common_mode_v": "請提供輸入共模範圍或目標。",
    "output_swing_v": "請提供輸出擺幅要求。",
    "temperature_range": "請提供溫度條件，例如 0C、27C、85C 或完整溫度範圍。",
}


def _issue(code: str, path: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(code=code, path=path, message=message, severity=severity)


def _infer_missing_information(parsed: ParsedSpecification) -> list[str]:
    missing = set(parsed.missing_information)
    family = parsed.circuit_family or "unknown"

    if family == "unknown":
        missing.add("circuit_family")
    if not parsed.process_node:
        missing.add("process_node")
    if parsed.environment["load_cap_f"] is None and family in {
        "two_stage_ota",
        "folded_cascode_ota",
        "telescopic_ota",
        "comparator",
        "unknown",
    }:
        missing.add("load_cap_f")
    if parsed.environment["temperature_c"] == [] and "across temperature" in parsed.raw_text.lower():
        missing.add("temperature_range")
    return sorted(missing)


def _compute_compile_confidence(parsed: ParsedSpecification, issue_count: int) -> float:
    confidence = 0.95
    confidence -= 0.08 * len(_infer_missing_information(parsed))
    confidence -= 0.12 * len(parsed.ambiguities)
    confidence -= 0.2 * issue_count
    return round(min(max(confidence, 0.0), 1.0), 4)


def _build_field_sources(parsed: ParsedSpecification) -> dict[str, str]:
    return {
        "task_id": "defaulted",
        "circuit_family": parsed.field_sources.get("circuit_family", "system_inferred"),
        "process_node": parsed.field_sources.get("process_node", "defaulted"),
        "supply_voltage_v": parsed.field_sources.get("supply_voltage_v", "defaulted"),
        "objectives": "system_inferred" if (parsed.objectives_maximize or parsed.objectives_minimize) else "defaulted",
        "hard_constraints": "user_provided" if parsed.hard_constraints else "defaulted",
        "environment": parsed.field_sources.get("environment", "defaulted"),
        "testbench_plan": "system_inferred",
        "design_variables": "system_inferred",
        "missing_information": "system_inferred",
        "notes": "system_inferred" if parsed.notes else "defaulted",
        "compile_confidence": "system_inferred",
    }


def build_design_spec(parsed: ParsedSpecification) -> DesignSpec:
    """Build a schema-compliant DesignSpec from normalized parser output."""

    hard_constraints = {
        metric: MetricRange.model_validate(constraint)
        for metric, constraint in parsed.hard_constraints.items()
    }
    objectives = Objectives(
        maximize=parsed.objectives_maximize,
        minimize=parsed.objectives_minimize,
    )

    draft = DesignSpec(
        task_id=uuid4().hex,
        circuit_family=parsed.circuit_family or "unknown",
        process_node=parsed.process_node,
        supply_voltage_v=parsed.supply_voltage_v,
        objectives=objectives,
        hard_constraints=hard_constraints,
        environment=parsed.environment,
        testbench_plan=["op"],
        design_variables=DESIGN_VARIABLES_BY_FAMILY[parsed.circuit_family or "unknown"],
        missing_information=_infer_missing_information(parsed),
        notes=parsed.notes,
        compile_confidence=0.0,
    )
    draft.testbench_plan = plan_testbenches(draft)
    draft.compile_confidence = _compute_compile_confidence(parsed, 0)
    return DesignSpec.model_validate(draft.model_dump())


def _build_clarification_request(parsed: ParsedSpecification, spec: DesignSpec | None) -> ClarificationRequest:
    missing = spec.missing_information if spec else _infer_missing_information(parsed)
    questions = [QUESTION_TEMPLATES[item] for item in missing if item in QUESTION_TEMPLATES]
    for ambiguity in parsed.ambiguities:
        questions.append(f"請澄清：{ambiguity}")
    return ClarificationRequest(
        missing_information=missing,
        ambiguities=parsed.ambiguities,
        suggested_questions=questions,
    )


def compile_spec(text: str, mode: str = "strict", max_repair_rounds: int = 3) -> CompileResponse:
    """Compile natural-language design requirements into a validated DesignSpec."""

    parsed = normalize_parsed_spec(parse_specification(text))
    parser_issues = [_issue("parser_error", "parser", error) for error in parsed.parser_errors]

    spec = build_design_spec(parsed)
    spec, repair_attempts = run_repair_loop(spec, max_rounds=max_repair_rounds)
    validation_report = validate_design_spec(spec)
    combined_issues = parser_issues + validation_report.issues
    issue_count = len(combined_issues)
    spec.compile_confidence = _compute_compile_confidence(parsed, issue_count)

    if parser_issues:
        status = "invalid"
        design_spec = None
        clarification_request = None
    elif mode == "interactive" and (spec.missing_information or parsed.ambiguities):
        status = "clarification_required"
        design_spec = None
        clarification_request = _build_clarification_request(parsed, spec)
    elif validation_report.valid:
        status = "compiled"
        design_spec = spec
        clarification_request = None
    else:
        status = "invalid"
        design_spec = None
        clarification_request = None

    report = CompilationReport(
        status=status,
        mode="interactive" if mode == "interactive" else "strict",
        field_sources=_build_field_sources(parsed),
        missing_fields=spec.missing_information,
        ambiguities=parsed.ambiguities,
        suggested_clarifications=_build_clarification_request(parsed, spec).suggested_questions
        if status != "compiled"
        else [],
        validation_issues=combined_issues,
        parser_notes=parsed.notes,
        repair_attempts=repair_attempts,
    )
    return CompileResponse(
        mode=report.mode,
        status=status,
        design_spec=design_spec,
        report=report,
        clarification_request=clarification_request,
    )
