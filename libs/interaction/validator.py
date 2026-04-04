"""Validation helpers for incoming design specs."""

from __future__ import annotations

from libs.schema.design_spec import (
    CIRCUIT_FAMILIES,
    CONSTRAINT_METRICS,
    DESIGN_VARIABLES_BY_FAMILY,
    DesignSpec,
    MISSING_INFO_ORDER,
    OBJECTIVE_METRICS,
    TESTBENCH_ORDER,
    ValidationIssue,
    ValidationReport,
)
from libs.tasking.default_testbench_planner import plan_testbenches


def _issue(code: str, path: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(code=code, path=path, message=message, severity=severity)


def _expected_missing_information(spec: DesignSpec) -> list[str]:
    expected: list[str] = []
    if spec.circuit_family == "unknown":
        expected.append("circuit_family")
    if spec.process_node is None:
        expected.append("process_node")
    if spec.environment.load_cap_f is None and spec.circuit_family in {
        "two_stage_ota",
        "folded_cascode_ota",
        "telescopic_ota",
        "comparator",
        "unknown",
    }:
        expected.append("load_cap_f")
    return sorted(set(expected), key=lambda item: (MISSING_INFO_ORDER.index(item), item))


def validate_design_spec(spec: DesignSpec) -> ValidationReport:
    """Return a structured validation report for a DesignSpec."""

    issues: list[ValidationIssue] = []

    if spec.circuit_family not in CIRCUIT_FAMILIES:
        issues.append(_issue("invalid_circuit_family", "circuit_family", "unsupported circuit family"))

    overlap = set(spec.objectives.maximize) & set(spec.objectives.minimize)
    if overlap:
        issues.append(
            _issue(
                "objective_overlap",
                "objectives",
                f"metrics cannot be both maximized and minimized: {sorted(overlap)}",
            )
        )

    invalid_objectives = [
        metric
        for metric in [*spec.objectives.maximize, *spec.objectives.minimize]
        if metric not in OBJECTIVE_METRICS
    ]
    if invalid_objectives:
        issues.append(_issue("invalid_objective_metric", "objectives", f"unsupported metrics: {invalid_objectives}"))

    for metric, metric_range in spec.hard_constraints.items():
        if metric not in CONSTRAINT_METRICS:
            issues.append(_issue("invalid_constraint_metric", f"hard_constraints.{metric}", "unsupported metric"))
            continue
        if metric_range.min is not None and metric_range.max is not None and metric_range.min > metric_range.max:
            issues.append(_issue("constraint_range_conflict", f"hard_constraints.{metric}", "min cannot exceed max"))
        if metric not in {"output_swing_v", "input_common_mode_v"}:
            for field_name in ("min", "max", "target"):
                value = getattr(metric_range, field_name)
                if value is not None and value < 0:
                    issues.append(
                        _issue(
                            "negative_constraint_value",
                            f"hard_constraints.{metric}.{field_name}",
                            "non-voltage constraints must be non-negative",
                        )
                    )

    expected_plan = plan_testbenches(spec)
    if spec.testbench_plan != expected_plan:
        issues.append(
            _issue(
                "testbench_plan_mismatch",
                "testbench_plan",
                f"expected {expected_plan}, received {spec.testbench_plan}",
            )
        )

    expected_variables = DESIGN_VARIABLES_BY_FAMILY[spec.circuit_family]
    if spec.design_variables != expected_variables:
        issues.append(
            _issue(
                "design_variables_mismatch",
                "design_variables",
                f"expected {expected_variables}, received {spec.design_variables}",
            )
        )

    expected_missing = _expected_missing_information(spec)
    missing_gap = sorted(set(expected_missing) - set(spec.missing_information))
    if missing_gap:
        issues.append(
            _issue(
                "missing_information_incomplete",
                "missing_information",
                f"expected missing information markers for {missing_gap}",
            )
        )

    for testbench in spec.testbench_plan:
        if testbench not in TESTBENCH_ORDER:
            issues.append(_issue("invalid_testbench_step", "testbench_plan", f"unsupported step {testbench}"))

    if not 0.0 <= spec.compile_confidence <= 1.0:
        issues.append(_issue("invalid_compile_confidence", "compile_confidence", "confidence must be within [0, 1]"))

    return ValidationReport(
        valid=not issues,
        issues=issues,
        error_types=sorted({issue.code for issue in issues}),
    )
