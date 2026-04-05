"""Acceptance and adapter-style test helpers for the task formalization layer."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import DesignTask, TaskCompileResponse
from libs.tasking.compiler import compile_design_task


class TaskAcceptanceCase(BaseModel):
    """Serializable task-formalization acceptance case."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    design_spec: DesignSpec
    task_type_hint: str | None = None


class TaskAcceptanceCaseResult(BaseModel):
    """Serializable per-case task-formalization result."""

    model_config = ConfigDict(extra="forbid")

    case: TaskAcceptanceCase
    raw_output: TaskCompileResponse
    schema_valid: bool
    problem_complete: bool
    semantic_consistent: bool
    solver_ready: bool
    result: str
    error_types: list[str] = Field(default_factory=list)
    warning_types: list[str] = Field(default_factory=list)


class TaskAcceptanceSummary(BaseModel):
    """Aggregated second-layer acceptance metrics."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed_cases: int
    schema_validity_rate: float
    problem_completeness_rate: float
    semantic_consistency_rate: float
    solver_readiness_rate: float
    unresolved_dependency_recall: float
    determinism_rate: float
    error_type_distribution: dict[str, int] = Field(default_factory=dict)
    warning_type_distribution: dict[str, int] = Field(default_factory=dict)


def fake_planner_consume(task: DesignTask) -> dict[str, int | str]:
    """Simulate a downstream planner consuming the compiled task."""

    if not task.design_space.variables:
        raise ValueError("planner cannot consume an empty design space")
    return {
        "variable_dimension": len(task.design_space.variables),
        "objective_mode": task.objective.objective_mode,
        "task_type": task.task_type,
    }


def fake_simulator_adapter(task: DesignTask) -> list[dict[str, str | int]]:
    """Simulate translation from EvaluationPlan to simulator calls."""

    if not task.evaluation_plan.analyses:
        raise ValueError("simulator adapter requires analyses")
    return [
        {
            "analysis_type": analysis.analysis_type,
            "order": analysis.order,
            "estimated_cost": analysis.estimated_cost,
        }
        for analysis in task.evaluation_plan.analyses
    ]


def fake_world_model_adapter(task: DesignTask) -> dict[str, object]:
    """Simulate world-model feature extraction from DesignTask."""

    if not task.topology.topology_mode:
        raise ValueError("world model adapter requires topology metadata")
    return {
        "circuit_family": task.circuit_family,
        "topology_mode": task.topology.topology_mode,
        "feature_keys": [variable.name for variable in task.design_space.variables],
        "reporting_metrics": task.objective.reporting_metrics,
    }


def evaluate_case(case: TaskAcceptanceCase) -> TaskAcceptanceCaseResult:
    """Compile one DesignSpec acceptance case and score the result."""

    output = compile_design_task(case.design_spec, task_type_hint=case.task_type_hint)
    schema_valid = output.design_task is not None
    problem_complete = bool(
        output.design_task
        and output.design_task.design_space.variables
        and output.design_task.evaluation_plan.metric_extractors
        and output.design_task.task_graph.nodes
    )
    semantic_consistent = not output.report.validation_errors
    solver_ready = bool(output.design_task and output.design_task.validation_status.is_valid)
    result = "pass" if output.status in {"compiled", "compiled_with_warnings"} else "fail"
    return TaskAcceptanceCaseResult(
        case=case,
        raw_output=output,
        schema_valid=schema_valid,
        problem_complete=problem_complete,
        semantic_consistent=semantic_consistent,
        solver_ready=solver_ready,
        result=result,
        error_types=[issue.code for issue in output.report.validation_errors],
        warning_types=[issue.code for issue in output.report.validation_warnings],
    )


def build_acceptance_summary(results: list[TaskAcceptanceCaseResult]) -> TaskAcceptanceSummary:
    """Aggregate acceptance metrics for task formalization."""

    total = len(results)
    passed = sum(1 for result in results if result.result == "pass")
    schema_valid = sum(1 for result in results if result.schema_valid)
    complete = sum(1 for result in results if result.problem_complete)
    semantic = sum(1 for result in results if result.semantic_consistent)
    ready = sum(1 for result in results if result.solver_ready)
    unresolved_recall = sum(
        1
        for result in results
        if result.case.category in {"underspecified", "ambiguous"} and result.raw_output.report.unresolved_dependencies
    )
    denominator = max(1, sum(1 for result in results if result.case.category in {"underspecified", "ambiguous"}))
    deterministic = sum(
        1
        for result in results
        if result.raw_output.design_task is not None
        and result.raw_output.design_task.task_id.startswith("task_")
    )
    error_counter = Counter(error for result in results for error in result.error_types)
    warning_counter = Counter(warning for result in results for warning in result.warning_types)

    return TaskAcceptanceSummary(
        total_cases=total,
        passed_cases=passed,
        schema_validity_rate=schema_valid / total if total else 0.0,
        problem_completeness_rate=complete / total if total else 0.0,
        semantic_consistency_rate=semantic / total if total else 0.0,
        solver_readiness_rate=ready / total if total else 0.0,
        unresolved_dependency_recall=unresolved_recall / denominator,
        determinism_rate=deterministic / total if total else 0.0,
        error_type_distribution=dict(error_counter),
        warning_type_distribution=dict(warning_counter),
    )
