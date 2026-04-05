"""Compile DesignSpec objects into formal DesignTask instances."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.schema.design_spec import DesignSpec
from libs.schema.design_task import (
    CandidateSeed,
    DesignTask,
    InitialState,
    RandomizationPolicy,
    ReproducibilitySpec,
    SolverHint,
    TaskCompilationReport,
    TaskCompileResponse,
    TaskMetadata,
    ValidationStatus,
)
from libs.tasking.constraint_compiler import compile_constraints
from libs.tasking.design_space_builder import build_design_space
from libs.tasking.difficulty_profiler import build_difficulty_profile
from libs.tasking.evaluation_plan_builder import build_evaluation_plan
from libs.tasking.objective_compiler import compile_objective
from libs.tasking.task_graph_builder import build_task_graph
from libs.tasking.task_type_resolver import resolve_task_type
from libs.tasking.topology_resolver import resolve_topology
from libs.tasking.validation import validate_design_task
from libs.utils.hashing import stable_hash


def _solver_hint(
    task_type: str,
    discrete_degree: float,
    variable_dimension: int,
    evaluation_cost: str,
    expected_feasibility: str,
) -> SolverHint:
    if task_type == "topology_sizing" or discrete_degree > 0.2:
        recommended_solver_family = "hybrid"
    elif evaluation_cost == "expensive" and variable_dimension <= 12:
        recommended_solver_family = "bayesopt"
    else:
        recommended_solver_family = "cmaes"

    needs_feasibility_first = expected_feasibility in {"low", "unknown"}
    if needs_feasibility_first:
        recommended_search_stage = "feasibility_first"
    elif task_type == "topology_sizing":
        recommended_search_stage = "coarse_exploration"
    else:
        recommended_search_stage = "direct_local_refinement"

    surrogate_friendly = evaluation_cost in {"moderate", "expensive"} and discrete_degree <= 0.5
    parallelism_hint = "multi_stage" if evaluation_cost == "expensive" else "batch" if variable_dimension >= 6 else "single"
    budget_hint = "high" if evaluation_cost == "expensive" else "medium" if evaluation_cost == "moderate" else "low"
    return SolverHint(
        recommended_solver_family=recommended_solver_family,
        recommended_search_stage=recommended_search_stage,
        surrogate_friendly=surrogate_friendly,
        needs_feasibility_first=needs_feasibility_first,
        parallelism_hint=parallelism_hint,
        budget_hint=budget_hint,
    )


def _initial_state(task_id: str, design_space_variables: list) -> InitialState:
    template_defaults = {
        variable.name: variable.default
        for variable in design_space_variables
        if variable.default is not None
    }
    seed_candidate = CandidateSeed(
        seed_id=f"{task_id}_seed_0",
        values=template_defaults,
        source="template_defaults",
    )
    return InitialState(
        init_strategy="template_default",
        seed_candidates=[seed_candidate],
        template_defaults=template_defaults,
        warm_start_source=None,
        randomization_policy=RandomizationPolicy(enabled=True, strategy="lhs", amplitude=0.05),
        reproducibility=ReproducibilitySpec(seed=17, tag="task-formalization-default"),
    )


def _field_sources(spec: DesignSpec, task_type_hint: str | None) -> dict[str, str]:
    return {
        "task_type": "user_override" if task_type_hint else "system_inferred",
        "topology": "template_default" if spec.circuit_family != "unknown" else "expert_rule",
        "design_space": "process_rule" if spec.process_node else "expert_rule",
        "objective": "system_inferred",
        "constraints": "system_inferred",
        "evaluation_plan": "system_inferred",
        "initial_state": "template_default",
        "task_graph": "system_inferred",
        "difficulty_profile": "system_inferred",
        "solver_hint": "system_inferred",
        "metadata": "system_inferred",
    }


def _acceptance_summary(task: DesignTask) -> dict[str, float | int | str]:
    return {
        "variable_dimension": task.difficulty_profile.variable_dimension,
        "hard_constraint_count": len(task.constraints.hard_constraints),
        "analysis_count": len(task.evaluation_plan.analyses),
        "unresolved_dependency_count": len(task.validation_status.unresolved_dependencies),
        "solver_ready": "yes" if task.validation_status.is_valid else "no",
    }


def compile_design_task(spec: DesignSpec, task_type_hint: str | None = None) -> TaskCompileResponse:
    """Compile a validated DesignSpec into a formal DesignTask."""

    source_spec_signature = stable_hash(spec.model_dump_json())
    task_id = f"task_{source_spec_signature[:12]}"

    task_type_resolution = resolve_task_type(spec, task_type_hint=task_type_hint)
    resolved_family = spec.circuit_family if spec.circuit_family != "unknown" else task_type_resolution.candidate_families[0]
    topology_resolution = resolve_topology(
        circuit_family=resolved_family,
        task_type_resolution=task_type_resolution,
    )
    design_space_resolution = build_design_space(
        circuit_family=resolved_family,
        process_node=spec.process_node,
        task_type_resolution=task_type_resolution,
    )
    objective_compilation = compile_objective(spec)
    constraint_compilation = compile_constraints(spec)
    evaluation_plan_compilation = build_evaluation_plan(
        spec=spec,
        objective=objective_compilation.objective,
        constraint_metrics=[constraint.metric for constraint in constraint_compilation.constraints.hard_constraints],
    )
    unresolved_dependencies = list(dict.fromkeys(task_type_resolution.unresolved_dependencies))
    repair_history = list(evaluation_plan_compilation.repair_history)

    placeholder_validation = ValidationStatus(
        is_valid=False,
        errors=[],
        warnings=[],
        unresolved_dependencies=unresolved_dependencies,
        repair_history=repair_history,
        completeness_score=0.0,
    )
    difficulty_profile = build_difficulty_profile(
        design_space=design_space_resolution.design_space,
        hard_constraint_count=len(constraint_compilation.constraints.hard_constraints),
        evaluation_plan=evaluation_plan_compilation.evaluation_plan,
        unresolved_dependencies=unresolved_dependencies,
    )
    solver_hint = _solver_hint(
        task_type=task_type_resolution.task_type,
        discrete_degree=difficulty_profile.discrete_degree,
        variable_dimension=difficulty_profile.variable_dimension,
        evaluation_cost=difficulty_profile.evaluation_cost,
        expected_feasibility=difficulty_profile.expected_feasibility,
    )
    task_graph = build_task_graph(
        needs_feasibility_first=solver_hint.needs_feasibility_first,
        surrogate_friendly=solver_hint.surrogate_friendly,
    )

    metadata = TaskMetadata(
        compile_timestamp=datetime.now(timezone.utc).isoformat(),
        schema_version="task-schema-v1",
        source_spec_signature=source_spec_signature,
        assumptions=[
            *task_type_resolution.assumptions,
            *topology_resolution.assumptions,
            *design_space_resolution.assumptions,
            *objective_compilation.assumptions,
            "evaluation plan is deterministic and simulator-grounded",
        ],
        provenance=[
            "task_type_resolver",
            "topology_resolver",
            "design_space_builder",
            "objective_compiler",
            "constraint_compiler",
            "evaluation_plan_builder",
            "difficulty_profiler",
            "task_graph_builder",
            "validation_engine",
        ],
    )

    draft_task = DesignTask(
        task_id=task_id,
        parent_spec_id=spec.task_id,
        task_type=task_type_resolution.task_type,
        circuit_family=spec.circuit_family,
        topology=topology_resolution.topology,
        design_space=design_space_resolution.design_space,
        objective=objective_compilation.objective,
        constraints=constraint_compilation.constraints,
        evaluation_plan=evaluation_plan_compilation.evaluation_plan,
        initial_state=_initial_state(task_id, design_space_resolution.design_space.variables),
        task_graph=task_graph,
        difficulty_profile=difficulty_profile,
        solver_hint=solver_hint,
        metadata=metadata,
        validation_status=placeholder_validation,
    )
    validation_status = validate_design_task(draft_task, source_spec=spec)
    design_task = draft_task.model_copy(update={"validation_status": validation_status})

    status = "compiled"
    if validation_status.errors:
        status = "invalid"
    elif validation_status.warnings or validation_status.unresolved_dependencies:
        status = "compiled_with_warnings"

    report = TaskCompilationReport(
        status=status,
        field_sources=_field_sources(spec, task_type_hint),
        unresolved_dependencies=validation_status.unresolved_dependencies,
        derived_fields=[
            *topology_resolution.derived_fields,
            *design_space_resolution.derived_fields,
            *objective_compilation.derived_fields,
            *constraint_compilation.derived_fields,
            *evaluation_plan_compilation.derived_fields,
            "initial_state",
            "task_graph",
            "difficulty_profile",
            "solver_hint",
            "metadata",
            "validation_status",
        ],
        validation_errors=validation_status.errors,
        validation_warnings=validation_status.warnings,
        acceptance_summary=_acceptance_summary(design_task),
    )
    return TaskCompileResponse(
        status=status,
        design_task=None if status == "invalid" else design_task,
        report=report,
    )
