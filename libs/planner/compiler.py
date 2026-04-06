"""Compile formal planning bundles from DesignTask and WorldModelBundle."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.schema.design_task import DesignTask
from libs.schema.planning import (
    BudgetController,
    CandidateSchema,
    EscalationPolicy,
    PhaseController,
    PhaseRule,
    PlanningBundle,
    PlanningCompilationReport,
    PlanningCompileResponse,
    PlanningMetadata,
    PlanningServingContract,
    PlanningValidationStatus,
    RolloutConfig,
    SearchPolicy,
    SearchStateSchema,
    SelectionPolicy,
    ServiceMethodSpec,
    TerminationPolicy,
    TraceSchema,
    WorldModelBinding,
)
from libs.schema.world_model import WorldModelBundle
from libs.utils.hashing import stable_hash
from libs.planner.phase_controller import determine_initial_phase
from libs.planner.validation import validate_planning_bundle


def _search_policy(task: DesignTask) -> SearchPolicy:
    feasibility_mode = "feasibility_first" if task.solver_hint.needs_feasibility_first or task.objective.priority_policy == "feasibility_first" else "balanced"
    exploration = "uncertainty_guided_beam" if task.solver_hint.surrogate_friendly else "bounded_diversity_scan"
    exploitation = "trust_region_local_refinement"
    rollout = "world_model_mpc" if task.solver_hint.recommended_solver_family in {"model_based_mpc", "hybrid"} else "single_step_rollout"
    restart = "best_infeasible_restart" if task.difficulty_profile.expected_feasibility in {"low", "unknown"} else "frontier_refresh"
    topology = "template_locked" if task.topology.topology_mode == "fixed" else "bounded_template_switch"
    return SearchPolicy(
        exploration_policy=exploration,
        exploitation_policy=exploitation,
        feasibility_policy=feasibility_mode,
        rollout_policy=rollout,
        restart_policy=restart,
        local_refinement_policy="sensitivity_prioritized",
        topology_policy=topology,
    )


def _rollout_config(task: DesignTask) -> RolloutConfig:
    horizon = 1 if task.difficulty_profile.evaluation_cost == "expensive" else 2
    if task.solver_hint.budget_hint == "high":
        horizon = min(horizon + 1, 3)
    return RolloutConfig(
        horizon=horizon,
        beam_width=2 if task.solver_hint.parallelism_hint == "batch" else 1,
        max_branching_factor=3 if task.difficulty_profile.variable_dimension <= 8 else 2,
        require_rollout_ready=True,
        allow_feasibility_rollout=True,
    )


def _selection_policy(task: DesignTask) -> SelectionPolicy:
    return SelectionPolicy(
        ranking_source="world_model_ranker",
        prioritize_feasibility=True,
        prioritize_diversity=task.topology.topology_mode != "fixed",
        min_feasible_probability=0.55 if task.solver_hint.needs_feasibility_first else 0.4,
        uncertainty_penalty=0.25,
        simulation_value_weight=0.3 if task.difficulty_profile.evaluation_cost == "expensive" else 0.2,
    )


def _escalation_policy(task: DesignTask) -> EscalationPolicy:
    threshold = 0.55 if task.difficulty_profile.evaluation_cost == "expensive" else 0.65
    return EscalationPolicy(
        min_simulation_value=threshold,
        allow_must_escalate_override=True,
        max_batch_size=1 if task.solver_hint.budget_hint == "low" else 2,
        allowed_service_tiers=["ranking_ready", "rollout_ready", "must_escalate"],
    )


def _budget_controller(task: DesignTask) -> BudgetController:
    budget_hint = task.solver_hint.budget_hint
    if budget_hint == "low":
        proxy_budget, rollout_budget, sim_budget, calibration_budget = 20, 8, 3, 1
    elif budget_hint == "high":
        proxy_budget, rollout_budget, sim_budget, calibration_budget = 80, 24, 12, 4
    else:
        proxy_budget, rollout_budget, sim_budget, calibration_budget = 40, 12, 6, 2
    return BudgetController(
        max_proxy_evaluations=proxy_budget,
        max_rollouts=rollout_budget,
        max_real_simulations=sim_budget,
        max_calibration_updates=calibration_budget,
        batch_size=2 if task.solver_hint.parallelism_hint == "batch" else 1,
        per_phase_proxy_caps={
            "feasibility_bootstrapping": proxy_budget // 2,
            "performance_refinement": proxy_budget // 3,
            "robustness_verification": max(2, proxy_budget // 6),
            "calibration_recovery": max(2, proxy_budget // 6),
            "terminated": 0,
        },
    )


def _phase_controller(task: DesignTask) -> PhaseController:
    initial = determine_initial_phase(task)
    return PhaseController(
        initial_phase=initial,
        allowed_transitions=[
            PhaseRule(from_phase="feasibility_bootstrapping", to_phase="performance_refinement", trigger="feasible_candidate_found"),
            PhaseRule(from_phase="performance_refinement", to_phase="robustness_verification", trigger="verified_feasible_candidate_available"),
            PhaseRule(from_phase="performance_refinement", to_phase="calibration_recovery", trigger="world_model_trust_degrades"),
            PhaseRule(from_phase="robustness_verification", to_phase="terminated", trigger="verification_goal_reached"),
            PhaseRule(from_phase="calibration_recovery", to_phase="performance_refinement", trigger="trust_restored"),
        ],
    )


def _termination_policy(task: DesignTask) -> TerminationPolicy:
    max_iterations = 12 if task.solver_hint.budget_hint == "low" else 20
    if task.solver_hint.budget_hint == "high":
        max_iterations = 32
    return TerminationPolicy(
        max_iterations=max_iterations,
        stagnation_patience=3 if task.solver_hint.needs_feasibility_first else 4,
        stop_on_verified_feasible=True,
        require_verification_phase=task.evaluation_plan.fidelity_policy == "staged_fidelity",
        min_feasible_count_before_terminate=1,
    )


def _serving_contract() -> PlanningServingContract:
    return PlanningServingContract(
        initialize_search=ServiceMethodSpec(input_schema="DesignTask+WorldModelBundle", output_schema="SearchInitializationResponse"),
        propose_candidates=ServiceMethodSpec(input_schema="SearchState", output_schema="CandidateBatchResponse"),
        evaluate_candidates=ServiceMethodSpec(input_schema="SearchState", output_schema="CandidateBatchResponse"),
        plan_next_actions=ServiceMethodSpec(input_schema="SearchState", output_schema="ActionPlanResponse"),
        select_for_simulation=ServiceMethodSpec(input_schema="SearchState", output_schema="SimulationSelectionResponse"),
        ingest_simulation_feedback=ServiceMethodSpec(input_schema="SearchState+TruthCalibrationRecord", output_schema="SimulationFeedbackResponse"),
        advance_phase=ServiceMethodSpec(input_schema="SearchState", output_schema="CandidateBatchResponse"),
        should_terminate=ServiceMethodSpec(input_schema="SearchState", output_schema="TerminationDecision"),
        get_best_result=ServiceMethodSpec(input_schema="SearchState", output_schema="PlanningBestResult"),
    )


def compile_planning_bundle(task: DesignTask, world_model_bundle: WorldModelBundle) -> PlanningCompileResponse:
    """Compile a formal PlanningBundle."""

    timestamp = datetime.now(timezone.utc).isoformat()
    signature = stable_hash(f"{task.model_dump_json()}|{world_model_bundle.world_model_id}")
    bundle = PlanningBundle(
        planning_id=f"plan_{signature[:12]}",
        parent_task_id=task.task_id,
        world_model_binding=WorldModelBinding(
            world_model_id=world_model_bundle.world_model_id,
            parent_task_id=task.task_id,
            supported_circuit_families=world_model_bundle.supported_circuit_families,
            supported_task_types=world_model_bundle.supported_task_types,
            service_methods=[
                "predict_metrics",
                "predict_feasibility",
                "predict_transition",
                "rollout",
                "rank_candidates",
                "estimate_simulation_value",
                "calibrate_with_truth",
                "validate_state",
            ],
            trust_policy_ref="world_model_bundle.trust_policy",
            calibration_readiness="screening",
        ),
        search_policy=_search_policy(task),
        search_state_schema=SearchStateSchema(
            required_fields=[
                "search_id",
                "task_id",
                "episode_id",
                "current_world_state",
                "candidate_pool_state",
                "frontier_state",
                "evaluated_state_refs",
                "pending_simulation_refs",
                "budget_state",
                "phase_state",
                "best_known_feasible",
                "best_known_infeasible",
                "strategy_context",
                "risk_context",
                "provenance",
            ]
        ),
        candidate_schema=CandidateSchema(
            required_fields=[
                "candidate_id",
                "task_id",
                "world_state_ref",
                "parent_candidate_id",
                "generation_depth",
                "proposal_source",
                "proposal_action_chain",
                "predicted_metrics",
                "predicted_feasibility",
                "predicted_uncertainty",
                "simulation_value_estimate",
                "priority_score",
                "dominance_status",
                "lifecycle_status",
                "evaluation_history",
                "decision_history",
                "artifact_refs",
            ]
        ),
        rollout_config=_rollout_config(task),
        selection_policy=_selection_policy(task),
        escalation_policy=_escalation_policy(task),
        budget_controller=_budget_controller(task),
        phase_controller=_phase_controller(task),
        termination_policy=_termination_policy(task),
        trace_schema=TraceSchema(
            required_fields=[
                "trace_id",
                "task_id",
                "episode_id",
                "step_index",
                "search_state_snapshot",
                "selected_candidate_id",
                "executed_action_chain",
                "world_model_queries",
                "simulation_decision",
                "simulation_result_ref",
                "reward_or_progress_signal",
                "decision_rationale",
                "trust_snapshot",
                "budget_snapshot",
                "outcome_tag",
            ]
        ),
        serving_contract=_serving_contract(),
        metadata=PlanningMetadata(
            compile_timestamp=timestamp,
            source_task_signature=stable_hash(task.model_dump_json()),
            source_world_model_signature=world_model_bundle.world_model_id,
            implementation_version="planning-layer-v1",
            assumptions=[
                "planning layer is feasibility-first and budget-aware",
                "all candidate evaluation flows through the third-layer world-model service",
                "real simulation escalation remains explicit and traceable",
            ],
            provenance=[
                "task_formalization_layer",
                "world_model_layer",
                "planning_compiler",
            ],
        ),
        validation_status=PlanningValidationStatus(
            is_valid=False,
            errors=[],
            warnings=[],
            unresolved_dependencies=[],
            repair_history=[],
            completeness_score=0.0,
        ),
    )
    validation = validate_planning_bundle(bundle, task, world_model_bundle)
    compiled_bundle = bundle.model_copy(update={"validation_status": validation})
    status = "compiled" if validation.is_valid and not validation.warnings else "compiled_with_warnings"
    if validation.errors:
        status = "invalid"
    report = PlanningCompilationReport(
        status=status,
        derived_fields=[
            "world_model_binding",
            "search_policy",
            "search_state_schema",
            "candidate_schema",
            "rollout_config",
            "selection_policy",
            "escalation_policy",
            "budget_controller",
            "phase_controller",
            "termination_policy",
            "trace_schema",
            "serving_contract",
            "metadata",
            "validation_status",
        ],
        validation_errors=validation.errors,
        validation_warnings=validation.warnings,
        acceptance_summary={
            "method_count": 9,
            "phase_count": len(bundle.search_state_schema.phase_values),
            "completeness_score": validation.completeness_score,
            "budget_hint": task.solver_hint.budget_hint,
        },
    )
    return PlanningCompileResponse(
        status=status,
        planning_bundle=None if status == "invalid" else compiled_bundle,
        report=report,
    )

