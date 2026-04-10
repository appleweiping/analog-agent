"""Compile formal fifth-layer simulation bundles."""

from __future__ import annotations

from datetime import datetime, timezone

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.planner.candidate_manager import find_candidate
from libs.schema.design_task import DesignTask
from libs.schema.planning import PlanningBundle, SearchState
from libs.schema.simulation import (
    BackendBinding,
    FeedbackContract,
    MonteCarloSettings,
    ResourceBudget,
    RobustnessPolicy,
    ServiceMethodSpec,
    SeverityRule,
    SimulationBundle,
    SimulationCompilationReport,
    SimulationCompileResponse,
    SimulationMetadata,
    SimulationRequest,
    SimulationServingContract,
    SimulationValidationStatus,
    VerificationPolicy,
)
from libs.utils.hashing import stable_hash
from libs.simulation.artifact_registry import initialize_artifact_registry
from libs.simulation.netlist_builder import realize_netlist_instance
from libs.simulation.testbench_builder import build_analysis_plan, build_measurement_contract
from libs.simulation.validation import validate_simulation_bundle


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_simulation_request(
    task: DesignTask,
    planning_bundle: PlanningBundle,
    search_state: SearchState,
    candidate_id: str,
    *,
    fidelity_level: str,
    backend_preference: str = "ngspice",
    escalation_reason: str = "planner_requested_truth_verification",
) -> SimulationRequest:
    """Build a formal SimulationRequest from fourth-layer state."""

    candidate = find_candidate(search_state.candidate_pool_state, candidate_id)
    if candidate is None:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    analysis_scope = [analysis.analysis_type for analysis in task.evaluation_plan.analyses]
    if fidelity_level == "quick_truth":
        analysis_scope = [analysis for analysis in analysis_scope if analysis in {"op", "ac"}]
    elif fidelity_level == "focused_validation":
        analysis_scope = [analysis for analysis in analysis_scope if analysis in {"op", "ac", "tran", "noise"}]
    elif fidelity_level == "targeted_failure_analysis":
        analysis_scope = [analysis for analysis in analysis_scope if analysis in {"op", "ac", "tran", "noise"}]

    return SimulationRequest(
        request_id=f"simreq_{stable_hash(f'{task.task_id}|{candidate_id}|{fidelity_level}')[:12]}",
        task_id=task.task_id,
        candidate_id=candidate_id,
        world_state_ref=candidate.world_state_ref,
        planner_context_ref=search_state.search_id,
        analysis_scope=analysis_scope,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
        environment_overrides={},
        measurement_targets=list(task.objective.reporting_metrics),
        escalation_reason=escalation_reason,
        priority_class="high" if candidate.predicted_uncertainty and candidate.predicted_uncertainty.must_escalate else "normal",
        resource_budget=ResourceBudget(
            timeout_seconds=90 if fidelity_level == "full_robustness_certification" else 45,
            max_parallel_analyses=2 if task.solver_hint.parallelism_hint == "batch" else 1,
            resource_tier="high" if fidelity_level == "full_robustness_certification" else "medium",
        ),
        provenance=[
            f"planner_phase={search_state.phase_state.current_phase}",
            f"candidate_lifecycle={candidate.lifecycle_status}",
            f"world_state_ref={candidate.world_state_ref}",
            f"planning_bundle={planning_bundle.planning_id}",
        ],
    )


def _verification_policy(task: DesignTask) -> VerificationPolicy:
    return VerificationPolicy(
        constraint_evaluation_rules=[
            "evaluate_all_hard_constraints_on_truth_metrics",
            "treat_missing_core_metric_as_failure",
            "preserve_constraint_group_membership_for_downstream_feedback",
        ],
        feasibility_aggregation="all_hard_constraints",
        severity_levels=[
            SeverityRule(condition="core_constraint_failed", severity="critical"),
            SeverityRule(condition="important_constraint_failed", severity="high"),
            SeverityRule(condition="auxiliary_constraint_failed", severity="medium"),
        ],
        pass_criteria=[
            "all_core_constraints_pass",
            "no_simulation_invalid_status",
        ],
    )


def _robustness_policy(task: DesignTask, fidelity_level: str) -> RobustnessPolicy:
    corner_values = [str(value) for value in task.evaluation_plan.corners_policy.values] or ["tt"]
    temp_values = [float(value) for value in task.evaluation_plan.temperature_policy.values] or [27.0]
    load_values = [float(value) for value in task.evaluation_plan.load_policy.values] or [2e-12]
    return RobustnessPolicy(
        required_corners=corner_values if fidelity_level == "full_robustness_certification" else corner_values[:1],
        temperature_range_c=temp_values if fidelity_level == "full_robustness_certification" else temp_values[:1],
        load_conditions=load_values if fidelity_level == "full_robustness_certification" else load_values[:1],
        monte_carlo_settings=MonteCarloSettings(
            enabled=fidelity_level == "full_robustness_certification",
            sample_count=16 if fidelity_level == "full_robustness_certification" else 0,
            sigma_level=3.0,
        ),
        escalation_policy=[
            "escalate_to_full_robustness_if_nominal_feasible",
            "emit_boundary_candidate_if_worst_case_margin_small",
        ],
    )


def _serving_contract() -> SimulationServingContract:
    return SimulationServingContract(
        realize_netlist=ServiceMethodSpec(input_schema="DesignTask+CandidateRecord+SimulationRequest", output_schema="NetlistInstance"),
        run_simulation=ServiceMethodSpec(input_schema="SimulationBundle+SimulationRequest", output_schema="MeasurementReport"),
        extract_measurements=ServiceMethodSpec(input_schema="raw_backend_outputs", output_schema="MeasurementReport"),
        verify_constraints=ServiceMethodSpec(input_schema="MeasurementReport+VerificationPolicy", output_schema="ConstraintAssessmentRecord[]"),
        certify_robustness=ServiceMethodSpec(input_schema="MeasurementReport+RobustnessPolicy", output_schema="RobustnessCertificate"),
        emit_calibration_feedback=ServiceMethodSpec(input_schema="VerificationResult", output_schema="CalibrationFeedback"),
        archive_artifacts=ServiceMethodSpec(input_schema="SimulationBundle", output_schema="ArtifactRegistry"),
        validate_backend=ServiceMethodSpec(input_schema="BackendBinding", output_schema="BackendValidationReport"),
    )


def compile_simulation_bundle(
    task: DesignTask,
    planning_bundle: PlanningBundle,
    search_state: SearchState,
    candidate_id: str,
    *,
    fidelity_level: str = "focused_validation",
    backend_preference: str = "ngspice",
    escalation_reason: str = "planner_requested_truth_verification",
) -> SimulationCompileResponse:
    """Compile the formal fifth-layer SimulationBundle."""

    candidate = find_candidate(search_state.candidate_pool_state, candidate_id)
    if candidate is None:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    request = build_simulation_request(
        task,
        planning_bundle,
        search_state,
        candidate_id,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
        escalation_reason=escalation_reason,
    )
    analysis_plan = build_analysis_plan(task, request)
    measurement_contract = build_measurement_contract(task, analysis_plan)
    netlist = realize_netlist_instance(
        task,
        candidate,
        backend=backend_preference,
        analyses=analysis_plan.ordered_analyses,
    )
    signature = stable_hash(f"{task.task_id}|{candidate_id}|{fidelity_level}|{backend_preference}")
    native_ngspice = (
        backend_preference == "ngspice"
        and native_ngspice_available()
        and task.circuit_family == "two_stage_ota"
        and task.topology.topology_mode == "fixed"
        and {analysis.analysis_type for analysis in analysis_plan.ordered_analyses}.issubset({"op", "ac"})
    )
    bundle = SimulationBundle(
        simulation_id=f"sim_{signature[:12]}",
        parent_task_id=task.task_id,
        candidate_id=candidate_id,
        planner_context_ref=search_state.search_id,
        backend_binding=BackendBinding(
            backend=backend_preference,
            backend_version="ngspice-native-batch-v1" if native_ngspice else "deterministic-backend-v1",
            capability_map=["op", "ac", "tran", "noise", "pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"],
            invocation_mode="native" if native_ngspice else "mock_truth",
            support_multi_analysis=True,
        ),
        netlist_instance=netlist,
        analysis_plan=analysis_plan,
        measurement_contract=measurement_contract,
        verification_policy=_verification_policy(task),
        robustness_policy=_robustness_policy(task, fidelity_level),
        artifact_registry=initialize_artifact_registry(f"sim_{signature[:12]}"),
        feedback_contract=FeedbackContract(
            emit_calibration_feedback=True,
            emit_planner_feedback=True,
            emit_failure_diagnostics=True,
        ),
        serving_contract=_serving_contract(),
        metadata=SimulationMetadata(
            compile_timestamp=_timestamp(),
            source_task_signature=stable_hash(task.model_dump_json()),
            source_candidate_signature=stable_hash(candidate.model_dump_json()),
            implementation_version="simulation-layer-v1",
            assumptions=[
                "ground-truth layer remains the only physical verification authority",
                "backend bindings preserve a unified schema across demonstrator truth and mock_truth modes",
                "verification outputs are emitted as structured objects for planner and world-model feedback",
                "native ngspice is currently enabled for two_stage_ota fixed-topology op/ac verification only" if native_ngspice else "bundle currently executes in mock_truth mode",
            ],
            provenance=[
                "task_formalization_layer",
                "planning_layer",
                "simulation_compiler",
            ],
        ),
        validation_status=SimulationValidationStatus(
            is_valid=False,
            errors=[],
            warnings=[],
            unresolved_dependencies=[],
            repair_history=[],
            completeness_score=0.0,
        ),
    )
    validation = validate_simulation_bundle(bundle, task, candidate)
    compiled_bundle = bundle.model_copy(update={"validation_status": validation})
    status = "compiled" if validation.is_valid and not validation.warnings else "compiled_with_warnings"
    if validation.errors:
        status = "invalid"
    report = SimulationCompilationReport(
        status=status,
        derived_fields=[
            "backend_binding",
            "netlist_instance",
            "analysis_plan",
            "measurement_contract",
            "verification_policy",
            "robustness_policy",
            "artifact_registry",
            "feedback_contract",
            "serving_contract",
            "metadata",
            "validation_status",
        ],
        validation_errors=validation.errors,
        validation_warnings=validation.warnings,
        acceptance_summary={
            "analysis_count": len(compiled_bundle.analysis_plan.ordered_analyses),
            "metric_count": len(compiled_bundle.measurement_contract.measurement_definitions),
            "completeness_score": validation.completeness_score,
            "backend": backend_preference,
        },
    )
    return SimulationCompileResponse(
        status=status,
        simulation_bundle=None if status == "invalid" else compiled_bundle,
        report=report,
    )
