"""Compile formal fifth-layer simulation bundles."""

from __future__ import annotations

from datetime import datetime, timezone

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.planner.candidate_manager import find_candidate
from libs.schema.design_task import DesignTask
from libs.schema.planning import PlanningBundle, SearchState
from libs.schema.simulation import (
    BackendBinding,
    ExecutionPolicy,
    FeedbackContract,
    FidelityLevel,
    FidelityPolicy,
    FidelitySelectionReason,
    MonteCarloSettings,
    ResourceBudget,
    RobustnessPolicy,
    ServiceMethodSpec,
    SeverityRule,
    SimulationProvenance,
    SimulationBundle,
    SimulationCompilationReport,
    SimulationCompileResponse,
    SimulationMetadata,
    SimulationRequest,
    SimulationServingContract,
    SimulationValidationStatus,
    ValidationStatus,
    VerificationExecutionProfile,
    VerificationPolicy,
)
from libs.utils.hashing import stable_hash
from libs.simulation.artifact_registry import initialize_artifact_registry
from libs.simulation.netlist_builder import realize_netlist_instance
from libs.simulation.testbench_builder import build_analysis_plan, build_measurement_contract
from libs.simulation.validation import validate_simulation_bundle


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def normalize_fidelity_level(fidelity_level: str) -> str:
    """Normalize legacy fidelity aliases onto the current formal names."""

    if fidelity_level == "focused_validation":
        return "focused_truth"
    return fidelity_level


def _supports_native_ngspice(task: DesignTask, analysis_types: set[str]) -> bool:
    return (
        native_ngspice_available()
        and task.topology.topology_mode == "fixed"
        and task.circuit_family in {"two_stage_ota", "folded_cascode_ota", "ldo", "bandgap"}
        and analysis_types.issubset({"op", "ac", "tran"})
    )


def _fidelity_policy() -> FidelityPolicy:
    return FidelityPolicy(
        default_fidelity="quick_truth",
        supported_fidelities=[
            FidelityLevel(name="quick_truth", canonical_name="quick_truth", purpose="fast_feasibility_screening"),
            FidelityLevel(name="focused_truth", canonical_name="focused_truth", purpose="stronger_nominal_validation"),
            FidelityLevel(name="focused_validation", canonical_name="focused_truth", purpose="legacy_alias_for_focused_truth"),
            FidelityLevel(name="full_robustness_certification", canonical_name="full_robustness_certification", purpose="broad_robustness_certification"),
            FidelityLevel(name="targeted_failure_analysis", canonical_name="targeted_failure_analysis", purpose="diagnostic_failure_followup"),
        ],
        escalation_rules=[
            "default_candidates_use_quick_truth",
            "near_feasible_candidates_may_upgrade_to_focused_truth",
            "measurement_anomalies_with_high_potential_may_upgrade_to_focused_truth",
        ],
    )


def _physical_validation_status(model_binding, *, invocation_mode: str) -> ValidationStatus:
    warnings: list[str] = []
    truth_level = model_binding.validity_level.truth_level
    validity_state = "strong"
    summary = "configured model binding provides stronger physical grounding"
    if truth_level == "demonstrator_truth":
        validity_state = "weak"
        summary = "builtin demonstrator model proves real SPICE participation but not industrial accuracy"
        warnings.append("demonstrator_truth_only")
    if model_binding.model_type == "external" and model_binding.model_source.locator in {"", "missing_external_model_card"}:
        validity_state = "invalid"
        summary = "external model binding requested but model source is missing"
        warnings.append("missing_external_model_source")
    if invocation_mode != "native":
        warnings.append("non_native_backend_execution")
        if validity_state == "strong":
            validity_state = "weak"
    return ValidationStatus(
        truth_level=truth_level,
        validity_state=validity_state,
        model_binding_present=bool(model_binding.backend_model_ref),
        summary=summary,
        warnings=warnings,
    )


def _simulation_provenance(bundle_backend: BackendBinding, request: SimulationRequest):
    model_binding = request.model_binding
    if model_binding is None:
        raise ValueError("simulation request must carry model_binding before provenance can be built")
    return SimulationProvenance(
        backend=bundle_backend.backend,
        backend_version=bundle_backend.backend_version,
        invocation_mode=bundle_backend.invocation_mode,
        fidelity_level=normalize_fidelity_level(request.fidelity_level),
        truth_level=model_binding.validity_level.truth_level,
        model_binding=model_binding,
        artifact_lineage=[],
        provenance_tags=[
            f"model_type={model_binding.model_type}",
            f"model_source={model_binding.model_source.source_type}",
            f"planner_context={request.planner_context_ref}",
        ],
    )


def build_simulation_request(
    task: DesignTask,
    planning_bundle: PlanningBundle,
    search_state: SearchState,
    candidate_id: str,
    *,
    fidelity_level: str,
    backend_preference: str = "ngspice",
    escalation_reason: str = "planner_requested_truth_verification",
    model_binding_overrides: dict[str, float | int | str | bool] | None = None,
) -> SimulationRequest:
    """Build a formal SimulationRequest from fourth-layer state."""

    requested_fidelity = fidelity_level
    fidelity_level = normalize_fidelity_level(fidelity_level)
    candidate = find_candidate(search_state.candidate_pool_state, candidate_id)
    if candidate is None:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    analysis_scope = [analysis.analysis_type for analysis in task.evaluation_plan.analyses]
    if fidelity_level == "quick_truth":
        analysis_scope = [analysis for analysis in analysis_scope if analysis in {"op", "ac"}]
    elif fidelity_level == "focused_truth":
        analysis_scope = [analysis for analysis in analysis_scope if analysis in {"op", "ac", "tran"}]
    elif fidelity_level == "targeted_failure_analysis":
        analysis_scope = [analysis for analysis in analysis_scope if analysis in {"op", "ac", "tran", "noise"}]

    return SimulationRequest(
        request_id=f"simreq_{stable_hash(f'{task.task_id}|{candidate_id}|{fidelity_level}')[:12]}",
        task_id=task.task_id,
        candidate_id=candidate_id,
        world_state_ref=candidate.world_state_ref,
        planner_context_ref=search_state.search_id,
        analysis_scope=analysis_scope,
        fidelity_level=requested_fidelity if requested_fidelity == "focused_validation" else fidelity_level,
        backend_preference=backend_preference,
        environment_overrides=dict(model_binding_overrides or {}),
        measurement_targets=list(task.objective.reporting_metrics) if fidelity_level == "quick_truth" else _ordered_unique([*task.objective.reporting_metrics, "slew_rate_v_per_us"]),
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
    fidelity_level = normalize_fidelity_level(fidelity_level)
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


def _execution_profile(request: SimulationRequest, analysis_plan, measurement_contract) -> VerificationExecutionProfile:
    resolved = normalize_fidelity_level(request.fidelity_level)
    if resolved == "quick_truth":
        reason = FidelitySelectionReason(reason_code="default_screening_path", detail="candidate sent to fast truth screening")
        policy_name = "quick_truth_screening"
    elif resolved == "focused_truth":
        reason = FidelitySelectionReason(reason_code="upgraded_validation_path", detail=request.escalation_reason)
        policy_name = "focused_truth_validation"
    elif resolved == "targeted_failure_analysis":
        reason = FidelitySelectionReason(reason_code="targeted_failure_followup", detail=request.escalation_reason)
        policy_name = "targeted_failure_analysis"
    else:
        reason = FidelitySelectionReason(reason_code="robustness_certification_path", detail=request.escalation_reason)
        policy_name = "full_robustness_certification"
    return VerificationExecutionProfile(
        requested_fidelity=request.fidelity_level,
        resolved_fidelity=resolved,
        execution_policy=ExecutionPolicy(
            policy_name=policy_name,
            ordered_analysis_scope=[analysis.analysis_type for analysis in analysis_plan.ordered_analyses],
            measurement_targets=[definition.metric for definition in measurement_contract.measurement_definitions],
            early_termination_rules=list(analysis_plan.early_termination_rules),
        ),
        selection_reason=reason,
        provenance=[f"escalation_reason={request.escalation_reason}", f"backend_preference={request.backend_preference}"],
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
    fidelity_level: str = "quick_truth",
    backend_preference: str = "ngspice",
    escalation_reason: str = "planner_requested_truth_verification",
    model_binding_overrides: dict[str, float | int | str | bool] | None = None,
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
        model_binding_overrides=model_binding_overrides,
    )
    analysis_plan = build_analysis_plan(task, request)
    measurement_contract = build_measurement_contract(task, analysis_plan)
    execution_profile = _execution_profile(request, analysis_plan, measurement_contract)
    netlist = realize_netlist_instance(
        task,
        candidate,
        backend=backend_preference,
        analyses=analysis_plan.ordered_analyses,
        model_binding_overrides=request.environment_overrides,
    )
    signature = stable_hash(f"{task.task_id}|{candidate_id}|{fidelity_level}|{backend_preference}")
    resolved_fidelity = normalize_fidelity_level(fidelity_level)
    native_ngspice = backend_preference == "ngspice" and _supports_native_ngspice(
        task,
        {analysis.analysis_type for analysis in analysis_plan.ordered_analyses},
    )
    request = request.model_copy(update={"model_binding": netlist.model_binding})
    backend_binding = BackendBinding(
        backend=backend_preference,
        backend_version="ngspice-native-batch-v1" if native_ngspice else "deterministic-backend-v1",
        capability_map=["op", "ac", "tran", "noise", "pvt_sweep", "load_sweep", "temperature_sweep", "monte_carlo"],
        invocation_mode="native" if native_ngspice else "mock_truth",
        support_multi_analysis=True,
    )
    simulation_provenance = _simulation_provenance(backend_binding, request)
    physical_validation = _physical_validation_status(
        netlist.model_binding,
        invocation_mode=backend_binding.invocation_mode,
    )
    bundle = SimulationBundle(
        simulation_id=f"sim_{signature[:12]}",
        parent_task_id=task.task_id,
        candidate_id=candidate_id,
        planner_context_ref=search_state.search_id,
        backend_binding=backend_binding,
        model_binding=netlist.model_binding,
        simulation_provenance=simulation_provenance,
        netlist_instance=netlist,
        analysis_plan=analysis_plan,
        measurement_contract=measurement_contract,
        fidelity_policy=_fidelity_policy(),
        execution_profile=execution_profile,
        verification_policy=_verification_policy(task),
        robustness_policy=_robustness_policy(task, resolved_fidelity),
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
                f"truth_level={physical_validation.truth_level}",
                f"validation_state={physical_validation.validity_state}",
                f"native ngspice is currently enabled for {task.circuit_family} fixed-topology quick/focused truth verification" if native_ngspice else "bundle currently executes in mock_truth mode",
            ],
            provenance=[
                "task_formalization_layer",
                "planning_layer",
                "simulation_compiler",
                f"model_binding={netlist.model_binding.backend_model_ref}",
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
            "truth_level": compiled_bundle.simulation_provenance.truth_level,
            "validation_state": physical_validation.validity_state,
        },
    )
    return SimulationCompileResponse(
        status=status,
        simulation_bundle=None if status == "invalid" else compiled_bundle,
        report=report,
    )
