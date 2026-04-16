"""Formal fifth-layer simulation and verification service."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.planner.candidate_manager import find_candidate
from libs.eval.stats import build_verification_stats_record
from libs.schema.design_task import DesignTask
from libs.schema.planning import PlanningBundle, SearchState
from libs.schema.simulation import (
    CalibrationFeedback,
    PlannerFeedback,
    SimulationBundle,
    SimulationCompileResponse,
    SimulationExecutionResponse,
    SimulationRequest,
    ValidationStatus,
    VerificationResult,
)
from libs.schema.world_model import WORLD_MODEL_METRICS, TruthCalibrationRecord, TruthConstraint, TruthMetric
from libs.simulation.artifact_registry import persist_json_artifact, persist_text_artifact
from libs.simulation.backend_router import execute_bundle, validate_backend
from libs.simulation.compiler import build_simulation_request, compile_simulation_bundle, normalize_fidelity_level
from libs.simulation.constraint_verifier import verify_constraints
from libs.simulation.failure_analyzer import attribute_failures
from libs.simulation.measurement_extractors import extract_measurement_report
from libs.simulation.robustness_evaluator import certify_robustness
from libs.simulation.validation import validate_verification_result
from libs.utils.hashing import stable_hash


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truth_fidelity(simulation_fidelity: str) -> str:
    simulation_fidelity = normalize_fidelity_level(simulation_fidelity)
    mapping = {
        "quick_truth": "quick_screening",
        "focused_truth": "partial_simulation",
        "full_robustness_certification": "full_ground_truth",
        "targeted_failure_analysis": "full_ground_truth",
    }
    return mapping[simulation_fidelity]


def _physical_validation_from_bundle(simulation_bundle: SimulationBundle, completion_status: str | None = None) -> ValidationStatus:
    base = simulation_bundle.simulation_provenance.model_binding
    warnings = list(simulation_bundle.simulation_provenance.provenance_tags)
    warnings.extend(simulation_bundle.metadata.assumptions[:1])
    validity_state = "strong"
    summary = "configured truth executed with explicit external model binding"
    if base.validity_level.truth_level == "demonstrator_truth":
        validity_state = "weak"
        summary = "demonstrator truth executed with builtin or simplified model binding"
        warnings.append("demonstrator_truth_only")
    if not base.backend_model_ref or base.binding_confidence < 0.3:
        validity_state = "invalid"
        summary = "model binding missing or too weak to claim reliable physical truth"
        warnings.append("model_binding_missing_or_weak")
    if completion_status in {"simulator_failure", "measurement_failure", "timeout"}:
        warnings.append(f"completion_status={completion_status}")
        if validity_state == "strong":
            validity_state = "weak"
    return ValidationStatus(
        truth_level=base.validity_level.truth_level,
        validity_state=validity_state,
        model_binding_present=bool(base.backend_model_ref),
        summary=summary,
        warnings=warnings,
    )


class SimulationService:
    """Ground-truth simulation and verification service."""

    def __init__(self, task: DesignTask, planning_bundle: PlanningBundle, search_state: SearchState) -> None:
        self.task = task
        self.planning_bundle = planning_bundle
        self.search_state = search_state

    def validate_backend(self, simulation_bundle: SimulationBundle):
        """Validate the selected backend binding."""

        return validate_backend(simulation_bundle.backend_binding)

    def realize_netlist(self, simulation_bundle: SimulationBundle) -> str:
        """Persist the rendered netlist artifact and return its artifact id."""

        registry, artifact_id = persist_text_artifact(
            simulation_bundle.artifact_registry,
            "netlist",
            "candidate.sp",
            simulation_bundle.netlist_instance.rendered_netlist,
            simulation_provenance=simulation_bundle.simulation_provenance,
            validation_status=_physical_validation_from_bundle(simulation_bundle),
        )
        simulation_bundle.artifact_registry = registry
        return artifact_id

    def run_simulation(self, simulation_bundle: SimulationBundle, simulation_request: SimulationRequest):
        """Execute backend analyses and return updated bundle plus raw outputs."""

        candidate = find_candidate(self.search_state.candidate_pool_state, simulation_request.candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate_id: {simulation_request.candidate_id}")
        return execute_bundle(simulation_bundle, simulation_request, self.task, candidate)

    def extract_measurements(self, simulation_bundle: SimulationBundle, parsed_outputs: list[dict[str, object]]):
        """Extract structured measurements from backend outputs."""

        return extract_measurement_report(simulation_bundle, parsed_outputs, candidate_id=simulation_bundle.candidate_id)

    def verify_constraints(self, measurement_report, simulation_bundle: SimulationBundle):
        """Verify DesignTask constraints against extracted truth metrics."""

        return verify_constraints(self.task, measurement_report, simulation_bundle.verification_policy)

    def certify_robustness(self, candidate_id: str, simulation_bundle: SimulationBundle):
        """Certify robustness according to the bundle policy."""

        candidate = find_candidate(self.search_state.candidate_pool_state, candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate_id: {candidate_id}")
        return certify_robustness(
            self.task,
            candidate,
            simulation_bundle.robustness_policy,
            fidelity_level=simulation_bundle.analysis_plan.fidelity_level,
        )

    def emit_calibration_feedback(
        self,
        simulation_bundle: SimulationBundle,
        measurement_report,
        assessments,
    ) -> CalibrationFeedback:
        """Emit structured calibration feedback for the third layer."""

        candidate = find_candidate(self.search_state.candidate_pool_state, simulation_bundle.candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate_id: {simulation_bundle.candidate_id}")
        truth_record = TruthCalibrationRecord(
            simulator_signature=f"{simulation_bundle.backend_binding.backend}:{simulation_bundle.backend_binding.backend_version}",
            analysis_fidelity=_truth_fidelity(simulation_bundle.analysis_plan.fidelity_level),
            truth_level=simulation_bundle.simulation_provenance.truth_level,
            validation_status=_physical_validation_from_bundle(simulation_bundle).validity_state,
            metrics=[
                TruthMetric(metric=metric.metric, value=metric.value)
                for metric in measurement_report.measured_metrics
                if metric.metric in WORLD_MODEL_METRICS
            ],
            constraints=[
                TruthConstraint(
                    constraint_name=item.constraint_name,
                    constraint_group=item.constraint_group,
                    is_satisfied=item.is_satisfied,
                    margin=item.margin,
                )
                for item in assessments
            ],
            artifact_refs=[record.artifact_id for record in simulation_bundle.artifact_registry.records],
            provenance_tags=[
                f"truth_level={simulation_bundle.simulation_provenance.truth_level}",
                f"fidelity={simulation_bundle.execution_profile.resolved_fidelity}",
                f"backend={simulation_bundle.backend_binding.backend}",
            ],
            timestamp=_timestamp(),
        )
        predicted_map = {
            metric.metric: metric.value
            for metric in (candidate.predicted_metrics.metrics if candidate.predicted_metrics else [])
        }
        residuals = {
            metric.metric: round(metric.value - predicted_map.get(metric.metric, metric.value), 6)
            for metric in measurement_report.measured_metrics
        }
        disagreements = [item.constraint_name for item in assessments if not item.is_satisfied]
        trust_flags = []
        if candidate.predicted_uncertainty and candidate.predicted_uncertainty.confidence >= 0.6 and disagreements:
            trust_flags.append("predicted_confident_but_truth_failed")
        if candidate.predicted_uncertainty and candidate.predicted_uncertainty.must_escalate:
            trust_flags.append("preexisting_must_escalate")
        return CalibrationFeedback(
            calibration_id=f"cal_{stable_hash(f'{simulation_bundle.simulation_id}|{simulation_bundle.candidate_id}')[:12]}",
            candidate_id=simulation_bundle.candidate_id,
            world_model_ref=self.planning_bundle.world_model_binding.world_model_id,
            truth_record=truth_record,
            residual_metrics=residuals,
            constraint_disagreement=disagreements,
            trust_violation_flags=trust_flags,
            usable_region_update=[f"family={self.task.circuit_family}", f"template={self.task.topology.template_id or 'unknown'}"],
            retrain_priority="high" if disagreements or trust_flags else "medium",
            provenance=[
                f"planner_search={self.search_state.search_id}",
                f"phase={self.search_state.phase_state.current_phase}",
                f"truth_level={simulation_bundle.simulation_provenance.truth_level}",
                f"validation_state={_physical_validation_from_bundle(simulation_bundle).validity_state}",
            ],
        )

    def _planner_feedback(self, simulation_bundle: SimulationBundle, assessments, robustness, calibration_feedback: CalibrationFeedback, failure) -> PlannerFeedback:
        if failure.primary_failure_class in {"measurement_failure", "analysis_failure", "simulation_invalid", "simulator_failure"}:
            lifecycle = "needs_more_simulation"
            strategy = ["inspect_measurement_path", "retry_or_escalate_truth_verification"]
            basis = "measurement_failure" if failure.primary_failure_class in {"measurement_failure", "analysis_failure"} else "simulation_failure"
            phase_hint = self.search_state.phase_state.current_phase
        else:
            all_pass = all(item.is_satisfied for item in assessments)
            robust_pass = robustness.certification_status in {"robust_certified", "partial_robust", "nominal_only"}
            resolved_fidelity = normalize_fidelity_level(simulation_bundle.analysis_plan.fidelity_level)
            if all_pass and robust_pass and resolved_fidelity == "full_robustness_certification":
                lifecycle = "verified"
            elif all_pass:
                lifecycle = "needs_more_simulation"
            elif any(abs(item.margin) <= 0.05 for item in assessments if item.assessment_basis == "measurement_value"):
                lifecycle = "boundary_candidate"
            else:
                lifecycle = "rejected"
            strategy = [
                "increase_safety_margin" if calibration_feedback.constraint_disagreement else "retain_current_direction",
                "prefer_high_truth_candidates_for_phase_transition" if lifecycle == "verified" else "revisit_candidate_neighborhood",
            ]
            basis = "verification_success" if lifecycle == "verified" else "design_failure"
            phase_hint = "robustness_verification" if lifecycle == "verified" else self.search_state.phase_state.current_phase
        resolved_fidelity = simulation_bundle.execution_profile.resolved_fidelity
        if failure.primary_failure_class == "measurement_failure":
            escalation_advice = ["upgrade_to_focused_truth"] if resolved_fidelity == "quick_truth" else ["repeat_focused_truth"]
            recommended_fidelity = "focused_truth"
            escalation_reason = "measurement_anomaly_requires_stronger_truth"
        elif lifecycle == "boundary_candidate":
            escalation_advice = ["upgrade_to_focused_truth"] if resolved_fidelity == "quick_truth" else ["retain_current_fidelity"]
            recommended_fidelity = "focused_truth"
            escalation_reason = "boundary_candidate_requires_higher_confidence"
        elif lifecycle == "verified":
            escalation_advice = ["retain_current_fidelity"]
            recommended_fidelity = resolved_fidelity
            escalation_reason = "verification_sufficient_for_current_phase"
        else:
            escalation_advice = ["defer_escalation"]
            recommended_fidelity = "quick_truth"
            escalation_reason = "default_screening_path"
        if simulation_bundle.simulation_provenance.truth_level == "demonstrator_truth":
            strategy.append("respect_demonstrator_truth_boundary")
            if "demonstrator_truth_only" not in calibration_feedback.trust_violation_flags:
                calibration_feedback.trust_violation_flags.append("demonstrator_truth_only")
        return PlannerFeedback(
            candidate_id=simulation_bundle.candidate_id,
            lifecycle_update=lifecycle,
            strategy_correction=strategy,
            phase_hint=phase_hint,
            trust_alerts=list(calibration_feedback.trust_violation_flags),
            feedback_basis=basis,
            escalation_advice=[
                {
                    "advice": advice,
                    "recommended_fidelity": recommended_fidelity,
                    "escalation_reason": escalation_reason,
                    "confidence": 0.72 if advice != "defer_escalation" else 0.55,
                }
                for advice in escalation_advice
            ],
            recommended_fidelity=recommended_fidelity,
            escalation_reason=escalation_reason,
            artifact_refs=[record.artifact_id for record in simulation_bundle.artifact_registry.records],
        )

    def _verify_compiled_bundle(self, simulation_bundle: SimulationBundle, simulation_request: SimulationRequest) -> SimulationExecutionResponse:
        """Run the full fifth-layer execution pipeline for a precompiled bundle."""

        netlist_artifact = self.realize_netlist(simulation_bundle)
        simulation_bundle, backend_report, parsed_outputs = self.run_simulation(simulation_bundle, simulation_request)
        measurement_report = self.extract_measurements(simulation_bundle, parsed_outputs)
        simulation_bundle.artifact_registry, measurement_artifact = persist_json_artifact(
            simulation_bundle.artifact_registry,
            "measurement_table",
            "measurement_report.json",
            measurement_report.model_dump(mode="json"),
            simulation_provenance=simulation_bundle.simulation_provenance,
            validation_status=_physical_validation_from_bundle(simulation_bundle),
        )
        assessments = self.verify_constraints(measurement_report, simulation_bundle)
        robustness = self.certify_robustness(simulation_bundle.candidate_id, simulation_bundle)
        measurement_failures = [result for result in measurement_report.measurement_results if result.status.status != "measured"]
        completion_status = "success" if all(record.success for record in measurement_report.executed_analyses) else "simulator_failure"
        if completion_status == "success" and measurement_failures and not measurement_report.measured_metrics:
            completion_status = "measurement_failure"
        elif completion_status == "success" and measurement_failures:
            completion_status = "partial_success"
        failure = attribute_failures(
            assessments,
            measurement_report,
            robustness,
            render_ready=simulation_bundle.netlist_instance.render_status == "ready",
            completion_status=completion_status,
        )
        calibration_feedback = self.emit_calibration_feedback(simulation_bundle, measurement_report, assessments)
        planner_feedback = self._planner_feedback(simulation_bundle, assessments, robustness, calibration_feedback, failure)
        physical_validation = _physical_validation_from_bundle(simulation_bundle, completion_status)
        simulation_bundle.artifact_registry, verification_artifact = persist_json_artifact(
            simulation_bundle.artifact_registry,
            "verification_report",
            "verification_result.json",
            {
                "candidate_id": simulation_bundle.candidate_id,
                "netlist_artifact": netlist_artifact,
                "assessment_count": len(assessments),
                "robustness_status": robustness.certification_status,
                "truth_level": physical_validation.truth_level,
                "validation_state": physical_validation.validity_state,
            },
            simulation_provenance=simulation_bundle.simulation_provenance,
            validation_status=physical_validation,
        )
        simulation_bundle.simulation_provenance = simulation_bundle.simulation_provenance.model_copy(
            update={
                "artifact_lineage": [record.artifact_id for record in simulation_bundle.artifact_registry.records],
                "provenance_tags": [
                    *simulation_bundle.simulation_provenance.provenance_tags,
                    f"validation_state={physical_validation.validity_state}",
                ],
            }
        )
        all_constraints_pass = all(item.is_satisfied for item in assessments if item.assessment_basis == "measurement_value")
        unavailable_assessments = [item for item in assessments if item.assessment_basis == "measurement_unavailable"]
        feasibility = "feasible_nominal"
        if completion_status == "simulator_failure" or failure.primary_failure_class in {"simulator_failure", "simulation_invalid"}:
            feasibility = "simulation_invalid"
        elif completion_status == "measurement_failure" or unavailable_assessments:
            feasibility = "measurement_invalid"
        elif not all_constraints_pass:
            feasibility = "constraint_fail"
        elif (
            simulation_bundle.analysis_plan.fidelity_level == "full_robustness_certification"
            and robustness.certification_status == "robust_certified"
        ):
            feasibility = "feasible_certified"
        elif (
            simulation_bundle.analysis_plan.fidelity_level == "full_robustness_certification"
            and robustness.certification_status == "robustness_failed"
        ):
            feasibility = "robustness_fail"
        result = VerificationResult(
            result_id=f"vres_{stable_hash(f'{simulation_bundle.simulation_id}|{simulation_bundle.candidate_id}')[:12]}",
            candidate_id=simulation_bundle.candidate_id,
            executed_fidelity=simulation_bundle.execution_profile.resolved_fidelity,
            backend_signature=f"{simulation_bundle.backend_binding.backend}:{simulation_bundle.backend_binding.backend_version}",
            simulation_provenance=simulation_bundle.simulation_provenance,
            validation_status=physical_validation,
            execution_profile=simulation_bundle.execution_profile,
            measurement_report=measurement_report,
            constraint_assessment=assessments,
            feasibility_status=feasibility,
            failure_attribution=failure,
            robustness_summary=robustness,
            diagnostic_features={
                "analysis_count": len(measurement_report.executed_analyses),
                "netlist_ready": simulation_bundle.netlist_instance.render_status == "ready",
                "critical_failure_count": sum(1 for item in assessments if not item.is_satisfied and item.severity == "critical"),
                "truth_level": physical_validation.truth_level,
                "validation_state": physical_validation.validity_state,
            },
            artifact_refs=[netlist_artifact, measurement_artifact, verification_artifact, *[record.artifact_id for record in simulation_bundle.artifact_registry.records]],
            calibration_payload=calibration_feedback,
            planner_feedback=planner_feedback,
            completion_status=completion_status,
        )
        result_validation = validate_verification_result(result)
        simulation_bundle.validation_status = result_validation
        verification_stats = build_verification_stats_record(
            simulation_bundle,
            simulation_request,
            result,
        )
        return SimulationExecutionResponse(
            simulation_bundle=simulation_bundle,
            simulation_request=simulation_request,
            backend_report=backend_report,
            verification_result=result,
            verification_stats=verification_stats,
        )

    def verify_candidate(
        self,
        candidate_id: str,
        *,
        fidelity_level: str = "quick_truth",
        backend_preference: str = "ngspice",
        escalation_reason: str = "planner_requested_truth_verification",
        model_binding_overrides: dict[str, float | int | str | bool] | None = None,
        paper_mode: bool = False,
    ) -> SimulationExecutionResponse:
        """Formal fifth-layer public entry for candidate verification."""

        compiled: SimulationCompileResponse = compile_simulation_bundle(
            self.task,
            self.planning_bundle,
            self.search_state,
            candidate_id,
            fidelity_level=fidelity_level,
            backend_preference=backend_preference,
            escalation_reason=escalation_reason,
            model_binding_overrides=model_binding_overrides,
            paper_mode=paper_mode,
        )
        if compiled.simulation_bundle is None:
            raise ValueError("simulation bundle failed to compile")
        request = build_simulation_request(
            self.task,
            self.planning_bundle,
            self.search_state,
            candidate_id,
            fidelity_level=fidelity_level,
            backend_preference=backend_preference,
            escalation_reason=escalation_reason,
            model_binding_overrides=model_binding_overrides,
        )
        request = request.model_copy(update={"model_binding": compiled.simulation_bundle.model_binding})
        return self._verify_compiled_bundle(compiled.simulation_bundle, request)
