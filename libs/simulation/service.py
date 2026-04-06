"""Formal fifth-layer simulation and verification service."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.planner.candidate_manager import find_candidate
from libs.schema.design_task import DesignTask
from libs.schema.planning import PlanningBundle, SearchState
from libs.schema.simulation import (
    CalibrationFeedback,
    PlannerFeedback,
    SimulationBundle,
    SimulationCompileResponse,
    SimulationExecutionResponse,
    SimulationRequest,
    VerificationResult,
)
from libs.schema.world_model import TruthCalibrationRecord, TruthConstraint, TruthMetric
from libs.simulation.artifact_registry import persist_json_artifact, persist_text_artifact
from libs.simulation.backend_router import execute_bundle, validate_backend
from libs.simulation.compiler import build_simulation_request, compile_simulation_bundle
from libs.simulation.constraint_verifier import verify_constraints
from libs.simulation.failure_analyzer import attribute_failures
from libs.simulation.measurement_extractors import extract_measurement_report
from libs.simulation.robustness_evaluator import certify_robustness
from libs.simulation.validation import validate_verification_result
from libs.utils.hashing import stable_hash


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truth_fidelity(simulation_fidelity: str) -> str:
    mapping = {
        "quick_truth": "quick_screening",
        "focused_validation": "partial_simulation",
        "full_robustness_certification": "full_ground_truth",
        "targeted_failure_analysis": "full_ground_truth",
    }
    return mapping[simulation_fidelity]


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

    def verify_constraints(self, metric_values: dict[str, float], simulation_bundle: SimulationBundle):
        """Verify DesignTask constraints against extracted truth metrics."""

        return verify_constraints(self.task, metric_values, simulation_bundle.verification_policy)

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
            metrics=[TruthMetric(metric=metric.metric, value=metric.value) for metric in measurement_report.measured_metrics],
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
            ],
        )

    def _planner_feedback(self, simulation_bundle: SimulationBundle, assessments, robustness, calibration_feedback: CalibrationFeedback) -> PlannerFeedback:
        all_pass = all(item.is_satisfied for item in assessments)
        robust_pass = robustness.certification_status in {"robust_certified", "partial_robust", "nominal_only"}
        if all_pass and robust_pass and simulation_bundle.analysis_plan.fidelity_level == "full_robustness_certification":
            lifecycle = "verified"
        elif all_pass:
            lifecycle = "needs_more_simulation"
        elif any(abs(item.margin) <= 0.05 for item in assessments):
            lifecycle = "boundary_candidate"
        else:
            lifecycle = "rejected"
        return PlannerFeedback(
            candidate_id=simulation_bundle.candidate_id,
            lifecycle_update=lifecycle,
            strategy_correction=[
                "increase_safety_margin" if calibration_feedback.constraint_disagreement else "retain_current_direction",
                "prefer_high_truth_candidates_for_phase_transition" if lifecycle == "verified" else "revisit_candidate_neighborhood",
            ],
            phase_hint="robustness_verification" if lifecycle == "verified" else self.search_state.phase_state.current_phase,
            trust_alerts=list(calibration_feedback.trust_violation_flags),
            artifact_refs=[record.artifact_id for record in simulation_bundle.artifact_registry.records],
        )

    def execute_compiled_bundle(self, simulation_bundle: SimulationBundle, simulation_request: SimulationRequest) -> SimulationExecutionResponse:
        """Run the full fifth-layer execution pipeline for a precompiled bundle."""

        netlist_artifact = self.realize_netlist(simulation_bundle)
        simulation_bundle, backend_report, parsed_outputs = self.run_simulation(simulation_bundle, simulation_request)
        measurement_report = self.extract_measurements(simulation_bundle, parsed_outputs)
        metric_values = {metric.metric: metric.value for metric in measurement_report.measured_metrics}
        assessments = self.verify_constraints(metric_values, simulation_bundle)
        robustness = self.certify_robustness(simulation_bundle.candidate_id, simulation_bundle)
        completion_status = "success" if all(record.success for record in measurement_report.executed_analyses) else "simulator_failure"
        failure = attribute_failures(
            assessments,
            measurement_report,
            robustness,
            render_ready=simulation_bundle.netlist_instance.render_status == "ready",
            completion_status=completion_status,
        )
        calibration_feedback = self.emit_calibration_feedback(simulation_bundle, measurement_report, assessments)
        planner_feedback = self._planner_feedback(simulation_bundle, assessments, robustness, calibration_feedback)
        simulation_bundle.artifact_registry, verification_artifact = persist_json_artifact(
            simulation_bundle.artifact_registry,
            "verification_report",
            "verification_result.json",
            {
                "candidate_id": simulation_bundle.candidate_id,
                "netlist_artifact": netlist_artifact,
                "assessment_count": len(assessments),
                "robustness_status": robustness.certification_status,
            },
        )
        all_constraints_pass = all(item.is_satisfied for item in assessments)
        feasibility = "feasible_nominal"
        if not all_constraints_pass:
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
            executed_fidelity=simulation_bundle.analysis_plan.fidelity_level,
            backend_signature=f"{simulation_bundle.backend_binding.backend}:{simulation_bundle.backend_binding.backend_version}",
            measurement_report=measurement_report,
            constraint_assessment=assessments,
            feasibility_status=feasibility,
            failure_attribution=failure,
            robustness_summary=robustness,
            diagnostic_features={
                "analysis_count": len(measurement_report.executed_analyses),
                "netlist_ready": simulation_bundle.netlist_instance.render_status == "ready",
                "critical_failure_count": sum(1 for item in assessments if not item.is_satisfied and item.severity == "critical"),
            },
            artifact_refs=[netlist_artifact, verification_artifact, *[record.artifact_id for record in simulation_bundle.artifact_registry.records]],
            calibration_payload=calibration_feedback,
            planner_feedback=planner_feedback,
            completion_status=completion_status,
        )
        result_validation = validate_verification_result(result)
        simulation_bundle.validation_status = result_validation
        return SimulationExecutionResponse(
            simulation_bundle=simulation_bundle,
            simulation_request=simulation_request,
            backend_report=backend_report,
            verification_result=result,
        )

    def execute(
        self,
        candidate_id: str,
        *,
        fidelity_level: str = "focused_validation",
        backend_preference: str = "ngspice",
        escalation_reason: str = "planner_requested_truth_verification",
    ) -> SimulationExecutionResponse:
        """Compile and execute a full fifth-layer request."""

        compiled: SimulationCompileResponse = compile_simulation_bundle(
            self.task,
            self.planning_bundle,
            self.search_state,
            candidate_id,
            fidelity_level=fidelity_level,
            backend_preference=backend_preference,
            escalation_reason=escalation_reason,
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
        )
        return self.execute_compiled_bundle(compiled.simulation_bundle, request)
