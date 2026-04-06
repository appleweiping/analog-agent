"""Service layer for the world model bundle."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    CandidateRanking,
    CalibrationUpdateResponse,
    ConstraintObservation,
    DesignAction,
    FeasibilityPrediction,
    HistoryContext,
    HistoryEntry,
    LocalPatchRecord,
    MetricsPrediction,
    RankedCandidate,
    RolloutResponse,
    RolloutStep,
    SimulationValueEstimate,
    TransitionPrediction,
    TruthCalibrationRecord,
    TrustAssessment,
    UsableRegion,
    WorldModelBundle,
    WorldState,
)
from libs.utils.hashing import stable_hash
from libs.world_model.feature_projection import build_metric_estimates, evaluate_constraints, project_metrics
from libs.world_model.state_builder import build_world_state
from libs.world_model.validation import validate_design_action, validate_world_state


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class WorldModelService:
    """Formal service façade around a compiled world-model bundle."""

    def __init__(self, bundle: WorldModelBundle, task: DesignTask) -> None:
        self.bundle = bundle
        self.task = task

    def validate_state(self, state: WorldState):
        return validate_world_state(self.bundle, self.task, state)

    def _trust_assessment(
        self,
        state: WorldState,
        constraints: list[ConstraintObservation],
        *,
        transition_confidence: float | None = None,
    ) -> TrustAssessment:
        boundary_risk = max((1.0 / (1.0 + abs(item.margin)) for item in constraints), default=0.0)
        uncertainty = _clip(
            0.6 * state.uncertainty_context.epistemic_score + 0.4 * state.uncertainty_context.aleatoric_score,
            0.0,
            1.0,
        )
        confidence = transition_confidence if transition_confidence is not None else _clip(1.0 - (0.65 * uncertainty + 0.35 * state.uncertainty_context.ood_score), 0.0, 1.0)
        reasons: list[str] = []
        must_escalate = False
        hard_block = False

        if state.uncertainty_context.ood_score >= self.bundle.trust_policy.screening_threshold.max_ood_score:
            reasons.append("ood_risk_high")
        if uncertainty >= self.bundle.trust_policy.screening_threshold.max_uncertainty_score:
            reasons.append("uncertainty_high")
        if any(item.constraint_group == "stability" and item.margin < 3.0 for item in constraints):
            reasons.append("stability_margin_near_boundary")
            must_escalate = True
        if state.provenance.state_origin == "model_rollout":
            reasons.append("rollout_generated_state")

        if confidence >= self.bundle.trust_policy.rollout_threshold.min_confidence and state.uncertainty_context.ood_score <= self.bundle.trust_policy.rollout_threshold.max_ood_score and uncertainty <= self.bundle.trust_policy.rollout_threshold.max_uncertainty_score:
            level = "high"
            tier = "rollout_ready"
        elif confidence >= self.bundle.trust_policy.ranking_threshold.min_confidence and state.uncertainty_context.ood_score <= self.bundle.trust_policy.ranking_threshold.max_ood_score and uncertainty <= self.bundle.trust_policy.ranking_threshold.max_uncertainty_score:
            level = "medium"
            tier = "ranking_ready"
        elif confidence >= self.bundle.trust_policy.screening_threshold.min_confidence:
            level = "low"
            tier = "screening_only"
        else:
            level = "blocked"
            tier = "hard_block"
            hard_block = True

        if must_escalate and tier != "hard_block":
            tier = "must_escalate"
            level = "low" if level == "medium" else level
        return TrustAssessment(
            trust_level=level,
            service_tier=tier,
            confidence=confidence,
            uncertainty_score=uncertainty,
            ood_score=state.uncertainty_context.ood_score,
            must_escalate=must_escalate,
            hard_block=hard_block,
            reasons=reasons,
        )

    def predict_metrics(self, state: WorldState) -> MetricsPrediction:
        validation = self.validate_state(state)
        if not validation.is_valid:
            raise ValueError(f"invalid world state: {[issue.message for issue in validation.errors]}")
        metric_values, auxiliary = project_metrics(self.task, state)
        trust = self._trust_assessment(state, evaluate_constraints(self.task, metric_values))
        estimates = build_metric_estimates(metric_values, trust.uncertainty_score, trust.confidence)
        return MetricsPrediction(
            state_id=state.state_id,
            task_id=state.task_id,
            metrics=estimates,
            auxiliary_features=auxiliary,
            trust_assessment=trust,
        )

    def predict_feasibility(self, state: WorldState) -> FeasibilityPrediction:
        metric_prediction = self.predict_metrics(state)
        constraints = evaluate_constraints(self.task, {metric.metric: metric.value for metric in metric_prediction.metrics})
        overall = 1.0
        failure_reasons: list[str] = []
        for item in constraints:
            overall *= max(0.05, item.satisfied_probability)
            if item.margin < 0.0:
                failure_reasons.append(item.constraint_name)
        trust = self._trust_assessment(state, constraints)
        return FeasibilityPrediction(
            state_id=state.state_id,
            task_id=state.task_id,
            overall_feasibility=_clip(overall, 0.0, 1.0),
            per_group_constraints=constraints,
            most_likely_failure_reasons=failure_reasons,
            confidence=trust.confidence,
            trust_assessment=trust,
        )

    def _apply_action(self, state: WorldState, action: DesignAction) -> dict[str, float | int | str | bool]:
        validation = validate_design_action(self.bundle, self.task, state, action)
        if not validation.is_valid:
            raise ValueError(f"invalid design action: {[issue.message for issue in validation.errors]}")

        values = {parameter.variable_name: parameter.value for parameter in state.parameter_state}
        targets = action.action_target.variable_names
        if action.action_operator == "freeze":
            return values
        if action.action_operator == "unfreeze":
            return values
        if action.action_operator == "promote_fidelity":
            return values
        if action.action_operator == "drop_candidate":
            return values

        for target in targets:
            current = values[target]
            if action.action_operator == "set":
                values[target] = action.action_payload.get("value", current)
            elif action.action_operator == "shift":
                values[target] = float(current) + float(action.action_payload.get("delta", 0.0))
            elif action.action_operator == "scale":
                values[target] = float(current) * float(action.action_payload.get("factor", 1.0))
            elif action.action_operator == "swap":
                values[target] = action.action_payload.get("value", current)
        return values

    def predict_transition(self, state: WorldState, action: DesignAction) -> TransitionPrediction:
        updated_parameters = self._apply_action(state, action)
        next_fidelity = state.evaluation_context.analysis_fidelity
        if action.action_operator == "promote_fidelity":
            next_fidelity = str(action.action_payload.get("target_fidelity", "partial_simulation"))
        provisional_next_state = build_world_state(
            self.task,
            updated_parameters,
            corner=state.environment_state.corner,
            temperature_c=state.environment_state.temperature_c,
            analysis_fidelity=next_fidelity,
            analysis_intent=state.evaluation_context.analysis_intent,
            output_load_ohm=state.environment_state.output_load_ohm,
            provenance_type="model_rollout",
            artifact_refs=state.provenance.artifact_refs,
            recent_actions=[*state.history_context.recent_actions,],
        )
        predicted_metrics = self.predict_metrics(provisional_next_state)
        predicted_feasibility = self.predict_feasibility(provisional_next_state)
        prior_metrics = {metric.metric: metric.value for metric in state.performance_observation}
        next_metrics = {metric.metric: metric.value for metric in predicted_metrics.metrics}
        metric_deltas = {metric: value - prior_metrics.get(metric, 0.0) for metric, value in next_metrics.items()}
        margin_deltas = {
            item.constraint_group: item.margin - next((previous.margin for previous in state.constraint_observation if previous.constraint_group == item.constraint_group), 0.0)
            for item in predicted_feasibility.per_group_constraints
        }
        op_deltas = {
            key: value - state.operating_point_state.gm.get(key, 0.0)
            for key, value in provisional_next_state.operating_point_state.gm.items()
        }
        history_entry = HistoryEntry(
            action_id=action.action_id,
            metric_deltas={metric: round(delta, 6) for metric, delta in metric_deltas.items()},
            failure_modes=predicted_feasibility.most_likely_failure_reasons,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        next_state = provisional_next_state.model_copy(
            update={
                "history_context": HistoryContext(
                    recent_actions=[*state.history_context.recent_actions, history_entry],
                    trajectory_depth=state.history_context.trajectory_depth + 1,
                    last_outcome="predicted_transition",
                )
            }
        )
        transition_signature = stable_hash(f"{state.state_id}|{action.action_id}|{next_state.state_id}")
        return TransitionPrediction(
            transition_id=f"transition_{transition_signature[:12]}",
            task_id=state.task_id,
            next_state=next_state,
            delta_features={
                "metric_deltas": metric_deltas,
                "margin_deltas": margin_deltas,
                "operating_point_deltas": op_deltas,
            },
            predicted_metrics=predicted_metrics.metrics,
            predicted_constraints=predicted_feasibility.per_group_constraints,
            analysis_fidelity=next_state.evaluation_context.analysis_fidelity,
            trust_assessment=predicted_feasibility.trust_assessment,
        )

    def estimate_simulation_value(self, state: WorldState) -> SimulationValueEstimate:
        feasibility = self.predict_feasibility(state)
        boundary_risk = max((1.0 / (1.0 + abs(item.margin)) for item in feasibility.per_group_constraints), default=0.0)
        value = _clip(0.4 * feasibility.overall_feasibility + 0.35 * feasibility.trust_assessment.uncertainty_score + 0.25 * boundary_risk, 0.0, 1.0)
        if feasibility.trust_assessment.must_escalate or value >= 0.75:
            decision = "prioritize"
            reasons = ["high_information_gain", *feasibility.trust_assessment.reasons]
        elif value >= 0.45:
            decision = "simulate"
            reasons = ["boundary_candidate", *feasibility.trust_assessment.reasons]
        else:
            decision = "defer"
            reasons = ["screening_sufficient"]
        return SimulationValueEstimate(
            state_id=state.state_id,
            estimated_value=value,
            decision=decision,
            reasons=reasons,
            trust_assessment=feasibility.trust_assessment,
        )

    def rank_candidates(self, states: list[WorldState]) -> CandidateRanking:
        ranked: list[RankedCandidate] = []
        for state in states:
            metrics = self.predict_metrics(state)
            feasibility = self.predict_feasibility(state)
            score = feasibility.overall_feasibility * 2.0 - feasibility.trust_assessment.uncertainty_score
            for term in self.task.objective.terms:
                metric_value = next((item.value for item in metrics.metrics if item.metric == term.metric), 0.0)
                score += (metric_value / max(abs(metric_value), 1.0)) * term.weight * (1.0 if term.direction == "maximize" else -1.0)
            ranked.append(
                RankedCandidate(
                    state_id=state.state_id,
                    score=round(score, 6),
                    feasible_probability=feasibility.overall_feasibility,
                    service_tier=feasibility.trust_assessment.service_tier,
                    recommended_action="simulate" if feasibility.trust_assessment.must_escalate else "keep",
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        trust = ranked and self.predict_feasibility(next(state for state in states if state.state_id == ranked[0].state_id)).trust_assessment
        return CandidateRanking(
            ranked_candidates=ranked,
            recommended_threshold=ranked[0].score if ranked else 0.0,
            trust_assessment=trust or TrustAssessment(
                trust_level="blocked",
                service_tier="hard_block",
                confidence=0.0,
                uncertainty_score=1.0,
                ood_score=1.0,
                must_escalate=False,
                hard_block=True,
                reasons=["no_candidates"],
            ),
        )

    def rollout(self, initial_state: WorldState, actions: list[DesignAction], horizon: int | None = None) -> RolloutResponse:
        horizon = horizon or len(actions)
        current = initial_state
        steps: list[RolloutStep] = []
        for index, action in enumerate(actions[:horizon]):
            transition = self.predict_transition(current, action)
            sim_value = self.estimate_simulation_value(transition.next_state)
            steps.append(RolloutStep(step_index=index, action=action, transition=transition, simulation_value=sim_value))
            current = transition.next_state
        terminal_trust = self.predict_feasibility(current).trust_assessment if steps else self.predict_feasibility(initial_state).trust_assessment
        return RolloutResponse(
            initial_state_id=initial_state.state_id,
            horizon=horizon,
            steps=steps,
            terminal_state=current,
            trust_assessment=terminal_trust,
        )

    def calibrate_with_truth(self, state: WorldState, truth: TruthCalibrationRecord) -> CalibrationUpdateResponse:
        prediction = self.predict_metrics(state)
        predicted_by_metric = {metric.metric: metric.value for metric in prediction.metrics}
        updated_metric_summaries = list(self.bundle.calibration_state.per_metric_error_summary)
        metric_index = {item.metric: position for position, item in enumerate(updated_metric_summaries)}

        for truth_metric in truth.metrics:
            error = abs(predicted_by_metric.get(truth_metric.metric, truth_metric.value) - truth_metric.value)
            relative = error / max(abs(truth_metric.value), 1e-12)
            if truth_metric.metric in metric_index:
                current = updated_metric_summaries[metric_index[truth_metric.metric]]
                updated_metric_summaries[metric_index[truth_metric.metric]] = current.model_copy(
                    update={
                        "mae": round((current.mae * current.sample_count + error) / (current.sample_count + 1), 6),
                        "relative_error": round((current.relative_error * current.sample_count + relative) / (current.sample_count + 1), 6),
                        "sample_count": current.sample_count + 1,
                    }
                )
            else:
                from libs.schema.world_model import MetricErrorSummary

                updated_metric_summaries.append(
                    MetricErrorSummary(
                        metric=truth_metric.metric,
                        mae=round(error, 6),
                        relative_error=round(relative, 6),
                        rank_correlation=1.0 if error < max(abs(truth_metric.value) * 0.1, 1e-6) else 0.5,
                        boundary_error=round(error, 6),
                        sample_count=1,
                    )
                )

        updated_ood = self.bundle.calibration_state.ood_statistics.model_copy(
            update={
                "query_count": self.bundle.calibration_state.ood_statistics.query_count + 1,
                "high_risk_count": self.bundle.calibration_state.ood_statistics.high_risk_count + (1 if prediction.trust_assessment.ood_score > 0.6 else 0),
                "high_risk_rate": round(
                    (
                        self.bundle.calibration_state.ood_statistics.high_risk_count
                        + (1 if prediction.trust_assessment.ood_score > 0.6 else 0)
                    )
                    / max(1, self.bundle.calibration_state.ood_statistics.query_count + 1),
                    4,
                ),
            }
        )
        patch = LocalPatchRecord(
            patch_id=f"patch_{stable_hash(state.state_id + truth.timestamp)[:10]}",
            reason="calibrate_with_truth",
            affected_metrics=[item.metric for item in truth.metrics],
            applied_at=datetime.now(timezone.utc).isoformat(),
        )
        updated_regions = list(self.bundle.calibration_state.usable_regions)
        if updated_regions:
            updated_regions[0] = updated_regions[0].model_copy(update={"readiness": "rollout_ready" if updated_ood.high_risk_rate < 0.4 else "screening"})
        else:
            updated_regions = [
                UsableRegion(
                    circuit_family=self.bundle.supported_circuit_families[0],
                    template_id=self.task.topology.template_id,
                    parameter_ranges={},
                    evaluation_intents=[state.evaluation_context.analysis_intent],
                    readiness="screening",
                )
            ]
        new_calibration_state = self.bundle.calibration_state.model_copy(
            update={
                "last_calibrated_at": truth.timestamp,
                "reference_simulator_signature": truth.simulator_signature,
                "per_metric_error_summary": updated_metric_summaries,
                "ood_statistics": updated_ood,
                "local_patch_history": [*self.bundle.calibration_state.local_patch_history, patch],
                "usable_regions": updated_regions,
            }
        )
        updated_bundle = self.bundle.model_copy(update={"calibration_state": new_calibration_state})
        self.bundle = updated_bundle
        trust = self.predict_feasibility(state).trust_assessment
        return CalibrationUpdateResponse(
            updated_bundle=updated_bundle,
            updated_metrics=updated_metric_summaries,
            updated_usable_regions=updated_regions,
            trust_assessment=trust,
        )
