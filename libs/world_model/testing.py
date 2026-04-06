"""Acceptance and adapter-style testing helpers for the world model layer."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_task import DesignTask
from libs.schema.world_model import (
    AcceptanceFailureRecord,
    DesignAction,
    TruthCalibrationRecord,
    WorldModelCompileResponse,
    WorldState,
)
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService


class WorldModelAcceptanceCase(BaseModel):
    """Serializable third-layer acceptance case."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    design_task: DesignTask
    initial_state: WorldState
    actions: list[DesignAction] = Field(default_factory=list)
    truth_record: TruthCalibrationRecord | None = None


class WorldModelAcceptanceResult(BaseModel):
    """Per-case acceptance outcome."""

    model_config = ConfigDict(extra="forbid")

    case: WorldModelAcceptanceCase
    compile_output: WorldModelCompileResponse
    schema_valid: bool
    predictive_valid: bool
    transition_valid: bool
    feasibility_reliable: bool
    uncertainty_trustworthy: bool
    planning_useful: bool
    result: str
    failures: list[AcceptanceFailureRecord] = Field(default_factory=list)


class WorldModelAcceptanceSummary(BaseModel):
    """Aggregated world-model acceptance statistics."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed_cases: int
    schema_validity_rate: float
    predictive_validity_rate: float
    transition_validity_rate: float
    feasibility_reliability_rate: float
    uncertainty_trustworthiness_rate: float
    planning_utility_rate: float
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)


def fake_planner_consume(service: WorldModelService, initial_state: WorldState, actions: list[DesignAction]) -> dict[str, float | int]:
    """Simulate a downstream planner consuming rollout and ranking outputs."""

    rollout = service.rollout(initial_state, actions, horizon=len(actions))
    ranking = service.rank_candidates([initial_state, rollout.terminal_state])
    return {
        "rollout_steps": len(rollout.steps),
        "top_candidate_score": ranking.ranked_candidates[0].score if ranking.ranked_candidates else 0.0,
    }


def fake_simulator_adapter(service: WorldModelService, state: WorldState) -> dict[str, str | float]:
    """Simulate the fifth-layer adapter consuming simulation-value advice."""

    simulation_value = service.estimate_simulation_value(state)
    return {"decision": simulation_value.decision, "estimated_value": simulation_value.estimated_value}


def fake_critic_consume(service: WorldModelService, state: WorldState) -> dict[str, object]:
    """Simulate critic consumption of feasibility and trust outputs."""

    feasibility = service.predict_feasibility(state)
    return {
        "overall_feasibility": feasibility.overall_feasibility,
        "failure_reasons": feasibility.most_likely_failure_reasons,
        "service_tier": feasibility.trust_assessment.service_tier,
    }


def evaluate_case(case: WorldModelAcceptanceCase) -> WorldModelAcceptanceResult:
    """Execute one acceptance case."""

    compile_output = compile_world_model_bundle(case.design_task)
    failures: list[AcceptanceFailureRecord] = []
    schema_valid = compile_output.world_model_bundle is not None
    if not schema_valid:
        failures.append(AcceptanceFailureRecord(code="schema_failure", message="bundle failed to compile"))
        return WorldModelAcceptanceResult(
            case=case,
            compile_output=compile_output,
            schema_valid=False,
            predictive_valid=False,
            transition_valid=False,
            feasibility_reliable=False,
            uncertainty_trustworthy=False,
            planning_useful=False,
            result="fail",
            failures=failures,
        )

    service = WorldModelService(compile_output.world_model_bundle, case.design_task)
    metrics = service.predict_metrics(case.initial_state)
    feasibility = service.predict_feasibility(case.initial_state)
    predictive_valid = bool(metrics.metrics)
    feasibility_reliable = bool(feasibility.per_group_constraints)
    uncertainty_trustworthy = feasibility.trust_assessment.service_tier in {"screening_only", "ranking_ready", "rollout_ready", "must_escalate", "hard_block"}
    transition_valid = True
    if case.actions:
        transition = service.predict_transition(case.initial_state, case.actions[0])
        transition_valid = bool(transition.delta_features.metric_deltas)
    planning_useful = bool(fake_planner_consume(service, case.initial_state, case.actions or [])["top_candidate_score"] >= 0.0)

    if not predictive_valid:
        failures.append(AcceptanceFailureRecord(code="metric_error_excessive", message="metrics head returned no outputs"))
    if not transition_valid:
        failures.append(AcceptanceFailureRecord(code="transition_direction_failure", message="transition head returned no usable deltas"))
    if not feasibility_reliable:
        failures.append(AcceptanceFailureRecord(code="constraint_margin_failure", message="feasibility head returned no constraint observations"))
    if not uncertainty_trustworthy:
        failures.append(AcceptanceFailureRecord(code="uncertainty_miscalibration", message="trust service tier is malformed"))
    if case.truth_record is not None:
        update = service.calibrate_with_truth(case.initial_state, case.truth_record)
        if not update.updated_bundle.calibration_state.local_patch_history:
            failures.append(AcceptanceFailureRecord(code="calibration_update_failure", message="calibration did not record a patch"))

    result = "pass" if not failures else "fail"
    return WorldModelAcceptanceResult(
        case=case,
        compile_output=compile_output,
        schema_valid=schema_valid,
        predictive_valid=predictive_valid,
        transition_valid=transition_valid,
        feasibility_reliable=feasibility_reliable,
        uncertainty_trustworthy=uncertainty_trustworthy,
        planning_useful=planning_useful,
        result=result,
        failures=failures,
    )


def build_acceptance_summary(results: list[WorldModelAcceptanceResult]) -> WorldModelAcceptanceSummary:
    """Aggregate acceptance metrics for the third layer."""

    total = len(results)
    passed = sum(1 for result in results if result.result == "pass")
    counter = Counter(failure.code for result in results for failure in result.failures)
    return WorldModelAcceptanceSummary(
        total_cases=total,
        passed_cases=passed,
        schema_validity_rate=sum(1 for result in results if result.schema_valid) / total if total else 0.0,
        predictive_validity_rate=sum(1 for result in results if result.predictive_valid) / total if total else 0.0,
        transition_validity_rate=sum(1 for result in results if result.transition_valid) / total if total else 0.0,
        feasibility_reliability_rate=sum(1 for result in results if result.feasibility_reliable) / total if total else 0.0,
        uncertainty_trustworthiness_rate=sum(1 for result in results if result.uncertainty_trustworthy) / total if total else 0.0,
        planning_utility_rate=sum(1 for result in results if result.planning_useful) / total if total else 0.0,
        failure_type_distribution=dict(counter),
    )
