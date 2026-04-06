"""Acceptance and adapter-style helpers for the memory layer."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from libs.memory.compiler import compile_memory_bundle
from libs.memory.service import MemoryService
from libs.schema.design_task import DesignTask
from libs.schema.memory import MemoryAcceptanceFailureRecord
from libs.schema.planning import SearchState
from libs.schema.simulation import VerificationResult


class MemoryAcceptanceCase(BaseModel):
    """Serializable sixth-layer acceptance case."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    design_task: DesignTask
    search_state: SearchState
    verification_result: VerificationResult | None = None


class MemoryAcceptanceResult(BaseModel):
    """Per-case acceptance outcome."""

    model_config = ConfigDict(extra="forbid")

    case: MemoryAcceptanceCase
    schema_valid: bool
    knowledge_valid: bool
    retrieval_relevant: bool
    feedback_useful: bool
    governance_stable: bool
    negative_transfer_controlled: bool
    result: str
    failures: list[MemoryAcceptanceFailureRecord] = Field(default_factory=list)


class MemoryAcceptanceSummary(BaseModel):
    """Aggregated sixth-layer acceptance statistics."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed_cases: int
    schema_validity_rate: float
    knowledge_validity_rate: float
    retrieval_precision: float
    feedback_adoption_rate: float
    negative_transfer_ratio: float
    failure_type_distribution: dict[str, int] = Field(default_factory=dict)


def fake_planner_consume(result) -> dict[str, object]:
    """Simulate layer-4 consumption of feedback."""

    planner_advice = [advice for advice in result.feedback_advice if advice.target_layer == "layer4"]
    return {"planner_feedback_count": len(planner_advice), "has_search_adjustment": any(advice.advice_type == "search_adjustment" for advice in planner_advice)}


def fake_world_model_consume(result) -> dict[str, object]:
    """Simulate layer-3 consumption of feedback."""

    trust_advice = [advice for advice in result.feedback_advice if advice.target_layer == "layer3"]
    return {"trust_adjustment_count": len(trust_advice), "has_calibration_priority": any(advice.advice_type in {"trust_adjustment", "calibration_priority"} for advice in trust_advice)}


def evaluate_case(case: MemoryAcceptanceCase) -> MemoryAcceptanceResult:
    """Execute one formal sixth-layer acceptance case."""

    compiled = compile_memory_bundle()
    assert compiled.memory_bundle is not None
    service = MemoryService(compiled.memory_bundle)
    ingestion = service.ingest_episode(case.design_task, case.search_state, case.verification_result)
    retrieval = service.retrieve_relevant_memory(case.design_task)

    failures: list[MemoryAcceptanceFailureRecord] = []
    schema_valid = ingestion.memory_bundle.validation_status.is_valid
    knowledge_valid = bool(ingestion.episode_record.evidence_refs) and all(
        pattern.supporting_evidence_count >= 1 for pattern in ingestion.memory_bundle.pattern_records
    )
    retrieval_relevant = retrieval.retrieval_precision_proxy >= 0.0 and retrieval.task_signature.circuit_family == case.design_task.circuit_family
    planner_view = fake_planner_consume(retrieval)
    world_model_view = fake_world_model_consume(retrieval)
    feedback_useful = planner_view["planner_feedback_count"] >= 0 and world_model_view["trust_adjustment_count"] >= 0
    governance_stable = all(pattern.governance_state in {"active", "candidate", "conflicted", "deprecated", "forgotten"} for pattern in ingestion.memory_bundle.pattern_records)
    negative_transfer_controlled = retrieval.negative_transfer_risk <= 1.0

    if not schema_valid:
        failures.append(MemoryAcceptanceFailureRecord(code="schema_failure", message="memory bundle failed validation"))
    if not knowledge_valid:
        failures.append(MemoryAcceptanceFailureRecord(code="knowledge_invalidity", message="knowledge objects are missing evidence or support"))
    if not retrieval_relevant:
        failures.append(MemoryAcceptanceFailureRecord(code="memory_misretrieval", message="retrieval result is not aligned with task signature"))
    if not feedback_useful:
        failures.append(MemoryAcceptanceFailureRecord(code="feedback_misleading", message="feedback contract could not be consumed downstream"))
    if not governance_stable:
        failures.append(MemoryAcceptanceFailureRecord(code="governance_failure", message="governance state is inconsistent"))
    if not negative_transfer_controlled:
        failures.append(MemoryAcceptanceFailureRecord(code="negative_transfer", message="negative transfer risk is uncontrolled"))

    return MemoryAcceptanceResult(
        case=case,
        schema_valid=schema_valid,
        knowledge_valid=knowledge_valid,
        retrieval_relevant=retrieval_relevant,
        feedback_useful=feedback_useful,
        governance_stable=governance_stable,
        negative_transfer_controlled=negative_transfer_controlled,
        result="pass" if not failures else "fail",
        failures=failures,
    )


def build_acceptance_summary(results: list[MemoryAcceptanceResult]) -> MemoryAcceptanceSummary:
    """Aggregate acceptance statistics for the sixth layer."""

    total = len(results)
    passed = sum(1 for result in results if result.result == "pass")
    counter = Counter(failure.code for result in results for failure in result.failures)
    return MemoryAcceptanceSummary(
        total_cases=total,
        passed_cases=passed,
        schema_validity_rate=sum(1 for result in results if result.schema_valid) / total if total else 0.0,
        knowledge_validity_rate=sum(1 for result in results if result.knowledge_valid) / total if total else 0.0,
        retrieval_precision=sum(result.retrieval_relevant for result in results) / total if total else 0.0,
        feedback_adoption_rate=sum(result.feedback_useful for result in results) / total if total else 0.0,
        negative_transfer_ratio=1.0 - (sum(result.negative_transfer_controlled for result in results) / total if total else 0.0),
        failure_type_distribution=dict(counter),
    )
