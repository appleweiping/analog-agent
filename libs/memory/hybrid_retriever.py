"""Hybrid retrieval for the memory layer."""

from __future__ import annotations

from libs.memory.retrieval_policy import signature_overlap
from libs.schema.memory import RetrievalHit, RetrievalResult, TaskSignature


def retrieve(bundle, task_signature: TaskSignature) -> RetrievalResult:
    """Retrieve relevant episodes, patterns, and reflections for a task signature."""

    episode_hits = []
    for episode in bundle.episode_records:
        overlap = signature_overlap(task_signature, episode.task_signature)
        score = (
            bundle.retrieval_policy.task_signature_weight * overlap
            + bundle.retrieval_policy.evidence_weight * min(1.0, len(episode.evidence_refs) / 5.0)
            + bundle.retrieval_policy.circuit_family_weight * (1.0 if episode.circuit_family == task_signature.circuit_family else 0.0)
            + bundle.retrieval_policy.task_type_weight * (1.0 if episode.task_type == task_signature.task_type else 0.0)
        )
        if score >= bundle.retrieval_policy.minimum_score:
            episode_hits.append(
                RetrievalHit(
                    source_type="episode",
                    source_id=episode.episode_memory_id,
                    score=score,
                    evidence_count=len(episode.evidence_refs),
                    rationale_tags=[episode.circuit_family, episode.task_type],
                )
            )
    pattern_hits = []
    for pattern in bundle.pattern_records:
        family_match = task_signature.circuit_family in pattern.applicability_scope.circuit_families
        task_match = task_signature.task_type in pattern.applicability_scope.task_types
        score = 0.5 * (1.0 if family_match else 0.0) + 0.25 * (1.0 if task_match else 0.0) + 0.25 * pattern.confidence_level
        if score >= bundle.retrieval_policy.minimum_score and pattern.governance_state in {"active", "candidate"}:
            pattern_hits.append(
                RetrievalHit(
                    source_type="pattern",
                    source_id=pattern.pattern_id,
                    score=score,
                    evidence_count=pattern.supporting_evidence_count,
                    rationale_tags=[pattern.pattern_type, *pattern.trigger_signature],
                )
            )
    reflection_hits = []
    for reflection in bundle.reflection_records:
        if any(episode_ref in {record.episode_memory_id for record in bundle.episode_records if record.task_signature.circuit_family == task_signature.circuit_family} for episode_ref in reflection.episode_scope):
            reflection_hits.append(
                RetrievalHit(
                    source_type="reflection",
                    source_id=reflection.reflection_id,
                    score=reflection.confidence_assessment.confidence_level,
                    evidence_count=reflection.confidence_assessment.evidence_count,
                    rationale_tags=list(reflection.provenance),
                )
            )

    episode_hits = sorted(episode_hits, key=lambda item: item.score, reverse=True)[: bundle.retrieval_policy.top_k]
    pattern_hits = sorted(pattern_hits, key=lambda item: item.score, reverse=True)[: bundle.retrieval_policy.top_k]
    reflection_hits = sorted(reflection_hits, key=lambda item: item.score, reverse=True)[: bundle.retrieval_policy.top_k]
    negative_transfer_risk = 1.0 - (episode_hits[0].score if episode_hits else 0.0)
    return RetrievalResult(
        task_signature=task_signature,
        episode_hits=episode_hits,
        pattern_hits=pattern_hits,
        reflection_hits=reflection_hits,
        feedback_advice=[],
        retrieval_precision_proxy=episode_hits[0].score if episode_hits else 0.0,
        negative_transfer_risk=negative_transfer_risk,
    )
