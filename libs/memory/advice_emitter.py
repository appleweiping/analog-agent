"""Formal feedback emission for the memory layer."""

from __future__ import annotations

from libs.schema.memory import ApplicabilityScope, FeedbackAdvice
from libs.utils.hashing import stable_hash


def emit_feedback(episode_record, patterns, reflection) -> list[FeedbackAdvice]:
    """Emit advisory feedback for layers 2-5 from patterns and reflection."""

    evidence_refs = [reference.evidence_id for reference in episode_record.evidence_refs]
    applicability_scope = (
        patterns[0].applicability_scope
        if patterns
        else ApplicabilityScope(
            circuit_families=[episode_record.circuit_family],
            task_types=[episode_record.task_type],
            topology_modes=[episode_record.task_signature.design_space_shape[3].split(":", 1)[1]] if len(episode_record.task_signature.design_space_shape) > 3 else [],
            difficulty_bands=[episode_record.constraint_profile.tightness],
            environment_tags=episode_record.task_signature.environment_profile,
        )
    )
    advice: list[FeedbackAdvice] = []
    for index, recommendation in enumerate(reflection.recommended_policy_updates):
        advice_type = recommendation.update_type if recommendation.update_type in {
            "initialization_hint",
            "search_adjustment",
            "trust_adjustment",
            "validation_focus",
            "calibration_priority",
            "budget_adjustment",
        } else "search_adjustment"
        advice.append(
            FeedbackAdvice(
                advice_id=f"advice_{stable_hash(reflection.reflection_id + str(index))[:12]}",
                target_layer=recommendation.target_layer,
                target_scope=episode_record.task_id,
                advice_type=advice_type,
                advice_payload=recommendation.payload,
                applicability_scope=applicability_scope,
                confidence_level=reflection.confidence_assessment.confidence_level,
                priority_level="high" if recommendation.target_layer in {"layer3", "layer4"} else "medium",
                expiry_condition="until_next_cross_episode_consolidation",
                evidence_refs=recommendation.evidence_refs or evidence_refs[:2],
            )
        )
    return advice
