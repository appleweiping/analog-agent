"""Governance helpers for long-term memory quality control."""

from __future__ import annotations

from libs.schema.memory import MemoryBundle, PatternMemoryRecord


def apply_quality_governance(bundle: MemoryBundle) -> MemoryBundle:
    """Apply confidence thresholds and conflict handling to patterns and reflections."""

    governed_patterns: list[PatternMemoryRecord] = []
    for pattern in bundle.pattern_records:
        governance_state = pattern.governance_state
        confidence_level = pattern.confidence_level
        if pattern.supporting_evidence_count < bundle.consolidation_policy.minimum_pattern_support:
            governance_state = "candidate"
            confidence_level = min(confidence_level, 0.49)
        elif confidence_level < bundle.quality_policy.minimum_pattern_confidence:
            governance_state = "candidate"
        if any(other.pattern_type == pattern.pattern_type and other.trigger_signature == pattern.trigger_signature and other.pattern_id != pattern.pattern_id for other in bundle.pattern_records):
            governance_state = "conflicted"
        governed_patterns.append(
            pattern.model_copy(update={"governance_state": governance_state, "confidence_level": confidence_level})
        )

    governed_reflections = []
    for reflection in bundle.reflection_records:
        confidence = reflection.confidence_assessment.confidence_level
        if confidence < bundle.quality_policy.minimum_reflection_confidence:
            reflection = reflection.model_copy(
                update={
                    "provenance": [*reflection.provenance, "downgraded_by_quality_governor"],
                }
            )
        governed_reflections.append(reflection)
    return bundle.model_copy(update={"pattern_records": governed_patterns, "reflection_records": governed_reflections})
