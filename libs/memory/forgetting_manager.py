"""Forgetting and decay helpers for the memory layer."""

from __future__ import annotations

from libs.schema.memory import MemoryBundle


def apply_forgetting(bundle: MemoryBundle) -> MemoryBundle:
    """Apply bounded growth, freshness decay, and record forgetting."""

    max_episodes = bundle.forgetting_policy.max_episode_records
    max_patterns = bundle.forgetting_policy.max_pattern_records

    episode_records = list(bundle.episode_records)[-max_episodes:]
    pattern_records = sorted(
        bundle.pattern_records,
        key=lambda record: (record.governance_state == "active", record.supporting_evidence_count, record.confidence_level),
    )[-max_patterns:]

    latest_index = len(episode_records)
    updated_patterns = []
    for pattern in pattern_records:
        gap = max(0, latest_index - pattern.supporting_evidence_count)
        freshness_state = pattern.freshness_state
        confidence = pattern.confidence_level
        governance_state = pattern.governance_state
        if gap >= bundle.forgetting_policy.expire_after_episode_gap:
            freshness_state = "expired"
            governance_state = "forgotten"
            confidence = max(0.0, confidence - 2 * bundle.forgetting_policy.confidence_decay)
        elif gap >= bundle.forgetting_policy.stale_after_episode_gap:
            freshness_state = "stale"
            confidence = max(0.0, confidence - bundle.forgetting_policy.confidence_decay)
        updated_patterns.append(
            pattern.model_copy(
                update={
                    "freshness_state": freshness_state,
                    "governance_state": governance_state,
                    "confidence_level": confidence,
                }
            )
        )

    indexing_state = bundle.indexing_state.model_copy(
        update={
            "episode_count": len(episode_records),
            "pattern_count": len(updated_patterns),
            "reflection_count": len(bundle.reflection_records),
        }
    )
    return bundle.model_copy(
        update={
            "episode_records": episode_records,
            "pattern_records": updated_patterns,
            "indexing_state": indexing_state,
        }
    )
