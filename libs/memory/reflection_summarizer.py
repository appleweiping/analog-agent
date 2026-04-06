"""Reflection generation for the memory and reflection layer."""

from __future__ import annotations

from libs.schema.memory import (
    ConfidenceAssessment,
    CounterfactualHypothesis,
    DiagnosisSummary,
    PatternMemoryRecord,
    PolicyUpdateRecommendation,
    ReflectionReport,
)
from libs.utils.hashing import stable_hash


def build_reflection_report(task_id: str, episode_record, relevant_patterns: list[PatternMemoryRecord]) -> ReflectionReport:
    """Generate a formal reflection report from one episode and mined patterns."""

    failure_findings = [*episode_record.dominant_failure_modes]
    success_findings = [sequence.action_families[0] for sequence in episode_record.effective_action_sequences if sequence.action_families]
    pattern_ids = [pattern.pattern_id for pattern in relevant_patterns]
    evidence_refs = [reference.evidence_id for reference in episode_record.evidence_refs]

    policy_updates = [
        PolicyUpdateRecommendation(
            target_layer="layer4",
            update_type="search_adjustment",
            payload={"focus": episode_record.dominant_failure_modes[0] if episode_record.dominant_failure_modes else "retain_diversity"},
            evidence_refs=evidence_refs[:2],
        )
    ]
    if episode_record.world_model_behavior_summary.disagreement_flags:
        policy_updates.append(
            PolicyUpdateRecommendation(
                target_layer="layer3",
                update_type="trust_adjustment",
                payload={"mode": "tighten", "reason": episode_record.world_model_behavior_summary.disagreement_flags[0]},
                evidence_refs=evidence_refs[:2],
            )
        )
    if episode_record.final_outcome.outcome_status != "verified_success":
        policy_updates.append(
            PolicyUpdateRecommendation(
                target_layer="layer5",
                update_type="validation_focus",
                payload={"target": episode_record.dominant_failure_modes[0] if episode_record.dominant_failure_modes else "boundary_constraints"},
                evidence_refs=evidence_refs[:2],
            )
        )
    else:
        policy_updates.append(
            PolicyUpdateRecommendation(
                target_layer="layer2",
                update_type="initialization_hint",
                payload={"source": "memory_episode_seed", "candidate_id": episode_record.final_outcome.best_candidate_id or "unknown"},
                evidence_refs=evidence_refs[:2],
            )
        )

    counterfactuals = [
        CounterfactualHypothesis(
            hypothesis_id=f"cf_{stable_hash(episode_record.episode_memory_id + str(index))[:12]}",
            premise=f"if {finding} had been handled earlier",
            expected_effect="improved phase progression and lower simulation waste",
            confidence=min(0.9, episode_record.confidence_score),
            evidence_refs=evidence_refs[:2],
        )
        for index, finding in enumerate(failure_findings[:2])
    ] or [
        CounterfactualHypothesis(
            hypothesis_id=f"cf_{stable_hash(episode_record.episode_memory_id)[:12]}",
            premise="if the strongest action sequence were replayed earlier",
            expected_effect="faster convergence to feasible regions",
            confidence=episode_record.confidence_score,
            evidence_refs=evidence_refs[:2],
        )
    ]

    return ReflectionReport(
        reflection_id=f"reflection_{stable_hash(task_id + episode_record.episode_memory_id)[:12]}",
        task_id=task_id,
        episode_scope=[episode_record.episode_memory_id],
        reflection_scope="episode",
        failure_synthesis=DiagnosisSummary(
            key_findings=failure_findings,
            dominant_patterns=pattern_ids,
            evidence_refs=evidence_refs,
        ),
        success_synthesis=DiagnosisSummary(
            key_findings=success_findings,
            dominant_patterns=pattern_ids,
            evidence_refs=evidence_refs,
        ),
        search_diagnosis=DiagnosisSummary(
            key_findings=[episode_record.search_summary.final_phase, f"trace_length={episode_record.search_summary.trace_length}"],
            dominant_patterns=pattern_ids,
            evidence_refs=evidence_refs,
        ),
        world_model_diagnosis=DiagnosisSummary(
            key_findings=episode_record.world_model_behavior_summary.disagreement_flags or ["trust_behavior_nominal"],
            dominant_patterns=[pattern.pattern_id for pattern in relevant_patterns if pattern.pattern_type == "calibration_pattern"],
            evidence_refs=evidence_refs,
        ),
        simulation_diagnosis=DiagnosisSummary(
            key_findings=episode_record.final_outcome.failure_classes or [episode_record.final_outcome.outcome_status],
            dominant_patterns=[pattern.pattern_id for pattern in relevant_patterns if pattern.pattern_type in {"failure_pattern", "robustness_pattern"}],
            evidence_refs=evidence_refs,
        ),
        counterfactual_hypotheses=counterfactuals,
        recommended_policy_updates=policy_updates,
        confidence_assessment=ConfidenceAssessment(
            evidence_count=len(evidence_refs),
            confidence_level=min(0.95, 0.55 + 0.05 * len(pattern_ids) + 0.1 * (1 if episode_record.final_outcome.outcome_status == "verified_success" else 0)),
            uncertainty_notes=["episode-scoped reflection", "advisory only"],
        ),
        evidence_refs=evidence_refs,
        provenance=["episode_consolidation", "pattern_mining", "reflection_engine"],
    )
