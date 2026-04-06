"""Pattern mining for the memory and reflection layer."""

from __future__ import annotations

from collections import defaultdict

from libs.schema.memory import ApplicabilityScope, InterventionRecord, PatternMemoryRecord
from libs.utils.hashing import stable_hash


def mine_patterns(episode_records, *, minimum_support: int) -> list[PatternMemoryRecord]:
    """Mine evidence-backed patterns from consolidated episodes."""

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for episode in episode_records:
        for failure in episode.dominant_failure_modes:
            grouped[("failure_pattern", failure)].append(episode)
        for sequence in episode.effective_action_sequences:
            if sequence.action_families:
                grouped[("success_pattern", sequence.action_families[0])].append(episode)
        grouped[("search_pattern", f"phase::{episode.search_summary.final_phase}")].append(episode)
        if episode.world_model_behavior_summary.disagreement_flags:
            grouped[("calibration_pattern", "trust_disagreement")].append(episode)
        if episode.simulation_budget_profile.budget_pressure >= 0.8:
            grouped[("budget_pattern", "high_budget_pressure")].append(episode)
        if episode.final_outcome.robustness_status == "robust_certified":
            grouped[("robustness_pattern", "robust_success")].append(episode)

    patterns: list[PatternMemoryRecord] = []
    for (pattern_type, trigger), episodes in grouped.items():
        if len(episodes) < minimum_support:
            continue
        family_scope = sorted({episode.circuit_family for episode in episodes})
        task_scope = sorted({episode.task_type for episode in episodes})
        supporting_refs = [episode.episode_memory_id for episode in episodes]
        intervention = []
        anti_patterns = []
        expected_effects = []
        risk_notes = []
        if pattern_type == "failure_pattern":
            intervention = [InterventionRecord(label="validation_focus", action="increase_validation_focus", payload={"target": trigger})]
            anti_patterns = [InterventionRecord(label="avoid_repeat_failure", action="avoid_failure_mode", payload={"failure_mode": trigger})]
            expected_effects = ["earlier_failure_detection", "reduced negative transfer"]
            risk_notes = ["apply only when task signature overlap is high"]
        elif pattern_type == "success_pattern":
            intervention = [InterventionRecord(label="search_adjustment", action="promote_action_family", payload={"action_family": trigger})]
            expected_effects = ["faster feasibility recovery", "improved warm-start direction"]
        elif pattern_type == "calibration_pattern":
            intervention = [InterventionRecord(label="trust_adjustment", action="tighten_world_model_trust", payload={"mode": "must_escalate"})]
            expected_effects = ["lower trust leakage", "better escalation discipline"]
        elif pattern_type == "budget_pattern":
            intervention = [InterventionRecord(label="budget_adjustment", action="front_load_feasibility_screening", payload={"policy": "feasibility_first"})]
            expected_effects = ["lower simulation waste"]
        elif pattern_type == "robustness_pattern":
            intervention = [InterventionRecord(label="initialization_hint", action="reuse_robust_seed_family", payload={"source": "robust_episode_memory"})]
            expected_effects = ["better robustness carry-over"]

        patterns.append(
            PatternMemoryRecord(
                pattern_id=f"pattern_{stable_hash(pattern_type + trigger + '|'.join(supporting_refs))[:12]}",
                pattern_type=pattern_type,
                applicability_scope=ApplicabilityScope(
                    circuit_families=family_scope,
                    task_types=task_scope,
                    topology_modes=sorted({episode.task_signature.design_space_shape[3].split(':', 1)[1] for episode in episodes if len(episode.task_signature.design_space_shape) > 3}),
                    difficulty_bands=sorted({episode.constraint_profile.tightness for episode in episodes}),
                    environment_tags=sorted({tag for episode in episodes for tag in episode.task_signature.environment_profile}),
                ),
                trigger_signature=[trigger],
                context_constraints=sorted({name for episode in episodes for name in episode.constraint_profile.hard_constraint_names}),
                observed_contexts=[episode.search_summary.final_phase for episode in episodes],
                recommended_interventions=intervention,
                anti_patterns=anti_patterns,
                expected_effects=expected_effects,
                risk_notes=risk_notes,
                supporting_evidence_count=len(episodes),
                supporting_episode_refs=supporting_refs,
                confidence_level=min(0.95, 0.45 + 0.1 * len(episodes)),
                freshness_state="fresh",
                activation_policy=[f"minimum_signature_overlap=0.55", f"minimum_support={minimum_support}"],
                governance_state="active",
            )
        )
    return patterns
