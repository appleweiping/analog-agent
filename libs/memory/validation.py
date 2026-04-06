"""Validation helpers for the memory and reflection layer."""

from __future__ import annotations

from libs.schema.memory import (
    FeedbackAdvice,
    MemoryBundle,
    MemoryValidationIssue,
    MemoryValidationStatus,
    PatternMemoryRecord,
    ReflectionReport,
)


def _issue(code: str, path: str, message: str, severity: str) -> MemoryValidationIssue:
    return MemoryValidationIssue(code=code, path=path, message=message, severity=severity)


def validate_pattern_record(pattern: PatternMemoryRecord, *, minimum_support: int) -> list[MemoryValidationIssue]:
    """Validate one mined pattern."""

    issues: list[MemoryValidationIssue] = []
    if pattern.supporting_evidence_count < minimum_support:
        issues.append(
            _issue(
                "pattern_evidence_threshold_failure",
                f"pattern_records[{pattern.pattern_id}]",
                "pattern support is below the configured evidence threshold",
                "warning",
            )
        )
    if not pattern.supporting_episode_refs:
        issues.append(
            _issue(
                "evidence_traceability_failure",
                f"pattern_records[{pattern.pattern_id}].supporting_episode_refs",
                "pattern is missing supporting episode references",
                "error",
            )
        )
    return issues


def validate_reflection_report(reflection: ReflectionReport) -> list[MemoryValidationIssue]:
    """Validate one reflection report."""

    issues: list[MemoryValidationIssue] = []
    if not reflection.recommended_policy_updates:
        issues.append(
            _issue(
                "reflection_policy_mapping_failure",
                f"reflection_records[{reflection.reflection_id}]",
                "reflection does not map to formal policy updates",
                "warning",
            )
        )
    if reflection.confidence_assessment.evidence_count <= 0 or not reflection.evidence_refs:
        issues.append(
            _issue(
                "evidence_traceability_failure",
                f"reflection_records[{reflection.reflection_id}]",
                "reflection is not evidence backed",
                "error",
            )
        )
    return issues


def validate_feedback_advice(advice: FeedbackAdvice) -> list[MemoryValidationIssue]:
    """Validate one emitted feedback advice."""

    issues: list[MemoryValidationIssue] = []
    if not advice.evidence_refs:
        issues.append(
            _issue(
                "evidence_traceability_failure",
                f"feedback[{advice.advice_id}]",
                "feedback advice is missing evidence references",
                "error",
            )
        )
    return issues


def validate_memory_bundle(bundle: MemoryBundle) -> MemoryValidationStatus:
    """Validate cross-object consistency for MemoryBundle."""

    errors: list[MemoryValidationIssue] = []
    warnings: list[MemoryValidationIssue] = []

    episode_ids = {record.episode_memory_id for record in bundle.episode_records}
    for episode in bundle.episode_records:
        if not episode.evidence_refs:
            errors.append(
                _issue(
                    "evidence_traceability_failure",
                    f"episode_records[{episode.episode_memory_id}]",
                    "episode memory record must carry evidence references",
                    "error",
                )
            )

    for pattern in bundle.pattern_records:
        for issue in validate_pattern_record(pattern, minimum_support=bundle.consolidation_policy.minimum_pattern_support):
            (errors if issue.severity == "error" else warnings).append(issue)
        missing = [ref for ref in pattern.supporting_episode_refs if ref not in episode_ids]
        if missing:
            errors.append(
                _issue(
                    "reference_consistency_failure",
                    f"pattern_records[{pattern.pattern_id}].supporting_episode_refs",
                    f"unknown episode references: {missing}",
                    "error",
                )
            )

    for reflection in bundle.reflection_records:
        for issue in validate_reflection_report(reflection):
            (errors if issue.severity == "error" else warnings).append(issue)
        missing = [ref for ref in reflection.episode_scope if ref not in episode_ids]
        if missing:
            errors.append(
                _issue(
                    "reference_consistency_failure",
                    f"reflection_records[{reflection.reflection_id}].episode_scope",
                    f"unknown episode references: {missing}",
                    "error",
                )
            )

    indexed = set(bundle.indexing_state.indexed_task_signatures)
    observed = {record.task_signature.difficulty_profile_hash for record in bundle.episode_records}
    if observed and not indexed:
        warnings.append(
            _issue(
                "cross_object_consistency_failure",
                "indexing_state.indexed_task_signatures",
                "indexing state is empty while episode records exist",
                "warning",
            )
        )
    if indexed and bundle.indexing_state.episode_count != len(bundle.episode_records):
        warnings.append(
            _issue(
                "cross_object_consistency_failure",
                "indexing_state.episode_count",
                "episode_count does not match episode_records length",
                "warning",
            )
        )

    completeness_components = [
        bool(bundle.scope_definition),
        bool(bundle.retrieval_policy),
        bool(bundle.consolidation_policy),
        bool(bundle.feedback_contract),
        bool(bundle.metadata),
        bool(bundle.episode_records),
        bool(bundle.pattern_records),
        bool(bundle.reflection_records),
    ]
    completeness_score = sum(1 for value in completeness_components if value) / len(completeness_components)
    return MemoryValidationStatus(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        unresolved_dependencies=[],
        repair_history=[],
        completeness_score=completeness_score,
    )
