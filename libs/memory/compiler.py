"""Compiler for the memory and reflection layer."""

from __future__ import annotations

from datetime import datetime, timezone

from libs.memory.validation import validate_memory_bundle
from libs.schema.memory import (
    ConsolidationPolicy,
    FeedbackContract,
    ForgettingPolicy,
    IndexingState,
    MemoryBundle,
    MemoryCompilationReport,
    MemoryCompileResponse,
    MemoryMetadata,
    MemoryScopeDefinition,
    MemoryValidationStatus,
    QualityPolicy,
    RetrievalPolicySpec,
    StoreSchema,
)
from libs.utils.hashing import stable_hash


def compile_memory_bundle() -> MemoryCompileResponse:
    """Compile an empty but fully structured sixth-layer bundle."""

    timestamp = datetime.now(timezone.utc).isoformat()
    bundle = MemoryBundle(
        memory_id=f"mem_{stable_hash(timestamp)[:12]}",
        scope_definition=MemoryScopeDefinition(),
        episode_store_schema=StoreSchema(
            key_field="episode_memory_id",
            required_fields=[
                "episode_memory_id",
                "task_id",
                "task_signature",
                "search_summary",
                "final_outcome",
                "evidence_refs",
                "confidence_score",
            ],
        ),
        pattern_store_schema=StoreSchema(
            key_field="pattern_id",
            required_fields=[
                "pattern_id",
                "pattern_type",
                "applicability_scope",
                "supporting_evidence_count",
                "supporting_episode_refs",
                "confidence_level",
                "governance_state",
            ],
        ),
        reflection_store_schema=StoreSchema(
            key_field="reflection_id",
            required_fields=[
                "reflection_id",
                "task_id",
                "episode_scope",
                "recommended_policy_updates",
                "confidence_assessment",
                "evidence_refs",
            ],
        ),
        retrieval_policy=RetrievalPolicySpec(),
        consolidation_policy=ConsolidationPolicy(),
        quality_policy=QualityPolicy(),
        forgetting_policy=ForgettingPolicy(),
        feedback_contract=FeedbackContract(),
        indexing_state=IndexingState(),
        metadata=MemoryMetadata(
            implementation_version="memory-reflection-v1",
            assumptions=[
                "memory layer emits advisory feedback only",
                "patterns must be evidence backed before activation",
                "governance explicitly controls decay, conflict, and forgetting",
            ],
            provenance=[
                "task_signature_builder",
                "consolidation_engine",
                "pattern_miner",
                "reflection_engine",
                "hybrid_retriever",
                "quality_governor",
                "validation_engine",
            ],
        ),
        validation_status=MemoryValidationStatus(
            is_valid=False,
            errors=[],
            warnings=[],
            unresolved_dependencies=[],
            repair_history=[],
            completeness_score=0.0,
        ),
    )
    validation = validate_memory_bundle(bundle)
    compiled_bundle = bundle.model_copy(update={"validation_status": validation})
    status = "compiled" if validation.is_valid and not validation.warnings else "compiled_with_warnings"
    if validation.errors:
        status = "invalid"
    return MemoryCompileResponse(
        status=status,
        memory_bundle=None if status == "invalid" else compiled_bundle,
        report=MemoryCompilationReport(
            status=status,
            derived_fields=[
                "scope_definition",
                "episode_store_schema",
                "pattern_store_schema",
                "reflection_store_schema",
                "retrieval_policy",
                "consolidation_policy",
                "quality_policy",
                "forgetting_policy",
                "feedback_contract",
                "indexing_state",
                "metadata",
                "validation_status",
            ],
            validation_errors=validation.errors,
            validation_warnings=validation.warnings,
            acceptance_summary={
                "mode_count": len(compiled_bundle.scope_definition.supported_modes),
                "advice_type_count": len(compiled_bundle.feedback_contract.supported_advice_types),
                "pattern_type_count": len(compiled_bundle.scope_definition.supported_pattern_types),
                "completeness_score": validation.completeness_score,
            },
        ),
    )
