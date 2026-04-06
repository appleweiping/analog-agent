"""Service layer for the memory and reflection bundle."""

from __future__ import annotations

from libs.memory.advice_emitter import emit_feedback
from libs.memory.consolidation_engine import consolidate_episode
from libs.memory.forgetting_manager import apply_forgetting
from libs.memory.hybrid_retriever import retrieve
from libs.memory.pattern_miner import mine_patterns
from libs.memory.quality_governor import apply_quality_governance
from libs.memory.reflection_summarizer import build_reflection_report
from libs.memory.task_signature import build_task_signature
from libs.memory.validation import validate_memory_bundle
from libs.schema.design_task import DesignTask
from libs.schema.memory import IngestionResponse, MemoryBundle, RetrievalResult
from libs.schema.planning import SearchState
from libs.schema.simulation import VerificationResult


class MemoryService:
    """Long-term knowledge system for cross-episode improvement."""

    def __init__(self, bundle: MemoryBundle) -> None:
        self.bundle = bundle

    def ingest_episode(
        self,
        task: DesignTask,
        search_state: SearchState,
        verification: VerificationResult | None = None,
    ) -> IngestionResponse:
        """Ingest one upstream episode and update all sixth-layer stores."""

        episode_record = consolidate_episode(task, search_state, verification)
        episode_records = [*self.bundle.episode_records, episode_record]
        candidate_bundle = self.bundle.model_copy(
            update={
                "episode_records": episode_records,
                "indexing_state": self.bundle.indexing_state.model_copy(
                    update={
                        "indexed_task_signatures": [*self.bundle.indexing_state.indexed_task_signatures, episode_record.task_signature.difficulty_profile_hash],
                        "episode_count": len(episode_records),
                        "last_consolidated_episode_id": episode_record.episode_memory_id,
                    }
                ),
            }
        )
        patterns = mine_patterns(candidate_bundle.episode_records, minimum_support=candidate_bundle.consolidation_policy.minimum_pattern_support)
        reflection = build_reflection_report(task.task_id, episode_record, patterns)
        feedback = emit_feedback(episode_record, patterns, reflection)
        updated_bundle = candidate_bundle.model_copy(
            update={
                "pattern_records": patterns,
                "reflection_records": [*candidate_bundle.reflection_records, reflection],
            }
        )
        updated_bundle = apply_quality_governance(updated_bundle)
        updated_bundle = apply_forgetting(updated_bundle)
        validation = validate_memory_bundle(updated_bundle)
        updated_bundle = updated_bundle.model_copy(update={"validation_status": validation})
        self.bundle = updated_bundle
        return IngestionResponse(
            memory_bundle=updated_bundle,
            episode_record=episode_record,
            new_patterns=patterns,
            reflection_report=reflection,
            emitted_feedback=feedback,
        )

    def retrieve_relevant_memory(self, task: DesignTask) -> RetrievalResult:
        """Retrieve task-relevant memory and advisory feedback."""

        task_signature = build_task_signature(task)
        result = retrieve(self.bundle, task_signature)
        advice = []
        reflection_lookup = {reflection.reflection_id: reflection for reflection in self.bundle.reflection_records}
        pattern_lookup = {pattern.pattern_id: pattern for pattern in self.bundle.pattern_records}
        episode_lookup = {record.episode_memory_id: record for record in self.bundle.episode_records}
        for hit in result.pattern_hits:
            pattern = pattern_lookup[hit.source_id]
            for episode_ref in pattern.supporting_episode_refs[:1]:
                episode = episode_lookup.get(episode_ref)
                if episode is None or not self.bundle.reflection_records:
                    continue
                related_reflection = next((reflection for reflection in self.bundle.reflection_records if episode_ref in reflection.episode_scope), None)
                if related_reflection is not None:
                    advice.extend(emit_feedback(episode, [pattern], related_reflection))
        for hit in result.reflection_hits:
            reflection = reflection_lookup[hit.source_id]
            episode = next((episode_lookup[ref] for ref in reflection.episode_scope if ref in episode_lookup), None)
            if episode is not None:
                advice.extend(emit_feedback(episode, [], reflection))
        return result.model_copy(update={"feedback_advice": advice})
