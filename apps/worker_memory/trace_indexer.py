"""Worker-facing indexing helper."""

from __future__ import annotations

from libs.schema.memory import EpisodeMemoryRecord, IndexingState


def index_trace(indexing_state: IndexingState, episode: EpisodeMemoryRecord) -> IndexingState:
    """Update formal indexing state from one episode record."""

    return indexing_state.model_copy(
        update={
            "indexed_task_signatures": [*indexing_state.indexed_task_signatures, episode.task_signature.difficulty_profile_hash],
            "episode_count": indexing_state.episode_count + 1,
            "last_consolidated_episode_id": episode.episode_memory_id,
        }
    )
