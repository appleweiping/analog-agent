"""Worker-facing episodic-store adapter."""

from __future__ import annotations

from libs.schema.memory import EpisodeMemoryRecord, MemoryBundle

class EpisodicStore:
    """Persist and retrieve structured episode memory records."""

    def write(self, bundle: MemoryBundle, episode: EpisodeMemoryRecord) -> MemoryBundle:
        return bundle.model_copy(
            update={
                "episode_records": [*bundle.episode_records, episode],
                "indexing_state": bundle.indexing_state.model_copy(
                    update={
                        "episode_count": len(bundle.episode_records) + 1,
                        "last_consolidated_episode_id": episode.episode_memory_id,
                    }
                ),
            }
        )
