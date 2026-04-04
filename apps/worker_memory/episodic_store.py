"""Episodic memory store placeholder."""

from __future__ import annotations


class EpisodicStore:
    """Persist and retrieve run episodes."""

    def write(self, episode: dict) -> dict:
        return {"store": "episodic", "status": "stub", "episode": episode}
