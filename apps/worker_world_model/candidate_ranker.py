"""Candidate ranking service for the world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import CandidateRanking, WorldModelBundle, WorldState
from libs.world_model.service import WorldModelService


def rank_candidates(bundle: WorldModelBundle, task: DesignTask, candidates: list[WorldState]) -> CandidateRanking:
    """Rank candidates through the formal world-model contract."""

    return WorldModelService(bundle, task).rank_candidates(candidates)
