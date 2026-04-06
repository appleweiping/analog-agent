"""Uncertainty estimation service."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import TrustAssessment, WorldModelBundle, WorldState
from libs.world_model.service import WorldModelService


class UncertaintyService:
    """Expose trust and uncertainty estimates from the world-model service."""

    def __init__(self, bundle: WorldModelBundle, task: DesignTask) -> None:
        self._service = WorldModelService(bundle, task)

    def estimate(self, state: WorldState) -> TrustAssessment:
        return self._service.predict_feasibility(state).trust_assessment
