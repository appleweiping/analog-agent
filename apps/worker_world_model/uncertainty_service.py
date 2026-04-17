"""Uncertainty estimation service."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import PredictionUncertaintySummary, WorldModelBundle, WorldState
from libs.world_model.service import WorldModelService


class UncertaintyService:
    """Expose trust and uncertainty estimates from the world-model service."""

    def __init__(self, bundle: WorldModelBundle, task: DesignTask) -> None:
        self._service = WorldModelService(bundle, task)

    def estimate(self, state: WorldState) -> PredictionUncertaintySummary:
        prediction = self._service.predict_metrics(state)
        if prediction.uncertainty_summary is None:
            raise ValueError("world-model prediction did not expose an uncertainty summary")
        return prediction.uncertainty_summary
