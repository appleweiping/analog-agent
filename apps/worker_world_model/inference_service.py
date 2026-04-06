"""World-model inference service."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import MetricsPrediction, WorldModelBundle, WorldState
from libs.world_model.service import WorldModelService


class InferenceService:
    """Predict metrics and feasibility with the formal world-model service."""

    def __init__(self, bundle: WorldModelBundle, task: DesignTask) -> None:
        self._service = WorldModelService(bundle, task)

    def predict(self, state: WorldState) -> MetricsPrediction:
        return self._service.predict_metrics(state)
