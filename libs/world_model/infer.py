"""Inference entrypoints for the formal world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import MetricsPrediction, WorldState
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService


def infer(task: DesignTask, state: WorldState) -> MetricsPrediction:
    """Compile a bundle for the task and run metric inference."""

    compiled = compile_world_model_bundle(task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    service = WorldModelService(compiled.world_model_bundle, task)
    return service.predict_metrics(state)
