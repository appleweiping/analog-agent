"""Calibration helpers for the formal world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import CalibrationUpdateResponse, TruthCalibrationRecord, WorldState
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService


def calibrate(task: DesignTask, state: WorldState, truth: TruthCalibrationRecord) -> CalibrationUpdateResponse:
    """Compile a bundle for the task and calibrate it with truth."""

    compiled = compile_world_model_bundle(task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    service = WorldModelService(compiled.world_model_bundle, task)
    return service.calibrate_with_truth(state, truth)
