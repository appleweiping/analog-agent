"""Training entrypoints for the formal world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model import WorldModelBundle
from libs.world_model.compiler import compile_world_model_bundle


def train(task: DesignTask) -> WorldModelBundle:
    """Return the compiled bundle for a task-scoped world model."""

    compiled = compile_world_model_bundle(task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    return compiled.world_model_bundle
