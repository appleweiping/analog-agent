"""Training entrypoints for the formal world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model_dataset import SurrogateTrainingConfig, SurrogateTrainingRun, WorldModelDatasetBundle
from libs.schema.world_model import WorldModelBundle
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.trainable_surrogate import train_tabular_surrogate


def train(task: DesignTask) -> WorldModelBundle:
    """Return the compiled bundle for a task-scoped world model."""

    compiled = compile_world_model_bundle(task)
    if compiled.world_model_bundle is None:
        raise ValueError("world model bundle failed to compile")
    return compiled.world_model_bundle


def train_from_dataset(
    dataset_bundle: WorldModelDatasetBundle,
    config: SurrogateTrainingConfig,
) -> SurrogateTrainingRun:
    """Train the first trainable surrogate baseline from a structured dataset bundle."""

    return train_tabular_surrogate(dataset_bundle, config)
