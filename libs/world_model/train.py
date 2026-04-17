"""Training entrypoints for the formal world model layer."""

from __future__ import annotations

from libs.schema.design_task import DesignTask
from libs.schema.world_model_dataset import SurrogateTrainingConfig, SurrogateTrainingRun, WorldModelDatasetBundle
from libs.schema.world_model import TrainedSurrogateCheckpoint, WorldModelBundle
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
    *,
    config_source: str = "<in_memory>",
    config_overrides: list[str] | None = None,
) -> SurrogateTrainingRun:
    """Train the first trainable surrogate baseline from a structured dataset bundle."""

    return train_tabular_surrogate(
        dataset_bundle,
        config,
        config_source=config_source,
        config_overrides=config_overrides,
    )


def attach_training_run(bundle: WorldModelBundle, training_run: SurrogateTrainingRun) -> WorldModelBundle:
    """Attach one trainable-surrogate artifact to an existing world-model bundle."""

    training_examples = list(training_run.model_payload.get("training_examples", []))
    supported_families = sorted(
        {
            str(example.get("family"))
            for example in training_examples
            if str(example.get("family", "")) in bundle.supported_circuit_families
        }
        or set(bundle.supported_circuit_families)
    )
    checkpoint = TrainedSurrogateCheckpoint(
        checkpoint_ref=f"trained-surrogate::{training_run.training_id}",
        backend_kind="trainable_tabular_surrogate",
        training_run_id=training_run.training_id,
        dataset_signature=training_run.reproducibility.dataset_signature,
        config_signature=training_run.reproducibility.config_signature,
        training_signature=training_run.reproducibility.training_signature,
        config_name=training_run.config.name,
        supported_families=supported_families,
        feature_keys=list(training_run.feature_keys),
        target_metrics=list(training_run.target_metrics),
        uncertainty_mode=training_run.config.uncertainty_mode,
        target_coverage=training_run.config.target_coverage,
        coverage_interval_scale=training_run.config.coverage_interval_scale,
        calibration_status="uncalibrated",
        model_payload={
            **dict(training_run.model_payload),
            "coverage_summary": [summary.model_dump(mode="json") for summary in training_run.coverage_summary],
            "confidence_alignment": training_run.confidence_alignment.model_dump(mode="json"),
        },
        notes=list(training_run.notes),
    )
    updated_training_state = bundle.training_state.model_copy(
        update={
            "dataset_signature": training_run.reproducibility.dataset_signature,
            "best_checkpoint_ref": checkpoint.checkpoint_ref,
            "trained_surrogate_checkpoint": checkpoint,
        }
    )
    updated_metadata = bundle.metadata.model_copy(
        update={
            "implementation_version": "trainable-tabular-surrogate-v1",
            "assumptions": [
                *bundle.metadata.assumptions,
                "trainable surrogate serving remains limited to families and metrics covered by the attached training artifact",
                "trainable surrogate uncertainty is raw neighbor spread unless later calibration is attached",
            ],
            "provenance": [*bundle.metadata.provenance, "trainable_surrogate_checkpoint"],
        }
    )
    return bundle.model_copy(update={"training_state": updated_training_state, "metadata": updated_metadata})


def build_trained_world_model_bundle(task: DesignTask, training_run: SurrogateTrainingRun) -> WorldModelBundle:
    """Compile a bundle for the task and attach the trainable-surrogate checkpoint."""

    bundle = train(task)
    return attach_training_run(bundle, training_run)
