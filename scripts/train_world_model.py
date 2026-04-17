"""Train the first tabular world-model surrogate from a structured dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.world_model_dataset import SurrogateTrainingConfig, WorldModelDatasetBundle
from libs.world_model.train import train_from_dataset


def _coerce_scalar(raw: str):
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _load_yaml_like(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            config[key] = items
        else:
            config[key] = _coerce_scalar(value)
    return config


def _apply_overrides(config: dict[str, object], overrides: list[str]) -> dict[str, object]:
    updated = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must look like key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        updated[key.strip()] = _coerce_scalar(raw_value.strip())
    return updated


def _evaluation_summary(training_run) -> dict[str, object]:
    return {
        "training_id": training_run.training_id,
        "dataset_id": training_run.dataset_id,
        "training_signature": training_run.reproducibility.training_signature,
        "backend": "trainable_tabular_surrogate",
        "split_summary": training_run.split_summary.model_dump(mode="json"),
        "per_metric_summary": [summary.model_dump(mode="json") for summary in training_run.per_metric_summary],
        "coverage_summary": [summary.model_dump(mode="json") for summary in training_run.coverage_summary],
        "confidence_alignment": training_run.confidence_alignment.model_dump(mode="json"),
        "evaluation_examples": [example.model_dump(mode="json") for example in training_run.evaluation_examples],
        "overall_mae": training_run.overall_mae,
        "overall_relative_mae": training_run.overall_relative_mae,
        "notes": list(training_run.notes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/world_model/tabular_surrogate.yaml")
    parser.add_argument("--output", default="artifacts/world_model/tabular_surrogate_training.json")
    parser.add_argument("--eval-output", default="artifacts/world_model/tabular_surrogate_eval.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    dataset_bundle = WorldModelDatasetBundle.model_validate_json(Path(args.dataset).read_text(encoding="utf-8"))
    config_payload = _apply_overrides(_load_yaml_like(Path(args.config)), list(args.overrides))
    config = SurrogateTrainingConfig.model_validate(config_payload)
    training_run = train_from_dataset(
        dataset_bundle,
        config,
        config_source=str(Path(args.config)),
        config_overrides=list(args.overrides),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(training_run.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    eval_output_path = Path(args.eval_output)
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output_path.write_text(json.dumps(_evaluation_summary(training_run), indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "training_id": training_run.training_id,
                "dataset_id": training_run.dataset_id,
                "training_signature": training_run.reproducibility.training_signature,
                "overall_mae": training_run.overall_mae,
                "overall_relative_mae": training_run.overall_relative_mae,
                "output": str(output_path),
                "eval_output": str(eval_output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
