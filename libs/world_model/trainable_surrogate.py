"""Trainable tabular surrogate baseline for Stage C world-model upgrades."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import fabs

from libs.schema.world_model_dataset import (
    SurrogateMetricSummary,
    SurrogateTrainingConfig,
    SurrogateTrainingRun,
    WorldModelDatasetBundle,
)
from libs.utils.hashing import stable_hash


def _numeric_environment_value(value) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _feature_vector(record, feature_keys: list[str]) -> list[float]:
    vector: list[float] = []
    for key in feature_keys:
        if key.startswith("env:"):
            vector.append(_numeric_environment_value(record.environment.get(key.split(":", 1)[1])))
        else:
            vector.append(float(record.normalized_parameters.get(key, 0.0)))
    return vector


def _target_map(record) -> dict[str, float]:
    return {metric.metric: float(metric.value) for metric in record.measured_metrics}


def _distance(lhs: list[float], rhs: list[float]) -> float:
    return sum(fabs(left - right) for left, right in zip(lhs, rhs, strict=False))


def _prediction_from_neighbors(neighbors, target_metrics: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    predictions: dict[str, float] = {}
    spreads: dict[str, float] = {}
    if not neighbors:
        for metric in target_metrics:
            predictions[metric] = 0.0
            spreads[metric] = 0.0
        return predictions, spreads

    for metric in target_metrics:
        weighted_values: list[tuple[float, float]] = []
        for distance, target_map, _family in neighbors:
            if metric not in target_map:
                continue
            weight = 1.0 / max(distance, 1e-6)
            weighted_values.append((target_map[metric], weight))
        if not weighted_values:
            predictions[metric] = 0.0
            spreads[metric] = 0.0
            continue
        total_weight = sum(weight for _value, weight in weighted_values)
        mean_value = sum(value * weight for value, weight in weighted_values) / max(total_weight, 1e-12)
        predictions[metric] = round(mean_value, 6)
        spreads[metric] = round(
            sum(abs(value - mean_value) * weight for value, weight in weighted_values) / max(total_weight, 1e-12),
            6,
        )
    return predictions, spreads


def train_tabular_surrogate(
    dataset_bundle: WorldModelDatasetBundle,
    config: SurrogateTrainingConfig,
) -> SurrogateTrainingRun:
    """Train a small but real tabular surrogate from exported experiment data."""

    feature_keys = list(dataset_bundle.feature_keys)
    target_metrics = config.target_metrics or list(dataset_bundle.target_metrics)
    train_records = [record for record in dataset_bundle.records if record.dataset_split == "train"]
    eval_records = [record for record in dataset_bundle.records if record.dataset_split == "eval"]

    if len(dataset_bundle.records) == 1:
        train_records = list(dataset_bundle.records)
        eval_records = list(dataset_bundle.records)
    elif not eval_records and dataset_bundle.records:
        eval_records = dataset_bundle.records[-config.minimum_eval_records :]
        train_records = dataset_bundle.records[: max(0, len(dataset_bundle.records) - len(eval_records))]
    elif not train_records and dataset_bundle.records:
        train_records = dataset_bundle.records[: max(1, len(dataset_bundle.records) - config.minimum_eval_records)]
        if not eval_records:
            eval_records = dataset_bundle.records[-config.minimum_eval_records :]
    if not train_records:
        raise ValueError("dataset does not contain any training records")
    if not eval_records:
        raise ValueError("dataset does not contain any evaluation records")

    training_points = [
        (_feature_vector(record, feature_keys), _target_map(record), record.family, record.record_id)
        for record in train_records
    ]

    metric_errors: defaultdict[str, list[float]] = defaultdict(list)
    metric_truths: defaultdict[str, list[float]] = defaultdict(list)
    example_predictions: list[dict[str, object]] = []
    for record in eval_records:
        vector = _feature_vector(record, feature_keys)
        same_family = [
            (distance, target_map, family, record_id)
            for train_vector, target_map, family, record_id in training_points
            for distance in [_distance(vector, train_vector)]
            if family == record.family
        ]
        candidate_neighbors = same_family or [
            (distance, target_map, family, record_id)
            for train_vector, target_map, family, record_id in training_points
            for distance in [_distance(vector, train_vector)]
        ]
        candidate_neighbors.sort(key=lambda item: item[0])
        neighbors = candidate_neighbors[: max(1, config.k_neighbors)]
        predictions, spreads = _prediction_from_neighbors(
            [(distance, target_map, family) for distance, target_map, family, _record_id in neighbors],
            target_metrics,
        )
        truth = _target_map(record)
        for metric in target_metrics:
            if metric in truth:
                metric_errors[metric].append(abs(predictions.get(metric, 0.0) - truth[metric]))
                metric_truths[metric].append(abs(truth[metric]))
        example_predictions.append(
            {
                "record_id": record.record_id,
                "family": record.family,
                "nearest_training_ids": [record_id for _distance_value, _target_map_value, _family_value, record_id in neighbors],
                "predictions": predictions,
                "uncertainty": spreads,
            }
        )

    per_metric_summary = [
        SurrogateMetricSummary(
            metric=metric,
            mae=round(sum(errors) / len(errors), 6),
            relative_mae=round(
                (sum(errors) / len(errors)) / max(1e-12, (sum(metric_truths[metric]) / len(metric_truths[metric]))),
                6,
            ),
            mean_target_value=round(sum(metric_truths[metric]) / len(metric_truths[metric]), 6),
            covered_eval_records=len(errors),
        )
        for metric, errors in sorted(metric_errors.items())
        if errors
    ]
    overall_mae = round(
        sum(summary.mae for summary in per_metric_summary) / max(1, len(per_metric_summary)),
        6,
    )
    overall_relative_mae = round(
        sum(summary.relative_mae for summary in per_metric_summary) / max(1, len(per_metric_summary)),
        6,
    )

    return SurrogateTrainingRun(
        training_id=f"wmtrain_{stable_hash(dataset_bundle.dataset_id + config.name)[:12]}",
        dataset_id=dataset_bundle.dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        training_record_count=len(train_records),
        evaluation_record_count=len(eval_records),
        feature_keys=feature_keys,
        target_metrics=target_metrics,
        per_metric_summary=per_metric_summary,
        overall_mae=overall_mae,
        overall_relative_mae=overall_relative_mae,
        model_payload={
            "model_family": config.model_family,
            "distance_metric": config.distance_metric,
            "k_neighbors": config.k_neighbors,
            "train_record_ids": [record.record_id for record in train_records],
            "eval_record_ids": [record.record_id for record in eval_records],
            "family_priors": {
                family: {
                    metric: round(
                        sum(target_map.get(metric, 0.0) for _vector, target_map, candidate_family, _record_id in training_points if candidate_family == family)
                        / max(1, sum(1 for _vector, _target_map, candidate_family, _record_id in training_points if candidate_family == family)),
                        6,
                    )
                    for metric in target_metrics
                }
                for family in sorted({record.family for record in train_records})
            },
            "example_predictions": example_predictions[: min(5, len(example_predictions))],
        },
        notes=[
            "first trainable baseline uses weighted nearest neighbors over normalized parameters",
            "same-family neighbors are preferred before cross-family fallback",
            "uncertainty is estimated from neighbor spread, not yet calibrated probabilistically",
        ],
    )
