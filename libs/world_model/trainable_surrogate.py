"""Trainable tabular surrogate baseline for Stage C world-model upgrades."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from math import fabs

from libs.schema.world_model import WorldState
from libs.schema.world_model_dataset import (
    SurrogateConfidenceAlignmentSummary,
    SurrogateCoverageSummary,
    SurrogateEvaluationExample,
    SurrogateMetricSummary,
    SurrogatePredictionInterval,
    SurrogateSplitSummary,
    SurrogateTrainingConfig,
    SurrogateTrainingReproducibility,
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


def build_record_feature_vector(record, feature_keys: list[str]) -> list[float]:
    """Project one dataset record into the configured feature vector."""

    vector: list[float] = []
    for key in feature_keys:
        if key.startswith("env:"):
            vector.append(_numeric_environment_value(record.environment.get(key.split(":", 1)[1])))
        else:
            vector.append(float(record.normalized_parameters.get(key, 0.0)))
    return vector


def build_state_feature_vector(state: WorldState, feature_keys: list[str]) -> list[float]:
    """Project one runtime world state into the trainable-surrogate feature space."""

    normalized_parameters = {
        parameter.variable_name: float(parameter.normalized_value)
        for parameter in state.parameter_state
    }
    environment = {
        "corner": state.environment_state.corner,
        "temperature_c": state.environment_state.temperature_c,
        "supply_voltage_v": state.environment_state.supply_voltage_v,
        "load_cap_f": state.environment_state.load_cap_f,
        "output_load_ohm": state.environment_state.output_load_ohm,
        "bias_mode": state.environment_state.bias_mode,
    }
    vector: list[float] = []
    for key in feature_keys:
        if key.startswith("env:"):
            vector.append(_numeric_environment_value(environment.get(key.split(":", 1)[1])))
        else:
            vector.append(float(normalized_parameters.get(key, 0.0)))
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


def _resolve_train_eval_records(
    dataset_bundle: WorldModelDatasetBundle,
    config: SurrogateTrainingConfig,
) -> tuple[list[object], list[object], SurrogateSplitSummary]:
    train_records = [record for record in dataset_bundle.records if record.dataset_split == "train"]
    eval_records = [record for record in dataset_bundle.records if record.dataset_split == "eval"]
    dataset_declares_train = bool(train_records)
    dataset_declares_eval = bool(eval_records)
    split_source = "dataset_declared"

    if len(dataset_bundle.records) == 1:
        train_records = list(dataset_bundle.records)
        eval_records = list(dataset_bundle.records)
        split_source = "single_record_reuse"
    elif not eval_records and dataset_bundle.records:
        eval_records = dataset_bundle.records[-config.minimum_eval_records :]
        train_records = dataset_bundle.records[: max(0, len(dataset_bundle.records) - len(eval_records))]
        split_source = "fallback_missing_eval"
    elif not train_records and dataset_bundle.records:
        train_records = dataset_bundle.records[: max(1, len(dataset_bundle.records) - config.minimum_eval_records)]
        if not eval_records:
            eval_records = dataset_bundle.records[-config.minimum_eval_records :]
        split_source = "fallback_missing_train"

    if not train_records:
        raise ValueError("dataset does not contain any training records")
    if not eval_records:
        raise ValueError("dataset does not contain any evaluation records")

    return train_records, eval_records, SurrogateSplitSummary(
        split_source=split_source,
        split_policy=dataset_bundle.split_policy,
        dataset_declares_train=dataset_declares_train,
        dataset_declares_eval=dataset_declares_eval,
        fallback_applied=split_source != "dataset_declared",
        train_record_ids=[record.record_id for record in train_records],
        eval_record_ids=[record.record_id for record in eval_records],
    )


def _relative_uncertainty(predicted_value: float, spread: float) -> float:
    scale = max(abs(predicted_value), 1e-9)
    return round(min(1.0, spread / scale), 6)


def _interval_half_width(predicted_value: float, spread: float, scale: float) -> float:
    return max(abs(predicted_value) * 0.03, spread * scale, 1e-9)


def predict_with_training_examples(
    *,
    feature_vector: list[float],
    family: str,
    target_metrics: list[str],
    training_examples: list[dict[str, object]],
    k_neighbors: int,
) -> dict[str, object]:
    """Run one weighted-kNN inference against serialized training examples."""

    same_family = [
        {
            "distance": _distance(feature_vector, [float(value) for value in example.get("features", [])]),
            "targets": {str(metric): float(value) for metric, value in dict(example.get("targets", {})).items()},
            "family": str(example.get("family", "unknown")),
            "record_id": str(example.get("record_id", "")),
        }
        for example in training_examples
        if str(example.get("family", "unknown")) == family
    ]
    candidate_neighbors = same_family or [
        {
            "distance": _distance(feature_vector, [float(value) for value in example.get("features", [])]),
            "targets": {str(metric): float(value) for metric, value in dict(example.get("targets", {})).items()},
            "family": str(example.get("family", "unknown")),
            "record_id": str(example.get("record_id", "")),
        }
        for example in training_examples
    ]
    candidate_neighbors.sort(key=lambda item: item["distance"])
    neighbors = candidate_neighbors[: max(1, k_neighbors)]
    predictions, spreads = _prediction_from_neighbors(
        [(float(item["distance"]), dict(item["targets"]), str(item["family"])) for item in neighbors],
        target_metrics,
    )
    mean_distance = sum(float(item["distance"]) for item in neighbors) / max(1, len(neighbors))
    support_score = round(1.0 / (1.0 + mean_distance), 6)
    same_family_ratio = round(
        sum(1 for item in neighbors if str(item["family"]) == family) / max(1, len(neighbors)),
        6,
    )
    confidence = round(min(1.0, 0.6 * support_score + 0.4 * same_family_ratio), 6)
    return {
        "predictions": predictions,
        "spreads": spreads,
        "confidence": confidence,
        "support_score": support_score,
        "same_family_ratio": same_family_ratio,
        "mean_neighbor_distance": round(mean_distance, 6),
        "nearest_training_ids": [str(item["record_id"]) for item in neighbors],
    }


def train_tabular_surrogate(
    dataset_bundle: WorldModelDatasetBundle,
    config: SurrogateTrainingConfig,
    *,
    config_source: str = "<in_memory>",
    config_overrides: list[str] | None = None,
) -> SurrogateTrainingRun:
    """Train a small but real tabular surrogate from exported experiment data."""

    feature_keys = list(dataset_bundle.feature_keys)
    target_metrics = config.target_metrics or list(dataset_bundle.target_metrics)
    train_records, eval_records, split_summary = _resolve_train_eval_records(dataset_bundle, config)
    training_examples = [
        {
            "record_id": record.record_id,
            "family": record.family,
            "features": build_record_feature_vector(record, feature_keys),
            "targets": _target_map(record),
        }
        for record in train_records
    ]

    dataset_signature = stable_hash(json.dumps(dataset_bundle.model_dump(mode="json"), sort_keys=True))
    resolved_config = config.model_dump(mode="json")
    config_signature = stable_hash(json.dumps(resolved_config, sort_keys=True))
    training_signature = stable_hash(
        "|".join(
            [
                dataset_signature,
                config_signature,
                ",".join(split_summary.train_record_ids),
                ",".join(split_summary.eval_record_ids),
            ]
        )
    )

    metric_errors: defaultdict[str, list[float]] = defaultdict(list)
    metric_truths: defaultdict[str, list[float]] = defaultdict(list)
    metric_widths: defaultdict[str, list[float]] = defaultdict(list)
    metric_relative_widths: defaultdict[str, list[float]] = defaultdict(list)
    metric_uncertainties: defaultdict[str, list[float]] = defaultdict(list)
    metric_coverages: defaultdict[str, list[bool]] = defaultdict(list)
    evaluation_examples: list[SurrogateEvaluationExample] = []
    confidence_values: list[float] = []
    interval_hits: list[bool] = []

    for record in eval_records:
        vector = build_record_feature_vector(record, feature_keys)
        inference = predict_with_training_examples(
            feature_vector=vector,
            family=record.family,
            target_metrics=target_metrics,
            training_examples=training_examples,
            k_neighbors=config.k_neighbors,
        )
        predictions = dict(inference["predictions"])
        spreads = dict(inference["spreads"])
        truth = _target_map(record)
        record_intervals: list[SurrogatePredictionInterval] = []

        for metric in target_metrics:
            if metric not in truth:
                continue
            predicted_value = float(predictions.get(metric, 0.0))
            spread = float(spreads.get(metric, 0.0))
            half_width = _interval_half_width(predicted_value, spread, config.coverage_interval_scale)
            lower_bound = round(predicted_value - half_width, 6)
            upper_bound = round(predicted_value + half_width, 6)
            covered = lower_bound <= truth[metric] <= upper_bound
            metric_errors[metric].append(abs(predicted_value - truth[metric]))
            metric_truths[metric].append(abs(truth[metric]))
            metric_widths[metric].append(upper_bound - lower_bound)
            metric_relative_widths[metric].append((upper_bound - lower_bound) / max(abs(truth[metric]), 1e-9))
            metric_uncertainties[metric].append(spread)
            metric_coverages[metric].append(covered)
            record_intervals.append(
                SurrogatePredictionInterval(
                    metric=metric,
                    predicted_value=predicted_value,
                    truth_value=round(truth[metric], 6),
                    uncertainty=round(spread, 6),
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    covered=covered,
                )
            )

        record_hit = bool(record_intervals) and all(interval.covered for interval in record_intervals)
        confidence_values.append(float(inference["confidence"]))
        interval_hits.append(record_hit)
        evaluation_examples.append(
            SurrogateEvaluationExample(
                record_id=record.record_id,
                family=record.family,
                nearest_training_ids=list(inference["nearest_training_ids"]),
                confidence=float(inference["confidence"]),
                interval_status="uncalibrated_neighbor_spread",
                prediction_intervals=record_intervals,
            )
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
    coverage_summary = [
        SurrogateCoverageSummary(
            metric=metric,
            target_coverage=config.target_coverage,
            empirical_coverage=round(
                sum(1.0 for covered in metric_coverages[metric] if covered) / max(1, len(metric_coverages[metric])),
                6,
            ),
            coverage_gap=round(
                (
                    sum(1.0 for covered in metric_coverages[metric] if covered) / max(1, len(metric_coverages[metric]))
                )
                - config.target_coverage,
                6,
            ),
            mean_interval_width=round(sum(metric_widths[metric]) / len(metric_widths[metric]), 6),
            mean_relative_interval_width=round(
                sum(metric_relative_widths[metric]) / len(metric_relative_widths[metric]),
                6,
            ),
            mean_uncertainty=round(sum(metric_uncertainties[metric]) / len(metric_uncertainties[metric]), 6),
            sample_count=len(metric_coverages[metric]),
            interval_status="uncalibrated_neighbor_spread",
        )
        for metric in sorted(metric_coverages)
        if metric_coverages[metric]
    ]
    overall_mae = round(
        sum(summary.mae for summary in per_metric_summary) / max(1, len(per_metric_summary)),
        6,
    )
    overall_relative_mae = round(
        sum(summary.relative_mae for summary in per_metric_summary) / max(1, len(per_metric_summary)),
        6,
    )
    confidence_alignment = SurrogateConfidenceAlignmentSummary(
        summary_name="all_target_metrics_interval_hit",
        mean_predicted_confidence=round(sum(confidence_values) / max(1, len(confidence_values)), 6),
        empirical_hit_rate=round(sum(1.0 for hit in interval_hits if hit) / max(1, len(interval_hits)), 6),
        reliability_gap=round(
            (sum(1.0 for hit in interval_hits if hit) / max(1, len(interval_hits)))
            - (sum(confidence_values) / max(1, len(confidence_values))),
            6,
        ),
        sample_count=len(interval_hits),
        calibration_status="uncalibrated",
    )

    return SurrogateTrainingRun(
        training_id=f"wmtrain_{training_signature[:12]}",
        dataset_id=dataset_bundle.dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        split_summary=split_summary,
        reproducibility=SurrogateTrainingReproducibility(
            dataset_signature=dataset_signature,
            config_signature=config_signature,
            training_signature=training_signature,
            config_source=config_source,
            config_overrides=list(config_overrides or []),
            resolved_config=resolved_config,
            dataset_split_policy=dataset_bundle.split_policy,
            split_source=split_summary.split_source,
            uncalibrated_baseline=True,
        ),
        training_record_count=len(train_records),
        evaluation_record_count=len(eval_records),
        feature_keys=feature_keys,
        target_metrics=target_metrics,
        per_metric_summary=per_metric_summary,
        evaluation_examples=evaluation_examples[: min(5, len(evaluation_examples))],
        coverage_summary=coverage_summary,
        confidence_alignment=confidence_alignment,
        overall_mae=overall_mae,
        overall_relative_mae=overall_relative_mae,
        model_payload={
            "model_family": config.model_family,
            "distance_metric": config.distance_metric,
            "k_neighbors": config.k_neighbors,
            "train_record_ids": [record.record_id for record in train_records],
            "eval_record_ids": [record.record_id for record in eval_records],
            "feature_keys": feature_keys,
            "target_metrics": target_metrics,
            "uncertainty_mode": config.uncertainty_mode,
            "target_coverage": config.target_coverage,
            "coverage_interval_scale": config.coverage_interval_scale,
            "training_examples": training_examples,
            "family_priors": {
                family: {
                    metric: round(
                        sum(
                            float(example["targets"].get(metric, 0.0))
                            for example in training_examples
                            if str(example["family"]) == family
                        )
                        / max(
                            1,
                            sum(1 for example in training_examples if str(example["family"]) == family),
                        ),
                        6,
                    )
                    for metric in target_metrics
                }
                for family in sorted({record.family for record in train_records})
            },
        },
        notes=[
            "first trainable baseline uses weighted nearest neighbors over normalized parameters",
            "same-family neighbors are preferred before cross-family fallback",
            "uncertainty intervals are derived from raw neighbor spread and are not yet probabilistically calibrated",
            "coverage and confidence summaries are reported honestly as uncalibrated baseline diagnostics",
        ],
    )
