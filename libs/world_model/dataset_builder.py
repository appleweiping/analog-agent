"""Dataset builders for trainable world-model upgrades."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from libs.schema.experiment import ExperimentResult, ExperimentSuiteResult, VerifiedCandidateSnapshot
from libs.schema.world_model_dataset import (
    DatasetMetricValue,
    FamilyDatasetSummary,
    WorldModelDatasetBundle,
    WorldModelDatasetRecord,
)
from libs.utils.hashing import stable_hash


def _metric_pairs(values: dict[str, float]) -> list[DatasetMetricValue]:
    return [
        DatasetMetricValue(metric=metric, value=round(float(value), 6))
        for metric, value in sorted(values.items())
    ]


def _prediction_gap_pairs(verification_stats) -> list[DatasetMetricValue]:
    pairs: list[DatasetMetricValue] = []
    for gap in verification_stats.prediction_ground_truth_gap:
        if gap.absolute_error is None:
            continue
        pairs.append(DatasetMetricValue(metric=gap.metric, value=round(float(gap.absolute_error), 6)))
    return sorted(pairs, key=lambda item: item.metric)


def _snapshot_index(result: ExperimentResult) -> dict[str, VerifiedCandidateSnapshot]:
    return {snapshot.candidate_id: snapshot for snapshot in result.verified_candidate_snapshots}


def _split_for_record(candidate_id: str, source_run_id: str) -> str:
    digest = stable_hash(f"{candidate_id}|{source_run_id}")
    return "eval" if int(digest[:2], 16) % 5 == 0 else "train"


def _build_records(result: ExperimentResult) -> list[WorldModelDatasetRecord]:
    snapshot_by_id = _snapshot_index(result)
    records: list[WorldModelDatasetRecord] = []
    for verification_stats in result.verification_stats:
        snapshot = snapshot_by_id.get(verification_stats.candidate_id)
        if snapshot is None:
            continue
        record_id = f"wmdata_{stable_hash(f'{result.run_id}|{verification_stats.candidate_id}')[:12]}"
        records.append(
            WorldModelDatasetRecord(
                record_id=record_id,
                dataset_split=_split_for_record(verification_stats.candidate_id, result.run_id),
                source_kind="experiment_verification",
                source_run_id=result.run_id,
                task_id=result.task_id,
                family=verification_stats.family,
                mode=result.mode,
                candidate_id=verification_stats.candidate_id,
                fidelity_level=verification_stats.fidelity_level,
                truth_level=verification_stats.truth_level,
                validation_status=verification_stats.validation_status,
                feasibility_status=verification_stats.feasibility_status,
                dominant_failure_mode=verification_stats.dominant_failure_mode,
                runtime_sec=round(float(verification_stats.runtime_sec), 6),
                parameter_values=snapshot.parameter_values,
                normalized_parameters=snapshot.normalized_parameters,
                environment=snapshot.environment,
                predicted_metrics=_metric_pairs(snapshot.predicted_metrics),
                measured_metrics=_metric_pairs(verification_stats.measured_metrics),
                prediction_gap=_prediction_gap_pairs(verification_stats),
                artifact_refs=verification_stats.artifact_refs,
            )
        )
    return records


def _cap_records_by_family(records: list[WorldModelDatasetRecord], max_records_per_family: int | None) -> tuple[list[WorldModelDatasetRecord], str]:
    if max_records_per_family is None:
        return records, "family_as_observed"
    grouped: defaultdict[str, list[WorldModelDatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    capped: list[WorldModelDatasetRecord] = []
    for family in sorted(grouped):
        capped.extend(grouped[family][:max_records_per_family])
    return capped, "family_balanced_cap"


def build_world_model_dataset(
    suite_results: list[ExperimentSuiteResult] | ExperimentSuiteResult,
    *,
    dataset_name: str = "world-model-dataset",
    max_records_per_family: int | None = None,
) -> WorldModelDatasetBundle:
    """Build a formal trainable-surrogate dataset from experiment suites."""

    suites = suite_results if isinstance(suite_results, list) else [suite_results]
    records: list[WorldModelDatasetRecord] = []
    source_run_ids: list[str] = []
    feature_keys: set[str] = set()
    target_metrics: set[str] = set()
    family_coverage: set[str] = set()
    for suite in suites:
        for result in suite.runs:
            source_run_ids.append(result.run_id)
            records.extend(_build_records(result))

    records, sampling_policy = _cap_records_by_family(records, max_records_per_family)
    for record in records:
        feature_keys.update(record.normalized_parameters.keys())
        for key, value in record.environment.items():
            if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
                feature_keys.add(f"env:{key}")
        target_metrics.update(metric.metric for metric in record.measured_metrics)
        family_coverage.add(record.family)

    family_grouped: defaultdict[str, list[WorldModelDatasetRecord]] = defaultdict(list)
    for record in records:
        family_grouped[record.family].append(record)
    family_summaries = []
    for family in sorted(family_grouped):
        family_records = family_grouped[family]
        family_summaries.append(
            FamilyDatasetSummary(
                family=family,
                record_count=len(family_records),
                train_count=sum(1 for record in family_records if record.dataset_split == "train"),
                eval_count=sum(1 for record in family_records if record.dataset_split == "eval"),
                modes=sorted({record.mode for record in family_records}),
                target_metrics=sorted(
                    {
                        metric.metric
                        for record in family_records
                        for metric in record.measured_metrics
                    }
                ),
            )
        )

    dataset_id = f"{dataset_name}_{stable_hash('|'.join(sorted(source_run_ids)))[:12]}"
    return WorldModelDatasetBundle(
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_scope="benchmark_multitask" if len(family_coverage) > 1 else "experiment_suite",
        sampling_policy=sampling_policy,
        split_policy="hash_mod_5_eval_split",
        source_run_ids=sorted(set(source_run_ids)),
        family_coverage=sorted(family_coverage),
        feature_keys=sorted(feature_keys),
        target_metrics=sorted(target_metrics),
        family_summaries=family_summaries,
        records=records,
        notes=[
            "records are sourced from real fifth-layer verification executions",
            "normalized_parameters are the primary trainable numeric features",
            "environment keys are preserved for future family-aware models",
        ],
        provenance=[
            "experiment_runner",
            "verification_stats",
            "verified_candidate_snapshots",
        ],
    )
