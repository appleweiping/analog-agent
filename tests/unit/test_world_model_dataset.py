from __future__ import annotations

import unittest
from types import SimpleNamespace

from libs.schema.experiment import ExperimentResult, MethodComponentConfig, VerifiedCandidateSnapshot
from libs.schema.stats import MetricGapRecord, VerificationStatsRecord
from libs.schema.world_model_dataset import SurrogateTrainingConfig
from libs.world_model.dataset_builder import build_world_model_dataset
from libs.world_model.trainable_surrogate import train_tabular_surrogate


class WorldModelDatasetTests(unittest.TestCase):
    def _component_config(self) -> MethodComponentConfig:
        return MethodComponentConfig(
            mode="full_system",
            use_world_model=True,
            use_calibration=True,
            use_fidelity_escalation=True,
        )

    def test_dataset_builder_exports_verified_records(self) -> None:
        result = ExperimentResult(
            run_id="run_ota2_0",
            mode="full_system",
            task_id="benchmark-ota2-v1",
            component_config=self._component_config(),
            simulation_call_count=1,
            candidate_count=1,
            best_feasible_found=True,
            best_metrics={"dc_gain_db": 80.0},
            verification_stats=[
                VerificationStatsRecord(
                    record_id="verify_1",
                    candidate_id="cand_1",
                    task_id="benchmark-ota2-v1",
                    family="two_stage_ota",
                    truth_level="demonstrator_truth",
                    fidelity_level="quick_truth",
                    analysis_types=["op", "ac"],
                    runtime_sec=0.5,
                    measured_metrics={"dc_gain_db": 80.0, "gbw_hz": 1.2e8},
                    measurement_statuses={"dc_gain_db": "measured"},
                    feasibility_status="feasible_nominal",
                    dominant_failure_mode="none",
                    prediction_ground_truth_gap=[
                        MetricGapRecord(
                            metric="dc_gain_db",
                            predicted_value=78.0,
                            ground_truth_value=80.0,
                            absolute_error=2.0,
                            relative_error=0.025,
                        )
                    ],
                    artifact_refs=["artifact://sim/1"],
                    validation_status="demonstrator_truth",
                )
            ],
            verified_candidate_snapshots=[
                VerifiedCandidateSnapshot(
                    candidate_id="cand_1",
                    task_id="benchmark-ota2-v1",
                    family="two_stage_ota",
                    parameter_values={"w_in": 8.0, "cc": 1.5e-12},
                    normalized_parameters={"w_in": 0.55, "cc": 0.35},
                    environment={"corner": "tt", "temperature_c": 27.0},
                    predicted_metrics={"dc_gain_db": 78.0, "gbw_hz": 1.1e8},
                    predicted_feasibility=0.82,
                    predicted_uncertainty=0.21,
                    artifact_refs=["artifact://candidate/1"],
                )
            ],
        )

        bundle = build_world_model_dataset(SimpleNamespace(runs=[result]))
        self.assertEqual(len(bundle.records), 1)
        self.assertEqual(bundle.records[0].family, "two_stage_ota")
        self.assertIn("w_in", bundle.feature_keys)
        self.assertIn("dc_gain_db", bundle.target_metrics)

    def test_trainable_surrogate_produces_metric_summary(self) -> None:
        bundle = build_world_model_dataset(
            [
                SimpleNamespace(
                    runs=[
                        ExperimentResult(
                            run_id=f"run_{index}",
                            mode="full_system",
                            task_id="benchmark-ota2-v1",
                            component_config=self._component_config(),
                            simulation_call_count=1,
                            candidate_count=1,
                            best_feasible_found=True,
                            best_metrics={"dc_gain_db": 70.0 + index},
                            verification_stats=[
                                VerificationStatsRecord(
                                    record_id=f"verify_{index}",
                                    candidate_id=f"cand_{index}",
                                    task_id="benchmark-ota2-v1",
                                    family="two_stage_ota",
                                    truth_level="demonstrator_truth",
                                    fidelity_level="quick_truth",
                                    analysis_types=["op", "ac"],
                                    runtime_sec=0.25,
                                    measured_metrics={"dc_gain_db": 70.0 + index, "power_w": 1.0e-4 + index * 1.0e-5},
                                    measurement_statuses={"dc_gain_db": "measured"},
                                    feasibility_status="feasible_nominal",
                                    dominant_failure_mode="none",
                                    prediction_ground_truth_gap=[],
                                    artifact_refs=[],
                                    validation_status="demonstrator_truth",
                                )
                            ],
                            verified_candidate_snapshots=[
                                VerifiedCandidateSnapshot(
                                    candidate_id=f"cand_{index}",
                                    task_id="benchmark-ota2-v1",
                                    family="two_stage_ota",
                                    parameter_values={"w_in": 6.0 + index},
                                    normalized_parameters={"w_in": 0.1 * (index + 1)},
                                    environment={"corner": "tt", "temperature_c": 27.0},
                                    predicted_metrics={"dc_gain_db": 69.0 + index},
                                    predicted_feasibility=0.8,
                                    predicted_uncertainty=0.2,
                                    artifact_refs=[],
                                )
                            ],
                        )
                        for index in range(5)
                    ]
                )
            ]
        )
        config = SurrogateTrainingConfig(
            name="tabular_surrogate_v1",
            model_family="tabular_knn",
            distance_metric="weighted_l1",
            target_metrics=["dc_gain_db", "power_w"],
            k_neighbors=2,
            train_fraction=0.8,
            minimum_eval_records=1,
            family_balanced=False,
        )
        run = train_tabular_surrogate(bundle, config)
        self.assertGreater(run.training_record_count, 0)
        self.assertGreater(run.evaluation_record_count, 0)
        self.assertGreater(len(run.per_metric_summary), 0)
        self.assertEqual(run.config.model_family, "tabular_knn")
        self.assertTrue(run.reproducibility.training_signature)
        self.assertIn(run.split_summary.split_source, {"dataset_declared", "single_record_reuse", "fallback_missing_eval", "fallback_missing_train"})
        self.assertTrue(run.coverage_summary)
        self.assertEqual(run.confidence_alignment.calibration_status, "uncalibrated")
        self.assertIn("training_examples", run.model_payload)


if __name__ == "__main__":
    unittest.main()
