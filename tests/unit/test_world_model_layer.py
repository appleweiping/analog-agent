"""Unit tests for the world model layer."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from apps.worker_world_model.uncertainty_service import UncertaintyService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.world_model import TruthCalibrationRecord, TruthMetric
from libs.schema.world_model_dataset import (
    DatasetMetricValue,
    FamilyDatasetSummary,
    SurrogateTrainingConfig,
    WorldModelDatasetBundle,
    WorldModelDatasetRecord,
)
from libs.tasking.compiler import compile_design_task
from libs.world_model.action_builder import build_design_action
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService
from libs.world_model.state_builder import build_world_state
from libs.world_model.train import build_trained_world_model_bundle, train_from_dataset


def build_standard_ota_task():
    spec = DesignSpec(
        task_id="wm-standard-ota",
        circuit_family="two_stage_ota",
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={
            "gbw_hz": MetricRange(min=1e8),
            "phase_margin_deg": MetricRange(min=60.0),
            "power_w": MetricRange(max=1e-3),
        },
        environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.93,
    )
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


def build_training_bundle() -> WorldModelDatasetBundle:
    records = []
    for index in range(4):
        split = "train" if index < 3 else "eval"
        records.append(
            WorldModelDatasetRecord(
                record_id=f"ota_record_{index}",
                dataset_split=split,
                source_kind="experiment_verification",
                source_run_id=f"ota_run_{index}",
                task_id="wm-standard-ota",
                family="two_stage_ota",
                mode="full_system",
                candidate_id=f"cand_{index}",
                fidelity_level="quick_truth",
                truth_level="demonstrator_truth",
                validation_status="demonstrator_truth",
                feasibility_status="feasible_nominal",
                dominant_failure_mode="none",
                runtime_sec=0.25,
                parameter_values={"w_in": 8e-6 + index * 1e-6, "cc": 1.0e-12 + index * 1.0e-13, "ibias": 5e-5 + index * 5e-6},
                normalized_parameters={"w_in": 0.35 + 0.1 * index, "cc": 0.25 + 0.08 * index, "ibias": 0.3 + 0.07 * index},
                environment={"corner": "tt", "temperature_c": 27.0},
                predicted_metrics=[
                    DatasetMetricValue(metric="gbw_hz", value=1.05e8 + index * 5e6),
                    DatasetMetricValue(metric="phase_margin_deg", value=60.0 + index),
                    DatasetMetricValue(metric="power_w", value=7.0e-4 + index * 2e-5),
                ],
                measured_metrics=[
                    DatasetMetricValue(metric="gbw_hz", value=1.08e8 + index * 5e6),
                    DatasetMetricValue(metric="phase_margin_deg", value=61.0 + index),
                    DatasetMetricValue(metric="power_w", value=7.2e-4 + index * 2e-5),
                ],
                prediction_gap=[
                    DatasetMetricValue(metric="gbw_hz", value=3e6),
                    DatasetMetricValue(metric="phase_margin_deg", value=1.0),
                    DatasetMetricValue(metric="power_w", value=2e-5),
                ],
                artifact_refs=[f"artifact://ota/{index}"],
            )
        )
    return WorldModelDatasetBundle(
        dataset_id="wm_day23_bundle",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_scope="experiment_suite",
        sampling_policy="family_as_observed",
        split_policy="declared_train_eval",
        source_run_ids=[f"ota_run_{index}" for index in range(4)],
        family_coverage=["two_stage_ota"],
        feature_keys=["w_in", "cc", "ibias", "env:temperature_c"],
        target_metrics=["gbw_hz", "phase_margin_deg", "power_w"],
        family_summaries=[
            FamilyDatasetSummary(
                family="two_stage_ota",
                record_count=4,
                train_count=3,
                eval_count=1,
                modes=["full_system"],
                target_metrics=["gbw_hz", "phase_margin_deg", "power_w"],
            )
        ],
        records=records,
        notes=[],
        provenance=["unit_test_fixture"],
    )


class WorldModelLayerTests(unittest.TestCase):
    def test_standard_case_compiles_bundle_and_predicts_metrics(self) -> None:
        task = build_standard_ota_task()
        compiled = compile_world_model_bundle(task)

        self.assertIn(compiled.status, {"compiled", "compiled_with_warnings"})
        self.assertIsNotNone(compiled.world_model_bundle)
        bundle = compiled.world_model_bundle
        assert bundle is not None

        state = build_world_state(task)
        service = WorldModelService(bundle, task)
        metric_prediction = service.predict_metrics(state)
        feasibility = service.predict_feasibility(state)

        self.assertGreaterEqual(len(metric_prediction.metrics), 5)
        self.assertIn("gbw_hz", [item.metric for item in metric_prediction.metrics])
        self.assertTrue(feasibility.per_group_constraints)
        self.assertIn(feasibility.trust_assessment.service_tier, {"screening_only", "ranking_ready", "rollout_ready", "must_escalate"})

    def test_boundary_feasibility_case_exposes_stability_risk(self) -> None:
        task = build_standard_ota_task()
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None

        state = build_world_state(
            task,
            parameter_values={"ibias": 2e-3, "cc": 0.15e-12, "w_in": 2.5e-6, "l_in": 0.8e-6},
        )
        feasibility = WorldModelService(bundle, task).predict_feasibility(state)
        stability = next(item for item in feasibility.per_group_constraints if item.constraint_group == "stability")

        self.assertLess(stability.margin, 2.0)
        self.assertLess(feasibility.overall_feasibility, 0.8)

    def test_action_transition_case_tracks_metric_direction(self) -> None:
        task = build_standard_ota_task()
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None

        state = build_world_state(task)
        action = build_design_action(
            task,
            action_family="parameter_update",
            target_kind="variable",
            variable_names=["ibias"],
            operator="scale",
            payload={"factor": 1.2},
            expected_scope=["operating_point", "power"],
            source="planner",
        )
        transition = WorldModelService(bundle, task).predict_transition(state, action)

        self.assertGreater(transition.delta_features.metric_deltas["gbw_hz"], 0.0)
        self.assertGreater(transition.delta_features.metric_deltas["power_w"], 0.0)

    def test_multi_environment_case_changes_predictions(self) -> None:
        task = build_standard_ota_task()
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None
        service = WorldModelService(bundle, task)

        nominal = build_world_state(task, corner="tt", temperature_c=27.0)
        stressed = build_world_state(task, corner="ss", temperature_c=125.0)
        nominal_metrics = {item.metric: item.value for item in service.predict_metrics(nominal).metrics}
        stressed_metrics = {item.metric: item.value for item in service.predict_metrics(stressed).metrics}

        self.assertLess(stressed_metrics["gbw_hz"], nominal_metrics["gbw_hz"])
        self.assertGreaterEqual(stressed_metrics["power_w"], nominal_metrics["power_w"] * 0.8)

    def test_ood_case_raises_trust_risk(self) -> None:
        task = build_standard_ota_task()
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None
        service = WorldModelService(bundle, task)

        ood_state = build_world_state(task, parameter_values={"ibias": 1.0})
        feasibility = service.predict_feasibility(ood_state)

        self.assertGreater(ood_state.uncertainty_context.ood_score, 0.0)
        self.assertNotEqual(feasibility.trust_assessment.service_tier, "rollout_ready")

    def test_calibration_case_updates_patch_history_and_error_summary(self) -> None:
        task = build_standard_ota_task()
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None
        service = WorldModelService(bundle, task)
        state = build_world_state(task)
        update = service.calibrate_with_truth(
            state,
            TruthCalibrationRecord(
                simulator_signature="ngspice-test",
                analysis_fidelity="full_ground_truth",
                truth_level="configured_truth",
                validation_status="strong",
                metrics=[
                    TruthMetric(metric="gbw_hz", value=9.5e7),
                    TruthMetric(metric="phase_margin_deg", value=61.0),
                ],
                constraints=[],
                artifact_refs=["artifact://spice/1"],
                provenance_tags=["test_fixture"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        )

        self.assertTrue(update.updated_bundle.calibration_state.local_patch_history)
        self.assertIn("gbw_hz", [item.metric for item in update.updated_metrics])

    def test_trained_surrogate_bundle_exposes_backend_and_uncertainty_summary(self) -> None:
        task = build_standard_ota_task()
        training_run = train_from_dataset(
            build_training_bundle(),
            SurrogateTrainingConfig(
                name="tabular_surrogate_v1",
                model_family="tabular_knn",
                distance_metric="weighted_l1",
                target_metrics=["gbw_hz", "phase_margin_deg", "power_w"],
                k_neighbors=2,
                train_fraction=0.8,
                minimum_eval_records=1,
                family_balanced=False,
            ),
            config_source="tests/unit/test_world_model_layer.py",
            config_overrides=["k_neighbors=2"],
        )
        bundle = build_trained_world_model_bundle(task, training_run)
        state = build_world_state(task)
        prediction = WorldModelService(bundle, task).predict_metrics(state)

        self.assertIsNotNone(prediction.surrogate_backend)
        self.assertEqual(prediction.surrogate_backend.backend_kind, "trainable_tabular_surrogate")
        self.assertIsNotNone(prediction.uncertainty_summary)
        assert prediction.uncertainty_summary is not None
        self.assertEqual(prediction.uncertainty_summary.summary_status, "uncalibrated_neighbor_spread")
        self.assertTrue(prediction.uncertainty_summary.per_metric)

    def test_uncertainty_service_returns_structured_summary(self) -> None:
        task = build_standard_ota_task()
        training_run = train_from_dataset(
            build_training_bundle(),
            SurrogateTrainingConfig(
                name="tabular_surrogate_v1",
                model_family="tabular_knn",
                distance_metric="weighted_l1",
                target_metrics=["gbw_hz", "phase_margin_deg", "power_w"],
                k_neighbors=2,
                train_fraction=0.8,
                minimum_eval_records=1,
                family_balanced=False,
            ),
        )
        bundle = build_trained_world_model_bundle(task, training_run)
        state = build_world_state(task)
        summary = UncertaintyService(bundle, task).estimate(state)

        self.assertEqual(summary.summary_status, "uncalibrated_neighbor_spread")
        self.assertGreaterEqual(summary.aggregate_support, 0.0)
        self.assertTrue(summary.per_metric)
