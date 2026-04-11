"""Unit tests for the world model layer."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.world_model import TruthCalibrationRecord, TruthMetric
from libs.tasking.compiler import compile_design_task
from libs.world_model.action_builder import build_design_action
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService
from libs.world_model.state_builder import build_world_state


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
