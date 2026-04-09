"""Integration tests for the Day-2 world-model truth binding path."""

from __future__ import annotations

import importlib.util
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx") and native_ngspice_available(),
    "fastapi/httpx or native ngspice are not available in this environment",
)
class WorldModelTruthBindingApiTests(unittest.TestCase):
    def test_verify_and_calibrate_hits_real_backend_and_updates_bundle(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task

        spec = DesignSpec(
            task_id="api-l3-l5-closure",
            circuit_family="two_stage_ota",
            process_node="65nm",
            supply_voltage_v=1.2,
            objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
            hard_constraints={
                "gbw_hz": MetricRange(min=8e7),
                "phase_margin_deg": MetricRange(min=55.0),
                "power_w": MetricRange(max=1.5e-3),
            },
            environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
            testbench_plan=["op", "ac", "tran"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.95,
        )
        task = compile_design_task(spec).design_task
        assert task is not None

        client = TestClient(app)
        response = client.post(
            "/world-model/verify-and-calibrate",
            json={
                "design_task": task.model_dump(),
                "fidelity_level": "focused_validation",
                "backend_preference": "ngspice",
                "escalation_reason": "day2_integration_test",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["simulation_execution"]["simulation_bundle"]["backend_binding"]["invocation_mode"], "native")
        self.assertEqual(payload["simulation_execution"]["backend_report"]["backend"], "ngspice")
        self.assertTrue(payload["simulation_execution"]["backend_report"]["is_available"])
        metric_names = {metric["metric"] for metric in payload["metrics_prediction"]["metrics"]}
        self.assertIn("gbw_hz", metric_names)
        self.assertIn("phase_margin_deg", metric_names)
        truth_metric_names = {
            metric["metric"] for metric in payload["simulation_execution"]["verification_result"]["calibration_payload"]["truth_record"]["metrics"]
        }
        self.assertIn("gbw_hz", truth_metric_names)
        self.assertTrue(payload["calibration_update"]["updated_bundle"]["calibration_state"]["local_patch_history"])
        self.assertEqual(payload["transition_record"]["state_t_plus_1"]["provenance"]["source_stage"], "simulated")
        self.assertTrue(payload["transition_record"]["raw_artifact_refs"])
