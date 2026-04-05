"""Integration tests for task-formalization API routes."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "fastapi/httpx are not installed in this environment",
)
class TaskingApiTests(unittest.TestCase):
    def test_compile_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        client = TestClient(app)
        response = client.post(
            "/tasking/compile",
            json={
                "design_spec": {
                    "task_id": "api-standard-ota",
                    "circuit_family": "two_stage_ota",
                    "process_node": "65nm",
                    "supply_voltage_v": 1.2,
                    "objectives": {"maximize": ["gbw_hz"], "minimize": ["power_w"]},
                    "hard_constraints": {
                        "gbw_hz": {"min": 100000000.0},
                        "phase_margin_deg": {"min": 60.0},
                        "power_w": {"max": 0.001},
                    },
                    "environment": {
                        "temperature_c": [27.0],
                        "corners": ["tt"],
                        "load_cap_f": 2e-12,
                        "supply_voltage_v": 1.2,
                    },
                    "testbench_plan": ["op", "ac"],
                    "design_variables": ["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                    "missing_information": [],
                    "notes": [],
                    "compile_confidence": 0.92,
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "compiled")
        self.assertEqual(payload["design_task"]["task_type"], "sizing")

    def test_validate_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task

        spec = DesignSpec(
            task_id="api-validate",
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
            testbench_plan=["op", "ac"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.92,
        )
        compiled = compile_design_task(spec)
        assert compiled.design_task is not None

        client = TestClient(app)
        response = client.post(
            "/tasking/validate",
            json={
                "design_spec": spec.model_dump(),
                "design_task": compiled.design_task.model_dump(),
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("is_valid", payload)
