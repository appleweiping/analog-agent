"""Integration tests for Day-4 experiment runner and API."""

from __future__ import annotations

import importlib.util
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx") and native_ngspice_available(),
    "fastapi/httpx or native ngspice are not available in this environment",
)
class ExperimentApiTests(unittest.TestCase):
    def test_run_suite_compares_three_modes(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task

        spec = DesignSpec(
            task_id="api-experiment-ota",
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
            testbench_plan=["op", "ac"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.95,
        )
        task = compile_design_task(spec).design_task
        assert task is not None

        client = TestClient(app)
        response = client.post(
            "/experiments/run-suite",
            json={
                "design_task": task.model_dump(),
                "modes": ["full_simulation_baseline", "no_world_model_baseline", "full_system"],
                "budget": {"max_simulations": 6, "max_candidates_per_step": 3},
                "steps": 3,
                "repeat_runs": 1,
                "fidelity_level": "focused_validation",
                "backend_preference": "ngspice",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        summaries = {item["mode"]: item for item in payload["summaries"]}
        self.assertGreater(
            summaries["full_simulation_baseline"]["average_simulation_call_count"],
            summaries["full_system"]["average_simulation_call_count"],
        )
        self.assertGreaterEqual(
            summaries["full_system"]["feasible_hit_rate"],
            1.0,
        )
