"""Integration tests for simulation-layer API routes."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "fastapi/httpx are not installed in this environment",
)
class SimulationApiTests(unittest.TestCase):
    def _context(self):
        from libs.planner.compiler import compile_planning_bundle
        from libs.planner.service import PlanningService
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task
        from libs.world_model.compiler import compile_world_model_bundle

        spec = DesignSpec(
            task_id="api-sim-ota",
            circuit_family="two_stage_ota",
            process_node="65nm",
            supply_voltage_v=1.2,
            objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
            hard_constraints={"gbw_hz": MetricRange(min=8e7), "phase_margin_deg": MetricRange(min=55.0)},
            environment=Environment(temperature_c=[27.0, 85.0], corners=["tt", "ff"], load_cap_f=2e-12, supply_voltage_v=1.2),
            testbench_plan=["op", "ac", "tran", "noise"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.94,
        )
        task = compile_design_task(spec).design_task
        assert task is not None
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
        candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
        return task, planning_bundle, search_state, candidate_id

    def test_compile_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        task, planning_bundle, search_state, candidate_id = self._context()
        client = TestClient(app)
        response = client.post(
            "/simulation/compile",
            json={
                "design_task": task.model_dump(),
                "planning_bundle": planning_bundle.model_dump(),
                "search_state": search_state.model_dump(),
                "candidate_id": candidate_id,
                "fidelity_level": "focused_validation",
                "backend_preference": "ngspice",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("simulation_bundle", payload)
        self.assertIn("report", payload)

    def test_verify_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        task, planning_bundle, search_state, candidate_id = self._context()
        client = TestClient(app)
        response = client.post(
            "/simulation/verify",
            json={
                "design_task": task.model_dump(),
                "planning_bundle": planning_bundle.model_dump(),
                "search_state": search_state.model_dump(),
                "candidate_id": candidate_id,
                "fidelity_level": "full_robustness_certification",
                "backend_preference": "xyce",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("verification_result", payload)
        self.assertIn("backend_report", payload)
        self.assertIn("simulation_request", payload)

    def test_verify_endpoint_hits_native_ngspice_path(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        task, planning_bundle, search_state, candidate_id = self._context()
        client = TestClient(app)
        response = client.post(
            "/simulation/verify",
            json={
                "design_task": task.model_dump(),
                "planning_bundle": planning_bundle.model_dump(),
                "search_state": search_state.model_dump(),
                "candidate_id": candidate_id,
                "fidelity_level": "focused_validation",
                "backend_preference": "ngspice",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["simulation_bundle"]["backend_binding"]["invocation_mode"], "native")
        self.assertEqual(payload["backend_report"]["backend"], "ngspice")
        self.assertTrue(payload["backend_report"]["is_available"])
        metric_names = {metric["metric"] for metric in payload["verification_result"]["measurement_report"]["measured_metrics"]}
        self.assertTrue({"dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"}.issubset(metric_names))
        self.assertEqual(payload["verification_result"]["completion_status"], "success")
