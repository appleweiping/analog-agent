"""Integration tests for planning-layer API routes."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "fastapi/httpx are not installed in this environment",
)
class PlanningApiTests(unittest.TestCase):
    def test_compile_and_initialize_endpoints(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task
        from libs.world_model.compiler import compile_world_model_bundle

        spec = DesignSpec(
            task_id="api-planning-ota",
            circuit_family="two_stage_ota",
            process_node="65nm",
            supply_voltage_v=1.2,
            objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
            hard_constraints={"gbw_hz": MetricRange(min=1e8), "phase_margin_deg": MetricRange(min=60.0)},
            environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
            testbench_plan=["op", "ac"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.92,
        )
        task = compile_design_task(spec).design_task
        assert task is not None
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None

        client = TestClient(app)
        compile_response = client.post("/planning/compile", json={"design_task": task.model_dump(), "world_model_bundle": world_model_bundle.model_dump()})
        self.assertEqual(compile_response.status_code, 200)
        self.assertIn("planning_bundle", compile_response.json())

        initialize_response = client.post("/planning/initialize", json={"design_task": task.model_dump(), "world_model_bundle": world_model_bundle.model_dump()})
        self.assertEqual(initialize_response.status_code, 200)
        self.assertIn("search_state", initialize_response.json())

    def test_propose_and_best_result_endpoints(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.planner.compiler import compile_planning_bundle
        from libs.planner.service import PlanningService
        from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
        from libs.tasking.compiler import compile_design_task
        from libs.world_model.compiler import compile_world_model_bundle

        spec = DesignSpec(
            task_id="api-planning-step",
            circuit_family="two_stage_ota",
            process_node="65nm",
            supply_voltage_v=1.2,
            objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
            hard_constraints={"phase_margin_deg": MetricRange(min=60.0)},
            environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
            testbench_plan=["op", "ac"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.9,
        )
        task = compile_design_task(spec).design_task
        assert task is not None
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state

        client = TestClient(app)
        propose_response = client.post(
            "/planning/propose",
            json={
                "design_task": task.model_dump(),
                "world_model_bundle": world_model_bundle.model_dump(),
                "planning_bundle": planning_bundle.model_dump(),
                "search_state": search_state.model_dump(),
            },
        )
        self.assertEqual(propose_response.status_code, 200)
        proposed_state = propose_response.json()["search_state"]

        best_response = client.post(
            "/planning/best-result",
            json={
                "design_task": task.model_dump(),
                "world_model_bundle": world_model_bundle.model_dump(),
                "planning_bundle": planning_bundle.model_dump(),
                "search_state": proposed_state,
            },
        )
        self.assertEqual(best_response.status_code, 200)
        self.assertIn("summary", best_response.json())
