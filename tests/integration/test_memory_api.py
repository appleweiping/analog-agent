"""Integration tests for memory-layer API routes."""

from __future__ import annotations

import importlib.util
import unittest


def _pipeline(task_id: str, family: str = "two_stage_ota"):
    from libs.planner.compiler import compile_planning_bundle
    from libs.planner.service import PlanningService
    from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
    from libs.simulation.service import SimulationService
    from libs.tasking.compiler import compile_design_task
    from libs.world_model.compiler import compile_world_model_bundle

    spec = DesignSpec(
        task_id=task_id,
        circuit_family=family,
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={"gbw_hz": MetricRange(min=8e7), "phase_margin_deg": MetricRange(min=55.0)},
        environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran", "noise"],
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
    planner = PlanningService(planning_bundle, task, world_model_bundle)
    search_state = planner.initialize_search().search_state
    search_state = planner.propose_candidates(search_state).search_state
    search_state = planner.evaluate_candidates(search_state).search_state
    verification = SimulationService(task, planning_bundle, search_state).execute(search_state.candidate_pool_state.candidates[0].candidate_id).verification_result
    return task, search_state, verification


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "fastapi/httpx are not installed in this environment",
)
class MemoryApiTests(unittest.TestCase):
    def test_compile_and_ingest_endpoints(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        client = TestClient(app)
        compile_response = client.post("/memory/compile", json={})
        self.assertEqual(compile_response.status_code, 200)
        memory_bundle = compile_response.json()["memory_bundle"]

        task, search_state, verification = _pipeline("api-memory-ingest")
        ingest_response = client.post(
            "/memory/ingest",
            json={
                "memory_bundle": memory_bundle,
                "design_task": task.model_dump(),
                "search_state": search_state.model_dump(),
                "verification_result": verification.model_dump(),
            },
        )
        self.assertEqual(ingest_response.status_code, 200)
        self.assertIn("episode_record", ingest_response.json())

    def test_retrieve_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        client = TestClient(app)
        memory_bundle = client.post("/memory/compile", json={}).json()["memory_bundle"]
        task, search_state, verification = _pipeline("api-memory-retrieve")
        updated_bundle = client.post(
            "/memory/ingest",
            json={
                "memory_bundle": memory_bundle,
                "design_task": task.model_dump(),
                "search_state": search_state.model_dump(),
                "verification_result": verification.model_dump(),
            },
        ).json()["memory_bundle"]

        retrieve_response = client.post(
            "/memory/retrieve",
            json={
                "memory_bundle": updated_bundle,
                "design_task": task.model_dump(),
            },
        )
        self.assertEqual(retrieve_response.status_code, 200)
        self.assertIn("feedback_advice", retrieve_response.json())


if __name__ == "__main__":
    unittest.main()
