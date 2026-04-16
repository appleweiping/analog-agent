"""Integration coverage for the Day-8 end-to-end acceptance runner."""

from __future__ import annotations

import importlib.util
import unittest

from apps.orchestrator.job_runner import run_full_system_acceptance
from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.system_binding import AcceptanceTaskConfig
from libs.tasking.compiler import compile_design_task


def _acceptance_task(task_id: str = "e2e-acceptance-ota"):
    spec = DesignSpec(
        task_id=task_id,
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
    return task


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class FullSystemAcceptanceIntegrationTests(unittest.TestCase):
    def test_single_task_closure_runs_end_to_end(self) -> None:
        result = run_full_system_acceptance(
            AcceptanceTaskConfig(
                design_task=_acceptance_task("e2e-single-ota"),
                max_steps=3,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            )
        )

        self.assertTrue(result.acceptance_summary.system_closed_loop_established)
        self.assertTrue(result.step_traces)
        self.assertTrue(result.cross_layer_traces)
        self.assertTrue(result.artifact_traces)
        self.assertIsNotNone(result.episode_record)
        self.assertIn("measurement_table", {artifact.artifact_type for artifact in result.artifact_traces})
        self.assertIn("verification_report", {artifact.artifact_type for artifact in result.artifact_traces})

    def test_candidate_lineage_is_consistent_across_layers(self) -> None:
        result = run_full_system_acceptance(
            AcceptanceTaskConfig(
                design_task=_acceptance_task("e2e-lineage-ota"),
                max_steps=3,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            )
        )

        traced_candidate_ids = {trace.candidate_id for trace in result.cross_layer_traces}
        step_candidate_ids = {candidate_id for step in result.step_traces for candidate_id in step.selected_candidate_ids}
        self.assertTrue(step_candidate_ids.issubset(traced_candidate_ids))
        self.assertTrue(all(trace.episode_memory_id == result.episode_memory_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.search_id == result.search_id for trace in result.cross_layer_traces))

    def test_fidelity_consistency_is_recorded_in_trace(self) -> None:
        result = run_full_system_acceptance(
            AcceptanceTaskConfig(
                design_task=_acceptance_task("e2e-fidelity-ota"),
                max_steps=4,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            )
        )

        executed = {trace.executed_fidelity for trace in result.cross_layer_traces}
        requested = {trace.requested_fidelity for trace in result.cross_layer_traces}
        self.assertIn("quick_truth", executed)
        self.assertTrue(executed.issubset({"quick_truth", "focused_truth"}))
        self.assertTrue(requested.issubset({"quick_truth", "focused_truth"}))

    def test_validation_status_propagates_to_memory(self) -> None:
        result = run_full_system_acceptance(
            AcceptanceTaskConfig(
                design_task=_acceptance_task("e2e-validation-ota"),
                max_steps=3,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            )
        )

        self.assertEqual(result.episode_record.final_outcome.truth_level, "demonstrator_truth")
        self.assertEqual(result.episode_record.final_outcome.validation_status, "weak")
        self.assertTrue(
            any(
                artifact.truth_level == "demonstrator_truth" and artifact.validation_status == "weak"
                for artifact in result.artifact_traces
            )
        )

    def test_artifact_traces_capture_native_replay_provenance(self) -> None:
        result = run_full_system_acceptance(
            AcceptanceTaskConfig(
                design_task=_acceptance_task("e2e-artifact-provenance-ota"),
                max_steps=3,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            )
        )

        self.assertTrue(result.artifact_traces)
        self.assertTrue(all(artifact.invocation_mode == "native" for artifact in result.artifact_traces))
        self.assertTrue(all(artifact.resolved_simulator_binary for artifact in result.artifact_traces))
        self.assertTrue(all(artifact.replayable for artifact in result.artifact_traces))
        self.assertTrue(any(artifact.replay_hint for artifact in result.artifact_traces))


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx") and native_ngspice_available(),
    "fastapi/httpx or native ngspice are not available in this environment",
)
class FullSystemAcceptanceApiTests(unittest.TestCase):
    def test_acceptance_api_returns_structured_trace(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        task = _acceptance_task("e2e-api-ota")
        client = TestClient(app)
        response = client.post(
            "/acceptance/run-full-system",
            json=AcceptanceTaskConfig(
                design_task=task,
                max_steps=3,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            ).model_dump(),
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["acceptance_summary"]["system_closed_loop_established"])
        self.assertTrue(payload["cross_layer_traces"])
        self.assertTrue(payload["artifact_traces"])


if __name__ == "__main__":
    unittest.main()
