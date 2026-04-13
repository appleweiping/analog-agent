"""Integration coverage for Day-9 research statistics hooks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.orchestrator.job_runner import run_full_system_acceptance
from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.eval.experiment_runner import run_experiment_suite
from libs.eval.stats import (
    aggregate_stats,
    build_failure_mode_summary,
    build_prediction_gap_summary,
    build_verification_stats_record,
    export_stats_csv,
    export_stats_json,
)
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.experiment import ExperimentBudget
from libs.schema.system_binding import AcceptanceTaskConfig
from libs.simulation.service import SimulationService
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService


def _task(task_id: str = "stats-ota"):
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
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class StatsPipelineIntegrationTests(unittest.TestCase):
    def _simulation_execution(self):
        task = _task("stats-verification-ota")
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
        candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="quick_truth",
            backend_preference="ngspice",
        )
        return task, execution

    def test_real_verification_generates_stats_record(self) -> None:
        _, execution = self._simulation_execution()
        stats = execution.verification_stats

        self.assertTrue(stats.record_id)
        self.assertEqual(stats.task_id, execution.simulation_bundle.parent_task_id)
        self.assertEqual(stats.truth_level, execution.verification_result.validation_status.truth_level)
        self.assertEqual(stats.validation_status, execution.verification_result.validation_status.validity_state)
        self.assertIn("gbw_hz", {gap.metric for gap in stats.prediction_ground_truth_gap})

    def test_experiment_suite_aggregates_stats(self) -> None:
        task = _task("stats-experiment-ota")
        suite = run_experiment_suite(
            task,
            modes=["full_simulation_baseline", "random_search_baseline", "bayesopt_baseline", "cmaes_baseline", "rl_baseline", "full_system"],
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            steps=2,
            repeat_runs=2,
            fidelity_level="quick_truth",
            backend_preference="ngspice",
        )

        self.assertIsNotNone(suite.aggregated_stats)
        assert suite.aggregated_stats is not None
        self.assertGreaterEqual(suite.aggregated_stats.total_real_simulation_calls, 1)
        self.assertIn("gbw_hz", suite.aggregated_stats.prediction_gap_summary.metrics_covered)

    def test_fidelity_and_prediction_gap_stats_are_structured(self) -> None:
        _, execution = self._simulation_execution()
        record = execution.verification_stats
        gap_summary = build_prediction_gap_summary([record])

        self.assertIn("gbw_hz", gap_summary.metrics_covered)
        self.assertIn(record.fidelity_level, {"quick_truth", "focused_truth"})

    def test_failure_mode_aggregation_uses_formal_categories(self) -> None:
        _, execution = self._simulation_execution()
        summary = build_failure_mode_summary([execution.verification_stats])

        self.assertIsInstance(summary.frequency, dict)
        self.assertTrue(all(mode in {"none", "design_failure", "stability_failure", "bandwidth_failure", "power_failure", "measurement_failure", "analysis_failure", "simulation_invalid", "netlist_failure"} for mode in {execution.verification_stats.dominant_failure_mode, *summary.frequency.keys()}))

    def test_stats_export_interfaces_write_json_and_csv(self) -> None:
        acceptance = run_full_system_acceptance(
            AcceptanceTaskConfig(
                design_task=_task("stats-acceptance-ota"),
                max_steps=2,
                default_fidelity="quick_truth",
                backend_preference="ngspice",
            )
        )
        stats = aggregate_stats(acceptance)
        self.assertGreaterEqual(stats.total_real_simulation_calls, 1)

        with tempfile.TemporaryDirectory() as tmp:
            json_path = export_stats_json(acceptance, Path(tmp) / "summary.json")
            csv_path = export_stats_csv(acceptance, Path(tmp) / "summary.csv")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
