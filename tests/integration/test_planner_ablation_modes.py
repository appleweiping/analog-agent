from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_experiment_suite, run_ota_planner_evidence


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class PlannerAblationIntegrationTests(unittest.TestCase):
    def test_planner_ablation_modes_run_from_single_suite_entry(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="planner_ablation",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-planner-ablation-ci",
        )
        self.assertEqual(
            suite.modes,
            ["full_system", "top_k_baseline", "no_fidelity_escalation", "no_phase_updates", "no_calibration_replanning", "no_rollout_planning"],
        )
        self.assertIsNotNone(suite.comparison)

    def test_mode_switches_affect_logged_planner_behavior(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="planner_ablation",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-planner-mode-behavior-ci",
        )
        no_phase = next(run for run in suite.runs if run.mode == "no_phase_updates")
        no_replan = next(run for run in suite.runs if run.mode == "no_calibration_replanning")
        no_rollout = next(run for run in suite.runs if run.mode == "no_rollout_planning")
        top_k = next(run for run in suite.runs if run.mode == "top_k_baseline")
        full = next(run for run in suite.runs if run.mode == "full_system")

        self.assertTrue(all(not record.phase_updates_enabled for record in no_phase.structured_log))
        self.assertTrue(all(record.phase_before == record.phase_after for record in no_phase.structured_log))
        self.assertTrue(all(not record.calibration_replanning_enabled for record in no_replan.structured_log))
        self.assertTrue(all(not record.rollout_planning_enabled for record in no_rollout.structured_log))
        self.assertTrue(all(not record.fidelity_escalation_enabled for record in top_k.structured_log))
        self.assertTrue(all(record.fidelity_usage.get("quick_truth", 0) >= 1 for record in top_k.structured_log))
        self.assertTrue(any(record.rollout_guidance_applied for record in full.structured_log))
        self.assertGreaterEqual(top_k.simulation_call_count, full.simulation_call_count)

    def test_planner_evidence_bundle_exports(self) -> None:
        bundle = run_ota_planner_evidence(
            steps=2,
            repeat_runs=1,
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            output_root="research/papers/ota2_v1_ci",
        )
        self.assertTrue(bundle.summary.fidelity_escalation_effective in {True, False})
        self.assertTrue(bundle.summary.planner_beats_top_k in {True, False})
        self.assertTrue(bundle.summary.phase_updates_effective in {True, False})
        self.assertTrue(bundle.summary.calibration_replanning_effective in {True, False})
        self.assertTrue(bundle.summary.rollout_guidance_effective in {True, False})
        self.assertTrue(any(figure.figure_id == "fig_planner_simulation_calls" for figure in bundle.figures))
        self.assertTrue(any(table.table_id == "tbl_planner_delta_vs_topk" for table in bundle.tables))


if __name__ == "__main__":
    unittest.main()
