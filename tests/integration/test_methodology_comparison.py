from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_experiment_suite


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MethodologyComparisonIntegrationTests(unittest.TestCase):
    def test_methodology_modes_can_run_from_one_suite_entry(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-methodology-ci",
        )

        self.assertEqual(
            suite.modes,
            ["full_system", "no_world_model", "no_calibration", "no_fidelity_escalation"],
        )
        self.assertEqual(sorted({run.mode for run in suite.runs}), sorted(suite.modes))
        self.assertIsNotNone(suite.comparison)

    def test_no_world_model_mode_disables_prediction_driven_decision(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-no-world-model-ci",
        )
        run = next(result for result in suite.runs if result.mode == "no_world_model")

        self.assertFalse(run.component_config.use_world_model)
        self.assertTrue(all(not record.world_model_enabled for record in run.structured_log))
        self.assertGreaterEqual(run.simulation_call_count, 1)

    def test_no_calibration_mode_keeps_world_model_static(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-no-calibration-ci",
        )
        run = next(result for result in suite.runs if result.mode == "no_calibration")

        self.assertTrue(run.component_config.use_world_model)
        self.assertFalse(run.component_config.use_calibration)
        self.assertEqual(run.calibration_update_count, 0)
        self.assertTrue(all(not record.calibration_enabled for record in run.structured_log))

    def test_comparison_result_is_structurally_consistent(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-comparison-ci",
        )
        assert suite.comparison is not None

        self.assertEqual(sorted(suite.comparison.modes), sorted(suite.modes))
        self.assertEqual(len(suite.comparison.mode_summaries), 4)
        self.assertEqual(len(suite.comparison.deltas), 3)
        self.assertTrue(
            all(
                summary.mode in {"full_system", "no_world_model", "no_calibration", "no_fidelity_escalation"}
                for summary in suite.comparison.mode_summaries
            )
        )
        no_fidelity = next(summary for summary in suite.comparison.mode_summaries if summary.mode == "no_fidelity_escalation")
        self.assertEqual(no_fidelity.focused_truth_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
