from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_experiment_suite


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class RlBaselineIntegrationTests(unittest.TestCase):
    def test_rl_baseline_runs_inside_unified_suite(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            modes=["rl_baseline"],
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-rl-ci",
        )

        self.assertEqual(suite.modes, ["rl_baseline"])
        self.assertIsNotNone(suite.aggregated_stats)
        run = suite.runs[0]
        self.assertEqual(run.mode, "rl_baseline")
        self.assertFalse(run.component_config.use_world_model)
        self.assertFalse(run.component_config.use_calibration)
        self.assertFalse(run.component_config.use_fidelity_escalation)
        self.assertTrue(run.component_config.use_rl_baseline)
        self.assertGreaterEqual(run.simulation_call_count, 1)
        self.assertTrue(all(not record.world_model_enabled for record in run.structured_log))
        self.assertTrue(all(not record.calibration_enabled for record in run.structured_log))
        self.assertTrue(all(not record.fidelity_escalation_enabled for record in run.structured_log))

    def test_rl_baseline_emits_quick_truth_real_stats(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            modes=["rl_baseline", "full_system"],
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-rl-stats-ci",
        )

        run = next(result for result in suite.runs if result.mode == "rl_baseline")
        self.assertTrue(run.verification_stats)
        self.assertTrue(all(record.fidelity_level == "quick_truth" for record in run.verification_stats))
        self.assertTrue(all(record.truth_level in {"demonstrator_truth", "configured_truth"} for record in run.verification_stats))
        self.assertIsNotNone(suite.aggregated_stats)
        assert suite.aggregated_stats is not None
        self.assertGreaterEqual(suite.aggregated_stats.total_real_simulation_calls, len(run.verification_stats))


if __name__ == "__main__":
    unittest.main()
