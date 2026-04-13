from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_submission_ready_freeze


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class SystemClosureReportIntegrationTests(unittest.TestCase):
    def test_submission_ready_freeze_runs_end_to_end(self) -> None:
        result = run_ota_submission_ready_freeze(
            acceptance_steps=2,
            experiment_steps=2,
            repeat_runs=1,
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
        )

        self.assertTrue(result.final_check_summary.submission_ready)
        self.assertTrue(result.final_check_summary.l2_to_l6_closed_loop)
        self.assertTrue(result.final_check_summary.ota_v1_acceptance_ok)
        self.assertTrue(result.final_check_summary.ota_v1_experiment_ok)
        self.assertTrue(result.final_check_summary.stats_export_ok)
        self.assertTrue(result.final_check_summary.method_comparison_ok)
        self.assertEqual(
            result.final_check_summary.closure_statement,
            "This system establishes a fully closed-loop analog design agent grounded in real SPICE verification under demonstrator-level physical validity.",
        )
        self.assertEqual(result.final_check_summary.current_truth_level, "demonstrator_truth")
        self.assertFalse(result.final_check_summary.real_pdk_connected)
        self.assertIsNotNone(result.method_conclusions)
        self.assertGreaterEqual(len(result.method_conclusions.mode_summaries), 3)


if __name__ == "__main__":
    unittest.main()
