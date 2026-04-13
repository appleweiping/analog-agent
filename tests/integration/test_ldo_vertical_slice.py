from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from libs.eval.stats import export_stats_csv, export_stats_json
from libs.vertical_slices.ldo import run_ldo_acceptance, run_ldo_experiment_suite
from libs.vertical_slices.ldo_spec import (
    build_ldo_v1_design_task,
    ldo_v1_measurement_contract_path,
    ldo_v1_netlist_template_path,
    ldo_v1_testbench_path,
    load_ldo_v1_config,
)


class LdoVerticalSliceIntegrationTest(unittest.TestCase):
    def test_ldo_acceptance_is_a_stable_ci_fixture(self) -> None:
        result = run_ldo_acceptance(max_steps=2, task_id="ldo-ci-acceptance")
        self.assertTrue(result.acceptance_summary.system_closed_loop_established)
        self.assertTrue(result.cross_layer_traces)
        self.assertIsNotNone(result.stats_summary)
        self.assertTrue(all(trace.candidate_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.verification_result_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.truth_level == "demonstrator_truth" for trace in result.cross_layer_traces))
        self.assertTrue(any("output_swing_v" in record.measured_metrics for record in result.verification_stats))

    def test_ldo_experiment_suite_generates_stats(self) -> None:
        suite = run_ldo_experiment_suite(steps=2, repeat_runs=1, task_id="benchmark-ldo-ci")
        self.assertIsNotNone(suite.aggregated_stats)
        self.assertEqual(suite.aggregated_stats.aggregation_scope, "benchmark_suite")
        self.assertGreaterEqual(suite.aggregated_stats.total_real_simulation_calls, 1)
        self.assertEqual(
            sorted(suite.modes),
            [
                "bayesopt_baseline",
                "cmaes_baseline",
                "full_simulation_baseline",
                "full_system",
                "no_world_model_baseline",
                "random_search_baseline",
                "rl_baseline",
                "top_k_baseline",
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = export_stats_json(suite, Path(tmpdir) / "ldo_stats.json")
            csv_path = export_stats_csv(suite, Path(tmpdir) / "ldo_stats.csv")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

    def test_ldo_vertical_slice_paths_and_measurement_contract_are_frozen(self) -> None:
        config = load_ldo_v1_config()
        self.assertEqual(config.version, "ldo_v1")
        self.assertEqual(config.defaults.fidelity_policy.default_fidelity, "quick_truth")
        self.assertEqual(config.defaults.fidelity_policy.promoted_fidelity, "focused_truth")
        self.assertEqual(config.defaults.model_binding.truth_level, "demonstrator_truth")
        self.assertEqual(config.measurement_targets, ["gbw_hz", "phase_margin_deg", "power_w", "output_swing_v"])
        self.assertTrue(ldo_v1_netlist_template_path().exists())
        self.assertTrue(ldo_v1_testbench_path("quick_truth").exists())
        self.assertTrue(ldo_v1_testbench_path("focused_truth").exists())
        self.assertTrue(ldo_v1_measurement_contract_path().exists())
        task = build_ldo_v1_design_task(task_id="ldo-ci-shape")
        self.assertEqual(task.circuit_family, "ldo")
        self.assertEqual(task.topology.template_id, "ldo_pmos_compensated_v1")
        self.assertEqual(
            [variable.name for variable in task.design_space.variables],
            ["w_pass", "l_pass", "w_err", "l_err", "ibias", "c_comp"],
        )
