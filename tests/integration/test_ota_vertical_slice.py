from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from libs.eval.stats import export_stats_csv, export_stats_json
from libs.vertical_slices.ota2 import run_ota_acceptance, run_ota_experiment_suite
from libs.vertical_slices.ota2_spec import (
    build_ota2_v1_design_task,
    load_ota2_v1_config,
    ota2_v1_measurement_contract_path,
    ota2_v1_netlist_template_path,
    ota2_v1_testbench_path,
)


class OtaVerticalSliceIntegrationTest(unittest.TestCase):
    def test_ota_acceptance_is_a_stable_ci_fixture(self) -> None:
        result = run_ota_acceptance(max_steps=2, task_id="ota2-ci-acceptance")
        self.assertTrue(result.acceptance_summary.system_closed_loop_established)
        self.assertTrue(result.cross_layer_traces)
        self.assertIsNotNone(result.stats_summary)
        self.assertTrue(all(trace.candidate_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.verification_result_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.truth_level == "demonstrator_truth" for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.validation_status in {"strong", "weak"} for trace in result.cross_layer_traces))
        self.assertTrue(any("gbw_hz" in record.measured_metrics for record in result.verification_stats))

    def test_ota_experiment_suite_always_generates_stats(self) -> None:
        suite = run_ota_experiment_suite(steps=2, repeat_runs=1, task_id="benchmark-ota2-ci")
        self.assertIsNotNone(suite.aggregated_stats)
        self.assertEqual(suite.aggregated_stats.aggregation_scope, "benchmark_suite")
        self.assertGreaterEqual(suite.aggregated_stats.total_real_simulation_calls, 1)
        self.assertEqual(sorted(suite.modes), ["bayesopt_baseline", "cmaes_baseline", "full_simulation_baseline", "full_system", "no_world_model_baseline", "random_search_baseline"])
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = export_stats_json(suite, Path(tmpdir) / "ota2_stats.json")
            csv_path = export_stats_csv(suite, Path(tmpdir) / "ota2_stats.csv")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

    def test_ota_vertical_slice_paths_and_measurement_contract_are_frozen(self) -> None:
        config = load_ota2_v1_config()
        self.assertEqual(config.version, "ota2_v1")
        self.assertEqual(config.defaults.fidelity_policy.default_fidelity, "quick_truth")
        self.assertEqual(config.defaults.fidelity_policy.promoted_fidelity, "focused_truth")
        self.assertEqual(config.defaults.model_binding.truth_level, "demonstrator_truth")
        self.assertEqual(config.measurement_targets, ["dc_gain_db", "gbw_hz", "phase_margin_deg", "power_w"])
        self.assertTrue(ota2_v1_netlist_template_path().exists())
        self.assertTrue(ota2_v1_testbench_path("quick_truth").exists())
        self.assertTrue(ota2_v1_testbench_path("focused_truth").exists())
        self.assertTrue(ota2_v1_measurement_contract_path().exists())
        task = build_ota2_v1_design_task(task_id="ota2-ci-shape")
        self.assertEqual(task.circuit_family, "two_stage_ota")
        self.assertEqual(task.topology.template_id, "ota2_miller_basic_v1")
        self.assertEqual(
            [variable.name for variable in task.design_space.variables],
            ["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        )
