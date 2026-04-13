from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from libs.eval.stats import export_stats_csv, export_stats_json
from libs.vertical_slices.bandgap import run_bandgap_acceptance, run_bandgap_experiment_suite
from libs.vertical_slices.bandgap_spec import (
    bandgap_v1_measurement_contract_path,
    bandgap_v1_netlist_template_path,
    bandgap_v1_testbench_path,
    build_bandgap_v1_design_task,
    load_bandgap_v1_config,
)


class BandgapVerticalSliceIntegrationTest(unittest.TestCase):
    def test_bandgap_acceptance_is_a_stable_ci_fixture(self) -> None:
        result = run_bandgap_acceptance(max_steps=2, task_id="bandgap-ci-acceptance")
        self.assertTrue(result.acceptance_summary.system_closed_loop_established)
        self.assertTrue(result.cross_layer_traces)
        self.assertIsNotNone(result.stats_summary)
        self.assertTrue(all(trace.candidate_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.verification_result_id for trace in result.cross_layer_traces))
        self.assertTrue(all(trace.truth_level == "demonstrator_truth" for trace in result.cross_layer_traces))
        self.assertTrue(any("power_w" in record.measured_metrics for record in result.verification_stats))

    def test_bandgap_experiment_suite_generates_stats(self) -> None:
        suite = run_bandgap_experiment_suite(steps=2, repeat_runs=1, task_id="benchmark-bandgap-ci")
        self.assertIsNotNone(suite.aggregated_stats)
        self.assertEqual(suite.aggregated_stats.aggregation_scope, "benchmark_suite")
        self.assertGreaterEqual(suite.aggregated_stats.total_real_simulation_calls, 1)
        self.assertEqual(sorted(suite.modes), ["bayesopt_baseline", "full_simulation_baseline", "full_system", "no_world_model_baseline", "random_search_baseline"])
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = export_stats_json(suite, Path(tmpdir) / "bandgap_stats.json")
            csv_path = export_stats_csv(suite, Path(tmpdir) / "bandgap_stats.csv")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

    def test_bandgap_vertical_slice_paths_and_measurement_contract_are_frozen(self) -> None:
        config = load_bandgap_v1_config()
        self.assertEqual(config.version, "bandgap_v1")
        self.assertEqual(config.defaults.fidelity_policy.default_fidelity, "quick_truth")
        self.assertEqual(config.defaults.fidelity_policy.promoted_fidelity, "focused_truth")
        self.assertEqual(config.defaults.model_binding.truth_level, "demonstrator_truth")
        self.assertEqual(
            config.measurement_targets,
            ["power_w", "temperature_coefficient_ppm_per_c", "line_regulation_mv_per_v"],
        )
        self.assertTrue(bandgap_v1_netlist_template_path().exists())
        self.assertTrue(bandgap_v1_testbench_path("quick_truth").exists())
        self.assertTrue(bandgap_v1_testbench_path("focused_truth").exists())
        self.assertTrue(bandgap_v1_measurement_contract_path().exists())
        task = build_bandgap_v1_design_task(task_id="bandgap-ci-shape")
        self.assertEqual(task.circuit_family, "bandgap")
        self.assertEqual(task.topology.template_id, "bandgap_brokaw_core_v1")
        self.assertEqual(
            [variable.name for variable in task.design_space.variables],
            ["area_ratio", "r1", "r2", "w_core", "l_core", "ibias"],
        )
