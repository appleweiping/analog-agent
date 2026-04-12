from __future__ import annotations

import unittest

from libs.eval.benchmark_registry import (
    list_benchmark_definitions,
    load_benchmark_suite_definition,
    load_benchmark_task_definition,
    runnable_benchmark_ids,
)


class BenchmarkRegistryTests(unittest.TestCase):
    def test_suite_definition_covers_multitask_scope(self) -> None:
        suite = load_benchmark_suite_definition()

        self.assertEqual(suite.primary_benchmark_id, "ota2")
        self.assertEqual(suite.benchmark_ids, ["ota2", "folded_cascode", "ldo", "bandgap"])
        self.assertIn("full_system", suite.supported_modes)
        self.assertIn("no_calibration", suite.supported_modes)

    def test_all_benchmark_definitions_validate(self) -> None:
        definitions = list_benchmark_definitions()
        benchmark_ids = [definition.benchmark_id for definition in definitions]

        self.assertEqual(
            benchmark_ids,
            ["ota2_v1", "folded_cascode_v1", "ldo_v1", "bandgap_v1"],
        )
        self.assertEqual(
            [definition.execution_readiness for definition in definitions],
            ["frozen_runnable", "frozen_runnable", "frozen_runnable", "spec_ready"],
        )

    def test_ota_folded_and_ldo_are_current_runnable_benchmarks(self) -> None:
        ota = load_benchmark_task_definition("ota2")
        folded = load_benchmark_task_definition("folded_cascode")
        ldo = load_benchmark_task_definition("ldo")
        bandgap = load_benchmark_task_definition("bandgap")

        self.assertEqual(ota.execution_readiness, "frozen_runnable")
        self.assertTrue(ota.vertical_slice_bound)
        self.assertEqual(folded.execution_readiness, "frozen_runnable")
        self.assertTrue(folded.vertical_slice_bound)
        self.assertEqual(ldo.execution_readiness, "frozen_runnable")
        self.assertTrue(ldo.vertical_slice_bound)
        self.assertEqual(bandgap.execution_readiness, "spec_ready")
        self.assertEqual(runnable_benchmark_ids(), ["ota2_v1", "folded_cascode_v1", "ldo_v1"])

    def test_cross_family_metric_contracts_are_explicit(self) -> None:
        folded = load_benchmark_task_definition("folded_cascode")
        ldo = load_benchmark_task_definition("ldo")
        bandgap = load_benchmark_task_definition("bandgap")

        self.assertIn("dc_gain_db", folded.measurement_contract.primary_metrics)
        self.assertIn("output_swing_v", ldo.measurement_contract.primary_metrics)
        self.assertIn("slew_rate_v_per_us", ldo.measurement_contract.auxiliary_metrics)
        self.assertIn("temperature_coefficient_ppm_per_c", bandgap.measurement_contract.auxiliary_metrics)


if __name__ == "__main__":
    unittest.main()
