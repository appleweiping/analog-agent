from __future__ import annotations

import unittest

from libs.eval.benchmark_protocol import (
    BASELINE_BENCHMARK_MODES,
    BASELINE_STRENGTH_TIERS,
    benchmark_modes_for_profile,
    benchmark_protocol_contract,
    default_benchmark_budget,
)


class BenchmarkProtocolTests(unittest.TestCase):
    def test_protocol_contract_exposes_shared_defaults(self) -> None:
        contract = benchmark_protocol_contract()

        self.assertEqual(contract["baseline_modes"], BASELINE_BENCHMARK_MODES)
        self.assertEqual(contract["default_budget"], default_benchmark_budget().model_dump())
        self.assertEqual(benchmark_modes_for_profile("baseline"), BASELINE_BENCHMARK_MODES)
        self.assertIn("top_k_baseline", benchmark_modes_for_profile("planner_ablation"))
        self.assertIn("no_calibration", benchmark_modes_for_profile("methodology"))

    def test_baseline_strength_tiers_keep_lightweight_internal_claim_boundary(self) -> None:
        contract = benchmark_protocol_contract()

        self.assertEqual(contract["baseline_strength_tiers"], BASELINE_STRENGTH_TIERS)
        self.assertEqual(BASELINE_STRENGTH_TIERS["full_simulation_baseline"], "upper_cost_reference")
        for mode in ["top_k_baseline", "random_search_baseline", "bayesopt_baseline", "cmaes_baseline", "rl_baseline"]:
            self.assertEqual(BASELINE_STRENGTH_TIERS[mode], "lightweight_internal")
            self.assertIn("Lightweight internal", contract["baseline_mode_narratives"][mode])


if __name__ == "__main__":
    unittest.main()
