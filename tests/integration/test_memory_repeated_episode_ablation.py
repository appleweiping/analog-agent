from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.vertical_slices.ota2 import run_ota_memory_ablation_suite, run_ota_memory_evidence


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MemoryRepeatedEpisodeAblationTests(unittest.TestCase):
    def test_repeated_episode_suite_applies_memory_warm_start(self) -> None:
        suite = run_ota_memory_ablation_suite(
            episodes=3,
            max_steps=2,
        )

        self.assertEqual(suite.modes, ["no_memory", "full_memory"])
        self.assertEqual(len(suite.episode_records), 6)

        full_memory_records = [record for record in suite.episode_records if record.mode == "full_memory"]
        self.assertEqual(full_memory_records[0].warm_start_applied, False)
        self.assertTrue(any(record.warm_start_applied for record in full_memory_records[1:]))
        self.assertTrue(any(record.retrieved_episode_count > 0 for record in full_memory_records[1:]))
        self.assertTrue(suite.summary.memory_uses_retrieval_in_practice)

    def test_memory_evidence_bundle_exports_figures_and_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_ota_memory_evidence(
                episodes=3,
                max_steps=2,
                output_root=root,
            )

            self.assertEqual(bundle.task_id, "benchmark-ota2-v1-memory-ablation")
            self.assertEqual(len(bundle.figures), 4)
            self.assertEqual(len(bundle.tables), 2)
            self.assertTrue((root / "memory_evidence_bundle.json").exists())
            self.assertTrue((root / "memory_figs" / "memory_real_simulation_calls_vs_episode.svg").exists())
            self.assertTrue((root / "memory_figs" / "memory_step_to_first_feasible_vs_episode.svg").exists())
            self.assertTrue((root / "memory_figs" / "memory_repeated_failures_vs_episode.svg").exists())
            self.assertTrue((root / "memory_tables" / "memory_mode_comparison.csv").exists())
            self.assertTrue((root / "memory_tables" / "memory_episode_breakdown.md").exists())
            self.assertIsInstance(bundle.summary.memory_reduces_simulation_calls, bool)
            self.assertIsInstance(bundle.summary.memory_reduces_step_to_first_feasible, bool)


if __name__ == "__main__":
    unittest.main()
