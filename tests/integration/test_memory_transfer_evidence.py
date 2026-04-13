from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.vertical_slices.memory_transfer import (
    run_folded_cascode_to_bandgap_memory_transfer_suite,
    run_folded_cascode_to_ota_memory_transfer_suite,
    run_folded_cascode_to_ldo_memory_transfer_suite,
    run_memory_transfer_evidence,
    run_ota_to_bandgap_memory_transfer_suite,
    run_ota_to_folded_cascode_memory_transfer_suite,
)


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MemoryTransferEvidenceTests(unittest.TestCase):
    def test_same_family_transfer_exports_and_stays_structured(self) -> None:
        suite = run_ota_to_folded_cascode_memory_transfer_suite(
            source_episodes=2,
            target_episodes=2,
            max_steps=2,
        )
        self.assertEqual(suite.transfer_kind, "same_family")
        self.assertEqual(suite.modes, ["no_memory", "governed_transfer", "no_governance", "forced_transfer"])
        self.assertTrue(any(record.warm_start_applied for record in suite.transfer_records if record.mode != "no_memory"))

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_memory_transfer_evidence(
                source_task_slug=suite.source_task_slug,
                target_task_slug=suite.target_task_slug,
                suite=suite,
                output_root=root,
            )
            self.assertTrue((root / "memory_transfer_evidence_bundle.json").exists())
            self.assertTrue((root / "memory_transfer_figs" / "memory_transfer_simulation_calls.svg").exists())
            self.assertTrue((root / "memory_transfer_tables" / "memory_transfer_comparison.csv").exists())
            self.assertEqual(bundle.transfer_kind, "same_family")

    def test_cross_family_transfer_exposes_harmful_forced_transfer(self) -> None:
        suite = run_ota_to_bandgap_memory_transfer_suite(
            source_episodes=2,
            target_episodes=2,
            max_steps=2,
        )
        summary_map = {summary.mode: summary for summary in suite.mode_summaries}
        self.assertEqual(suite.transfer_kind, "cross_family")
        self.assertGreaterEqual(summary_map["forced_transfer"].harmful_transfer_rate, summary_map["governed_transfer"].harmful_transfer_rate)
        self.assertGreaterEqual(summary_map["no_governance"].harmful_transfer_rate, summary_map["governed_transfer"].harmful_transfer_rate)
        self.assertIsInstance(suite.summary.governance_blocks_harmful_transfer, bool)
        self.assertIsInstance(suite.summary.no_governance_exposes_harmful_transfer, bool)
        self.assertIsInstance(suite.summary.forced_transfer_exposes_negative_transfer, bool)

    def test_reverse_same_family_transfer_is_structured(self) -> None:
        suite = run_folded_cascode_to_ota_memory_transfer_suite(
            source_episodes=2,
            target_episodes=2,
            max_steps=2,
        )
        self.assertEqual(suite.transfer_kind, "same_family")
        self.assertEqual(suite.source_task_slug, "folded_cascode-v1")
        self.assertEqual(suite.target_task_slug, "ota2-v1")
        self.assertTrue(any(record.warm_start_applied for record in suite.transfer_records if record.mode == "governed_transfer"))

    def test_folded_cross_family_transfer_is_governed(self) -> None:
        ldo_suite = run_folded_cascode_to_ldo_memory_transfer_suite(
            source_episodes=2,
            target_episodes=2,
            max_steps=2,
        )
        bandgap_suite = run_folded_cascode_to_bandgap_memory_transfer_suite(
            source_episodes=2,
            target_episodes=2,
            max_steps=2,
        )
        for suite in (ldo_suite, bandgap_suite):
            summary_map = {summary.mode: summary for summary in suite.mode_summaries}
            self.assertEqual(suite.transfer_kind, "cross_family")
            self.assertGreaterEqual(summary_map["no_governance"].harmful_transfer_rate, summary_map["governed_transfer"].harmful_transfer_rate)
            self.assertGreaterEqual(summary_map["forced_transfer"].harmful_transfer_rate, summary_map["governed_transfer"].harmful_transfer_rate)


if __name__ == "__main__":
    unittest.main()
