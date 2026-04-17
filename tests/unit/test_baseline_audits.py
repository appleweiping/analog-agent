from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_baseline_parity import build_status as build_parity_status
from scripts.build_baseline_narrative_package import build_package, export_package
from scripts.confirm_common_benchmark_protocol import build_status as build_protocol_status


class BaselineAuditTests(unittest.TestCase):
    def test_baseline_parity_audit_reports_ready_contract(self) -> None:
        status = build_parity_status()

        self.assertEqual(status["stage"], "Stage E")
        self.assertEqual(status["audit"], "baseline_parity")
        self.assertEqual(status["parity_status"], "baseline_parity_ready")
        self.assertTrue(status["all_benchmarks_frozen_runnable"])
        self.assertFalse(status["missing_supported_modes"])
        self.assertTrue(status["baseline_mode_narratives_ready"])

    def test_common_protocol_confirmation_reports_consistent_contract(self) -> None:
        status = build_protocol_status()

        self.assertEqual(status["stage"], "Stage E")
        self.assertEqual(status["audit"], "common_protocol")
        self.assertEqual(status["protocol_status"], "common_protocol_confirmed")
        self.assertEqual(status["backend_preference_values"], ["ngspice"])
        self.assertEqual(status["truth_level_values"], ["demonstrator_truth"])
        self.assertTrue(status["reporting_axes_ready"])
        self.assertTrue(status["template_contract_ready"])

    def test_baseline_narrative_package_exports(self) -> None:
        package = build_package()
        self.assertIn("full_simulation_baseline", package["narrative_sections"])
        self.assertIn("top_k_baseline", package["baseline_modes"])

        with TemporaryDirectory() as tmpdir:
            outputs = export_package(Path(tmpdir))
            self.assertTrue(Path(outputs["json_output_path"]).exists())
            self.assertTrue(Path(outputs["markdown_output_path"]).exists())


if __name__ == "__main__":
    unittest.main()
