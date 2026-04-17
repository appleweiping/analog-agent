"""Focused tests for replay manifests and rerunability helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from libs.schema.simulation import ArtifactReplayManifest, PhysicalClaimScope
from libs.simulation.replay_tools import assess_replay_manifest
from scripts.check_native_artifact_rerunability import build_status


class ReplayToolsTests(unittest.TestCase):
    def test_assess_replay_manifest_reports_rerunnable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            binary = root / "ngspice.exe"
            netlist = root / "candidate.sp"
            log = root / "candidate.log"
            measurement = root / "measurement_report.json"
            verification = root / "verification_result.json"
            for path in (binary, netlist, log, measurement, verification):
                path.write_text("placeholder", encoding="utf-8")
            manifest = ArtifactReplayManifest(
                simulation_id="sim_test",
                candidate_id="cand_test",
                request_id="req_test",
                run_directory=str(root),
                invocation_mode="native",
                replayable=True,
                resolved_simulator_binary=str(binary),
                truth_level="demonstrator_truth",
                validation_status="weak",
                fidelity_level="quick_truth",
                physical_claim_scope=PhysicalClaimScope(
                    nominal_profile="single_point_nominal",
                    truth_claim_tier="demonstrator_only",
                ),
                netlist_paths=[str(netlist)],
                log_paths=[str(log)],
                raw_output_paths=[],
                measurement_report_path=str(measurement),
                verification_report_path=str(verification),
                replay_commands=[f'"{binary}" -b "{netlist}" -o "{log}"'],
            )
            manifest_path = root / "replay_manifest.json"
            manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")

            status = assess_replay_manifest(manifest_path)

        self.assertTrue(status["rerunnable_now"])
        self.assertEqual(status["truth_claim_tier"], "demonstrator_only")
        self.assertEqual(status["nominal_profile"], "single_point_nominal")

    def test_rerunability_summary_scans_multiple_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            artifacts_root = Path(tempdir)
            run_dir = artifacts_root / "sim_example"
            run_dir.mkdir(parents=True, exist_ok=True)
            binary = run_dir / "ngspice.exe"
            netlist = run_dir / "candidate.sp"
            log = run_dir / "candidate.log"
            binary.write_text("placeholder", encoding="utf-8")
            netlist.write_text("placeholder", encoding="utf-8")
            log.write_text("placeholder", encoding="utf-8")
            (run_dir / "measurement_report.json").write_text("{}", encoding="utf-8")
            (run_dir / "verification_result.json").write_text("{}", encoding="utf-8")
            manifest = ArtifactReplayManifest(
                simulation_id="sim_example",
                candidate_id="cand_example",
                request_id="req_example",
                run_directory=str(run_dir),
                invocation_mode="native",
                replayable=True,
                resolved_simulator_binary=str(binary),
                truth_level="demonstrator_truth",
                validation_status="weak",
                fidelity_level="quick_truth",
                netlist_paths=[str(netlist)],
                log_paths=[str(log)],
                raw_output_paths=[],
                measurement_report_path=str(run_dir / "measurement_report.json"),
                verification_report_path=str(run_dir / "verification_result.json"),
                replay_commands=[f'"{binary}" -b "{netlist}" -o "{log}"'],
            )
            (run_dir / "replay_manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")

            summary = build_status(artifacts_root)

        self.assertEqual(summary["manifest_count"], 1)
        self.assertEqual(summary["replayable_manifest_count"], 1)
        self.assertEqual(summary["rerunnable_now_count"], 1)


if __name__ == "__main__":
    unittest.main()
