from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.planner_evidence import build_planner_ablation_evidence_bundle
from scripts.audit_planner_rollout_claim import build_audit
from tests.unit.test_planner_evidence import PlannerEvidenceUnitTests


class RolloutClaimAuditUnitTests(unittest.TestCase):
    def test_rollout_claim_audit_limits_scope_to_short_horizon_guidance(self) -> None:
        fixture = PlannerEvidenceUnitTests()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_planner_ablation_evidence_bundle(
                fixture._suite(),
                figures_dir=root / "figs",
                tables_dir=root / "tables",
                json_output_path=root / "planner_evidence_bundle.json",
            )
            audit = build_audit(bundle)

            self.assertEqual(audit["claim_scope"], "short_horizon_world_model_guidance")
            self.assertEqual(audit["claim_status"], "support_short_horizon_rollout_guidance_claim")
            self.assertTrue(audit["structural_ready"])
            self.assertTrue(audit["rollout_guidance_observable"])
            self.assertTrue(audit["rollout_guidance_improves_convergence"])
            self.assertTrue(audit["rollout_guidance_preserves_or_improves_feasible_hit_rate"])
            self.assertIn("general_mpc_superiority", audit["avoid_claims"])


if __name__ == "__main__":
    unittest.main()
