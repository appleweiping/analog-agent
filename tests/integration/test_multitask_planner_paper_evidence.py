from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.bandgap import run_bandgap_planner_evidence
from libs.vertical_slices.folded_cascode import run_folded_cascode_planner_evidence
from libs.vertical_slices.ldo import run_ldo_planner_evidence


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MultiTaskPlannerEvidenceTests(unittest.TestCase):
    def _assert_bundle_outputs(self, root: Path, bundle) -> None:
        self.assertTrue((root / "planner_evidence_bundle.json").exists())
        self.assertTrue((root / "planner_figs" / "planner_simulation_calls.svg").exists())
        self.assertTrue((root / "planner_figs" / "planner_feasible_hit_rate.svg").exists())
        self.assertTrue((root / "planner_figs" / "planner_focused_truth_ratio.svg").exists())
        self.assertTrue((root / "planner_figs" / "planner_phase_change_rate.svg").exists())
        self.assertTrue((root / "planner_figs" / "planner_rollout_guidance_rate.svg").exists())
        self.assertTrue((root / "planner_tables" / "planner_ablation_comparison.csv").exists())
        self.assertTrue((root / "planner_tables" / "planner_step_behavior.csv").exists())
        self.assertTrue(bundle.summary.notes)

    def test_folded_cascode_planner_evidence_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_folded_cascode_planner_evidence(
                steps=2,
                repeat_runs=1,
                budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
                output_root=root,
            )
            self._assert_bundle_outputs(root, bundle)

    def test_ldo_planner_evidence_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_ldo_planner_evidence(
                steps=2,
                repeat_runs=1,
                budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
                output_root=root,
            )
            self._assert_bundle_outputs(root, bundle)

    def test_bandgap_planner_evidence_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_bandgap_planner_evidence(
                steps=2,
                repeat_runs=1,
                budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
                output_root=root,
            )
            self._assert_bundle_outputs(root, bundle)


if __name__ == "__main__":
    unittest.main()
