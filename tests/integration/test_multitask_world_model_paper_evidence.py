from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.bandgap import run_bandgap_world_model_evidence
from libs.vertical_slices.folded_cascode import run_folded_cascode_world_model_evidence
from libs.vertical_slices.ldo import run_ldo_world_model_evidence


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class MultiTaskWorldModelPaperEvidenceTests(unittest.TestCase):
    def _assert_bundle_outputs(self, bundle, root: Path) -> None:
        self.assertTrue((root / "world_model_evidence_bundle.json").exists())
        self.assertTrue((root / "figs" / "world_model_prediction_gap_vs_step.svg").exists())
        self.assertTrue((root / "figs" / "world_model_simulation_calls.svg").exists())
        self.assertTrue((root / "figs" / "world_model_feasible_hit_rate.svg").exists())
        self.assertTrue((root / "figs" / "world_model_trust_guided_selection.svg").exists())
        self.assertTrue((root / "tables" / "world_model_method_comparison.csv").exists())
        self.assertTrue((root / "tables" / "world_model_budget_comparison.csv").exists())
        self.assertTrue(bundle.summary.notes)
        gap_figure = next(figure for figure in bundle.figures if figure.figure_id == "fig_world_model_prediction_gap_vs_step")
        self.assertGreaterEqual(len(gap_figure.series[0].x_values), 2)

    def test_folded_cascode_world_model_evidence_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_folded_cascode_world_model_evidence(
                steps=2,
                repeat_runs=1,
                budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
                output_root=root,
            )
            self._assert_bundle_outputs(bundle, root)

    def test_ldo_world_model_evidence_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_ldo_world_model_evidence(
                steps=2,
                repeat_runs=1,
                budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
                output_root=root,
            )
            self._assert_bundle_outputs(bundle, root)

    def test_bandgap_world_model_evidence_exports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = run_bandgap_world_model_evidence(
                steps=2,
                repeat_runs=1,
                budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
                output_root=root,
            )
            self._assert_bundle_outputs(bundle, root)


if __name__ == "__main__":
    unittest.main()
