from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.eval.paper_evidence import build_world_model_evidence_bundle
from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_experiment_suite


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class WorldModelPaperEvidenceTests(unittest.TestCase):
    def test_world_model_evidence_bundle_exports_figures_and_tables(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-paper-evidence-ci",
            force_full_steps=True,
        )
        baseline_suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="baseline",
            modes=["full_simulation_baseline", "no_world_model_baseline", "full_system"],
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-paper-evidence-baseline-ci",
        )

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_world_model_evidence_bundle(
                suite,
                baseline_suite=baseline_suite,
                figures_dir=root / "figs",
                tables_dir=root / "tables",
                json_output_path=root / "world_model_evidence_bundle.json",
            )

            self.assertEqual(bundle.task_id, suite.task_id)
            self.assertEqual(len(bundle.figures), 8)
            self.assertEqual(len(bundle.tables), 8)
            self.assertTrue((root / "world_model_evidence_bundle.json").exists())
            self.assertTrue((root / "figs" / "world_model_prediction_gap_vs_step.svg").exists())
            self.assertTrue((root / "figs" / "world_model_simulation_calls.svg").exists())
            self.assertTrue((root / "figs" / "world_model_feasible_hit_rate.svg").exists())
            self.assertTrue((root / "figs" / "world_model_trust_guided_selection.svg").exists())
            self.assertTrue((root / "figs" / "world_model_reliability_alignment.svg").exists())
            self.assertTrue((root / "figs" / "world_model_metric_gap_profile.svg").exists())
            self.assertTrue((root / "figs" / "world_model_ranking_efficiency.svg").exists())
            self.assertTrue((root / "figs" / "world_model_calibration_utility.svg").exists())
            self.assertTrue((root / "tables" / "world_model_method_comparison.csv").exists())
            self.assertTrue((root / "tables" / "world_model_method_comparison.md").exists())
            self.assertTrue((root / "tables" / "prediction_gap_by_step.csv").exists())
            self.assertTrue((root / "tables" / "trust_guided_selection_profile.csv").exists())
            self.assertTrue((root / "tables" / "world_model_budget_comparison.csv").exists())
            self.assertTrue((root / "tables" / "world_model_reliability_alignment.csv").exists())
            self.assertTrue((root / "tables" / "world_model_metric_gap_profile.csv").exists())
            self.assertTrue((root / "tables" / "world_model_ranking_utility.csv").exists())
            self.assertTrue((root / "tables" / "world_model_calibration_utility.csv").exists())
            gap_figure = next(figure for figure in bundle.figures if figure.figure_id == "fig_world_model_prediction_gap_vs_step")
            self.assertGreaterEqual(len(gap_figure.series[0].x_values), 2)

    def test_world_model_evidence_summary_has_methodology_flags(self) -> None:
        suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-paper-evidence-flags-ci",
            force_full_steps=True,
        )
        baseline_suite = run_ota_experiment_suite(
            steps=2,
            repeat_runs=1,
            comparison_profile="baseline",
            modes=["full_simulation_baseline", "no_world_model_baseline", "full_system"],
            budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
            task_id="benchmark-ota2-paper-evidence-flags-baseline-ci",
        )

        with TemporaryDirectory() as tmpdir:
            bundle = build_world_model_evidence_bundle(
                suite,
                baseline_suite=baseline_suite,
                figures_dir=Path(tmpdir) / "figs",
                tables_dir=Path(tmpdir) / "tables",
                json_output_path=Path(tmpdir) / "world_model_evidence_bundle.json",
            )
            self.assertIsInstance(bundle.summary.world_model_reduces_simulations, bool)
            self.assertIsInstance(bundle.summary.calibration_reduces_prediction_gap, bool)
            self.assertIsInstance(bundle.summary.prediction_gap_beats_no_world_model, bool)
            self.assertIsInstance(bundle.summary.trust_guides_selection_behavior, bool)
            self.assertIsInstance(bundle.summary.ranking_improves_efficiency, bool)
            self.assertIsInstance(bundle.summary.calibration_updates_observable, bool)
            self.assertTrue(bundle.summary.notes)

    def test_calibration_curve_improves_after_initial_step(self) -> None:
        suite = run_ota_experiment_suite(
            steps=3,
            repeat_runs=1,
            comparison_profile="methodology",
            budget=ExperimentBudget(max_simulations=6, max_candidates_per_step=3),
            task_id="benchmark-ota2-paper-evidence-calibration-ci",
            force_full_steps=True,
        )
        baseline_suite = run_ota_experiment_suite(
            steps=3,
            repeat_runs=1,
            comparison_profile="baseline",
            modes=["full_simulation_baseline", "no_world_model_baseline", "full_system"],
            budget=ExperimentBudget(max_simulations=6, max_candidates_per_step=3),
            task_id="benchmark-ota2-paper-evidence-calibration-baseline-ci",
        )

        with TemporaryDirectory() as tmpdir:
            bundle = build_world_model_evidence_bundle(
                suite,
                baseline_suite=baseline_suite,
                figures_dir=Path(tmpdir) / "figs",
                tables_dir=Path(tmpdir) / "tables",
                json_output_path=Path(tmpdir) / "world_model_evidence_bundle.json",
            )
            gap_figure = next(figure for figure in bundle.figures if figure.figure_id == "fig_world_model_prediction_gap_vs_step")
            full_system_series = next(series for series in gap_figure.series if series.label == "full_system")
            no_calibration_series = next(series for series in gap_figure.series if series.label == "no_calibration")

            self.assertGreaterEqual(len(full_system_series.y_values), 3)
            self.assertGreaterEqual(len(no_calibration_series.y_values), 3)
            self.assertLess(full_system_series.y_values[1], no_calibration_series.y_values[1])
            self.assertLess(full_system_series.y_values[2], no_calibration_series.y_values[2])


if __name__ == "__main__":
    unittest.main()
