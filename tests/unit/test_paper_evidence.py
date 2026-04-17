from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.paper_evidence import build_world_model_evidence_bundle
from libs.schema.experiment import (
    ExperimentAggregateSummary,
    ExperimentLogRecord,
    ExperimentResult,
    ExperimentSuiteResult,
    MethodComparisonResult,
    MethodComponentConfig,
    MethodConclusionSummary,
    MethodDeltaSummary,
    MethodModeSummary,
)


def _config(mode: str, *, use_world_model: bool, use_calibration: bool, use_fidelity: bool) -> MethodComponentConfig:
    return MethodComponentConfig(
        mode=mode,
        use_world_model=use_world_model,
        use_calibration=use_calibration,
        use_fidelity_escalation=use_fidelity,
        use_phase_updates=True,
        use_calibration_replanning=use_calibration,
        use_rollout_planning=use_world_model,
    )


def _log(step_index: int, mode: str, *, gap: float, confidence: float, uncertainty: float, sim_value: float, pred_feas: float, feasible: bool) -> ExperimentLogRecord:
    return ExperimentLogRecord(
        step_index=step_index,
        mode=mode,
        candidate_ids=[f"{mode}-cand-{step_index}"],
        predicted_truth_gap={"gbw_hz": gap, "phase_margin_deg": gap / 10.0},
        simulation_selection_ratio=0.5,
        feasible_hit=feasible,
        failure_type_distribution={},
        fidelity_usage={"focused_truth": 1},
        calibration_updates_applied=1 if mode == "full_system" else 0,
        world_model_enabled=mode != "no_world_model",
        calibration_enabled=mode == "full_system",
        fidelity_escalation_enabled=mode != "no_fidelity_escalation",
        phase_updates_enabled=True,
        calibration_replanning_enabled=mode == "full_system",
        rollout_planning_enabled=mode != "no_world_model",
        selected_mean_uncertainty=uncertainty,
        selected_mean_confidence=confidence,
        selected_mean_simulation_value=sim_value,
        selected_mean_predicted_feasibility=pred_feas,
    )


def _result(
    mode: str,
    *,
    config: MethodComponentConfig,
    sim_calls: int,
    feasible_rate: float,
    convergence: int,
    efficiency: float,
    selection_ratio: float,
    gaps: list[float],
    confidence: float,
    uncertainty: float,
    sim_value: float,
    pred_feas: float,
    calibration_updates: int = 0,
) -> ExperimentResult:
    logs = [
        _log(
            index,
            mode,
            gap=gaps[index],
            confidence=confidence,
            uncertainty=uncertainty,
            sim_value=sim_value,
            pred_feas=pred_feas,
            feasible=feasible_rate >= 0.5,
        )
        for index in range(len(gaps))
    ]
    return ExperimentResult(
        run_id=f"run-{mode}",
        mode=mode,
        task_id="synthetic-world-model-task",
        component_config=config,
        simulation_call_count=sim_calls,
        candidate_count=6,
        best_feasible_found=feasible_rate >= 0.5,
        best_metrics={"gbw_hz": 1.0e8, "power_w": 8e-4},
        convergence_step=convergence,
        predicted_truth_gap={"gbw_hz": gaps[-1], "phase_margin_deg": gaps[-1] / 10.0},
        simulation_selection_ratio=selection_ratio,
        feasible_hit_rate=feasible_rate,
        failure_type_distribution={},
        efficiency_score=efficiency,
        prediction_gap_by_step=[
            {"gbw_hz": gap, "phase_margin_deg": gap / 10.0}
            for gap in gaps
        ],
        calibration_update_count=calibration_updates,
        focused_truth_call_count=2,
        structured_log=logs,
        verification_stats=[],
        verified_candidate_snapshots=[],
        stats_record=None,
    )


class PaperEvidenceUnitTests(unittest.TestCase):
    def _methodology_suite(self) -> ExperimentSuiteResult:
        modes = ["full_system", "no_world_model", "no_calibration", "no_fidelity_escalation"]
        configs = {
            "full_system": _config("full_system", use_world_model=True, use_calibration=True, use_fidelity=True),
            "no_world_model": _config("no_world_model", use_world_model=False, use_calibration=False, use_fidelity=True),
            "no_calibration": _config("no_calibration", use_world_model=True, use_calibration=False, use_fidelity=True),
            "no_fidelity_escalation": _config("no_fidelity_escalation", use_world_model=True, use_calibration=True, use_fidelity=False),
        }
        runs = [
            _result("full_system", config=configs["full_system"], sim_calls=4, feasible_rate=1.0, convergence=2, efficiency=0.84, selection_ratio=0.52, gaps=[0.32, 0.2, 0.12], confidence=0.92, uncertainty=0.22, sim_value=0.74, pred_feas=0.88, calibration_updates=2),
            _result("no_world_model", config=configs["no_world_model"], sim_calls=6, feasible_rate=0.75, convergence=3, efficiency=0.58, selection_ratio=0.66, gaps=[0.54, 0.48, 0.42], confidence=0.62, uncertainty=0.51, sim_value=0.55, pred_feas=0.68, calibration_updates=0),
            _result("no_calibration", config=configs["no_calibration"], sim_calls=5, feasible_rate=0.75, convergence=3, efficiency=0.67, selection_ratio=0.6, gaps=[0.46, 0.35, 0.28], confidence=0.74, uncertainty=0.37, sim_value=0.63, pred_feas=0.76, calibration_updates=0),
            _result("no_fidelity_escalation", config=configs["no_fidelity_escalation"], sim_calls=5, feasible_rate=0.75, convergence=3, efficiency=0.63, selection_ratio=0.58, gaps=[0.42, 0.31, 0.24], confidence=0.79, uncertainty=0.33, sim_value=0.66, pred_feas=0.8, calibration_updates=1),
        ]
        mode_summaries = [
            MethodModeSummary(
                mode=run.mode,
                component_config=run.component_config,
                run_count=1,
                simulation_call_count=float(run.simulation_call_count),
                feasible_hit_rate=run.feasible_hit_rate,
                average_prediction_gap=dict(run.predicted_truth_gap),
                average_best_metrics=dict(run.best_metrics),
                average_convergence_step=float(run.convergence_step or 0),
                average_calibration_update_count=float(run.calibration_update_count),
                focused_truth_ratio=0.5 if run.mode != "no_fidelity_escalation" else 0.0,
                escalation_count=run.simulation_call_count,
                failure_type_distribution={},
            )
            for run in runs
        ]
        deltas = [
            MethodDeltaSummary(
                baseline_mode="no_world_model",
                compared_mode="full_system",
                simulation_call_delta=-2.0,
                feasible_hit_rate_delta=0.25,
                prediction_gap_delta={"gbw_hz": -0.3},
                best_metric_delta={"gbw_hz": 1.0e7},
                focused_truth_ratio_delta=0.2,
                calibration_update_delta=2.0,
            )
        ]
        comparison = MethodComparisonResult(
            task_id="synthetic-world-model-task",
            modes=modes,
            mode_summaries=mode_summaries,
            deltas=deltas,
            conclusions=MethodConclusionSummary(
                world_model_effective=True,
                calibration_effective=True,
                fidelity_effective=True,
                top_k_baseline_effective=False,
                phase_updates_effective=True,
                calibration_replanning_effective=True,
                rollout_effective=True,
                conclusion_notes=["synthetic_fixture"],
            ),
        )
        summaries = [
            ExperimentAggregateSummary(
                mode=run.mode,
                run_count=1,
                average_simulation_call_count=float(run.simulation_call_count),
                feasible_hit_rate=run.feasible_hit_rate,
                average_efficiency_score=run.efficiency_score,
                average_convergence_step=float(run.convergence_step or 0),
                average_selection_ratio=run.simulation_selection_ratio,
                average_best_metrics=dict(run.best_metrics),
                failure_type_distribution={},
                average_prediction_gap=dict(run.predicted_truth_gap),
                average_calibration_update_count=float(run.calibration_update_count),
                average_focused_truth_call_count=float(run.focused_truth_call_count),
            )
            for run in runs
        ]
        return ExperimentSuiteResult(
            task_id="synthetic-world-model-task",
            modes=modes,
            runs=runs,
            summaries=summaries,
            aggregated_stats=None,
            comparison=comparison,
        )

    def _baseline_suite(self) -> ExperimentSuiteResult:
        modes = ["full_simulation_baseline", "no_world_model_baseline", "full_system"]
        runs = [
            _result("full_simulation_baseline", config=_config("full_simulation_baseline", use_world_model=False, use_calibration=False, use_fidelity=False), sim_calls=8, feasible_rate=1.0, convergence=2, efficiency=0.49, selection_ratio=1.0, gaps=[0.0, 0.0], confidence=0.2, uncertainty=0.9, sim_value=1.0, pred_feas=1.0),
            _result("no_world_model_baseline", config=_config("no_world_model_baseline", use_world_model=False, use_calibration=False, use_fidelity=True), sim_calls=6, feasible_rate=0.75, convergence=3, efficiency=0.57, selection_ratio=0.66, gaps=[0.4, 0.35], confidence=0.6, uncertainty=0.5, sim_value=0.55, pred_feas=0.67),
            _result("full_system", config=_config("full_system", use_world_model=True, use_calibration=True, use_fidelity=True), sim_calls=4, feasible_rate=1.0, convergence=2, efficiency=0.84, selection_ratio=0.52, gaps=[0.2, 0.12], confidence=0.92, uncertainty=0.22, sim_value=0.74, pred_feas=0.88, calibration_updates=2),
        ]
        summaries = [
            ExperimentAggregateSummary(
                mode=run.mode,
                run_count=1,
                average_simulation_call_count=float(run.simulation_call_count),
                feasible_hit_rate=run.feasible_hit_rate,
                average_efficiency_score=run.efficiency_score,
                average_convergence_step=float(run.convergence_step or 0),
                average_selection_ratio=run.simulation_selection_ratio,
                average_best_metrics=dict(run.best_metrics),
                failure_type_distribution={},
                average_prediction_gap=dict(run.predicted_truth_gap),
                average_calibration_update_count=float(run.calibration_update_count),
                average_focused_truth_call_count=float(run.focused_truth_call_count),
            )
            for run in runs
        ]
        return ExperimentSuiteResult(
            task_id="synthetic-world-model-task",
            modes=modes,
            runs=runs,
            summaries=summaries,
            aggregated_stats=None,
            comparison=None,
        )

    def test_world_model_evidence_bundle_exports_extended_stage_c_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_world_model_evidence_bundle(
                self._methodology_suite(),
                baseline_suite=self._baseline_suite(),
                figures_dir=root / "figs",
                tables_dir=root / "tables",
                json_output_path=root / "world_model_evidence_bundle.json",
                core_gap_metrics=["gbw_hz", "phase_margin_deg"],
            )

            self.assertEqual(len(bundle.figures), 8)
            self.assertEqual(len(bundle.tables), 8)
            self.assertTrue((root / "figs" / "world_model_reliability_alignment.svg").exists())
            self.assertTrue((root / "figs" / "world_model_metric_gap_profile.svg").exists())
            self.assertTrue((root / "figs" / "world_model_ranking_efficiency.svg").exists())
            self.assertTrue((root / "figs" / "world_model_calibration_utility.svg").exists())
            self.assertTrue((root / "tables" / "world_model_reliability_alignment.csv").exists())
            self.assertTrue((root / "tables" / "world_model_metric_gap_profile.csv").exists())
            self.assertTrue((root / "tables" / "world_model_ranking_utility.csv").exists())
            self.assertTrue((root / "tables" / "world_model_calibration_utility.csv").exists())

    def test_world_model_evidence_summary_exposes_stage_c_flags(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bundle = build_world_model_evidence_bundle(
                self._methodology_suite(),
                baseline_suite=self._baseline_suite(),
                figures_dir=Path(tmpdir) / "figs",
                tables_dir=Path(tmpdir) / "tables",
                json_output_path=Path(tmpdir) / "world_model_evidence_bundle.json",
                core_gap_metrics=["gbw_hz", "phase_margin_deg"],
            )

            self.assertTrue(bundle.summary.world_model_reduces_simulations)
            self.assertTrue(bundle.summary.calibration_reduces_prediction_gap)
            self.assertTrue(bundle.summary.prediction_gap_beats_no_world_model)
            self.assertTrue(bundle.summary.ranking_improves_efficiency)
            self.assertTrue(bundle.summary.calibration_updates_observable)
            self.assertTrue(bundle.summary.notes)


if __name__ == "__main__":
    unittest.main()
