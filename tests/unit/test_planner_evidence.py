from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from libs.eval.planner_evidence import build_planner_ablation_evidence_bundle
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


def _config(
    mode: str,
    *,
    use_fidelity: bool = True,
    use_phase_updates: bool = True,
    use_replanning: bool = True,
    use_rollout: bool = True,
    use_top_k: bool = False,
) -> MethodComponentConfig:
    return MethodComponentConfig(
        mode=mode,
        use_world_model=True,
        use_calibration=True,
        use_fidelity_escalation=use_fidelity,
        use_phase_updates=use_phase_updates,
        use_calibration_replanning=use_replanning,
        use_rollout_planning=use_rollout,
        use_top_k_baseline=use_top_k,
    )


def _log(
    step_index: int,
    mode: str,
    *,
    feasible: bool,
    fidelity_enabled: bool,
    phase_enabled: bool,
    phase_changed: bool,
    replan_enabled: bool,
    replan_required: bool,
    rollout_enabled: bool,
    rollout_applied: bool,
    rollout_value: float,
    confidence: float,
    uncertainty: float,
    sim_value: float,
    pred_feas: float,
) -> ExperimentLogRecord:
    return ExperimentLogRecord(
        step_index=step_index,
        mode=mode,
        candidate_ids=[f"{mode}-cand-{step_index}"],
        predicted_truth_gap={"gbw_hz": 0.2},
        simulation_selection_ratio=0.5,
        feasible_hit=feasible,
        failure_type_distribution={},
        fidelity_usage={"focused_truth": 1 if fidelity_enabled else 0, "quick_truth": 1},
        calibration_updates_applied=1 if replan_enabled else 0,
        world_model_enabled=True,
        calibration_enabled=True,
        fidelity_escalation_enabled=fidelity_enabled,
        phase_updates_enabled=phase_enabled,
        calibration_replanning_enabled=replan_enabled,
        rollout_planning_enabled=rollout_enabled,
        phase_before="explore",
        phase_after="refine" if phase_changed else "explore",
        phase_changed=phase_changed,
        calibration_required_after_step=replan_required,
        rollout_guidance_applied=rollout_applied,
        rollout_guidance_value=rollout_value,
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
    focused_truth_ratio: float,
    phase_change_rate: float,
    replan_rate: float,
    rollout_rate: float,
    rollout_value: float,
    failures: dict[str, int],
) -> ExperimentResult:
    logs = [
        _log(
            index,
            mode,
            feasible=feasible_rate >= 0.5,
            fidelity_enabled=config.use_fidelity_escalation,
            phase_enabled=config.use_phase_updates,
            phase_changed=index < round(phase_change_rate * 2),
            replan_enabled=config.use_calibration_replanning,
            replan_required=index < round(replan_rate * 2),
            rollout_enabled=config.use_rollout_planning,
            rollout_applied=index < round(rollout_rate * 2),
            rollout_value=rollout_value,
            confidence=0.88 if mode == "full_system" else 0.7,
            uncertainty=0.22 if mode == "full_system" else 0.38,
            sim_value=0.8 if mode == "full_system" else 0.62,
            pred_feas=0.9 if mode == "full_system" else 0.72,
        )
        for index in range(2)
    ]
    return ExperimentResult(
        run_id=f"run-{mode}",
        mode=mode,
        task_id="synthetic-planner-task",
        component_config=config,
        simulation_call_count=sim_calls,
        candidate_count=6,
        best_feasible_found=feasible_rate >= 0.5,
        best_metrics={"gbw_hz": 1.0e8},
        convergence_step=convergence,
        predicted_truth_gap={"gbw_hz": 0.2},
        simulation_selection_ratio=0.5,
        feasible_hit_rate=feasible_rate,
        failure_type_distribution=failures,
        efficiency_score=efficiency,
        prediction_gap_by_step=[{"gbw_hz": 0.2}, {"gbw_hz": 0.1}],
        calibration_update_count=2 if config.use_calibration_replanning else 0,
        focused_truth_call_count=round(focused_truth_ratio * 4),
        structured_log=logs,
        verification_stats=[],
        verified_candidate_snapshots=[],
        stats_record=None,
    )


class PlannerEvidenceUnitTests(unittest.TestCase):
    def _suite(self) -> ExperimentSuiteResult:
        configs = {
            "full_system": _config("full_system"),
            "top_k_baseline": _config("top_k_baseline", use_fidelity=False, use_phase_updates=False, use_replanning=False, use_rollout=False, use_top_k=True),
            "no_fidelity_escalation": _config("no_fidelity_escalation", use_fidelity=False),
            "no_phase_updates": _config("no_phase_updates", use_phase_updates=False),
            "no_calibration_replanning": _config("no_calibration_replanning", use_replanning=False),
            "no_rollout_planning": _config("no_rollout_planning", use_rollout=False),
        }
        runs = [
            _result("full_system", config=configs["full_system"], sim_calls=4, feasible_rate=1.0, convergence=2, efficiency=0.84, focused_truth_ratio=0.5, phase_change_rate=1.0, replan_rate=0.5, rollout_rate=1.0, rollout_value=0.82, failures={"stability_failure": 1}),
            _result("top_k_baseline", config=configs["top_k_baseline"], sim_calls=6, feasible_rate=0.75, convergence=4, efficiency=0.45, focused_truth_ratio=0.0, phase_change_rate=0.0, replan_rate=0.0, rollout_rate=0.0, rollout_value=0.0, failures={"stability_failure": 2, "measurement_failure": 1}),
            _result("no_fidelity_escalation", config=configs["no_fidelity_escalation"], sim_calls=5, feasible_rate=0.75, convergence=3, efficiency=0.66, focused_truth_ratio=0.0, phase_change_rate=0.5, replan_rate=0.5, rollout_rate=1.0, rollout_value=0.76, failures={"stability_failure": 2}),
            _result("no_phase_updates", config=configs["no_phase_updates"], sim_calls=5, feasible_rate=0.75, convergence=4, efficiency=0.61, focused_truth_ratio=0.5, phase_change_rate=0.0, replan_rate=0.5, rollout_rate=1.0, rollout_value=0.74, failures={"drive_bandwidth_failure": 2}),
            _result("no_calibration_replanning", config=configs["no_calibration_replanning"], sim_calls=5, feasible_rate=0.75, convergence=4, efficiency=0.63, focused_truth_ratio=0.5, phase_change_rate=0.5, replan_rate=0.0, rollout_rate=1.0, rollout_value=0.75, failures={"measurement_failure": 2}),
            _result("no_rollout_planning", config=configs["no_rollout_planning"], sim_calls=5, feasible_rate=0.75, convergence=4, efficiency=0.64, focused_truth_ratio=0.5, phase_change_rate=0.5, replan_rate=0.5, rollout_rate=0.0, rollout_value=0.0, failures={"stability_failure": 2, "design_failure": 1}),
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
                focused_truth_ratio=0.5 if run.mode != "top_k_baseline" and run.mode != "no_fidelity_escalation" else 0.0,
                escalation_count=run.simulation_call_count,
                failure_type_distribution=dict(run.failure_type_distribution),
            )
            for run in runs
        ]
        comparison = MethodComparisonResult(
            task_id="synthetic-planner-task",
            modes=[run.mode for run in runs],
            mode_summaries=mode_summaries,
            deltas=[
                MethodDeltaSummary(
                    baseline_mode="top_k_baseline",
                    compared_mode="full_system",
                    simulation_call_delta=-2.0,
                    feasible_hit_rate_delta=0.25,
                    prediction_gap_delta={"gbw_hz": -0.1},
                    best_metric_delta={"gbw_hz": 1.0e7},
                    focused_truth_ratio_delta=0.5,
                    calibration_update_delta=2.0,
                )
            ],
            conclusions=MethodConclusionSummary(
                world_model_effective=True,
                calibration_effective=True,
                fidelity_effective=True,
                top_k_baseline_effective=True,
                phase_updates_effective=True,
                calibration_replanning_effective=True,
                rollout_effective=True,
                conclusion_notes=["synthetic_planner_fixture"],
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
            task_id="synthetic-planner-task",
            modes=[run.mode for run in runs],
            runs=runs,
            summaries=summaries,
            aggregated_stats=None,
            comparison=comparison,
        )

    def test_planner_evidence_bundle_exports_stage_d_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = build_planner_ablation_evidence_bundle(
                self._suite(),
                figures_dir=root / "figs",
                tables_dir=root / "tables",
                json_output_path=root / "planner_evidence_bundle.json",
            )

            self.assertEqual(len(bundle.figures), 14)
            self.assertEqual(len(bundle.tables), 10)
            self.assertTrue((root / "figs" / "planner_topk_efficiency.svg").exists())
            self.assertTrue((root / "figs" / "planner_fidelity_tradeoff.svg").exists())
            self.assertTrue((root / "figs" / "planner_phase_convergence.svg").exists())
            self.assertTrue((root / "figs" / "planner_calibration_utility.svg").exists())
            self.assertTrue((root / "figs" / "planner_rollout_claim_audit.svg").exists())
            self.assertTrue((root / "figs" / "planner_failure_pressure.svg").exists())
            self.assertTrue((root / "figs" / "planner_efficiency_frontier.svg").exists())
            self.assertTrue((root / "tables" / "planner_topk_utility.csv").exists())
            self.assertTrue((root / "tables" / "planner_fidelity_tradeoff.csv").exists())
            self.assertTrue((root / "tables" / "planner_phase_utility.csv").exists())
            self.assertTrue((root / "tables" / "planner_calibration_utility.csv").exists())
            self.assertTrue((root / "tables" / "planner_rollout_claim_audit.csv").exists())
            self.assertTrue((root / "tables" / "planner_failure_mode_summary.csv").exists())
            self.assertTrue((root / "tables" / "planner_efficiency_synthesis.csv").exists())

    def test_planner_evidence_summary_exposes_stage_d_flags(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bundle = build_planner_ablation_evidence_bundle(
                self._suite(),
                figures_dir=Path(tmpdir) / "figs",
                tables_dir=Path(tmpdir) / "tables",
                json_output_path=Path(tmpdir) / "planner_evidence_bundle.json",
            )

            self.assertTrue(bundle.summary.planner_beats_top_k)
            self.assertTrue(bundle.summary.planner_reduces_simulations_vs_top_k)
            self.assertTrue(bundle.summary.planner_preserves_or_improves_feasible_hit_rate_vs_top_k)
            self.assertTrue(bundle.summary.planner_improves_efficiency_vs_top_k)
            self.assertTrue(bundle.summary.fidelity_escalation_effective)
            self.assertTrue(bundle.summary.fidelity_escalation_reduces_simulations)
            self.assertTrue(bundle.summary.phase_updates_effective)
            self.assertTrue(bundle.summary.phase_updates_improve_convergence)
            self.assertTrue(bundle.summary.phase_updates_observable)
            self.assertTrue(bundle.summary.calibration_replanning_effective)
            self.assertTrue(bundle.summary.calibration_replanning_improves_convergence)
            self.assertTrue(bundle.summary.calibration_replanning_observable)
            self.assertTrue(bundle.summary.rollout_guidance_effective)
            self.assertTrue(bundle.summary.rollout_guidance_improves_convergence)
            self.assertTrue(bundle.summary.rollout_guidance_preserves_or_improves_feasible_hit_rate)
            self.assertTrue(bundle.summary.rollout_guidance_observable)
            self.assertTrue(bundle.summary.rollout_claim_supported_without_mpc_overclaim)
            self.assertTrue(bundle.summary.rollout_claim_limited_to_short_horizon_guidance)
            self.assertTrue(bundle.summary.rollout_evidence_real_not_placeholder)
            self.assertFalse(bundle.summary.rollout_placeholder_risk)
            self.assertEqual(bundle.summary.rollout_claim_scope, "short_horizon_world_model_guidance")
            self.assertEqual(bundle.summary.rollout_claim_status, "supported_short_horizon_rollout_guidance")
            self.assertEqual(bundle.summary.dominant_failure_mode, "stability_failure")
            self.assertTrue(bundle.summary.planner_reduces_failure_pressure)
            self.assertTrue(bundle.summary.failure_synthesis_ready)
            self.assertTrue(bundle.summary.efficiency_synthesis_ready)
            self.assertTrue(bundle.summary.efficiency_frontier_consistent)


if __name__ == "__main__":
    unittest.main()
