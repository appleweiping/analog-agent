"""Run the OTA v1 submission-ready freeze check and write the closure report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.schema.experiment import ExperimentBudget
from libs.vertical_slices.ota2 import run_ota_submission_ready_freeze
from libs.vertical_slices.ota2_spec import load_ota2_v1_config


def _bool_line(label: str, value: bool, detail: str | None = None) -> str:
    suffix = f"：{detail}" if detail else ""
    return f"- {label}：{'yes' if value else 'no'}{suffix}"


def _render_report(result) -> str:
    config = load_ota2_v1_config()
    check = result.final_check_summary
    method = result.method_conclusions
    baseline_stats = result.baseline_suite.aggregated_stats
    methodology_stats = result.methodology_suite.aggregated_stats
    return "\n".join(
        [
            "# April 20th System Closure Report",
            "",
            "## Final Status",
            "",
            check.closure_statement,
            "",
            f"- submission_ready：{'yes' if check.submission_ready else 'no'}",
            f"- current_truth_level：`{check.current_truth_level}`",
            f"- ota_vertical_slice：`{config.version}`",
            "",
            "## Layer Status",
            "",
            "### L5",
            _bool_line("ngspice real backend is the primary path", check.l5_real_backend_primary),
            _bool_line("quick_truth is established", check.l5_quick_truth_established),
            _bool_line("focused_truth is established", check.l5_focused_truth_established),
            _bool_line("measurement contract is stable", check.l5_measurement_contract_stable),
            "",
            "### L3",
            _bool_line("real calibration feedback is consumed", check.l3_consumes_real_calibration_feedback),
            _bool_line("world model is calibratable rather than static", check.l3_world_model_is_calibratable),
            "",
            "### L4",
            _bool_line("search state updates from real verification", check.l4_updates_search_from_verification),
            _bool_line("budget-aware and fidelity-aware decision is active", check.l4_budget_and_fidelity_aware),
            "",
            "### L6",
            _bool_line("real episode memory is persisted", check.l6_persists_real_episode_memory),
            _bool_line("demonstrator vs configured truth is distinguished", check.l6_distinguishes_truth_levels),
            "",
            "### System",
            _bool_line("L2→L6 closed loop is established", check.l2_to_l6_closed_loop),
            _bool_line("formal acceptance suite exists", check.acceptance_suite_available),
            _bool_line("statistics foundation exists", check.stats_foundation_available),
            "",
            "## Physical Validity Statement",
            "",
            f"- current truth level：`{check.current_truth_level}`",
            f"- real PDK connected：{'yes' if check.real_pdk_connected else 'no'}",
            "- default backend：`ngspice`",
            "- default model binding：`builtin`",
            "- physical validity scope：The current system proves that real SPICE participates in the loop under demonstrator-level physical validity. It does not claim industrial-grade accuracy, full PDK fidelity, or production sign-off equivalence.",
            "",
            "## Methodology Conclusions",
            "",
            f"- world model：{'有用' if method.conclusions.world_model_effective else '无明显作用'}",
            f"- calibration：{'有用' if method.conclusions.calibration_effective else '无明显作用'}",
            f"- fidelity escalation：{'有用' if method.conclusions.fidelity_effective else '无明显作用'}",
            "",
            "Conclusion notes:",
            *[f"- {note}" for note in method.conclusions.conclusion_notes],
            "",
            "## Experiment Capability",
            "",
            "- supported vertical slice：`two_stage_ota / ota2_v1`",
            "- baseline comparison：Day4 baseline runner is available",
            "- methodology comparison：Day11 component-switch runner is available",
            "- automatic stats：Day9 stats export is available",
            f"- multi-task support：{'yes' if check.multi_task_supported else 'no'}",
            "",
            "## Final System Check",
            "",
            _bool_line("run_ota_acceptance()", check.ota_v1_acceptance_ok),
            _bool_line("run_ota_experiment_suite()", check.ota_v1_experiment_ok),
            _bool_line("stats export", check.stats_export_ok),
            _bool_line("MethodComparisonResult", check.method_comparison_ok),
            "",
            "## Freeze Rules",
            "",
            "- OTA v1 is the frozen paper-facing path and must not be modified in-place.",
            "- Current L3-L4-L5-L6 schema and their OTA v1 behavior are frozen for submission.",
            "- Any future change must either create a new version (for example `v2`) or preserve OTA v1 results and semantics exactly.",
            "",
            "## Known Limits",
            "",
            "- Only `two_stage_ota` is currently supported as the stable submission path.",
            "- Current truth level is `demonstrator_truth`.",
            "- No real external PDK/model-card is connected by default.",
            "- No full PVT or Monte Carlo verification is included.",
            "- The world model remains heuristic/proxy-based rather than a trained high-capacity predictor.",
            "- Fidelity support currently stops at `quick_truth` and `focused_truth`.",
            "",
            "## Future Work",
            "",
            "- Extend the vertical-slice path to LDO and bandgap.",
            "- Upgrade to `configured_truth` with external model cards / real PDK bindings.",
            "- Add multi-backend consistency checks with Xyce and Spectre-compatible execution.",
            "- Replace the current heuristic world model with a stronger learned model.",
            "- Expand fidelity beyond quick/focused truth to full PVT and Monte Carlo.",
            "",
            "## Aggregate Snapshot",
            "",
            f"- baseline total_real_simulation_calls：{baseline_stats.total_real_simulation_calls if baseline_stats is not None else 'n/a'}",
            f"- methodology total_real_simulation_calls：{methodology_stats.total_real_simulation_calls if methodology_stats is not None else 'n/a'}",
            f"- baseline simulation_reduction_ratio：{baseline_stats.simulation_reduction_ratio if baseline_stats is not None else 'n/a'}",
            f"- methodology focused_truth_ratio：{methodology_stats.fidelity_summary.focused_truth_ratio if methodology_stats is not None else 'n/a'}",
            "",
        ]
    )


def main() -> None:
    output_dir = REPO_ROOT / "research" / "implementation_notes"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_ota_submission_ready_freeze(
        acceptance_steps=2,
        experiment_steps=2,
        repeat_runs=1,
        budget=ExperimentBudget(max_simulations=4, max_candidates_per_step=2),
    )
    report_path = output_dir / "April_20th_system_closure_report.md"
    summary_path = output_dir / "April_20th_system_closure_summary.json"
    report_path.write_text(_render_report(result), encoding="utf-8")
    summary_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(summary_path), "submission_ready": result.final_check_summary.submission_ready}, indent=2))


if __name__ == "__main__":
    main()
