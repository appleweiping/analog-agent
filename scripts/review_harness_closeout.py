"""Review whether the project should continue experiments or move to writing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.benchmark_protocol import (
    BASELINE_BENCHMARK_MODES,
    DEFAULT_BENCHMARK_MAX_SIMULATIONS,
    DEFAULT_BENCHMARK_REPEAT_RUNS,
    DEFAULT_BENCHMARK_STEPS,
)
from libs.eval.benchmark_registry import runnable_benchmark_ids
from scripts.check_configured_truth_readiness import build_status as configured_truth_status
from scripts.review_stage_c_world_model_readiness import build_status as world_model_status
from scripts.review_stage_d_planner_package import build_status as planner_status

REQUIRED_DOCS = [
    "AGENTS.md",
    "README.md",
    "docs/configured_truth_user_action_boundary.md",
    "docs/repo-map.md",
    "docs/related_work_map.md",
    "docs/stop_conditions.md",
]

REQUIRED_REVIEW_SCRIPTS = [
    "scripts/review_stage_b_truth_groundwork.py",
    "scripts/review_stage_c_world_model_readiness.py",
    "scripts/review_stage_d_planner_package.py",
    "scripts/review_stage_e_benchmark_package.py",
    "scripts/review_harness_closeout.py",
]

REFERENCE_ANCHORS = [
    "AnalogGym",
    "AICircuit",
    "AnalogCoder",
    "AnalogGenie",
    "OpenFASOC",
    "ALIGN",
    "VerilogEval",
    "RTLFixer",
    "HaVen",
    "ChipNeMo",
    "I-JEPA",
    "V-JEPA 2",
    "TD-MPC2",
    "DreamerV3",
]


def _path_exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def _reference_anchor_status() -> dict[str, object]:
    text = (REPO_ROOT / "docs" / "related_work_map.md").read_text(encoding="utf-8")
    missing = [anchor for anchor in REFERENCE_ANCHORS if anchor not in text]
    return {
        "anchor_count": len(REFERENCE_ANCHORS),
        "missing_anchors": missing,
        "anchors_ready": not missing,
    }


def _benchmark_depth_status() -> dict[str, object]:
    return {
        "default_steps": DEFAULT_BENCHMARK_STEPS,
        "default_repeat_runs": DEFAULT_BENCHMARK_REPEAT_RUNS,
        "default_max_simulations": DEFAULT_BENCHMARK_MAX_SIMULATIONS,
        "runnable_benchmarks": runnable_benchmark_ids(),
        "baseline_modes": list(BASELINE_BENCHMARK_MODES),
        "smoke_scale_only": DEFAULT_BENCHMARK_REPEAT_RUNS < 30 or DEFAULT_BENCHMARK_MAX_SIMULATIONS < 20,
    }


def build_status() -> dict[str, object]:
    """Build a structural closeout review without mutating files."""

    docs_status = {path: _path_exists(path) for path in REQUIRED_DOCS}
    script_status = {path: _path_exists(path) for path in REQUIRED_REVIEW_SCRIPTS}
    truth = configured_truth_status()
    world_model = world_model_status()
    planner = planner_status()
    related_work = _reference_anchor_status()
    benchmark_depth = _benchmark_depth_status()

    blockers: list[str] = []
    if not all(docs_status.values()):
        blockers.append("required_harness_docs_missing")
    if not all(script_status.values()):
        blockers.append("required_review_scripts_missing")
    if truth["readiness_state"] != "configured_truth_ready":
        blockers.append("configured_truth_not_ready")
    if benchmark_depth["smoke_scale_only"]:
        blockers.append("benchmark_depth_smoke_scale_only")
    if not world_model["ready_for_stage_d"]:
        blockers.append("world_model_stage_not_ready")
    if not planner["ready_for_stage_e"]:
        blockers.append("planner_stage_not_ready")
    if not related_work["anchors_ready"]:
        blockers.append("related_work_anchor_map_incomplete")

    ready_for_writing = not blockers
    return {
        "stage": "Harness Closeout",
        "closeout_status": "ready_for_writing" if ready_for_writing else "need_more_evidence",
        "ready_for_writing": ready_for_writing,
        "safe_current_claim": "calibrated surrogate-guided analog sizing loop under explicit SPICE truth boundaries",
        "required_docs": docs_status,
        "required_review_scripts": script_status,
        "configured_truth_state": truth["readiness_state"],
        "benchmark_depth": benchmark_depth,
        "world_model_stage_status": world_model["stage_status"],
        "planner_stage_status": planner["stage_status"],
        "related_work": related_work,
        "blockers": blockers,
        "next_plan": (
            "freeze experiments and move to writing"
            if ready_for_writing
            else "resolve blockers before claiming the experiment stage is complete"
        ),
        "stop_decision": "stop_and_write" if ready_for_writing else "continue_experiments",
    }


def main() -> None:
    print(json.dumps(build_status(), indent=2))


if __name__ == "__main__":
    main()
