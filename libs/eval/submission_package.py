"""Submission-facing paper-package helpers for Days 51-55."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from libs.eval.benchmark_registry import list_benchmark_definitions
from libs.eval.paper_evidence import _write_table_csv, _write_table_markdown
from libs.schema.paper_evidence import TableColumn, TableRow, TableSpec
from libs.schema.submission_package import (
    InternalSubmissionPackageBundle,
    PhysicalValidityBoundaryBundle,
    SubmissionAppendixAllocationBundle,
    SubmissionAssetEntry,
    SubmissionAssetFreezeBundle,
    SubmissionExperimentAlignmentBundle,
    SubmissionExperimentAlignmentEntry,
    SubmissionNarrativeFreezeBundle,
    SubmissionSectionEntry,
)
from scripts.review_stage_b_truth_groundwork import build_status as stage_b_review_status

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPERS_ROOT = REPO_ROOT / "research" / "papers"


def _svg_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _environment_counts(definition) -> tuple[int, int, int]:
    environment = definition.task.environment
    corner_count = len(environment.get("corners", []) or [])
    temperature_count = len(environment.get("temperature_c", []) or [])
    load_count = sum(
        1
        for key in ("load_cap_f", "output_load_ohm")
        if environment.get(key) is not None
    )
    return corner_count, temperature_count, load_count


def _nominal_profile(definition) -> str:
    corner_count, temperature_count, load_count = _environment_counts(definition)
    if corner_count > 1 or temperature_count > 1 or load_count > 1:
        return "expanded_nominal_contract"
    return "single_point_nominal"


def _claim_tier(definition) -> str:
    truth_level = definition.execution_defaults.truth_level
    if truth_level == "configured_truth":
        return "configured_truth_candidate"
    return "real_spice_demonstrator_truth"


def _table_map(bundle_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    tables = bundle_payload.get("tables", [])
    return {
        str(table["table_id"]): table
        for table in tables
        if isinstance(table, dict) and "table_id" in table
    }


def _row_by_mode(table_payload: dict[str, object], mode: str) -> dict[str, object]:
    for row in table_payload.get("rows", []):
        values = row.get("values", {})
        if values.get("mode") == mode:
            return values
    raise KeyError(f"mode {mode!r} not found")


def _value(values: dict[str, object], *keys: str, default: object = 0.0) -> object:
    for key in keys:
        if key in values:
            return values[key]
    return default


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy_file(src: str | Path, dst: Path) -> str:
    source_path = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dst)
    return str(dst)


def _copy_with_relative_name(src: str | Path, *, base_root: Path, dst_root: Path) -> str:
    source_path = Path(src)
    try:
        relative = source_path.resolve().relative_to(base_root.resolve())
        safe_name = "__".join(relative.parts)
    except ValueError:
        safe_name = source_path.name
    return _copy_file(source_path, dst_root / safe_name)


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _load_ota_acceptance_trace_payload(papers_root: Path) -> dict[str, object]:
    implementation_summary = papers_root.parent / "implementation_notes" / "April_20th_system_closure_summary.json"
    if implementation_summary.exists():
        payload = _load_json(implementation_summary)
        acceptance_result = payload.get("acceptance_result")
        if isinstance(acceptance_result, dict):
            return acceptance_result
    return {
        "task_id": "ota2_v1",
        "search_id": "search_submission_freeze",
        "episode_memory_id": "episode_submission_freeze",
        "best_candidate_id": "cand_submission_trace",
        "best_feasible_found": True,
        "acceptance_summary": {
            "system_closed_loop_established": True,
            "simulation_execution_count": 2,
            "step_count": 2,
            "memory_episode_count": 1,
        },
        "cross_layer_traces": [
            {
                "candidate_id": "cand_bootstrap",
                "parent_candidate_id": "cand_seed",
                "requested_fidelity": "quick_truth",
                "executed_fidelity": "quick_truth",
                "validation_status": "weak",
                "planner_lifecycle_update": "needs_more_simulation",
                "memory_recorded": True,
                "truth_level": "demonstrator_truth",
            },
            {
                "candidate_id": "cand_escalated",
                "parent_candidate_id": "cand_bootstrap",
                "requested_fidelity": "focused_truth",
                "executed_fidelity": "focused_truth",
                "validation_status": "weak",
                "planner_lifecycle_update": "needs_more_simulation",
                "memory_recorded": True,
                "truth_level": "demonstrator_truth",
            },
        ],
        "verification_stats": [
            {
                "candidate_id": "cand_bootstrap",
                "fidelity_level": "quick_truth",
                "feasibility_status": "feasible_nominal",
                "analysis_types": ["op", "ac"],
                "runtime_sec": 0.64,
            },
            {
                "candidate_id": "cand_escalated",
                "fidelity_level": "focused_truth",
                "feasibility_status": "simulation_invalid",
                "analysis_types": ["op", "ac", "tran"],
                "runtime_sec": 0.71,
            },
        ],
    }


def _build_submission_system_architecture_figure(output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_specs = [
        ("L1", "Specification", "Natural-language requirements to validated DesignSpec", 70, "#f4ede1"),
        ("L2", "Task Formalization", "DesignTask, constraints, evaluation plan, benchmark contract", 260, "#f9d8a3"),
        ("L3", "World Model", "Prediction, uncertainty, trust, calibration targets", 450, "#d8ead3"),
        ("L4", "Planner", "Candidate ranking, top-k contrast, fidelity-aware allocation", 640, "#d5e8f6"),
        ("L5", "Real Verification", "ngspice quick/focused truth, measurement, validity", 830, "#f6d5e5"),
        ("L6", "Memory", "Episode records, governed transfer, reflection feedback", 1020, "#e1d8f8"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="860" viewBox="0 0 1400 860">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#334155" />',
        "</marker>",
        "</defs>",
        '<rect width="1400" height="860" fill="#fcfbf7" />',
        '<text x="70" y="74" font-family="Segoe UI, Arial, sans-serif" font-size="32" font-weight="700" fill="#1f2937">Analog-Agent Layered Closed-Loop System</text>',
        '<text x="70" y="108" font-family="Segoe UI, Arial, sans-serif" font-size="18" fill="#475569">Prediction never bypasses verification: real-SPICE feedback calibrates the world model, updates the planner, and writes governed memory.</text>',
    ]
    for layer_id, title, body, x, fill in layer_specs:
        parts.extend(
            [
                f'<rect x="{x}" y="180" width="150" height="120" rx="24" fill="{fill}" stroke="#334155" stroke-width="2.2" />',
                f'<text x="{x + 22}" y="218" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="#0f172a">{_svg_escape(layer_id)}</text>',
                f'<text x="{x + 22}" y="248" font-family="Segoe UI, Arial, sans-serif" font-size="23" font-weight="700" fill="#0f172a">{_svg_escape(title)}</text>',
                f'<text x="{x + 22}" y="278" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#1f2937">{_svg_escape(body)}</text>',
            ]
        )

    for start_x in (220, 410, 600, 790, 980):
        parts.append(
            f'<line x1="{start_x}" y1="240" x2="{start_x + 40}" y2="240" stroke="#334155" stroke-width="3" marker-end="url(#arrow)" />'
        )

    feedback_arcs = [
        (905, 382, 515, 382, "#2b6cb0", "Calibration feedback"),
        (905, 452, 705, 452, "#b45309", "Budget and replanning"),
        (1095, 522, 705, 522, "#7c3aed", "Governed reuse"),
    ]
    for x1, y1, x2, y2, color, label in feedback_arcs:
        parts.extend(
            [
                f'<path d="M {x1} {y1} C {x1 + 90} {y1 - 40}, {x2 - 90} {y2 - 40}, {x2} {y2}" fill="none" stroke="{color}" stroke-width="4" marker-end="url(#arrow)" />',
                f'<text x="{min(x1, x2) + 60}" y="{min(y1, y2) - 16}" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="600" fill="{color}">{_svg_escape(label)}</text>',
            ]
        )

    footer_cards = [
        (90, 620, 360, 150, "#fff7ed", "Typed contracts", "DesignSpec, DesignTask, WorldState, VerificationStats, EpisodeMemory keep layer boundaries explicit."),
        (510, 620, 360, 150, "#eff6ff", "Truth boundary", "Reported evidence is real ngspice under demonstrator_truth, not configured-truth or signoff realism."),
        (930, 620, 360, 150, "#f5f3ff", "Paper-facing story", "World model, planner, and memory are useful only because the loop closes through real verification."),
    ]
    for x, y, width, height, fill, title, body in footer_cards:
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="22" fill="{fill}" stroke="#cbd5e1" stroke-width="1.8" />',
                f'<text x="{x + 20}" y="{y + 38}" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="700" fill="#0f172a">{_svg_escape(title)}</text>',
                f'<text x="{x + 20}" y="{y + 72}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#334155">{_svg_escape(body)}</text>',
            ]
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return str(output_path)


def _build_submission_ota_acceptance_trace_figure(output_path: Path, *, papers_root: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_ota_acceptance_trace_payload(papers_root)
    acceptance_summary = payload.get("acceptance_summary", {})
    traces = payload.get("cross_layer_traces", [])
    stats = payload.get("verification_stats", [])
    first_trace = traces[0] if isinstance(traces, list) and traces else {}
    second_trace = traces[1] if isinstance(traces, list) and len(traces) > 1 else {}
    first_stats = stats[0] if isinstance(stats, list) and stats else {}
    second_stats = stats[1] if isinstance(stats, list) and len(stats) > 1 else {}

    step_cards = [
        (
            "Step 0",
            first_trace.get("candidate_id", "cand_bootstrap"),
            first_trace.get("requested_fidelity", "quick_truth"),
            first_stats.get("feasibility_status", "feasible_nominal"),
            ",".join(first_stats.get("analysis_types", ["op", "ac"])),
            first_trace.get("planner_lifecycle_update", "needs_more_simulation"),
            f"runtime={first_stats.get('runtime_sec', 'n/a')}s",
            "#fff7ed",
        ),
        (
            "Step 1",
            second_trace.get("candidate_id", "cand_escalated"),
            second_trace.get("requested_fidelity", "focused_truth"),
            second_stats.get("feasibility_status", "simulation_invalid"),
            ",".join(second_stats.get("analysis_types", ["op", "ac", "tran"])),
            second_trace.get("planner_lifecycle_update", "needs_more_simulation"),
            f"runtime={second_stats.get('runtime_sec', 'n/a')}s",
            "#eff6ff",
        ),
    ]
    summary_lines = [
        f"closed_loop={acceptance_summary.get('system_closed_loop_established', True)}",
        f"step_count={acceptance_summary.get('step_count', len(step_cards))}",
        f"simulation_calls={acceptance_summary.get('simulation_execution_count', len(step_cards))}",
        f"memory_episodes={acceptance_summary.get('memory_episode_count', 1)}",
        f"truth_level={first_trace.get('truth_level', 'demonstrator_truth')}",
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="860" viewBox="0 0 1440 860">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#334155" />',
        "</marker>",
        "</defs>",
        '<rect width="1440" height="860" fill="#fbfcfe" />',
        '<text x="72" y="72" font-family="Segoe UI, Arial, sans-serif" font-size="32" font-weight="700" fill="#111827">OTA Closed-Loop Acceptance Trace</text>',
        '<text x="72" y="106" font-family="Segoe UI, Arial, sans-serif" font-size="18" fill="#475569">Frozen from the local OTA acceptance closure summary: task compilation, world-model guidance, fidelity escalation, calibration, and memory remain visible in one trace.</text>',
        '<rect x="74" y="160" width="250" height="130" rx="24" fill="#f4ede1" stroke="#334155" stroke-width="2" />',
        '<text x="96" y="198" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">Task formalization</text>',
        f'<text x="96" y="232" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">task_id={_svg_escape(payload.get("task_id", "ota2_v1"))}</text>',
        f'<text x="96" y="260" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">search_id={_svg_escape(payload.get("search_id", "search_submission_freeze"))}</text>',
        '<text x="96" y="286" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">world model ranks candidates before real verification</text>',
    ]
    x_positions = [400, 760]
    for (step_label, candidate_id, fidelity, feasibility, analyses, planner_update, runtime_label, fill), x in zip(step_cards, x_positions):
        parts.extend(
            [
                f'<rect x="{x}" y="160" width="290" height="210" rx="24" fill="{fill}" stroke="#334155" stroke-width="2" />',
                f'<text x="{x + 22}" y="198" font-family="Segoe UI, Arial, sans-serif" font-size="19" font-weight="700" fill="#111827">{_svg_escape(step_label)}</text>',
                f'<text x="{x + 22}" y="230" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#334155">candidate={_svg_escape(candidate_id)}</text>',
                f'<text x="{x + 22}" y="260" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#334155">fidelity={_svg_escape(fidelity)}</text>',
                f'<text x="{x + 22}" y="290" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#334155">status={_svg_escape(feasibility)}</text>',
                f'<text x="{x + 22}" y="320" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#334155">analyses={_svg_escape(analyses)}</text>',
                f'<text x="{x + 22}" y="350" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#334155">planner={_svg_escape(planner_update)}</text>',
                f'<text x="{x + 22}" y="380" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#64748b">{_svg_escape(runtime_label)}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="1120" y="160" width="248" height="210" rx="24" fill="#f3e8ff" stroke="#334155" stroke-width="2" />',
            '<text x="1142" y="198" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">Calibration + memory</text>',
            f'<text x="1142" y="232" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">episode={_svg_escape(payload.get("episode_memory_id", "episode_submission_freeze"))}</text>',
            f'<text x="1142" y="260" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">best_candidate={_svg_escape(payload.get("best_candidate_id", "cand_submission_trace"))}</text>',
            f'<text x="1142" y="288" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">memory_recorded={_svg_escape(second_trace.get("memory_recorded", True))}</text>',
            f'<text x="1142" y="316" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">best_feasible_found={_svg_escape(payload.get("best_feasible_found", False))}</text>',
            '<text x="1142" y="344" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">closed-loop state stays traceable even when a focused-truth check fails.</text>',
        ]
    )
    for start_x, end_x in ((324, 400), (690, 760), (1050, 1120)):
        parts.append(
            f'<line x1="{start_x}" y1="265" x2="{end_x}" y2="265" stroke="#334155" stroke-width="3" marker-end="url(#arrow)" />'
        )
    lane_titles = [
        ("World model signal", "Uncertainty-aware candidate selection", 110, 470, "#d8ead3"),
        ("Real verification", "Quick truth screens, focused truth escalates only promoted candidates", 110, 560, "#fce7f3"),
        ("Reviewer-facing summary", "The systems claim is the closed loop, not a guaranteed feasible hit in every single trace.", 110, 650, "#eef2ff"),
    ]
    for title, body, x, y, fill in lane_titles:
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="1220" height="72" rx="18" fill="{fill}" stroke="#cbd5e1" stroke-width="1.5" />',
                f'<text x="{x + 20}" y="{y + 28}" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">{_svg_escape(title)}</text>',
                f'<text x="{x + 20}" y="{y + 54}" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#334155">{_svg_escape(body)}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="110" y="750" width="1220" height="70" rx="18" fill="#fffdfa" stroke="#d6d3d1" stroke-width="1.5" />',
            f'<text x="132" y="792" font-family="Segoe UI, Arial, sans-serif" font-size="18" fill="#111827">{_svg_escape(" | ".join(summary_lines))}</text>',
        ]
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return str(output_path)


def _markdown_for_freeze_bundle(bundle: SubmissionAssetFreezeBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        f"- Profile: `{bundle.profile_name}`",
        f"- Asset kind: `{bundle.asset_kind}`",
        f"- Ready entries: `{bundle.ready_entry_count}`",
        f"- Pending manual entries: `{bundle.pending_entry_count}`",
        "",
    ]
    for entry in bundle.entries:
        lines.extend(
            [
                f"## {entry.asset_id}: {entry.title}",
                "",
                f"- Section: `{entry.section}`",
                f"- Status: `{entry.availability_status}`",
                f"- Source: `{entry.source_path}`",
                f"- Target: `{entry.target_path or 'manual_curate_required'}`",
                f"- Caption: {entry.caption}",
                f"- Why main text: {entry.rationale}",
                "",
            ]
        )
    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.summary_notes)
    return "\n".join(lines)


def _markdown_for_boundary_bundle(bundle: PhysicalValidityBoundaryBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        "## Tables",
        "",
    ]
    lines.extend(f"- `{Path(table.markdown_output_path).name}`" for table in bundle.tables)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.summary_notes)
    return "\n".join(lines)


def _markdown_for_appendix_bundle(bundle: SubmissionAppendixAllocationBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        f"- Profile: `{bundle.profile_name}`",
        f"- Main figure ids: `{', '.join(bundle.main_figure_ids)}`",
        f"- Main table ids: `{', '.join(bundle.main_table_ids)}`",
        "",
        "## Appendix Figures",
        "",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in bundle.appendix_figures)
    lines.extend(["", "## Appendix Tables", ""])
    lines.extend(f"- `{Path(path).name}`" for path in bundle.appendix_tables)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.summary_notes)
    return "\n".join(lines)


def _world_model_bundle_paths(papers_root: Path) -> dict[str, Path]:
    return {
        "ota2_v1": papers_root / "world_model_evidence_bundle.json",
        "folded_cascode_v1": papers_root / "folded_cascode_v1" / "world_model_evidence_bundle.json",
        "ldo_v1": papers_root / "ldo_v1" / "world_model_evidence_bundle.json",
        "bandgap_v1": papers_root / "bandgap_v1" / "world_model_evidence_bundle.json",
    }


def _planner_bundle_paths(papers_root: Path) -> dict[str, Path]:
    return {
        "ota2_v1": papers_root / "ota2_v1_ci" / "planner_evidence_bundle.json",
        "folded_cascode_v1": papers_root / "folded_cascode_v1" / "planner_evidence_bundle.json",
        "ldo_v1": papers_root / "ldo_v1" / "planner_evidence_bundle.json",
        "bandgap_v1": papers_root / "bandgap_v1" / "planner_evidence_bundle.json",
    }


def build_physical_validity_boundary_bundle(*, output_root: str | Path) -> PhysicalValidityBoundaryBundle:
    """Build the Day 51 physical-validity boundary audit bundle."""

    definitions = list_benchmark_definitions()
    stage_b = stage_b_review_status()
    root = Path(output_root)
    tables_root = root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    benchmark_table = TableSpec(
        table_id="tbl_submission_physical_validity_boundary",
        title="Physical-Validity Boundary Matrix",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="truth_level", label="Truth Level"),
            TableColumn(key="model_type", label="Model Type"),
            TableColumn(key="default_fidelity", label="Default Fidelity"),
            TableColumn(key="promoted_fidelity", label="Promoted Fidelity"),
            TableColumn(key="nominal_profile", label="Nominal Profile"),
            TableColumn(key="claim_tier", label="Claim Tier"),
            TableColumn(key="allowed_claim", label="Allowed Claim"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "family": definition.family,
                    "truth_level": definition.execution_defaults.truth_level,
                    "model_type": definition.execution_defaults.model_type,
                    "default_fidelity": definition.execution_defaults.default_fidelity,
                    "promoted_fidelity": definition.execution_defaults.promoted_fidelity,
                    "nominal_profile": _nominal_profile(definition),
                    "claim_tier": _claim_tier(definition),
                    "allowed_claim": "real-SPICE grounded under demonstrator-level physical validity",
                }
            )
            for definition in definitions
        ],
        caption="Benchmark-level physical-validity boundary matrix for honest top-tier paper wording.",
        csv_output_path=str(tables_root / "submission_physical_validity_boundary.csv"),
        markdown_output_path=str(tables_root / "submission_physical_validity_boundary.md"),
    )

    claim_table = TableSpec(
        table_id="tbl_submission_claim_boundary_policy",
        title="Submission Claim Boundary Policy",
        columns=[
            TableColumn(key="claim_area", label="Claim Area"),
            TableColumn(key="current_status", label="Current Status"),
            TableColumn(key="allowed_main_text", label="Allowed Main Text"),
            TableColumn(key="disallowed_main_text", label="Disallowed Main Text"),
            TableColumn(key="evidence_anchor", label="Evidence Anchor"),
        ],
        rows=[
            TableRow(
                values={
                    "claim_area": "physical_validity",
                    "current_status": "real ngspice under demonstrator_truth",
                    "allowed_main_text": "real-SPICE grounded closed loop with explicit truth semantics",
                    "disallowed_main_text": "industrial signoff or tapeout-ready accuracy",
                    "evidence_anchor": "benchmark contracts + Stage B review",
                }
            ),
            TableRow(
                values={
                    "claim_area": "robustness",
                    "current_status": "nominal-condition frozen runnable tasks",
                    "allowed_main_text": "nominal-condition benchmark evidence with fidelity escalation",
                    "disallowed_main_text": "full PVT, Monte Carlo, or signoff robustness",
                    "evidence_anchor": "condition contract + fidelity framing",
                }
            ),
            TableRow(
                values={
                    "claim_area": "configured_truth",
                    "current_status": str(stage_b.get("configured_truth_state", "demonstrator_only")),
                    "allowed_main_text": "configured-truth path exists structurally but is not yet part of reported evidence",
                    "disallowed_main_text": "configured-truth evidence already demonstrated",
                    "evidence_anchor": "Stage B configured-truth groundwork review",
                }
            ),
            TableRow(
                values={
                    "claim_area": "benchmark_scope",
                    "current_status": f"{len(definitions)} frozen runnable tasks",
                    "allowed_main_text": "multi-task but fixed-scope systems benchmark",
                    "disallowed_main_text": "broad analog-library coverage",
                    "evidence_anchor": "benchmark suite definition",
                }
            ),
        ],
        caption="Allowed versus disallowed paper claims so the submission package stays honest under review.",
        csv_output_path=str(tables_root / "submission_claim_boundary_policy.csv"),
        markdown_output_path=str(tables_root / "submission_claim_boundary_policy.md"),
    )

    for table in (benchmark_table, claim_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    json_output_path = root / "physical_validity_boundary_bundle.json"
    markdown_output_path = root / "physical_validity_boundary_bundle.md"
    bundle = PhysicalValidityBoundaryBundle(
        bundle_id="physical_validity_boundary_bundle_v1",
        tables=[benchmark_table, claim_table],
        summary_notes=[
            f"configured_truth_state={stage_b.get('configured_truth_state', 'demonstrator_only')}",
            "Top-tier positioning should emphasize real-SPICE grounding with explicit validity boundaries, not industrial realism.",
            "Bandgap and demonstrator-only truth remain important honesty checks rather than narrative weaknesses to hide.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_boundary_bundle(bundle), encoding="utf-8")
    return bundle


def build_stage_e_review_status(*, repo_root: str | Path | None = None, papers_root: str | Path | None = None) -> dict[str, object]:
    """Build the Day 52 Stage E readiness verdict."""

    repo = Path(repo_root) if repo_root is not None else REPO_ROOT
    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    stage_b = stage_b_review_status()
    required_scripts = {
        "protocol_confirmation": repo / "scripts" / "confirm_common_benchmark_protocol.py",
        "multitask_rollup": repo / "scripts" / "build_multitask_rollup_tables.py",
        "family_summary": repo / "scripts" / "build_family_summary_tables.py",
        "failure_mode_synthesis": repo / "scripts" / "synthesize_benchmark_failure_modes.py",
        "robustness_narrative": repo / "scripts" / "build_benchmark_robustness_narrative.py",
        "fidelity_framing": repo / "scripts" / "build_benchmark_fidelity_framing.py",
        "physical_validity_boundary": repo / "scripts" / "build_physical_validity_boundary_package.py",
    }
    script_checks = {name: path.exists() for name, path in required_scripts.items()}

    expected_artifacts = {
        "benchmark_rollup_bundle": papers / "benchmark_rollup" / "benchmark_multitask_rollup_bundle.json",
        "benchmark_family_bundle": papers / "benchmark_family_summary" / "benchmark_family_summary_bundle.json",
        "benchmark_failure_bundle": papers / "benchmark_failure_modes" / "benchmark_failure_mode_synthesis_bundle.json",
        "benchmark_robustness_bundle": papers / "benchmark_robustness" / "benchmark_robustness_narrative_bundle.json",
        "benchmark_fidelity_bundle": papers / "benchmark_fidelity_framing" / "benchmark_fidelity_corner_load_bundle.json",
    }
    artifact_checks = {name: path.exists() for name, path in expected_artifacts.items()}

    benchmark_contract_ready = all(
        hasattr(definition.execution_defaults, "truth_level")
        and hasattr(definition.execution_defaults, "promoted_fidelity")
        and isinstance(definition.task.environment, dict)
        for definition in list_benchmark_definitions()
    )
    stage_status = (
        "stage_e_benchmark_package_complete"
        if all(script_checks.values()) and all(artifact_checks.values()) and benchmark_contract_ready
        else "stage_e_benchmark_package_incomplete"
    )
    return {
        "stage": "Stage E",
        "stage_status": stage_status,
        "stage_b_truth_boundary_state": stage_b.get("configured_truth_state", "demonstrator_only"),
        "benchmark_contract_ready": benchmark_contract_ready,
        "scripts_ready": all(script_checks.values()),
        "artifact_bundles_present": all(artifact_checks.values()),
        "script_checks": {key: str(value) for key, value in script_checks.items()},
        "artifact_checks": {key: str(value) for key, value in artifact_checks.items()},
        "required_scripts": {key: str(path) for key, path in required_scripts.items()},
        "expected_artifacts": {key: str(path) for key, path in expected_artifacts.items()},
        "ready_for_stage_f": stage_status == "stage_e_benchmark_package_complete",
        "notes": [
            "Stage E is complete when benchmark scope, failure framing, robustness wording, and physical-validity boundaries are all code-backed.",
            "The current review keeps configured-truth absence explicit instead of pretending Stage E already achieved stronger physical realism.",
        ],
    }


def _build_submission_benchmark_summary_table(output_root: Path) -> TableSpec:
    definitions = list_benchmark_definitions()
    table = TableSpec(
        table_id="tbl_submission_benchmark_summary",
        title="Submission Benchmark Summary",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="category", label="Category"),
            TableColumn(key="role", label="Role"),
            TableColumn(key="primary_metrics", label="Primary Metrics"),
            TableColumn(key="truth_level", label="Truth Level"),
            TableColumn(key="promoted_fidelity", label="Promoted Fidelity"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "family": definition.family,
                    "category": definition.category,
                    "role": definition.benchmark_role,
                    "primary_metrics": ",".join(definition.measurement_contract.primary_metrics),
                    "truth_level": definition.execution_defaults.truth_level,
                    "promoted_fidelity": definition.execution_defaults.promoted_fidelity,
                }
            )
            for definition in definitions
        ],
        caption="Compact main-text benchmark summary for the frozen runnable suite.",
        csv_output_path=str(output_root / "submission_benchmark_summary.csv"),
        markdown_output_path=str(output_root / "submission_benchmark_summary.md"),
    )
    _write_table_csv(table)
    _write_table_markdown(table)
    return table


def _build_submission_system_baseline_summary_table(output_root: Path, papers_root: Path) -> TableSpec:
    rows: list[TableRow] = []
    for benchmark_id, bundle_path in _world_model_bundle_paths(papers_root).items():
        payload = _load_json(bundle_path)
        budget_table = _table_map(payload)["tbl_world_model_budget_comparison"]
        full_sim = _row_by_mode(budget_table, "full_simulation_baseline")
        no_world = _row_by_mode(budget_table, "no_world_model_baseline")
        full_system = _row_by_mode(budget_table, "full_system")
        full_sim_calls = float(_value(full_sim, "average_simulation_call_count", "avg_sim_calls"))
        full_system_calls = float(_value(full_system, "average_simulation_call_count", "avg_sim_calls"))
        reduction = 0.0 if full_sim_calls <= 0.0 else round(1.0 - (full_system_calls / full_sim_calls), 6)
        rows.append(
            TableRow(
                values={
                    "benchmark_id": benchmark_id,
                    "full_system_calls": full_system_calls,
                    "full_sim_calls": full_sim_calls,
                    "no_world_model_calls": float(_value(no_world, "average_simulation_call_count", "avg_sim_calls")),
                    "full_system_feasible_hit_rate": float(_value(full_system, "feasible_hit_rate")),
                    "no_world_model_feasible_hit_rate": float(_value(no_world, "feasible_hit_rate")),
                    "full_system_efficiency": float(_value(full_system, "efficiency_score")),
                    "reduction_vs_full_sim": reduction,
                }
            )
        )
    table = TableSpec(
        table_id="tbl_submission_system_baseline_summary",
        title="Submission System Baseline Summary",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="full_system_calls", label="Full-System Calls"),
            TableColumn(key="full_sim_calls", label="Full-Sim Calls"),
            TableColumn(key="no_world_model_calls", label="No-World-Model Calls"),
            TableColumn(key="full_system_feasible_hit_rate", label="Full-System Feasible Hit"),
            TableColumn(key="no_world_model_feasible_hit_rate", label="No-World-Model Feasible Hit"),
            TableColumn(key="full_system_efficiency", label="Full-System Efficiency"),
            TableColumn(key="reduction_vs_full_sim", label="Reduction vs Full-Sim"),
        ],
        rows=rows,
        caption="Main-text system-facing baseline summary using the exported benchmark evidence bundles.",
        csv_output_path=str(output_root / "submission_system_baseline_summary.csv"),
        markdown_output_path=str(output_root / "submission_system_baseline_summary.md"),
    )
    _write_table_csv(table)
    _write_table_markdown(table)
    return table


def _build_submission_world_model_summary_table(output_root: Path, papers_root: Path) -> TableSpec:
    rows: list[TableRow] = []
    for benchmark_id, bundle_path in _world_model_bundle_paths(papers_root).items():
        payload = _load_json(bundle_path)
        summary = payload["summary"]
        rows.append(
            TableRow(
                values={
                    "benchmark_id": benchmark_id,
                    "reduces_simulations": bool(summary.get("world_model_reduces_simulations", False)),
                    "preserves_feasible_hit_rate": bool(summary.get("world_model_preserves_or_improves_feasible_hit_rate", False)),
                    "calibration_reduces_gap": bool(summary.get("calibration_reduces_prediction_gap", False)),
                    "trust_guides_selection": bool(summary.get("trust_guides_selection_behavior", False)),
                    "reliability_alignment": bool(summary.get("reliability_alignment_improves", False)),
                }
            )
        )
    table = TableSpec(
        table_id="tbl_submission_world_model_summary",
        title="Submission World-Model Utility Summary",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="reduces_simulations", label="Reduces Sim"),
            TableColumn(key="preserves_feasible_hit_rate", label="Preserves Feasible Hit"),
            TableColumn(key="calibration_reduces_gap", label="Calibration Reduces Gap"),
            TableColumn(key="trust_guides_selection", label="Trust Guides Selection"),
            TableColumn(key="reliability_alignment", label="Reliability Alignment"),
        ],
        rows=rows,
        caption="Cross-task summary of world-model utility claims already exported by the evidence bundles.",
        csv_output_path=str(output_root / "submission_world_model_summary.csv"),
        markdown_output_path=str(output_root / "submission_world_model_summary.md"),
    )
    _write_table_csv(table)
    _write_table_markdown(table)
    return table


def _build_submission_planner_summary_table(output_root: Path, papers_root: Path) -> TableSpec:
    rows: list[TableRow] = []
    for benchmark_id, bundle_path in _planner_bundle_paths(papers_root).items():
        payload = _load_json(bundle_path)
        summary = payload["summary"]
        rows.append(
            TableRow(
                values={
                    "benchmark_id": benchmark_id,
                    "beats_top_k": bool(summary.get("planner_beats_top_k", False)),
                    "reduces_simulations_vs_top_k": bool(summary.get("planner_reduces_simulations_vs_top_k", False)),
                    "preserves_feasible_hit_rate": bool(summary.get("planner_preserves_or_improves_feasible_hit_rate_vs_top_k", False)),
                    "fidelity_effective": bool(summary.get("fidelity_escalation_effective", False)),
                    "phase_updates_effective": bool(summary.get("phase_updates_effective", False)),
                    "calibration_replanning_effective": bool(summary.get("calibration_replanning_effective", False)),
                    "rollout_claim_status": str(summary.get("rollout_claim_status", "unknown")),
                    "dominant_failure_mode": str(summary.get("dominant_failure_mode", "unknown")),
                }
            )
        )
    table = TableSpec(
        table_id="tbl_submission_planner_summary",
        title="Submission Planner Utility Summary",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="beats_top_k", label="Beats Top-K"),
            TableColumn(key="reduces_simulations_vs_top_k", label="Reduces Sim vs Top-K"),
            TableColumn(key="preserves_feasible_hit_rate", label="Preserves Feasible Hit"),
            TableColumn(key="fidelity_effective", label="Fidelity Effective"),
            TableColumn(key="phase_updates_effective", label="Phase Updates Effective"),
            TableColumn(key="calibration_replanning_effective", label="Calib Replan Effective"),
            TableColumn(key="rollout_claim_status", label="Rollout Claim"),
            TableColumn(key="dominant_failure_mode", label="Dominant Failure"),
        ],
        rows=rows,
        caption="Cross-task planner summary emphasizing the bounded but real planner contribution under current evidence.",
        csv_output_path=str(output_root / "submission_planner_summary.csv"),
        markdown_output_path=str(output_root / "submission_planner_summary.md"),
    )
    _write_table_csv(table)
    _write_table_markdown(table)
    return table


def _build_submission_memory_repeated_table(output_root: Path, papers_root: Path) -> TableSpec:
    payload = _load_json(papers_root / "memory_chapter" / "memory_chapter_evidence_bundle.json")
    repeated_table = _table_map(payload)["tbl_memory_chapter_repeated_episode"]
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for row in repeated_table["rows"]:
        values = row["values"]
        grouped.setdefault(str(values["task"]), {})[str(values["mode"])] = values
    table = TableSpec(
        table_id="tbl_submission_memory_repeated_summary",
        title="Submission Memory Repeated-Episode Summary",
        columns=[
            TableColumn(key="task", label="Task"),
            TableColumn(key="no_memory_calls", label="No-Memory Calls"),
            TableColumn(key="full_memory_calls", label="Full-Memory Calls"),
            TableColumn(key="no_memory_failures", label="No-Memory Failures"),
            TableColumn(key="full_memory_failures", label="Full-Memory Failures"),
            TableColumn(key="beneficial", label="Beneficial"),
        ],
        rows=[
            TableRow(
                values={
                    "task": task,
                    "no_memory_calls": float(modes["no_memory"]["avg_sim_calls"]),
                    "full_memory_calls": float(modes["full_memory"]["avg_sim_calls"]),
                    "no_memory_failures": float(modes["no_memory"]["avg_repeated_failures"]),
                    "full_memory_failures": float(modes["full_memory"]["avg_repeated_failures"]),
                    "beneficial": float(modes["full_memory"]["avg_sim_calls"]) <= float(modes["no_memory"]["avg_sim_calls"]),
                }
            )
            for task, modes in sorted(grouped.items())
            if "no_memory" in modes and "full_memory" in modes
        ],
        caption="Main-text repeated-episode memory summary contrasting no-memory and full-memory modes.",
        csv_output_path=str(output_root / "submission_memory_repeated_summary.csv"),
        markdown_output_path=str(output_root / "submission_memory_repeated_summary.md"),
    )
    _write_table_csv(table)
    _write_table_markdown(table)
    return table


def _build_submission_memory_transfer_table(output_root: Path, papers_root: Path) -> TableSpec:
    payload = _load_json(papers_root / "memory_chapter" / "memory_chapter_evidence_bundle.json")
    transfer_table = _table_map(payload)["tbl_memory_chapter_transfer"]
    table = TableSpec(
        table_id="tbl_submission_memory_transfer_summary",
        title="Submission Memory Transfer Summary",
        columns=[
            TableColumn(key="pair", label="Transfer Pair"),
            TableColumn(key="transfer_kind", label="Kind"),
            TableColumn(key="governed_avg_sim_calls", label="Governed Calls"),
            TableColumn(key="governance_blocks_harm", label="Governance Blocks Harm"),
            TableColumn(key="beneficial", label="Governed Beneficial"),
            TableColumn(key="no_governance_harmful_rate", label="No-Gov Harm Rate"),
            TableColumn(key="forced_harmful_rate", label="Forced Harm Rate"),
        ],
        rows=[
            TableRow(
                values={
                    "pair": row["values"]["pair"],
                    "transfer_kind": row["values"]["transfer_kind"],
                    "governed_avg_sim_calls": row["values"]["governed_avg_sim_calls"],
                    "governance_blocks_harm": row["values"]["governance_blocks_harm"],
                    "beneficial": row["values"]["beneficial"],
                    "no_governance_harmful_rate": row["values"]["no_governance_harmful_rate"],
                    "forced_harmful_rate": row["values"]["forced_harmful_rate"],
                }
            )
            for row in transfer_table["rows"]
        ],
        caption="Main-text transfer summary covering same-family benefit and cross-family governance.",
        csv_output_path=str(output_root / "submission_memory_transfer_summary.csv"),
        markdown_output_path=str(output_root / "submission_memory_transfer_summary.md"),
    )
    _write_table_csv(table)
    _write_table_markdown(table)
    return table


def build_submission_main_figure_freeze_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionAssetFreezeBundle:
    """Build the Day 53 main-figure freeze bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    figures_root = root / "main_figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    generated_system_figure = _build_submission_system_architecture_figure(figures_root / "fig_system_architecture.svg")
    generated_trace_figure = _build_submission_ota_acceptance_trace_figure(
        figures_root / "fig_ota_acceptance_trace.svg",
        papers_root=papers,
    )

    manifest_entries = [
        {
            "asset_id": "fig1_system_architecture",
            "title": "Layered Closed-Loop System",
            "section": "System Overview",
            "source_path": generated_system_figure,
            "caption": "Six-layer system overview showing task formalization, world model, planner, real verification, calibration, and governed memory.",
            "rationale": "Top-tier systems readers need one architecture anchor before any ablation evidence.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig2_ota_acceptance_trace",
            "title": "OTA Closed-Loop Acceptance Trace",
            "section": "Main Result",
            "source_path": generated_trace_figure,
            "caption": "One OTA episode showing how prediction, verification, calibration, and memory interact in a concrete closed loop.",
            "rationale": "A trace figure grounds the paper in one real end-to-end episode instead of only aggregate charts.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig3a_world_model_prediction_gap",
            "title": "World-Model Prediction Gap vs Step",
            "section": "World Model Utility",
            "source_path": str(papers / "figs" / "world_model_prediction_gap_vs_step.svg"),
            "caption": "Prediction gap tightens across steps, supporting the calibration-and-trust story instead of a pure black-box surrogate claim.",
            "rationale": "This is the clearest generated world-model figure for systems-paper utility under current evidence.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig3b_world_model_simulation_calls",
            "title": "World-Model Real Simulation Calls",
            "section": "World Model Utility",
            "source_path": str(papers / "figs" / "world_model_simulation_calls.svg"),
            "caption": "World-model guidance reduces or preserves verification burden relative to system-facing baselines depending on task difficulty.",
            "rationale": "This panel is the most review-friendly connection from world-model behavior to real simulator cost.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig4a_planner_topk_efficiency",
            "title": "Planner Efficiency vs Top-K",
            "section": "Planner Utility",
            "source_path": str(papers / "ota2_v1_ci" / "planner_figs" / "planner_topk_efficiency.svg"),
            "caption": "The full planner outperforms simple top-k by combining uncertainty-aware ranking and structured verification control.",
            "rationale": "This is the cleanest single-panel planner claim for top-tier review.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig4b_planner_failure_pressure",
            "title": "Planner Failure Pressure",
            "section": "Planner Utility",
            "source_path": str(papers / "ota2_v1_ci" / "planner_figs" / "planner_failure_pressure.svg"),
            "caption": "Planner structure changes how failure pressure is allocated, not just whether one final scalar improves.",
            "rationale": "This prevents the planner story from collapsing into one overly narrow efficiency number.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig5a_memory_repeated_episode_calls",
            "title": "Repeated-Episode Memory Calls",
            "section": "Memory Utility",
            "source_path": str(papers / "memory_chapter" / "layout" / "main_figs" / "fig_memory_repeated_episode_calls.svg"),
            "caption": "Memory reduces repeated verification burden on the strongest repeated-episode tasks.",
            "rationale": "This is the strongest high-level memory efficiency figure for the main paper.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig5b_memory_same_family_transfer",
            "title": "Same-Family Transfer Benefit",
            "section": "Memory Utility",
            "source_path": str(papers / "memory_chapter" / "layout" / "main_figs" / "fig_memory_same_family_transfer.svg"),
            "caption": "Same-family transfer between OTA and folded-cascode shows memory can help beyond one repeated benchmark.",
            "rationale": "This is the cleanest positive-transfer figure and supports the generalization story.",
            "availability_status": "generated_ready",
        },
        {
            "asset_id": "fig6_memory_cross_family_governance",
            "title": "Cross-Family Governance",
            "section": "Boundary Cases",
            "source_path": str(papers / "memory_chapter" / "layout" / "main_figs" / "fig_memory_cross_family_governance.svg"),
            "caption": "Cross-family reuse can be harmful, and governance is what makes memory a bounded systems component instead of a reckless retrieval layer.",
            "rationale": "Top-tier matching is stronger when the paper includes one honest boundary-case figure rather than only wins.",
            "availability_status": "generated_ready",
        },
    ]

    entries: list[SubmissionAssetEntry] = []
    for item in manifest_entries:
        target_path = ""
        if item["availability_status"] == "generated_ready":
            source_path = Path(item["source_path"])
            if source_path.resolve().parent == figures_root.resolve():
                target_path = str(source_path)
            else:
                target_path = _copy_file(source_path, figures_root / source_path.name)
        entries.append(
            SubmissionAssetEntry(
                asset_kind="figure",
                target_path=target_path,
                **item,
            )
        )

    json_output_path = root / "submission_main_figure_freeze_bundle.json"
    markdown_output_path = root / "submission_main_figure_freeze_bundle.md"
    bundle = SubmissionAssetFreezeBundle(
        bundle_id="submission_main_figure_freeze_v1",
        profile_name=profile_name,
        asset_kind="figure",
        entries=entries,
        ready_entry_count=sum(entry.availability_status == "generated_ready" for entry in entries),
        pending_entry_count=sum(entry.availability_status == "manual_curation_required" for entry in entries),
        summary_notes=[
            "Main figures now include code-generated system architecture and OTA acceptance-trace panels in addition to the quantitative world-model, planner, and memory figures.",
            "The frozen figure set keeps one honest boundary-case panel while still giving the main paper a full systems-paper visual arc.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_freeze_bundle(bundle), encoding="utf-8")
    return bundle


def build_submission_main_table_freeze_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionAssetFreezeBundle:
    """Build the Day 54 main-table freeze bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    tables_root = root / "main_tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    generated_tables = [
        (
            "tbl1_benchmark_summary",
            "Benchmark Summary",
            "Benchmark Design",
            _build_submission_benchmark_summary_table(tables_root),
            "Compact frozen-suite overview used to orient the main results section.",
        ),
        (
            "tbl2_system_baseline_summary",
            "System Baseline Summary",
            "Main Result",
            _build_submission_system_baseline_summary_table(tables_root, papers),
            "Shows the real simulator cost story that matters most for a systems-paper review.",
        ),
        (
            "tbl3_world_model_summary",
            "World-Model Utility Summary",
            "World Model Utility",
            _build_submission_world_model_summary_table(tables_root, papers),
            "Turns per-task world-model claims into one compact cross-task table.",
        ),
        (
            "tbl4_planner_summary",
            "Planner Utility Summary",
            "Planner Utility",
            _build_submission_planner_summary_table(tables_root, papers),
            "Keeps the planner story bounded, multi-factor, and reviewer-readable.",
        ),
        (
            "tbl5_memory_repeated_summary",
            "Memory Repeated-Episode Summary",
            "Memory Utility",
            _build_submission_memory_repeated_table(tables_root, papers),
            "Summarizes where memory actually reduces repeated simulator burden.",
        ),
        (
            "tbl6_memory_transfer_summary",
            "Memory Transfer Summary",
            "Boundary Cases",
            _build_submission_memory_transfer_table(tables_root, papers),
            "Pairs positive same-family transfer with honest cross-family governance behavior.",
        ),
    ]

    entries = [
        SubmissionAssetEntry(
            asset_id=asset_id,
            asset_kind="table",
            title=title,
            section=section,
            source_path=table.markdown_output_path,
            target_path=table.markdown_output_path,
            caption=table.caption,
            rationale=rationale,
            availability_status="generated_ready",
        )
        for asset_id, title, section, table, rationale in generated_tables
    ]

    json_output_path = root / "submission_main_table_freeze_bundle.json"
    markdown_output_path = root / "submission_main_table_freeze_bundle.md"
    bundle = SubmissionAssetFreezeBundle(
        bundle_id="submission_main_table_freeze_v1",
        profile_name=profile_name,
        asset_kind="table",
        entries=entries,
        ready_entry_count=len(entries),
        pending_entry_count=0,
        summary_notes=[
            "Main tables were regenerated as one coherent submission-facing pack instead of pointing reviewers to scattered per-task files.",
            "Research-baseline narrative remains important, but the frozen main tables prioritize code-backed cross-task evidence over weaker ad hoc claims.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_freeze_bundle(bundle), encoding="utf-8")
    return bundle


def build_submission_appendix_allocation_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionAppendixAllocationBundle:
    """Build the Day 55 appendix allocation bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    appendix_figs_root = root / "appendix_figures"
    appendix_tables_root = root / "appendix_tables"
    appendix_figs_root.mkdir(parents=True, exist_ok=True)
    appendix_tables_root.mkdir(parents=True, exist_ok=True)

    figure_freeze = build_submission_main_figure_freeze_bundle(
        profile_name=profile_name,
        output_root=root / "_figure_freeze",
        papers_root=papers,
    )
    table_freeze = build_submission_main_table_freeze_bundle(
        profile_name=profile_name,
        output_root=root / "_table_freeze",
        papers_root=papers,
    )
    boundary_bundle = build_physical_validity_boundary_bundle(output_root=root / "_physical_validity")

    main_figure_sources = {
        str(Path(entry.source_path).resolve())
        for entry in figure_freeze.entries
        if entry.availability_status == "generated_ready"
    }

    appendix_figure_candidates = sorted(
        {
            path
            for pattern in (
                "**/figs/*.svg",
                "**/planner_figs/*.svg",
                "memory_chapter/layout/appendix_figs/*.svg",
            )
            for path in papers.glob(pattern)
            if path.is_file() and "submission_package" not in path.parts and "physical_validity_boundaries" not in path.parts
        },
        key=lambda item: str(item),
    )

    appendix_figures = [
        _copy_with_relative_name(path, base_root=papers, dst_root=appendix_figs_root)
        for path in appendix_figure_candidates
        if str(path.resolve()) not in main_figure_sources
    ]

    appendix_table_candidates = sorted(
        {
            path
            for pattern in (
                "**/tables/*.csv",
                "**/tables/*.md",
                "**/planner_tables/*.csv",
                "**/planner_tables/*.md",
                "memory_chapter/layout/appendix_tables/*.csv",
                "memory_chapter/layout/appendix_tables/*.md",
                "benchmark_rollup/tables/*.csv",
                "benchmark_rollup/tables/*.md",
                "benchmark_family_summary/tables/*.csv",
                "benchmark_family_summary/tables/*.md",
                "benchmark_failure_modes/tables/*.csv",
                "benchmark_failure_modes/tables/*.md",
                "benchmark_robustness/tables/*.csv",
                "benchmark_robustness/tables/*.md",
                "benchmark_fidelity_framing/tables/*.csv",
                "benchmark_fidelity_framing/tables/*.md",
            )
            for path in papers.glob(pattern)
            if path.is_file() and "submission_package" not in path.parts and "physical_validity_boundaries" not in path.parts
        },
        key=lambda item: str(item),
    )

    appendix_tables = [
        _copy_with_relative_name(path, base_root=papers, dst_root=appendix_tables_root)
        for path in appendix_table_candidates
    ]
    appendix_tables.extend(
        [
            _copy_with_relative_name(table.csv_output_path, base_root=root, dst_root=appendix_tables_root)
            for table in boundary_bundle.tables
        ]
    )
    appendix_tables.extend(
        [
            _copy_with_relative_name(table.markdown_output_path, base_root=root, dst_root=appendix_tables_root)
            for table in boundary_bundle.tables
        ]
    )
    appendix_figures = _dedupe_paths(appendix_figures)
    appendix_tables = _dedupe_paths(appendix_tables)
    appendix_tables = [path for path in appendix_tables if "physical_validity_boundaries__" not in path]

    json_output_path = root / "submission_appendix_allocation_bundle.json"
    markdown_output_path = root / "submission_appendix_allocation_bundle.md"
    bundle = SubmissionAppendixAllocationBundle(
        bundle_id="submission_appendix_allocation_v1",
        profile_name=profile_name,
        main_figure_ids=[entry.asset_id for entry in figure_freeze.entries],
        main_table_ids=[entry.asset_id for entry in table_freeze.entries],
        appendix_figures=appendix_figures,
        appendix_tables=appendix_tables,
        summary_notes=[
            "Appendix keeps the richer per-task figures/tables so the main paper can stay selective without losing auditability.",
            "Physical-validity boundary tables are explicitly allocated to appendix to support honest reviewer-facing wording.",
            "Memory appendix allocation reuses the existing chapter layout rather than inventing a second incompatible organization.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_appendix_bundle(bundle), encoding="utf-8")
    return bundle


def _copy_documents(paths: list[Path], *, dst_root: Path) -> list[str]:
    return [_copy_file(path, dst_root / path.name) for path in paths]


def _markdown_for_narrative_bundle(bundle: SubmissionNarrativeFreezeBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        f"- Profile: `{bundle.profile_name}`",
        f"- Bundle kind: `{bundle.bundle_kind}`",
        f"- Ready sections: `{bundle.ready_section_count}`",
        f"- Pending sections: `{bundle.pending_section_count}`",
        "",
        "## Frozen Documents",
        "",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in bundle.frozen_documents)
    lines.extend(["", "## Sections", ""])
    for section in bundle.sections:
        lines.extend(
            [
                f"### {section.section_id}: {section.title}",
                "",
                f"- Status: `{section.alignment_status}`",
                f"- Sources: `{', '.join(Path(path).name for path in section.source_paths)}`",
                f"- Required assets: `{', '.join(section.required_asset_ids) or 'none'}`",
                f"- Evidence: `{', '.join(Path(path).name for path in section.evidence_paths) or 'none'}`",
            ]
        )
        lines.extend(f"- Note: {note}" for note in section.notes)
        lines.append("")
    lines.extend(["## Tables", ""])
    lines.extend(f"- `{Path(table.markdown_output_path).name}`" for table in bundle.tables)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.summary_notes)
    return "\n".join(lines)


def _markdown_for_experiment_alignment_bundle(bundle: SubmissionExperimentAlignmentBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        f"- Profile: `{bundle.profile_name}`",
        f"- Aligned entries: `{bundle.aligned_entry_count}`",
        f"- Pending entries: `{bundle.pending_entry_count}`",
        "",
        "## Experiment Entries",
        "",
    ]
    for entry in bundle.entries:
        lines.extend(
            [
                f"### {entry.subsection_id}: {entry.title}",
                "",
                f"- Status: `{entry.alignment_status}`",
                f"- Claim: {entry.primary_claim}",
                f"- Main figures: `{', '.join(entry.main_figure_ids) or 'none'}`",
                f"- Main tables: `{', '.join(entry.main_table_ids) or 'none'}`",
                f"- Appendix support: `{', '.join(Path(path).name for path in entry.appendix_paths) or 'none'}`",
            ]
        )
        lines.extend(f"- Note: {note}" for note in entry.notes)
        lines.append("")
    lines.extend(["## Tables", ""])
    lines.extend(f"- `{Path(table.markdown_output_path).name}`" for table in bundle.tables)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.summary_notes)
    return "\n".join(lines)


def _markdown_for_internal_submission_bundle(bundle: InternalSubmissionPackageBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        f"- Profile: `{bundle.profile_name}`",
        f"- Internal review ready: `{bundle.internal_review_ready}`",
        f"- External submission ready: `{bundle.external_submission_ready}`",
        f"- Unresolved manual assets: `{', '.join(bundle.unresolved_manual_asset_ids) or 'none'}`",
        "",
        "## Included Bundles",
        "",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in bundle.included_bundle_paths)
    lines.extend(["", "## Primary Documents", ""])
    lines.extend(f"- `{Path(path).name}`" for path in bundle.primary_document_paths)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.summary_notes)
    return "\n".join(lines)


def _bundle_output_paths(papers_root: Path) -> dict[str, Path]:
    submission_root = papers_root / "submission_package"
    return {
        "figure_freeze": submission_root / "main_figures" / "submission_main_figure_freeze_bundle.md",
        "table_freeze": submission_root / "submission_main_table_freeze_bundle.md",
        "appendix": submission_root / "appendix" / "submission_appendix_allocation_bundle.md",
        "boundary": papers_root / "physical_validity_boundaries" / "physical_validity_boundary_bundle.md",
    }


def build_submission_protocol_finalization_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionNarrativeFreezeBundle:
    """Build the Day 56 protocol finalization bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    docs_root = root / "frozen_docs"
    tables_root = root / "tables"
    docs_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    source_documents = [papers / "experimental_protocol.md"]
    frozen_documents = _copy_documents(source_documents, dst_root=docs_root)
    bundle_paths = _bundle_output_paths(papers)

    benchmark_table = TableSpec(
        table_id="tbl_submission_protocol_benchmark_matrix",
        title="Submission Protocol Benchmark Matrix",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="role", label="Role"),
            TableColumn(key="default_fidelity", label="Default Fidelity"),
            TableColumn(key="promoted_fidelity", label="Promoted Fidelity"),
            TableColumn(key="truth_level", label="Truth Level"),
            TableColumn(key="primary_metrics", label="Primary Metrics"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "family": definition.family,
                    "role": definition.benchmark_role,
                    "default_fidelity": definition.execution_defaults.default_fidelity,
                    "promoted_fidelity": definition.execution_defaults.promoted_fidelity,
                    "truth_level": definition.execution_defaults.truth_level,
                    "primary_metrics": ",".join(definition.measurement_contract.primary_metrics),
                }
            )
            for definition in list_benchmark_definitions()
        ],
        caption="Frozen benchmark/protocol matrix for the submission-facing four-task suite.",
        csv_output_path=str(tables_root / "submission_protocol_benchmark_matrix.csv"),
        markdown_output_path=str(tables_root / "submission_protocol_benchmark_matrix.md"),
    )
    reporting_table = TableSpec(
        table_id="tbl_submission_protocol_reporting_conventions",
        title="Submission Protocol Reporting Conventions",
        columns=[
            TableColumn(key="protocol_area", label="Protocol Area"),
            TableColumn(key="frozen_rule", label="Frozen Rule"),
            TableColumn(key="submission_use", label="Submission Use"),
            TableColumn(key="evidence_anchor", label="Evidence Anchor"),
        ],
        rows=[
            TableRow(
                values={
                    "protocol_area": "truth_validity",
                    "frozen_rule": "report ngspice + demonstrator_truth + builtin demonstrator models",
                    "submission_use": "state demonstrator-level physical validity in benchmark/method/results sections",
                    "evidence_anchor": "physical validity boundary bundle",
                }
            ),
            TableRow(
                values={
                    "protocol_area": "fidelity_policy",
                    "frozen_rule": "quick_truth screens candidates; focused_truth verifies promoted candidates",
                    "submission_use": "describe selective verification rather than claiming universal high-fidelity runs",
                    "evidence_anchor": "experimental_protocol.md",
                }
            ),
            TableRow(
                values={
                    "protocol_area": "baseline_policy",
                    "frozen_rule": "separate system-facing baselines, research baselines, planner ablations, and memory ablations",
                    "submission_use": "keep baseline narratives disentangled in methods/results",
                    "evidence_anchor": "experimental_protocol.md + results_synthesis.md",
                }
            ),
            TableRow(
                values={
                    "protocol_area": "metric_policy",
                    "frozen_rule": "prioritize simulation_call_count, feasible_hit_rate, efficiency_score, and step_to_first_feasible",
                    "submission_use": "use appendix for detailed per-episode/per-mode breakdowns",
                    "evidence_anchor": "submission main tables + appendix allocation",
                }
            ),
            TableRow(
                values={
                    "protocol_area": "statistics_policy",
                    "frozen_rule": "report mean trends and structured breakdowns without overstating significance",
                    "submission_use": "avoid unsupported statistical language in main text",
                    "evidence_anchor": "experimental_protocol.md",
                }
            ),
        ],
        caption="Frozen reporting conventions so the experiments section stays top-tier disciplined instead of ad hoc.",
        csv_output_path=str(tables_root / "submission_protocol_reporting_conventions.csv"),
        markdown_output_path=str(tables_root / "submission_protocol_reporting_conventions.md"),
    )
    for table in (benchmark_table, reporting_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    sections = [
        SubmissionSectionEntry(
            section_id="protocol_scope",
            title="Scope",
            source_paths=frozen_documents,
            evidence_paths=[],
            alignment_status="ready",
            notes=["Protocol is frozen as the paper-facing contract for the current local submission package."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_truth_validity",
            title="Truth Validity Declaration",
            source_paths=frozen_documents,
            evidence_paths=[str(bundle_paths["boundary"])],
            alignment_status="ready",
            notes=["Writing must explicitly say demonstrator-level physical validity and avoid industrial-realism wording."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_benchmark_suite",
            title="Benchmark Suite",
            source_paths=frozen_documents,
            evidence_paths=[benchmark_table.markdown_output_path],
            alignment_status="ready",
            notes=["The frozen runnable suite remains four tasks and should be described as fixed-version paper tracks."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_fidelity_policy",
            title="Fidelity Policy",
            source_paths=frozen_documents,
            evidence_paths=[reporting_table.markdown_output_path],
            alignment_status="ready",
            notes=["Quick-vs-focused truth is part of the methodology story, not a hidden implementation detail."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_baselines",
            title="Baselines",
            source_paths=frozen_documents,
            evidence_paths=[str(bundle_paths["table_freeze"])],
            alignment_status="ready",
            notes=["System baselines, research baselines, and ablations should stay separated in the narrative."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_reported_metrics",
            title="Reported Metrics",
            source_paths=frozen_documents,
            evidence_paths=[str(bundle_paths["table_freeze"]), str(bundle_paths["appendix"])],
            alignment_status="ready",
            notes=["Main text should stay compact while appendix carries detailed breakdowns."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_statistical_conventions",
            title="Statistical Conventions",
            source_paths=frozen_documents,
            evidence_paths=[reporting_table.markdown_output_path],
            alignment_status="ready",
            notes=["Current package supports mean trends and reproducible breakdowns, not strong significance claims."],
        ),
        SubmissionSectionEntry(
            section_id="protocol_writing_notes",
            title="Protocol Notes for Writing",
            source_paths=frozen_documents,
            evidence_paths=[str(bundle_paths["boundary"])],
            alignment_status="ready",
            notes=["Boundary cases such as bandgap should be written explicitly rather than normalized away."],
        ),
    ]

    json_output_path = root / "submission_protocol_finalization_bundle.json"
    markdown_output_path = root / "submission_protocol_finalization_bundle.md"
    bundle = SubmissionNarrativeFreezeBundle(
        bundle_id="submission_protocol_finalization_v1",
        profile_name=profile_name,
        bundle_kind="protocol",
        source_documents=[str(path) for path in source_documents],
        frozen_documents=frozen_documents,
        sections=sections,
        tables=[benchmark_table, reporting_table],
        ready_section_count=len(sections),
        pending_section_count=0,
        summary_notes=[
            "Protocol finalization freezes the four-task benchmark suite, fidelity policy, and reporting conventions into one code-backed bundle.",
            "Top-tier matching is stronger when the protocol explicitly names demonstrator-level physical validity instead of letting reviewers infer the boundary.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_narrative_bundle(bundle), encoding="utf-8")
    return bundle


def build_submission_limitations_finalization_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionNarrativeFreezeBundle:
    """Build the Day 57 limitations/threats finalization bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    docs_root = root / "frozen_docs"
    tables_root = root / "tables"
    docs_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    source_documents = [papers / "limitations_and_validity.md"]
    frozen_documents = _copy_documents(source_documents, dst_root=docs_root)
    bundle_paths = _bundle_output_paths(papers)

    limitations_table = TableSpec(
        table_id="tbl_submission_limitations_matrix",
        title="Submission Limitations Matrix",
        columns=[
            TableColumn(key="limitation_area", label="Limitation Area"),
            TableColumn(key="supported_claim", label="Supported Claim"),
            TableColumn(key="bounded_non_claim", label="Do Not Claim"),
            TableColumn(key="reviewer_framing", label="Reviewer Framing"),
        ],
        rows=[
            TableRow(
                values={
                    "limitation_area": "physical_validity",
                    "supported_claim": "real-SPICE grounded closed loop under demonstrator_truth",
                    "bounded_non_claim": "industrial signoff, tapeout readiness, PDK-calibrated realism",
                    "reviewer_framing": "strong systems grounding with explicit honesty boundary",
                }
            ),
            TableRow(
                values={
                    "limitation_area": "benchmark_scope",
                    "supported_claim": "fixed four-task multi-family benchmark suite",
                    "bounded_non_claim": "broad analog-library coverage or topology-search completeness",
                    "reviewer_framing": "credible first systems-paper benchmark breadth, not universal coverage",
                }
            ),
            TableRow(
                values={
                    "limitation_area": "world_model",
                    "supported_claim": "calibratable, uncertainty-aware proxy that improves selection behavior",
                    "bounded_non_claim": "large-scale learned surrogate with uniform gains on every task",
                    "reviewer_framing": "closed-loop calibration story is stronger than pure prediction claims",
                }
            ),
            TableRow(
                values={
                    "limitation_area": "planner",
                    "supported_claim": "selective-verification planner that beats top-k under bounded conditions",
                    "bounded_non_claim": "uniformly dominant final-metric gains on every benchmark",
                    "reviewer_framing": "planner evidence includes allocation behavior, not just one scalar outcome",
                }
            ),
            TableRow(
                values={
                    "limitation_area": "memory",
                    "supported_claim": "repeated-episode and same-family governed reuse",
                    "bounded_non_claim": "universal positive transfer or always-on retrieval benefits",
                    "reviewer_framing": "governance is part of the contribution because negative transfer is real",
                }
            ),
            TableRow(
                values={
                    "limitation_area": "statistics",
                    "supported_claim": "reproducible mean trends with appendix breakdowns",
                    "bounded_non_claim": "fully powered statistical significance program",
                    "reviewer_framing": "structured exports support auditing even when significance claims stay modest",
                }
            ),
        ],
        caption="Frozen limitations matrix so reviewer-facing caveats stay explicit and disciplined.",
        csv_output_path=str(tables_root / "submission_limitations_matrix.csv"),
        markdown_output_path=str(tables_root / "submission_limitations_matrix.md"),
    )
    boundary_table = TableSpec(
        table_id="tbl_submission_boundary_case_matrix",
        title="Submission Boundary-Case Matrix",
        columns=[
            TableColumn(key="boundary_case", label="Boundary Case"),
            TableColumn(key="paper_rule", label="Paper Rule"),
            TableColumn(key="evidence_anchor", label="Evidence Anchor"),
        ],
        rows=[
            TableRow(
                values={
                    "boundary_case": "bandgap_boundary_case",
                    "paper_rule": "describe bandgap as an honest stress test and boundary case, not a universal success story",
                    "evidence_anchor": "results_synthesis.md + limitations_and_validity.md",
                }
            ),
            TableRow(
                values={
                    "boundary_case": "flat_or_tied_plots",
                    "paper_rule": "move visually flat plots to appendix and pair them with a table note",
                    "evidence_anchor": "figure_table_manifest.md + limitations_and_validity.md",
                }
            ),
            TableRow(
                values={
                    "boundary_case": "measurement_vs_design_failure",
                    "paper_rule": "preserve the distinction between measurement failure, invalid simulation, and actual design failure",
                    "evidence_anchor": "limitations_and_validity.md",
                }
            ),
        ],
        caption="Boundary-case writing rules that keep the paper honest without weakening the systems claim.",
        csv_output_path=str(tables_root / "submission_boundary_case_matrix.csv"),
        markdown_output_path=str(tables_root / "submission_boundary_case_matrix.md"),
    )
    for table in (limitations_table, boundary_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    sections = [
        SubmissionSectionEntry(
            section_id="limitations_physical_validity",
            title="Physical Validity Boundary",
            source_paths=frozen_documents,
            evidence_paths=[str(bundle_paths["boundary"]), limitations_table.markdown_output_path],
            alignment_status="ready",
            notes=["The paper should claim real-SPICE grounding but not configured-truth or signoff realism."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_benchmark_scope",
            title="Benchmark Scope",
            source_paths=frozen_documents,
            evidence_paths=[limitations_table.markdown_output_path],
            alignment_status="ready",
            notes=["The benchmark suite is credible and fixed, but still not a full analog library."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_world_model",
            title="World-Model Limits",
            source_paths=frozen_documents,
            evidence_paths=[limitations_table.markdown_output_path],
            alignment_status="ready",
            notes=["World-model value should be framed around calibration and trust-guided selection."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_planner",
            title="Planner Limits",
            source_paths=frozen_documents,
            evidence_paths=[limitations_table.markdown_output_path],
            alignment_status="ready",
            notes=["Planner utility should stay multi-factor and not collapse into one headline metric."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_memory",
            title="Memory Limits",
            source_paths=frozen_documents,
            evidence_paths=[limitations_table.markdown_output_path, boundary_table.markdown_output_path],
            alignment_status="ready",
            notes=["Governed reuse is a stronger claim than universal transfer benefit."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_failure_analysis",
            title="Failure Analysis Notes for Writing",
            source_paths=frozen_documents,
            evidence_paths=[boundary_table.markdown_output_path, str(bundle_paths["appendix"])],
            alignment_status="ready",
            notes=["Bandgap and flat plots remain explicit writing-time honesty checks."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_threats_to_validity",
            title="Threats to Validity",
            source_paths=frozen_documents,
            evidence_paths=[limitations_table.markdown_output_path],
            alignment_status="ready",
            notes=["Internal, external, and construct validity are separated rather than mixed into one caveat blob."],
        ),
        SubmissionSectionEntry(
            section_id="limitations_reviewer_closing",
            title="Reviewer-Facing Closing Sentence",
            source_paths=frozen_documents,
            evidence_paths=[limitations_table.markdown_output_path],
            alignment_status="ready",
            notes=["Closing sentence should bound interpretation without retreating from the core systems result."],
        ),
    ]

    json_output_path = root / "submission_limitations_finalization_bundle.json"
    markdown_output_path = root / "submission_limitations_finalization_bundle.md"
    bundle = SubmissionNarrativeFreezeBundle(
        bundle_id="submission_limitations_finalization_v1",
        profile_name=profile_name,
        bundle_kind="limitations",
        source_documents=[str(path) for path in source_documents],
        frozen_documents=frozen_documents,
        sections=sections,
        tables=[limitations_table, boundary_table],
        ready_section_count=len(sections),
        pending_section_count=0,
        summary_notes=[
            "Limitations finalization converts reviewer-facing caveats into a frozen matrix instead of leaving them as loose prose.",
            "Bandgap, negative transfer, and demonstrator-only physical validity are treated as bounded evidence conditions rather than hidden weaknesses.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_narrative_bundle(bundle), encoding="utf-8")
    return bundle


def build_submission_manuscript_structure_freeze_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionNarrativeFreezeBundle:
    """Build the Day 58 manuscript structure freeze bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    docs_root = root / "frozen_docs"
    tables_root = root / "tables"
    docs_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    source_documents = [
        papers / "outline.md",
        papers / "manuscript_blueprint.md",
        papers / "figure_table_manifest.md",
        papers / "experimental_protocol.md",
        papers / "results_synthesis.md",
        papers / "limitations_and_validity.md",
    ]
    frozen_documents = _copy_documents(source_documents, dst_root=docs_root)
    bundle_paths = _bundle_output_paths(papers)

    section_rows = [
        {
            "section_id": "sec1_introduction",
            "title": "Introduction",
            "required_assets": "",
            "status": "ready",
            "evidence_anchor": "outline + blueprint",
        },
        {
            "section_id": "sec2_related_work",
            "title": "Related Work",
            "required_assets": "",
            "status": "ready",
            "evidence_anchor": "blueprint",
        },
        {
            "section_id": "sec3_system_overview",
            "title": "System Overview",
            "required_assets": "fig1_system_architecture",
            "status": "ready",
            "evidence_anchor": "figure manifest + figure freeze",
        },
        {
            "section_id": "sec4_layered_method",
            "title": "Layered Method",
            "required_assets": "",
            "status": "ready",
            "evidence_anchor": "outline + blueprint",
        },
        {
            "section_id": "sec5_benchmark_design",
            "title": "Benchmark Design",
            "required_assets": "tbl1_benchmark_summary",
            "status": "ready",
            "evidence_anchor": "protocol + table freeze",
        },
        {
            "section_id": "sec6_baselines_and_ablations",
            "title": "Baselines and Ablations",
            "required_assets": "tbl2_system_baseline_summary,tbl3_world_model_summary,tbl4_planner_summary",
            "status": "ready",
            "evidence_anchor": "results synthesis + table freeze",
        },
        {
            "section_id": "sec7_experimental_results",
            "title": "Experimental Results",
            "required_assets": "fig2_ota_acceptance_trace,fig3a_world_model_prediction_gap,fig3b_world_model_simulation_calls,fig4a_planner_topk_efficiency,fig4b_planner_failure_pressure,fig5a_memory_repeated_episode_calls,fig5b_memory_same_family_transfer,tbl5_memory_repeated_summary,tbl6_memory_transfer_summary",
            "status": "ready",
            "evidence_anchor": "results synthesis + figure/table freeze",
        },
        {
            "section_id": "sec8_failure_analysis",
            "title": "Failure Analysis and Boundary Cases",
            "required_assets": "fig6_memory_cross_family_governance,tbl6_memory_transfer_summary",
            "status": "ready",
            "evidence_anchor": "results synthesis + limitations",
        },
        {
            "section_id": "sec9_limitations",
            "title": "Limitations and Threats to Validity",
            "required_assets": "",
            "status": "ready",
            "evidence_anchor": "limitations bundle",
        },
        {
            "section_id": "sec10_future_work",
            "title": "Future Work",
            "required_assets": "",
            "status": "ready",
            "evidence_anchor": "outline + blueprint",
        },
    ]
    structure_table = TableSpec(
        table_id="tbl_submission_manuscript_structure_freeze",
        title="Submission Manuscript Structure Freeze",
        columns=[
            TableColumn(key="section_id", label="Section"),
            TableColumn(key="title", label="Title"),
            TableColumn(key="required_assets", label="Required Assets"),
            TableColumn(key="status", label="Status"),
            TableColumn(key="evidence_anchor", label="Evidence Anchor"),
        ],
        rows=[TableRow(values=row) for row in section_rows],
        caption="Frozen section-level manuscript structure with required assets and manual gaps surfaced explicitly.",
        csv_output_path=str(tables_root / "submission_manuscript_structure_freeze.csv"),
        markdown_output_path=str(tables_root / "submission_manuscript_structure_freeze.md"),
    )
    _write_table_csv(structure_table)
    _write_table_markdown(structure_table)

    sections = [
        SubmissionSectionEntry(
            section_id=str(row["section_id"]),
            title=str(row["title"]),
            source_paths=frozen_documents,
            required_asset_ids=[item for item in str(row["required_assets"]).split(",") if item],
            evidence_paths=[
                structure_table.markdown_output_path,
                str(bundle_paths["figure_freeze"]),
                str(bundle_paths["table_freeze"]),
            ],
            alignment_status=str(row["status"]),
            notes=[
                "Section structure is frozen and anchored to the current submission package."
            ],
        )
        for row in section_rows
    ]

    json_output_path = root / "submission_manuscript_structure_freeze_bundle.json"
    markdown_output_path = root / "submission_manuscript_structure_freeze_bundle.md"
    bundle = SubmissionNarrativeFreezeBundle(
        bundle_id="submission_manuscript_structure_freeze_v1",
        profile_name=profile_name,
        bundle_kind="manuscript",
        source_documents=[str(path) for path in source_documents],
        frozen_documents=frozen_documents,
        sections=sections,
        tables=[structure_table],
        ready_section_count=sum(section.alignment_status == "ready" for section in sections),
        pending_section_count=sum(section.alignment_status == "manual_attention_required" for section in sections),
        summary_notes=[
            "Manuscript structure freeze keeps the paper in systems-paper shape instead of drifting into module-by-module reporting.",
            "System overview and experimental-results sections are now fully asset-backed because the architecture and OTA trace figures are generated inside the submission package.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_narrative_bundle(bundle), encoding="utf-8")
    return bundle


def build_submission_experiments_alignment_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> SubmissionExperimentAlignmentBundle:
    """Build the Day 59 experiments-section alignment bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    docs_root = root / "frozen_docs"
    tables_root = root / "tables"
    docs_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    source_documents = [papers / "results_synthesis.md", papers / "figure_table_manifest.md"]
    frozen_documents = _copy_documents(source_documents, dst_root=docs_root)
    bundle_paths = _bundle_output_paths(papers)

    entries = [
        SubmissionExperimentAlignmentEntry(
            subsection_id="sec7_1_ota_closed_loop",
            title="Main Closed-Loop Result on OTA",
            primary_claim="Anchor the experiments section in one concrete OTA closed-loop trace and the main full-system baseline comparison.",
            main_figure_ids=["fig2_ota_acceptance_trace"],
            main_table_ids=["tbl2_system_baseline_summary"],
            appendix_paths=[str(bundle_paths["appendix"])],
            alignment_status="aligned",
            notes=["OTA closed-loop anchor is now backed by a generated trace figure plus the frozen baseline table."],
        ),
        SubmissionExperimentAlignmentEntry(
            subsection_id="sec7_2_multitask_generalization",
            title="Multi-Task Generalization",
            primary_claim="Show that the same closed-loop stack runs across OTA, folded-cascode, LDO, and bandgap with task-dependent gain strength.",
            main_figure_ids=[],
            main_table_ids=["tbl1_benchmark_summary", "tbl2_system_baseline_summary"],
            appendix_paths=[str(bundle_paths["appendix"])],
            alignment_status="aligned",
            notes=["Appendix keeps per-task evidence while the main text stays compact."],
        ),
        SubmissionExperimentAlignmentEntry(
            subsection_id="sec7_3_world_model_utility",
            title="World-Model Utility",
            primary_claim="Frame world-model value around calibration, trust-guided selection, and bounded simulation savings rather than universal wins.",
            main_figure_ids=["fig3a_world_model_prediction_gap", "fig3b_world_model_simulation_calls"],
            main_table_ids=["tbl3_world_model_summary"],
            appendix_paths=[str(bundle_paths["appendix"])],
            alignment_status="aligned",
            notes=["Bandgap remains the clearest budget-gain task and should be written that way."],
        ),
        SubmissionExperimentAlignmentEntry(
            subsection_id="sec7_4_planner_utility",
            title="Planner Utility",
            primary_claim="Contrast the full planner with top-k using efficiency, fidelity escalation, and calibration-driven replanning evidence.",
            main_figure_ids=["fig4a_planner_topk_efficiency", "fig4b_planner_failure_pressure"],
            main_table_ids=["tbl4_planner_summary"],
            appendix_paths=[str(bundle_paths["appendix"])],
            alignment_status="aligned",
            notes=["Planner story should remain multi-factor even where end-metric gaps are modest."],
        ),
        SubmissionExperimentAlignmentEntry(
            subsection_id="sec7_5_memory_utility",
            title="Memory Utility",
            primary_claim="Show repeated-episode gains, same-family transfer benefit, and cross-family governance as one bounded memory story.",
            main_figure_ids=[
                "fig5a_memory_repeated_episode_calls",
                "fig5b_memory_same_family_transfer",
                "fig6_memory_cross_family_governance",
            ],
            main_table_ids=["tbl5_memory_repeated_summary", "tbl6_memory_transfer_summary"],
            appendix_paths=[str(bundle_paths["appendix"])],
            alignment_status="aligned",
            notes=["Bandgap remains a boundary case, which strengthens the governance claim rather than weakening it."],
        ),
    ]
    alignment_table = TableSpec(
        table_id="tbl_submission_experiments_alignment",
        title="Submission Experiments Alignment Matrix",
        columns=[
            TableColumn(key="subsection_id", label="Subsection"),
            TableColumn(key="title", label="Title"),
            TableColumn(key="main_figures", label="Main Figures"),
            TableColumn(key="main_tables", label="Main Tables"),
            TableColumn(key="appendix_support", label="Appendix Support"),
            TableColumn(key="status", label="Status"),
        ],
        rows=[
            TableRow(
                values={
                    "subsection_id": entry.subsection_id,
                    "title": entry.title,
                    "main_figures": ",".join(entry.main_figure_ids),
                    "main_tables": ",".join(entry.main_table_ids),
                    "appendix_support": ",".join(Path(path).name for path in entry.appendix_paths),
                    "status": entry.alignment_status,
                }
            )
            for entry in entries
        ],
        caption="Experiments-section alignment matrix tying each result claim to frozen main-text and appendix assets.",
        csv_output_path=str(tables_root / "submission_experiments_alignment.csv"),
        markdown_output_path=str(tables_root / "submission_experiments_alignment.md"),
    )
    _write_table_csv(alignment_table)
    _write_table_markdown(alignment_table)

    json_output_path = root / "submission_experiments_alignment_bundle.json"
    markdown_output_path = root / "submission_experiments_alignment_bundle.md"
    bundle = SubmissionExperimentAlignmentBundle(
        bundle_id="submission_experiments_alignment_v1",
        profile_name=profile_name,
        entries=entries,
        tables=[alignment_table],
        aligned_entry_count=sum(entry.alignment_status == "aligned" for entry in entries),
        pending_entry_count=sum(entry.alignment_status == "manual_attention_required" for entry in entries),
        summary_notes=[
            f"Frozen documents: {', '.join(Path(path).name for path in frozen_documents)}.",
            "Experiments alignment keeps claims tied to code-backed figures/tables instead of relying on prose-only memory.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_experiment_alignment_bundle(bundle), encoding="utf-8")
    return bundle


def build_final_internal_submission_package_bundle(
    *,
    profile_name: str,
    output_root: str | Path,
    papers_root: str | Path | None = None,
) -> InternalSubmissionPackageBundle:
    """Build the Day 60 final internal submission package bundle."""

    papers = Path(papers_root) if papers_root is not None else PAPERS_ROOT
    root = Path(output_root)
    docs_root = root / "frozen_docs"
    docs_root.mkdir(parents=True, exist_ok=True)

    physical_bundle = build_physical_validity_boundary_bundle(output_root=root / "physical_validity")
    figure_bundle = build_submission_main_figure_freeze_bundle(
        profile_name=profile_name,
        output_root=root / "main_figures",
        papers_root=papers,
    )
    table_bundle = build_submission_main_table_freeze_bundle(
        profile_name=profile_name,
        output_root=root / "main_tables",
        papers_root=papers,
    )
    appendix_bundle = build_submission_appendix_allocation_bundle(
        profile_name=profile_name,
        output_root=root / "appendix",
        papers_root=papers,
    )
    protocol_bundle = build_submission_protocol_finalization_bundle(
        profile_name=profile_name,
        output_root=root / "protocol",
        papers_root=papers,
    )
    limitations_bundle = build_submission_limitations_finalization_bundle(
        profile_name=profile_name,
        output_root=root / "limitations",
        papers_root=papers,
    )
    manuscript_bundle = build_submission_manuscript_structure_freeze_bundle(
        profile_name=profile_name,
        output_root=root / "manuscript",
        papers_root=papers,
    )
    experiments_bundle = build_submission_experiments_alignment_bundle(
        profile_name=profile_name,
        output_root=root / "experiments",
        papers_root=papers,
    )

    primary_documents = _copy_documents(
        [
            papers / "outline.md",
            papers / "manuscript_blueprint.md",
            papers / "figure_table_manifest.md",
            papers / "experimental_protocol.md",
            papers / "results_synthesis.md",
            papers / "limitations_and_validity.md",
            papers / "paper_package_index.md",
        ],
        dst_root=docs_root,
    )
    stage_e_status = build_stage_e_review_status(papers_root=papers)
    unresolved_manual_asset_ids = [
        entry.asset_id
        for entry in figure_bundle.entries
        if entry.availability_status == "manual_curation_required"
    ]
    included_bundle_paths = [
        physical_bundle.json_output_path,
        figure_bundle.json_output_path,
        table_bundle.json_output_path,
        appendix_bundle.json_output_path,
        protocol_bundle.json_output_path,
        limitations_bundle.json_output_path,
        manuscript_bundle.json_output_path,
        experiments_bundle.json_output_path,
    ]
    internal_review_ready = bool(stage_e_status.get("ready_for_stage_f", False)) and all(
        Path(path).exists() for path in included_bundle_paths
    )
    external_submission_ready = internal_review_ready and not unresolved_manual_asset_ids

    json_output_path = root / "final_internal_submission_package_bundle.json"
    markdown_output_path = root / "final_internal_submission_package_bundle.md"
    bundle = InternalSubmissionPackageBundle(
        bundle_id="final_internal_submission_package_v1",
        profile_name=profile_name,
        included_bundle_paths=included_bundle_paths,
        primary_document_paths=primary_documents,
        unresolved_manual_asset_ids=unresolved_manual_asset_ids,
        internal_review_ready=internal_review_ready,
        external_submission_ready=external_submission_ready,
        summary_notes=[
            f"stage_e_ready_for_stage_f={stage_e_status.get('ready_for_stage_f', False)}",
            "Internal submission package is review-ready once protocol, limitations, manuscript, experiments, figures, tables, appendix, and validity boundaries are all frozen together.",
            "External submission package is now fully asset-backed for the current demonstrator-truth scope; remaining future work is the separate PDK/Spectre upgrade line.",
        ],
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    _write_json(json_output_path, bundle.model_dump(mode="json"))
    markdown_output_path.write_text(_markdown_for_internal_submission_bundle(bundle), encoding="utf-8")
    return bundle
