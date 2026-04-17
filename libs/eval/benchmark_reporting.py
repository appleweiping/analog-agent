"""Build benchmark-facing rollup tables and narrative packages for Stage E."""

from __future__ import annotations

import json
from pathlib import Path

from libs.eval.benchmark_protocol import benchmark_protocol_contract
from libs.eval.benchmark_registry import list_benchmark_definitions, load_benchmark_suite_definition
from libs.eval.paper_evidence import _write_table_csv, _write_table_markdown
from libs.memory.failure_taxonomy import FAILURE_TAXONOMY
from libs.schema.benchmark_evidence import BenchmarkEvidenceBundle, BenchmarkNarrativeSection
from libs.schema.paper_evidence import TableColumn, TableRow, TableSpec


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
    if definition.execution_defaults.truth_level == "configured_truth" and (corner_count > 1 or temperature_count > 1 or load_count > 1):
        return "configured_robustness_candidate"
    if corner_count > 1 or temperature_count > 1 or load_count > 1:
        return "robustness_style_contract"
    return "single_point_nominal"


def _robustness_claim_status(definition) -> str:
    profile = _nominal_profile(definition)
    if profile == "configured_robustness_candidate":
        return "configured_robustness_candidate"
    if profile == "robustness_style_contract":
        return "robustness_style_not_physical_validity_strong"
    return "nominal_only_contract"


def _family_group(definitions):
    grouped: dict[str, list] = {}
    for definition in definitions:
        grouped.setdefault(definition.family, []).append(definition)
    return dict(sorted(grouped.items()))


def _failure_focus_for_category(category: str) -> list[str]:
    if category == "amplifier":
        return ["stability_failure", "drive_bandwidth_failure", "measurement_failure"]
    if category == "regulator":
        return ["drive_bandwidth_failure", "robustness_failure", "measurement_failure"]
    return ["robustness_failure", "measurement_failure", "operating_region_failure"]


def _bundle_markdown(bundle: BenchmarkEvidenceBundle) -> str:
    lines = [
        f"# {bundle.bundle_id}",
        "",
        f"- Suite: `{bundle.suite_id}`",
        f"- Scope: `{bundle.scope}`",
        "",
        "## Tables",
        "",
    ]
    lines.extend(f"- `{Path(table.csv_output_path).name}`" for table in bundle.tables)
    lines.extend(["", "## Narrative", ""])
    for section in bundle.narrative_sections:
        lines.extend([f"### {section.title}", "", section.body, ""])
    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in bundle.notes)
    return "\n".join(lines)


def _finalize_bundle(
    *,
    bundle_id: str,
    scope: str,
    tables: list[TableSpec],
    narrative_sections: list[BenchmarkNarrativeSection],
    notes: list[str],
    output_root: str | Path,
) -> BenchmarkEvidenceBundle:
    suite = load_benchmark_suite_definition()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    json_output_path = root / f"{bundle_id}.json"
    markdown_output_path = root / f"{bundle_id}.md"
    for table in tables:
        _write_table_csv(table)
        _write_table_markdown(table)
    bundle = BenchmarkEvidenceBundle(
        bundle_id=bundle_id,
        suite_id=suite.suite_id,
        scope=scope,
        tables=tables,
        narrative_sections=narrative_sections,
        notes=notes,
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    json_output_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    markdown_output_path.write_text(_bundle_markdown(bundle), encoding="utf-8")
    return bundle


def build_multitask_rollup_bundle(*, output_root: str | Path) -> BenchmarkEvidenceBundle:
    suite = load_benchmark_suite_definition()
    definitions = list_benchmark_definitions()
    protocol = benchmark_protocol_contract()
    tables_root = Path(output_root) / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    rollup_table = TableSpec(
        table_id="tbl_benchmark_multitask_rollup",
        title="Frozen Runnable Multi-Task Benchmark Rollup",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="category", label="Category"),
            TableColumn(key="role", label="Role"),
            TableColumn(key="truth_level", label="Truth Level"),
            TableColumn(key="promoted_fidelity", label="Promoted Fidelity"),
            TableColumn(key="primary_metric_count", label="Primary Metric Count"),
            TableColumn(key="corner_count", label="Corners"),
            TableColumn(key="temperature_count", label="Temps"),
            TableColumn(key="load_count", label="Loads"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "family": definition.family,
                    "category": definition.category,
                    "role": definition.benchmark_role,
                    "truth_level": definition.execution_defaults.truth_level,
                    "promoted_fidelity": definition.execution_defaults.promoted_fidelity,
                    "primary_metric_count": len(definition.measurement_contract.primary_metrics),
                    "corner_count": _environment_counts(definition)[0],
                    "temperature_count": _environment_counts(definition)[1],
                    "load_count": _environment_counts(definition)[2],
                }
            )
            for definition in definitions
        ],
        caption="Cross-task benchmark rollup for the frozen runnable suite used in paper-facing reporting.",
        csv_output_path=str(tables_root / "benchmark_multitask_rollup.csv"),
        markdown_output_path=str(tables_root / "benchmark_multitask_rollup.md"),
    )

    roster_table = TableSpec(
        table_id="tbl_benchmark_protocol_roster",
        title="Benchmark Protocol and Mode Roster",
        columns=[
            TableColumn(key="profile", label="Profile"),
            TableColumn(key="modes", label="Modes"),
            TableColumn(key="default_steps", label="Steps"),
            TableColumn(key="default_repeat_runs", label="Repeat Runs"),
            TableColumn(key="default_budget", label="Default Budget"),
        ],
        rows=[
            TableRow(
                values={
                    "profile": "baseline",
                    "modes": ",".join(protocol["baseline_modes"]),
                    "default_steps": protocol["default_steps"],
                    "default_repeat_runs": protocol["default_repeat_runs"],
                    "default_budget": f"sim={protocol['default_budget']['max_simulations']},cand={protocol['default_budget']['max_candidates_per_step']}",
                }
            ),
            TableRow(
                values={
                    "profile": "methodology",
                    "modes": ",".join(protocol["methodology_modes"]),
                    "default_steps": protocol["default_steps"],
                    "default_repeat_runs": protocol["default_repeat_runs"],
                    "default_budget": f"sim={protocol['default_budget']['max_simulations']},cand={protocol['default_budget']['max_candidates_per_step']}",
                }
            ),
            TableRow(
                values={
                    "profile": "planner_ablation",
                    "modes": ",".join(protocol["planner_ablation_modes"]),
                    "default_steps": protocol["default_steps"],
                    "default_repeat_runs": protocol["default_repeat_runs"],
                    "default_budget": f"sim={protocol['default_budget']['max_simulations']},cand={protocol['default_budget']['max_candidates_per_step']}",
                }
            ),
        ],
        caption="Shared mode roster and default benchmark protocol used across the frozen runnable suite.",
        csv_output_path=str(tables_root / "benchmark_protocol_roster.csv"),
        markdown_output_path=str(tables_root / "benchmark_protocol_roster.md"),
    )

    narrative = [
        BenchmarkNarrativeSection(
            section_id="multitask_scope",
            title="Multi-Task Scope",
            body=(
                f"The frozen runnable suite currently covers {len(definitions)} tasks spanning amplifier, regulator, and reference families, "
                f"with `{suite.primary_benchmark_id}` designated as the paper-primary benchmark."
            ),
        )
    ]
    notes = [
        "This rollup package summarizes the benchmark contract and roster; it does not claim that all measured result tables have already been regenerated in the current environment.",
    ]
    return _finalize_bundle(
        bundle_id="benchmark_multitask_rollup_bundle",
        scope="multitask_rollup",
        tables=[rollup_table, roster_table],
        narrative_sections=narrative,
        notes=notes,
        output_root=output_root,
    )


def build_family_summary_bundle(*, output_root: str | Path) -> BenchmarkEvidenceBundle:
    definitions = list_benchmark_definitions()
    grouped = _family_group(definitions)
    tables_root = Path(output_root) / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    summary_table = TableSpec(
        table_id="tbl_benchmark_family_summary",
        title="Benchmark Family Summary",
        columns=[
            TableColumn(key="family", label="Family"),
            TableColumn(key="task_count", label="Task Count"),
            TableColumn(key="categories", label="Categories"),
            TableColumn(key="roles", label="Roles"),
            TableColumn(key="truth_levels", label="Truth Levels"),
            TableColumn(key="promoted_fidelity", label="Promoted Fidelity"),
        ],
        rows=[
            TableRow(
                values={
                    "family": family,
                    "task_count": len(items),
                    "categories": ",".join(sorted({item.category for item in items})),
                    "roles": ",".join(sorted({item.benchmark_role for item in items})),
                    "truth_levels": ",".join(sorted({item.execution_defaults.truth_level for item in items})),
                    "promoted_fidelity": ",".join(sorted({item.execution_defaults.promoted_fidelity for item in items})),
                }
            )
            for family, items in grouped.items()
        ],
        caption="Family-level summary across the frozen runnable benchmark suite.",
        csv_output_path=str(tables_root / "benchmark_family_summary.csv"),
        markdown_output_path=str(tables_root / "benchmark_family_summary.md"),
    )

    metric_table = TableSpec(
        table_id="tbl_benchmark_family_metric_coverage",
        title="Benchmark Family Metric Coverage",
        columns=[
            TableColumn(key="family", label="Family"),
            TableColumn(key="primary_metrics", label="Primary Metrics"),
            TableColumn(key="auxiliary_metrics", label="Auxiliary Metrics"),
            TableColumn(key="reporting_metrics", label="Reporting Metrics"),
        ],
        rows=[
            TableRow(
                values={
                    "family": family,
                    "primary_metrics": ",".join(sorted({metric for item in items for metric in item.measurement_contract.primary_metrics})),
                    "auxiliary_metrics": ",".join(sorted({metric for item in items for metric in item.measurement_contract.auxiliary_metrics})) or "none",
                    "reporting_metrics": ",".join(sorted({metric for item in items for metric in item.measurement_contract.reporting_metrics})),
                }
            )
            for family, items in grouped.items()
        ],
        caption="Family-level metric coverage used for cross-family paper framing.",
        csv_output_path=str(tables_root / "benchmark_family_metric_coverage.csv"),
        markdown_output_path=str(tables_root / "benchmark_family_metric_coverage.md"),
    )

    narrative = [
        BenchmarkNarrativeSection(
            section_id="family_coverage",
            title="Family Coverage",
            body="The current benchmark suite covers amplifier, regulator, and reference families with explicit primary/reporting metric contracts for each frozen runnable task.",
        )
    ]
    return _finalize_bundle(
        bundle_id="benchmark_family_summary_bundle",
        scope="family_summary",
        tables=[summary_table, metric_table],
        narrative_sections=narrative,
        notes=["Family summaries are contract-facing cross-task summaries, not measured aggregate performance tables."],
        output_root=output_root,
    )


def build_failure_mode_synthesis_bundle(*, output_root: str | Path) -> BenchmarkEvidenceBundle:
    suite = load_benchmark_suite_definition()
    definitions = list_benchmark_definitions()
    categories = sorted({definition.category for definition in definitions})
    tables_root = Path(output_root) / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    taxonomy_table = TableSpec(
        table_id="tbl_benchmark_failure_mode_taxonomy",
        title="Benchmark Failure-Mode Taxonomy Coverage",
        columns=[
            TableColumn(key="failure_mode", label="Failure Mode"),
            TableColumn(key="description", label="Description"),
            TableColumn(key="amplifier_relevance", label="Amplifier"),
            TableColumn(key="regulator_relevance", label="Regulator"),
            TableColumn(key="reference_relevance", label="Reference"),
            TableColumn(key="reporting_axis_present", label="Reporting Axis"),
        ],
        rows=[
            TableRow(
                values={
                    "failure_mode": mode,
                    "description": description,
                    "amplifier_relevance": mode in _failure_focus_for_category("amplifier"),
                    "regulator_relevance": mode in _failure_focus_for_category("regulator"),
                    "reference_relevance": mode in _failure_focus_for_category("reference"),
                    "reporting_axis_present": "failure_mode_distribution" in suite.reporting_axes,
                }
            )
            for mode, description in sorted(FAILURE_TAXONOMY.items())
        ],
        caption="Contract-level failure-mode taxonomy coverage for the benchmark suite.",
        csv_output_path=str(tables_root / "benchmark_failure_mode_taxonomy.csv"),
        markdown_output_path=str(tables_root / "benchmark_failure_mode_taxonomy.md"),
    )

    family_focus_table = TableSpec(
        table_id="tbl_benchmark_family_failure_focus",
        title="Benchmark Family Failure Focus",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="category", label="Category"),
            TableColumn(key="expected_failure_focus", label="Expected Failure Focus"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "family": definition.family,
                    "category": definition.category,
                    "expected_failure_focus": ",".join(_failure_focus_for_category(definition.category)),
                }
            )
            for definition in definitions
        ],
        caption="Expected failure-focus framing by benchmark, used to keep cross-task result discussion honest and comparable.",
        csv_output_path=str(tables_root / "benchmark_family_failure_focus.csv"),
        markdown_output_path=str(tables_root / "benchmark_family_failure_focus.md"),
    )

    narrative = [
        BenchmarkNarrativeSection(
            section_id="failure_synthesis",
            title="Failure Synthesis",
            body="The benchmark package now makes failure reporting explicit at the taxonomy level, with category-aware expectations for amplifier, regulator, and reference tasks.",
        )
    ]
    return _finalize_bundle(
        bundle_id="benchmark_failure_mode_synthesis_bundle",
        scope="failure_mode_synthesis",
        tables=[taxonomy_table, family_focus_table],
        narrative_sections=narrative,
        notes=["This synthesis is contract-facing and category-aware; it is not a measured aggregate failure histogram across all reruns."],
        output_root=output_root,
    )


def build_robustness_narrative_bundle(*, output_root: str | Path) -> BenchmarkEvidenceBundle:
    definitions = list_benchmark_definitions()
    tables_root = Path(output_root) / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    scope_table = TableSpec(
        table_id="tbl_benchmark_robustness_scope",
        title="Benchmark Robustness Claim Scope",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="truth_level", label="Truth Level"),
            TableColumn(key="nominal_profile", label="Nominal Profile"),
            TableColumn(key="corner_count", label="Corners"),
            TableColumn(key="temperature_count", label="Temps"),
            TableColumn(key="load_count", label="Loads"),
            TableColumn(key="robustness_claim_status", label="Claim Status"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "truth_level": definition.execution_defaults.truth_level,
                    "nominal_profile": _nominal_profile(definition),
                    "corner_count": _environment_counts(definition)[0],
                    "temperature_count": _environment_counts(definition)[1],
                    "load_count": _environment_counts(definition)[2],
                    "robustness_claim_status": _robustness_claim_status(definition),
                }
            )
            for definition in definitions
        ],
        caption="Robustness-claim framing derived from the frozen benchmark contract and current truth-level scope.",
        csv_output_path=str(tables_root / "benchmark_robustness_scope.csv"),
        markdown_output_path=str(tables_root / "benchmark_robustness_scope.md"),
    )

    narrative = [
        BenchmarkNarrativeSection(
            section_id="robustness_overview",
            title="Robustness Narrative",
            body=(
                "Current frozen runnable benchmarks are honest nominal contracts: they are demonstrator-truth tasks with single-corner, single-temperature default environments, "
                "so robustness language should be framed as future expansion rather than a current strong physical-validity claim."
            ),
        )
    ]
    return _finalize_bundle(
        bundle_id="benchmark_robustness_narrative_bundle",
        scope="robustness_narrative",
        tables=[scope_table],
        narrative_sections=narrative,
        notes=["Robustness framing is intentionally conservative until configured-truth and richer condition coverage are staged."],
        output_root=output_root,
    )


def build_fidelity_corner_load_bundle(*, output_root: str | Path) -> BenchmarkEvidenceBundle:
    definitions = list_benchmark_definitions()
    tables_root = Path(output_root) / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    fidelity_table = TableSpec(
        table_id="tbl_benchmark_fidelity_framing",
        title="Benchmark Fidelity and Condition Framing",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="default_fidelity", label="Default Fidelity"),
            TableColumn(key="promoted_fidelity", label="Promoted Fidelity"),
            TableColumn(key="testbench_plan", label="Analyses"),
            TableColumn(key="corner_count", label="Corners"),
            TableColumn(key="temperature_count", label="Temps"),
            TableColumn(key="load_count", label="Loads"),
            TableColumn(key="nominal_profile", label="Nominal Profile"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "default_fidelity": definition.execution_defaults.default_fidelity,
                    "promoted_fidelity": definition.execution_defaults.promoted_fidelity,
                    "testbench_plan": ",".join(definition.task.testbench_plan),
                    "corner_count": _environment_counts(definition)[0],
                    "temperature_count": _environment_counts(definition)[1],
                    "load_count": _environment_counts(definition)[2],
                    "nominal_profile": _nominal_profile(definition),
                }
            )
            for definition in definitions
        ],
        caption="Shared fidelity/corner/load framing across frozen runnable benchmarks.",
        csv_output_path=str(tables_root / "benchmark_fidelity_framing.csv"),
        markdown_output_path=str(tables_root / "benchmark_fidelity_framing.md"),
    )

    condition_table = TableSpec(
        table_id="tbl_benchmark_condition_contract",
        title="Benchmark Condition Contract",
        columns=[
            TableColumn(key="benchmark_id", label="Benchmark"),
            TableColumn(key="corners", label="Corners"),
            TableColumn(key="temperatures_c", label="Temperatures C"),
            TableColumn(key="load_cap_f", label="Load Cap F"),
            TableColumn(key="output_load_ohm", label="Output Load Ohm"),
        ],
        rows=[
            TableRow(
                values={
                    "benchmark_id": definition.benchmark_id,
                    "corners": ",".join(definition.task.environment.get("corners", []) or []),
                    "temperatures_c": ",".join(str(value) for value in definition.task.environment.get("temperature_c", []) or []),
                    "load_cap_f": str(definition.task.environment.get("load_cap_f", "")),
                    "output_load_ohm": str(definition.task.environment.get("output_load_ohm", "")),
                }
            )
            for definition in definitions
        ],
        caption="Explicit corner/load environment contract for the current frozen benchmark suite.",
        csv_output_path=str(tables_root / "benchmark_condition_contract.csv"),
        markdown_output_path=str(tables_root / "benchmark_condition_contract.md"),
    )

    narrative = [
        BenchmarkNarrativeSection(
            section_id="fidelity_condition_framing",
            title="Fidelity, Corner, and Load Framing",
            body="Benchmark condition framing is now explicit: each frozen runnable task exposes its default/promoted fidelity, analyses, and nominal corner/load contract so later paper claims stay within the true tested scope.",
        )
    ]
    return _finalize_bundle(
        bundle_id="benchmark_fidelity_corner_load_bundle",
        scope="fidelity_corner_load_framing",
        tables=[fidelity_table, condition_table],
        narrative_sections=narrative,
        notes=["Current frozen runnable tasks remain nominal-condition contracts and should be described that way in the paper package."],
        output_root=output_root,
    )
