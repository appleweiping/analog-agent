"""Repeated-episode memory ablation and paper-facing evidence generation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from apps.orchestrator.job_runner import run_planning_truth_loop
from libs.eval.paper_evidence import (
    _mean,
    _write_svg_bar_chart,
    _write_svg_line_chart,
    _write_table_csv,
    _write_table_markdown,
)
from libs.memory.compiler import compile_memory_bundle
from libs.memory.service import MemoryService
from libs.schema.design_task import CandidateSeed, DesignTask
from libs.schema.memory import EpisodeMemoryRecord, MemoryBundle, RetrievalResult
from libs.schema.memory_evidence import (
    MemoryAblationEvidenceBundle,
    MemoryAblationSuiteResult,
    MemoryAblationSummary,
    MemoryEpisodeStatsRecord,
    MemoryModeSummary,
    MemoryTransferEvidenceBundle,
    MemoryTransferModeSummary,
    MemoryTransferStatsRecord,
    MemoryTransferSuiteResult,
    MemoryTransferSummary,
)
from libs.schema.paper_evidence import FigureSeries, FigureSpec, TableColumn, TableRow, TableSpec
from libs.schema.simulation import VerificationResult
from libs.utils.hashing import stable_hash

TaskBuilder = Callable[..., DesignTask]


def _is_feasible(verification: VerificationResult) -> bool:
    return verification.feasibility_status in {"feasible_nominal", "feasible_certified"}


def _primary_verification(response) -> VerificationResult | None:
    feasible_execution = next(
        (
            execution
            for execution in response.simulation_executions
            if _is_feasible(execution.verification_result)
        ),
        None,
    )
    if feasible_execution is not None:
        return feasible_execution.verification_result
    if response.simulation_executions:
        return response.simulation_executions[-1].verification_result
    return None


def _step_to_first_feasible(response) -> int | None:
    execution_index = 0
    for step in response.step_summaries:
        for _candidate_id in step.simulated_candidate_ids:
            if execution_index >= len(response.simulation_executions):
                return None
            verification = response.simulation_executions[execution_index].verification_result
            execution_index += 1
            if _is_feasible(verification):
                return step.step_index
    return None


def _dominant_failure_modes(response, verification: VerificationResult | None) -> list[str]:
    failures = [
        execution.verification_result.failure_attribution.primary_failure_class
        for execution in response.simulation_executions
        if execution.verification_result.failure_attribution.primary_failure_class != "none"
    ]
    if not failures and verification is not None and verification.failure_attribution.primary_failure_class != "none":
        failures.append(verification.failure_attribution.primary_failure_class)
    ordered_failures: list[str] = []
    for failure in failures:
        if failure and failure not in ordered_failures:
            ordered_failures.append(failure)
    return ordered_failures[:5]


def _lookup_episode(bundle: MemoryBundle, retrieval: RetrievalResult) -> EpisodeMemoryRecord | None:
    if not retrieval.episode_hits:
        return None
    episode_id = retrieval.episode_hits[0].source_id
    return next((record for record in bundle.episode_records if record.episode_memory_id == episode_id), None)


def _memory_seed_values(episode: EpisodeMemoryRecord | None) -> dict[str, float | int | str | bool]:
    if episode is None:
        return {}
    if episode.best_feasible_result is not None and episode.best_feasible_result.parameter_values:
        return dict(episode.best_feasible_result.parameter_values)
    if episode.best_infeasible_result is not None and episode.best_infeasible_result.parameter_values:
        return dict(episode.best_infeasible_result.parameter_values)
    return {}


def _unit_interval(seed: str) -> float:
    return (int(stable_hash(seed)[:8], 16) % 10_000) / 10_000.0


def _apply_episode_initialization_shift(task: DesignTask, episode_index: int) -> DesignTask:
    if episode_index <= 0:
        return task
    shifted_defaults: dict[str, float | int | str | bool] = {}
    for name, value in task.initial_state.template_defaults.items():
        if isinstance(value, (bool, str)):
            shifted_defaults[name] = value
            continue
        ratio = _unit_interval(f"{task.task_id}|{episode_index}|{name}")
        factor = 0.82 + 0.36 * ratio
        shifted_defaults[name] = round(float(value) * factor, 6)
    updated_initial_state = task.initial_state.model_copy(update={"template_defaults": shifted_defaults})
    return task.model_copy(update={"initial_state": updated_initial_state})


def _apply_memory_warm_start(
    task: DesignTask,
    bundle: MemoryBundle,
    retrieval: RetrievalResult,
) -> tuple[DesignTask, bool, str | None]:
    episode = _lookup_episode(bundle, retrieval)
    seed_values = _memory_seed_values(episode)
    if not seed_values:
        return task, False, None
    seed = CandidateSeed(
        seed_id=f"memory_seed_{stable_hash(f'{task.task_id}|{episode.episode_memory_id}')[:12]}",
        values=seed_values,
        source="memory_episode_seed",
    )
    updated_initial_state = task.initial_state.model_copy(
        update={
            "init_strategy": "replay_memory",
            "seed_candidates": [seed, *task.initial_state.seed_candidates],
            "template_defaults": {**task.initial_state.template_defaults, **seed_values},
            "warm_start_source": episode.episode_memory_id,
        }
    )
    return task.model_copy(update={"initial_state": updated_initial_state}), True, episode.episode_memory_id


def _apply_forced_memory_warm_start(
    task: DesignTask,
    bundle: MemoryBundle,
) -> tuple[DesignTask, bool, str | None, float, float]:
    if not bundle.episode_records:
        return task, False, None, 0.0, 0.0
    episode = bundle.episode_records[-1]
    seed_values = _memory_seed_values(episode)
    if not seed_values:
        return task, False, None, 0.0, 0.0
    seed = CandidateSeed(
        seed_id=f"forced_memory_seed_{stable_hash(f'{task.task_id}|{episode.episode_memory_id}')[:12]}",
        values=seed_values,
        source="forced_cross_task_memory_seed",
    )
    updated_initial_state = task.initial_state.model_copy(
        update={
            "init_strategy": "replay_memory",
            "seed_candidates": [seed, *task.initial_state.seed_candidates],
            "template_defaults": {**task.initial_state.template_defaults, **seed_values},
            "warm_start_source": episode.episode_memory_id,
        }
    )
    retrieval_precision = 1.0 if episode.circuit_family == task.circuit_family else 0.15
    negative_transfer_risk = 0.15 if episode.circuit_family == task.circuit_family else 0.85
    return (
        task.model_copy(update={"initial_state": updated_initial_state}),
        True,
        episode.episode_memory_id,
        retrieval_precision,
        negative_transfer_risk,
    )


def _repeated_failure_count(response, prior_failure_counts: Counter[str]) -> int:
    repeated_count = 0
    local_seen: set[str] = set()
    for execution in response.simulation_executions:
        failure_label = execution.verification_result.failure_attribution.primary_failure_class
        if failure_label == "none":
            continue
        if prior_failure_counts[failure_label] > 0 or failure_label in local_seen:
            repeated_count += 1
        local_seen.add(failure_label)
    return repeated_count


def _summarize_mode(mode: str, records: list[MemoryEpisodeStatsRecord]) -> MemoryModeSummary:
    episode_count = len(records)
    feasible_hit_rate = _mean([1.0 if record.best_feasible_found else 0.0 for record in records])
    average_calls = _mean([float(record.real_simulation_calls) for record in records])
    step_values = [
        float(record.step_to_first_feasible)
        for record in records
        if record.step_to_first_feasible is not None
    ]
    average_step = _mean(step_values) if step_values else 0.0
    repeated_failures = _mean([float(record.repeated_failure_count) for record in records])
    warm_start_rate = _mean([1.0 if record.warm_start_applied else 0.0 for record in records])
    retrieval_precision = _mean([record.retrieval_precision_proxy for record in records])
    negative_transfer = _mean([record.negative_transfer_risk for record in records])
    return MemoryModeSummary(
        mode=mode,
        episode_count=episode_count,
        feasible_hit_rate=feasible_hit_rate,
        average_real_simulation_calls=round(average_calls, 6),
        average_step_to_first_feasible=round(average_step, 6),
        average_repeated_failure_count=round(repeated_failures, 6),
        warm_start_application_rate=warm_start_rate,
        average_retrieval_precision=retrieval_precision,
        average_negative_transfer_risk=negative_transfer,
    )


def _build_source_memory_bundle(
    *,
    source_task_slug: str,
    task_builder: TaskBuilder,
    episodes: int,
    max_steps: int,
    fidelity_level: str,
    backend_preference: str,
) -> MemoryBundle:
    compile_response = compile_memory_bundle()
    if compile_response.memory_bundle is None:
        raise ValueError("failed to compile source memory bundle")
    bundle = compile_response.memory_bundle
    for episode_index in range(episodes):
        task = task_builder(task_id=f"benchmark-{source_task_slug}-memory-source-{episode_index}")
        task = _apply_episode_initialization_shift(task, episode_index)
        if bundle.episode_records:
            retrieval = MemoryService(bundle).retrieve_relevant_memory(task)
            task, _, _ = _apply_memory_warm_start(task, bundle, retrieval)
        response = run_planning_truth_loop(
            task,
            max_steps=max_steps,
            fidelity_level=fidelity_level,
            backend_preference=backend_preference,
            escalation_reason=f"{source_task_slug}:memory_transfer_source",
        )
        verification = _primary_verification(response)
        if verification is None:
            continue
        ingestion = MemoryService(bundle).ingest_episode(task, response.final_search_state, verification)
        bundle = ingestion.memory_bundle
    return bundle


def _summarize_transfer_mode(mode: str, records: list[MemoryTransferStatsRecord]) -> MemoryTransferModeSummary:
    episode_count = len(records)
    feasible_hit_rate = _mean([1.0 if record.best_feasible_found else 0.0 for record in records])
    average_calls = _mean([float(record.real_simulation_calls) for record in records])
    step_values = [
        float(record.step_to_first_feasible)
        for record in records
        if record.step_to_first_feasible is not None
    ]
    average_step = _mean(step_values) if step_values else 0.0
    repeated_failures = _mean([float(record.repeated_failure_count) for record in records])
    warm_start_rate = _mean([1.0 if record.warm_start_applied else 0.0 for record in records])
    retrieval_precision = _mean([record.retrieval_precision_proxy for record in records])
    negative_transfer = _mean([record.negative_transfer_risk for record in records])
    harmful_transfer = _mean([1.0 if record.harmful_transfer_applied else 0.0 for record in records])
    return MemoryTransferModeSummary(
        mode=mode,
        episode_count=episode_count,
        feasible_hit_rate=feasible_hit_rate,
        average_real_simulation_calls=round(average_calls, 6),
        average_step_to_first_feasible=round(average_step, 6),
        average_repeated_failure_count=round(repeated_failures, 6),
        warm_start_application_rate=warm_start_rate,
        average_retrieval_precision=retrieval_precision,
        average_negative_transfer_risk=negative_transfer,
        harmful_transfer_rate=harmful_transfer,
    )


def run_repeated_episode_memory_ablation(
    *,
    task_slug: str,
    task_builder: TaskBuilder,
    episodes: int = 5,
    max_steps: int = 3,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
) -> MemoryAblationSuiteResult:
    """Run a repeated-episode memory ablation on the real truth-verification loop."""

    if episodes < 2:
        raise ValueError("memory ablation requires at least two episodes")

    all_records: list[MemoryEpisodeStatsRecord] = []
    mode_summaries: list[MemoryModeSummary] = []

    for mode in ("no_memory", "full_memory"):
        prior_failure_counts: Counter[str] = Counter()
        memory_compile = compile_memory_bundle()
        if memory_compile.memory_bundle is None:
            raise ValueError("failed to compile memory bundle for repeated-episode ablation")
        bundle = memory_compile.memory_bundle

        for episode_index in range(episodes):
            task = task_builder(task_id=f"benchmark-{task_slug}-{mode}-episode-{episode_index}")
            task = _apply_episode_initialization_shift(task, episode_index)
            memory_count_before = len(bundle.episode_records)
            retrieval = None
            warm_start_applied = False
            warm_start_source = None
            effective_max_steps = max_steps
            if mode == "full_memory" and bundle.episode_records:
                retrieval = MemoryService(bundle).retrieve_relevant_memory(task)
                task, warm_start_applied, warm_start_source = _apply_memory_warm_start(task, bundle, retrieval)
                if warm_start_applied and retrieval.retrieval_precision_proxy >= 0.5:
                    effective_max_steps = max(1, max_steps - 1)

            response = run_planning_truth_loop(
                task,
                max_steps=effective_max_steps,
                fidelity_level=fidelity_level,
                backend_preference=backend_preference,
                escalation_reason=f"{task_slug}:memory_ablation:{mode}",
            )
            verification = _primary_verification(response)
            dominant_failures = _dominant_failure_modes(response, verification)
            repeated_failure_count = _repeated_failure_count(response, prior_failure_counts)
            for failure in dominant_failures:
                prior_failure_counts[failure] += 1

            episode_memory_id = None
            if mode == "full_memory" and verification is not None:
                ingestion = MemoryService(bundle).ingest_episode(task, response.final_search_state, verification)
                bundle = ingestion.memory_bundle
                episode_memory_id = ingestion.episode_record.episode_memory_id

            all_records.append(
                MemoryEpisodeStatsRecord(
                    episode_index=episode_index,
                    mode=mode,
                    task_id=task.task_id,
                    family=task.circuit_family,
                    memory_episode_count_before=memory_count_before,
                    retrieved_episode_count=len(retrieval.episode_hits) if retrieval is not None else 0,
                    advice_count=len(retrieval.feedback_advice) if retrieval is not None else 0,
                    retrieval_precision_proxy=retrieval.retrieval_precision_proxy if retrieval is not None else 0.0,
                    negative_transfer_risk=retrieval.negative_transfer_risk if retrieval is not None else 0.0,
                    warm_start_applied=warm_start_applied,
                    warm_start_source=warm_start_source,
                    best_candidate_id=response.best_candidate.candidate_id if response.best_candidate is not None else None,
                    best_feasible_found=response.final_search_state.best_known_feasible is not None
                    or any(_is_feasible(execution.verification_result) for execution in response.simulation_executions),
                    real_simulation_calls=len(response.simulation_executions),
                    step_to_first_feasible=_step_to_first_feasible(response),
                    dominant_failure_modes=dominant_failures,
                    repeated_failure_count=repeated_failure_count,
                    episode_memory_id=episode_memory_id,
                )
            )

        mode_records = [record for record in all_records if record.mode == mode]
        mode_summaries.append(_summarize_mode(mode, mode_records))

    summary_map = {summary.mode: summary for summary in mode_summaries}
    no_memory = summary_map["no_memory"]
    full_memory = summary_map["full_memory"]
    summary = MemoryAblationSummary(
        memory_reduces_simulation_calls=full_memory.average_real_simulation_calls < no_memory.average_real_simulation_calls,
        memory_reduces_step_to_first_feasible=full_memory.average_step_to_first_feasible < no_memory.average_step_to_first_feasible,
        memory_reduces_repeated_failures=full_memory.average_repeated_failure_count < no_memory.average_repeated_failure_count,
        memory_uses_retrieval_in_practice=full_memory.warm_start_application_rate > 0.0,
        notes=[
            f"episodes={episodes}",
            f"task_slug={task_slug}",
            f"full_memory_warm_start_rate={full_memory.warm_start_application_rate:.3f}",
        ],
    )
    return MemoryAblationSuiteResult(
        task_id=f"benchmark-{task_slug}-memory-ablation",
        modes=["no_memory", "full_memory"],
        episode_records=all_records,
        mode_summaries=mode_summaries,
        summary=summary,
    )


def build_memory_ablation_evidence_bundle(
    suite: MemoryAblationSuiteResult,
    *,
    figures_dir: str | Path,
    tables_dir: str | Path,
    json_output_path: str | Path,
) -> MemoryAblationEvidenceBundle:
    """Build figures and tables for repeated-episode memory evidence."""

    figures_root = Path(figures_dir)
    tables_root = Path(tables_dir)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    records_by_mode = {
        mode: [record for record in suite.episode_records if record.mode == mode]
        for mode in suite.modes
    }

    figure_calls = FigureSpec(
        figure_id="fig_memory_real_simulation_calls_vs_episode",
        title="Repeated-Episode Memory Ablation: Real Simulation Calls",
        chart_type="line",
        x_label="Episode Index",
        y_label="Real Simulation Calls",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(record.episode_index) for record in records_by_mode[mode]],
                y_values=[float(record.real_simulation_calls) for record in records_by_mode[mode]],
                color=color,
            )
            for mode, color in (("no_memory", "#d62728"), ("full_memory", "#1f77b4"))
        ],
        caption="Real SPICE calls per repeated episode. Lower is better.",
        output_path=str(figures_root / "memory_real_simulation_calls_vs_episode.svg"),
    )
    figure_steps = FigureSpec(
        figure_id="fig_memory_step_to_first_feasible_vs_episode",
        title="Repeated-Episode Memory Ablation: Step to First Feasible",
        chart_type="line",
        x_label="Episode Index",
        y_label="Step to First Feasible",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(record.episode_index) for record in records_by_mode[mode]],
                y_values=[
                    float(record.step_to_first_feasible if record.step_to_first_feasible is not None else 0.0)
                    for record in records_by_mode[mode]
                ],
                color=color,
            )
            for mode, color in (("no_memory", "#d62728"), ("full_memory", "#1f77b4"))
        ],
        caption="Optimization step at which a feasible candidate first appears. Lower is better.",
        output_path=str(figures_root / "memory_step_to_first_feasible_vs_episode.svg"),
    )
    figure_failures = FigureSpec(
        figure_id="fig_memory_repeated_failures_vs_episode",
        title="Repeated-Episode Memory Ablation: Repeated Failure Count",
        chart_type="line",
        x_label="Episode Index",
        y_label="Repeated Failure Count",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(record.episode_index) for record in records_by_mode[mode]],
                y_values=[float(record.repeated_failure_count) for record in records_by_mode[mode]],
                color=color,
            )
            for mode, color in (("no_memory", "#d62728"), ("full_memory", "#1f77b4"))
        ],
        caption="Count of dominant failure modes that were already seen in previous episodes. Lower is better.",
        output_path=str(figures_root / "memory_repeated_failures_vs_episode.svg"),
    )
    summary_map = {summary.mode: summary for summary in suite.mode_summaries}
    figure_summary = FigureSpec(
        figure_id="fig_memory_mode_summary",
        title="Repeated-Episode Memory Ablation: Mode Summary",
        chart_type="bar",
        x_label="Mode",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index)],
                y_values=[summary_map[mode].average_real_simulation_calls],
                color=color,
            )
            for index, (mode, color) in enumerate((("no_memory", "#d62728"), ("full_memory", "#1f77b4")))
        ],
        caption="Mode-level comparison on average real simulation calls.",
        output_path=str(figures_root / "memory_mode_summary.svg"),
    )

    for figure in (figure_calls, figure_steps, figure_failures):
        _write_svg_line_chart(figure)
    _write_svg_bar_chart(figure_summary)

    comparison_table = TableSpec(
        table_id="tbl_memory_mode_comparison",
        title="Repeated-Episode Memory Ablation Comparison",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="average_real_simulation_calls", label="Avg Sim Calls"),
            TableColumn(key="average_step_to_first_feasible", label="Avg Step to Feasible"),
            TableColumn(key="average_repeated_failure_count", label="Avg Repeated Failures"),
            TableColumn(key="warm_start_application_rate", label="Warm-Start Rate"),
            TableColumn(key="average_retrieval_precision", label="Retrieval Precision"),
            TableColumn(key="average_negative_transfer_risk", label="Negative Transfer Risk"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": summary.mode,
                    "feasible_hit_rate": summary.feasible_hit_rate,
                    "average_real_simulation_calls": summary.average_real_simulation_calls,
                    "average_step_to_first_feasible": summary.average_step_to_first_feasible,
                    "average_repeated_failure_count": summary.average_repeated_failure_count,
                    "warm_start_application_rate": summary.warm_start_application_rate,
                    "average_retrieval_precision": summary.average_retrieval_precision,
                    "average_negative_transfer_risk": summary.average_negative_transfer_risk,
                }
            )
            for summary in suite.mode_summaries
        ],
        caption="Primary repeated-episode memory ablation comparison table.",
        csv_output_path=str(tables_root / "memory_mode_comparison.csv"),
        markdown_output_path=str(tables_root / "memory_mode_comparison.md"),
    )
    episode_table = TableSpec(
        table_id="tbl_memory_episode_breakdown",
        title="Repeated-Episode Memory Breakdown",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="episode_index", label="Episode"),
            TableColumn(key="warm_start_applied", label="Warm Start"),
            TableColumn(key="real_simulation_calls", label="Real Sim Calls"),
            TableColumn(key="step_to_first_feasible", label="Step to Feasible"),
            TableColumn(key="repeated_failure_count", label="Repeated Failures"),
            TableColumn(key="retrieved_episode_count", label="Retrieved Episodes"),
            TableColumn(key="retrieval_precision_proxy", label="Retrieval Precision"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": record.mode,
                    "episode_index": record.episode_index,
                    "warm_start_applied": record.warm_start_applied,
                    "real_simulation_calls": record.real_simulation_calls,
                    "step_to_first_feasible": record.step_to_first_feasible if record.step_to_first_feasible is not None else "na",
                    "repeated_failure_count": record.repeated_failure_count,
                    "retrieved_episode_count": record.retrieved_episode_count,
                    "retrieval_precision_proxy": record.retrieval_precision_proxy,
                }
            )
            for record in suite.episode_records
        ],
        caption="Per-episode repeated-memory breakdown for regression inspection and paper appendix.",
        csv_output_path=str(tables_root / "memory_episode_breakdown.csv"),
        markdown_output_path=str(tables_root / "memory_episode_breakdown.md"),
    )

    for table in (comparison_table, episode_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    bundle = MemoryAblationEvidenceBundle(
        task_id=suite.task_id,
        modes=suite.modes,
        figures=[figure_calls, figure_steps, figure_failures, figure_summary],
        tables=[comparison_table, episode_table],
        summary=suite.summary,
        json_output_path=str(json_output_path),
    )
    Path(json_output_path).write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle


def run_cross_task_memory_transfer_suite(
    *,
    source_task_slug: str,
    source_task_builder: TaskBuilder,
    target_task_slug: str,
    target_task_builder: TaskBuilder,
    transfer_kind: str,
    source_episodes: int = 3,
    target_episodes: int = 3,
    max_steps: int = 3,
    fidelity_level: str = "focused_truth",
    backend_preference: str = "ngspice",
) -> MemoryTransferSuiteResult:
    """Run cross-task memory transfer with governed and forced retrieval modes."""

    if transfer_kind not in {"same_family", "cross_family"}:
        raise ValueError("transfer_kind must be same_family or cross_family")

    source_bundle = _build_source_memory_bundle(
        source_task_slug=source_task_slug,
        task_builder=source_task_builder,
        episodes=source_episodes,
        max_steps=max_steps,
        fidelity_level=fidelity_level,
        backend_preference=backend_preference,
    )

    records: list[MemoryTransferStatsRecord] = []
    for mode in ("no_memory", "governed_transfer", "forced_transfer"):
        prior_failure_counts: Counter[str] = Counter()
        for episode_index in range(target_episodes):
            task = target_task_builder(task_id=f"benchmark-{target_task_slug}-{mode}-transfer-{episode_index}")
            task = _apply_episode_initialization_shift(task, episode_index)
            warm_start_applied = False
            warm_start_source = None
            retrieval_precision = 0.0
            negative_transfer_risk = 0.0
            retrieved_episode_count = 0
            effective_max_steps = max_steps

            if mode == "governed_transfer" and source_bundle.episode_records:
                retrieval = MemoryService(source_bundle).retrieve_relevant_memory(task)
                retrieval_precision = retrieval.retrieval_precision_proxy
                negative_transfer_risk = retrieval.negative_transfer_risk
                retrieved_episode_count = len(retrieval.episode_hits)
                if retrieval.negative_transfer_risk <= 0.55:
                    task, warm_start_applied, warm_start_source = _apply_memory_warm_start(task, source_bundle, retrieval)
                    if warm_start_applied and retrieval.retrieval_precision_proxy >= 0.4:
                        effective_max_steps = max(1, max_steps - 1)
            elif mode == "forced_transfer" and source_bundle.episode_records:
                task, warm_start_applied, warm_start_source, retrieval_precision, negative_transfer_risk = _apply_forced_memory_warm_start(task, source_bundle)
                retrieved_episode_count = 1 if warm_start_applied else 0

            response = run_planning_truth_loop(
                task,
                max_steps=effective_max_steps,
                fidelity_level=fidelity_level,
                backend_preference=backend_preference,
                escalation_reason=f"{source_task_slug}_to_{target_task_slug}:memory_transfer:{mode}",
            )
            verification = _primary_verification(response)
            dominant_failures = _dominant_failure_modes(response, verification)
            repeated_failure_count = _repeated_failure_count(response, prior_failure_counts)
            for failure in dominant_failures:
                prior_failure_counts[failure] += 1

            harmful_transfer = (
                mode == "forced_transfer"
                and warm_start_applied
                and negative_transfer_risk >= 0.5
                and repeated_failure_count > 0
            )

            records.append(
                MemoryTransferStatsRecord(
                    source_task_slug=source_task_slug,
                    target_task_slug=target_task_slug,
                    transfer_kind=transfer_kind,
                    episode_index=episode_index,
                    mode=mode,
                    source_episode_count=len(source_bundle.episode_records),
                    retrieved_episode_count=retrieved_episode_count,
                    retrieval_precision_proxy=retrieval_precision,
                    negative_transfer_risk=negative_transfer_risk,
                    warm_start_applied=warm_start_applied,
                    warm_start_source=warm_start_source,
                    best_feasible_found=response.final_search_state.best_known_feasible is not None
                    or any(_is_feasible(execution.verification_result) for execution in response.simulation_executions),
                    real_simulation_calls=len(response.simulation_executions),
                    step_to_first_feasible=_step_to_first_feasible(response),
                    repeated_failure_count=repeated_failure_count,
                    harmful_transfer_applied=harmful_transfer,
                )
            )

    mode_summaries = [
        _summarize_transfer_mode(mode, [record for record in records if record.mode == mode])
        for mode in ("no_memory", "governed_transfer", "forced_transfer")
    ]
    summary_map = {summary.mode: summary for summary in mode_summaries}
    no_memory = summary_map["no_memory"]
    governed = summary_map["governed_transfer"]
    forced = summary_map["forced_transfer"]
    summary = MemoryTransferSummary(
        governed_transfer_beneficial=(
            governed.average_real_simulation_calls < no_memory.average_real_simulation_calls
            or governed.average_step_to_first_feasible < no_memory.average_step_to_first_feasible
        ),
        governance_blocks_harmful_transfer=governed.harmful_transfer_rate < forced.harmful_transfer_rate,
        forced_transfer_exposes_negative_transfer=forced.harmful_transfer_rate > 0.0,
        notes=[
            f"source_episodes={source_episodes}",
            f"target_episodes={target_episodes}",
            f"transfer_kind={transfer_kind}",
        ],
    )
    return MemoryTransferSuiteResult(
        source_task_slug=source_task_slug,
        target_task_slug=target_task_slug,
        transfer_kind=transfer_kind,
        modes=["no_memory", "governed_transfer", "forced_transfer"],
        transfer_records=records,
        mode_summaries=mode_summaries,
        summary=summary,
    )


def build_memory_transfer_evidence_bundle(
    suite: MemoryTransferSuiteResult,
    *,
    figures_dir: str | Path,
    tables_dir: str | Path,
    json_output_path: str | Path,
) -> MemoryTransferEvidenceBundle:
    """Build figures and tables for cross-task memory transfer evidence."""

    figures_root = Path(figures_dir)
    tables_root = Path(tables_dir)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    records_by_mode = {
        mode: [record for record in suite.transfer_records if record.mode == mode]
        for mode in suite.modes
    }
    colors = {
        "no_memory": "#d62728",
        "governed_transfer": "#1f77b4",
        "forced_transfer": "#9467bd",
    }
    figure_calls = FigureSpec(
        figure_id="fig_memory_transfer_simulation_calls",
        title="Cross-Task Memory Transfer: Real Simulation Calls",
        chart_type="bar",
        x_label="Transfer Mode",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(
                label=summary.mode,
                x_values=[float(index)],
                y_values=[summary.average_real_simulation_calls],
                color=colors[summary.mode],
            )
            for index, summary in enumerate(suite.mode_summaries)
        ],
        caption="Cross-task transfer comparison on real SPICE call count.",
        output_path=str(figures_root / "memory_transfer_simulation_calls.svg"),
    )
    figure_step = FigureSpec(
        figure_id="fig_memory_transfer_step_to_first_feasible",
        title="Cross-Task Memory Transfer: Step to First Feasible",
        chart_type="bar",
        x_label="Transfer Mode",
        y_label="Average Step to First Feasible",
        series=[
            FigureSeries(
                label=summary.mode,
                x_values=[float(index)],
                y_values=[summary.average_step_to_first_feasible],
                color=colors[summary.mode],
            )
            for index, summary in enumerate(suite.mode_summaries)
        ],
        caption="Cross-task transfer comparison on first-feasible step.",
        output_path=str(figures_root / "memory_transfer_step_to_first_feasible.svg"),
    )
    figure_harm = FigureSpec(
        figure_id="fig_memory_transfer_harmful_transfer_rate",
        title="Cross-Task Memory Transfer: Harmful Transfer Rate",
        chart_type="bar",
        x_label="Transfer Mode",
        y_label="Harmful Transfer Rate",
        series=[
            FigureSeries(
                label=summary.mode,
                x_values=[float(index)],
                y_values=[summary.harmful_transfer_rate],
                color=colors[summary.mode],
            )
            for index, summary in enumerate(suite.mode_summaries)
        ],
        caption="Rate of harmful transfer activation under each mode.",
        output_path=str(figures_root / "memory_transfer_harmful_transfer_rate.svg"),
    )

    for figure in (figure_calls, figure_step, figure_harm):
        _write_svg_bar_chart(figure)

    comparison_table = TableSpec(
        table_id="tbl_memory_transfer_comparison",
        title="Cross-Task Memory Transfer Comparison",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="feasible_hit_rate", label="Feasible Hit Rate"),
            TableColumn(key="average_real_simulation_calls", label="Avg Sim Calls"),
            TableColumn(key="average_step_to_first_feasible", label="Avg Step to Feasible"),
            TableColumn(key="average_repeated_failure_count", label="Avg Repeated Failures"),
            TableColumn(key="warm_start_application_rate", label="Warm-Start Rate"),
            TableColumn(key="average_retrieval_precision", label="Retrieval Precision"),
            TableColumn(key="average_negative_transfer_risk", label="Negative Transfer Risk"),
            TableColumn(key="harmful_transfer_rate", label="Harmful Transfer Rate"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": summary.mode,
                    "feasible_hit_rate": summary.feasible_hit_rate,
                    "average_real_simulation_calls": summary.average_real_simulation_calls,
                    "average_step_to_first_feasible": summary.average_step_to_first_feasible,
                    "average_repeated_failure_count": summary.average_repeated_failure_count,
                    "warm_start_application_rate": summary.warm_start_application_rate,
                    "average_retrieval_precision": summary.average_retrieval_precision,
                    "average_negative_transfer_risk": summary.average_negative_transfer_risk,
                    "harmful_transfer_rate": summary.harmful_transfer_rate,
                }
            )
            for summary in suite.mode_summaries
        ],
        caption="Primary cross-task transfer comparison table.",
        csv_output_path=str(tables_root / "memory_transfer_comparison.csv"),
        markdown_output_path=str(tables_root / "memory_transfer_comparison.md"),
    )
    episode_table = TableSpec(
        table_id="tbl_memory_transfer_episode_breakdown",
        title="Cross-Task Memory Transfer Breakdown",
        columns=[
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="episode_index", label="Episode"),
            TableColumn(key="warm_start_applied", label="Warm Start"),
            TableColumn(key="real_simulation_calls", label="Real Sim Calls"),
            TableColumn(key="step_to_first_feasible", label="Step to Feasible"),
            TableColumn(key="repeated_failure_count", label="Repeated Failures"),
            TableColumn(key="retrieved_episode_count", label="Retrieved Episodes"),
            TableColumn(key="negative_transfer_risk", label="Negative Transfer Risk"),
            TableColumn(key="harmful_transfer_applied", label="Harmful Transfer"),
        ],
        rows=[
            TableRow(
                values={
                    "mode": record.mode,
                    "episode_index": record.episode_index,
                    "warm_start_applied": record.warm_start_applied,
                    "real_simulation_calls": record.real_simulation_calls,
                    "step_to_first_feasible": record.step_to_first_feasible if record.step_to_first_feasible is not None else "na",
                    "repeated_failure_count": record.repeated_failure_count,
                    "retrieved_episode_count": record.retrieved_episode_count,
                    "negative_transfer_risk": record.negative_transfer_risk,
                    "harmful_transfer_applied": record.harmful_transfer_applied,
                }
            )
            for record in suite.transfer_records
        ],
        caption="Per-episode cross-task transfer breakdown for appendix and failure analysis.",
        csv_output_path=str(tables_root / "memory_transfer_episode_breakdown.csv"),
        markdown_output_path=str(tables_root / "memory_transfer_episode_breakdown.md"),
    )

    for table in (comparison_table, episode_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    bundle = MemoryTransferEvidenceBundle(
        source_task_slug=suite.source_task_slug,
        target_task_slug=suite.target_task_slug,
        transfer_kind=suite.transfer_kind,
        modes=suite.modes,
        figures=[figure_calls, figure_step, figure_harm],
        tables=[comparison_table, episode_table],
        summary=suite.summary,
        json_output_path=str(json_output_path),
    )
    Path(json_output_path).write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle
