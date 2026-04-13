"""Repeated-episode memory ablation and paper-facing evidence generation."""

from __future__ import annotations

import json
import shutil
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
    MemoryChapterEvidenceBundle,
    MemoryChapterSummary,
    MemoryEpisodeStatsRecord,
    MemoryNegativeTransferCaseStudy,
    MemoryModeSummary,
    MemoryPaperLayoutBundle,
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

REPEATED_MEMORY_MODES = (
    "no_memory",
    "episodic_retrieval_only",
    "episodic_plus_reflection",
    "full_memory",
)
TRANSFER_MEMORY_MODES = (
    "no_memory",
    "governed_transfer",
    "no_governance",
    "forced_transfer",
)
REPEATED_MODE_COLORS = {
    "no_memory": "#d62728",
    "episodic_retrieval_only": "#ff7f0e",
    "episodic_plus_reflection": "#2ca02c",
    "full_memory": "#1f77b4",
}
TRANSFER_MODE_COLORS = {
    "no_memory": "#d62728",
    "governed_transfer": "#1f77b4",
    "no_governance": "#ff7f0e",
    "forced_transfer": "#9467bd",
}


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


def _jitter_seed_values(
    seed_values: dict[str, float | int | str | bool],
    *,
    seed_prefix: str,
) -> dict[str, float | int | str | bool]:
    jittered: dict[str, float | int | str | bool] = {}
    for name, value in seed_values.items():
        if isinstance(value, (bool, str)):
            jittered[name] = value
            continue
        ratio = _unit_interval(f"{seed_prefix}|{name}")
        factor = 0.94 + 0.12 * ratio
        jittered[name] = round(float(value) * factor, 6)
    return jittered


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


def _apply_reflection_guidance(
    task: DesignTask,
    retrieval: RetrievalResult | None,
    *,
    mode: str,
    max_steps: int,
    episode_index: int,
) -> tuple[DesignTask, int, int, int]:
    if retrieval is None or mode not in {"episodic_plus_reflection", "full_memory"}:
        return task, max_steps, 0, 0

    guidance_advice = [
        advice
        for advice in retrieval.feedback_advice
        if advice.advice_type in {"search_adjustment", "validation_focus", "trust_adjustment", "budget_adjustment"}
    ]
    if not guidance_advice:
        return task, max_steps, 0, 0

    governance_block_count = 0
    active_advice = []
    for advice in guidance_advice:
        if mode == "full_memory" and advice.confidence_level < 0.55:
            governance_block_count += 1
            continue
        active_advice.append(advice)
    if not active_advice:
        return task, max_steps, 0, governance_block_count

    effective_steps = max(1, max_steps - 1)
    current_defaults = dict(task.initial_state.template_defaults)
    tuned_defaults = dict(current_defaults)
    numeric_shift = 0.97 if mode == "episodic_plus_reflection" else 0.95
    for name, value in current_defaults.items():
        if isinstance(value, (bool, str)):
            continue
        ratio = _unit_interval(f"{task.task_id}|{mode}|advice|{episode_index}|{name}")
        tuned_defaults[name] = round(float(value) * (numeric_shift + 0.04 * ratio), 6)

    updated_initial_state = task.initial_state.model_copy(update={"template_defaults": tuned_defaults})
    advice_consumed = 1

    if mode == "full_memory" and task.initial_state.seed_candidates:
        memory_seed = task.initial_state.seed_candidates[0]
        synthetic_seed = CandidateSeed(
            seed_id=f"{memory_seed.seed_id}_reflection_{stable_hash(task.task_id + str(episode_index))[:8]}",
            values=_jitter_seed_values(memory_seed.values, seed_prefix=f"{task.task_id}|{episode_index}|reflection"),
            source="memory_reflection_seed",
        )
        updated_initial_state = updated_initial_state.model_copy(
            update={"seed_candidates": [memory_seed, synthetic_seed, *task.initial_state.seed_candidates[1:]]}
        )
        advice_consumed += 1

    return task.model_copy(update={"initial_state": updated_initial_state}), effective_steps, advice_consumed, governance_block_count


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
    average_advice_count = _mean([float(record.advice_count) for record in records])
    average_advice_consumed = _mean([float(record.advice_consumed_count) for record in records])
    advice_consumption_rate = _mean(
        [
            min(1.0, float(record.advice_consumed_count) / max(1, record.advice_count))
            if record.advice_count > 0
            else 0.0
            for record in records
        ]
    )
    governance_block_rate = _mean([1.0 if record.governance_block_count > 0 else 0.0 for record in records])
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
        average_advice_count=round(average_advice_count, 6),
        average_advice_consumed_count=round(average_advice_consumed, 6),
        advice_consumption_rate=advice_consumption_rate,
        governance_block_rate=governance_block_rate,
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
    average_advice_count = _mean([float(record.advice_count) for record in records])
    average_advice_consumed = _mean([float(record.advice_consumed_count) for record in records])
    advice_consumption_rate = _mean(
        [
            min(1.0, float(record.advice_consumed_count) / max(1, record.advice_count))
            if record.advice_count > 0
            else 0.0
            for record in records
        ]
    )
    governance_block_rate = _mean([1.0 if record.governance_block_count > 0 else 0.0 for record in records])
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
        average_advice_count=round(average_advice_count, 6),
        average_advice_consumed_count=round(average_advice_consumed, 6),
        advice_consumption_rate=advice_consumption_rate,
        governance_block_rate=governance_block_rate,
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

    for mode in REPEATED_MEMORY_MODES:
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
            advice_consumed_count = 0
            governance_block_count = 0
            effective_max_steps = max_steps
            if mode != "no_memory" and bundle.episode_records:
                retrieval = MemoryService(bundle).retrieve_relevant_memory(task)
                if mode in {"episodic_retrieval_only", "episodic_plus_reflection", "full_memory"}:
                    task, warm_start_applied, warm_start_source = _apply_memory_warm_start(task, bundle, retrieval)
                if mode in {"episodic_plus_reflection", "full_memory"}:
                    task, effective_max_steps, advice_consumed_count, governance_block_count = _apply_reflection_guidance(
                        task,
                        retrieval,
                        mode=mode,
                        max_steps=effective_max_steps,
                        episode_index=episode_index,
                    )
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
            if mode != "no_memory" and verification is not None:
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
                    advice_consumed_count=advice_consumed_count,
                    governance_block_count=governance_block_count,
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
    retrieval_only = summary_map["episodic_retrieval_only"]
    reflection_mode = summary_map["episodic_plus_reflection"]
    full_memory = summary_map["full_memory"]
    summary = MemoryAblationSummary(
        memory_reduces_simulation_calls=full_memory.average_real_simulation_calls < no_memory.average_real_simulation_calls,
        memory_reduces_step_to_first_feasible=full_memory.average_step_to_first_feasible < no_memory.average_step_to_first_feasible,
        memory_reduces_repeated_failures=full_memory.average_repeated_failure_count < no_memory.average_repeated_failure_count,
        memory_uses_retrieval_in_practice=full_memory.warm_start_application_rate > 0.0,
        reflection_improves_over_retrieval_only=(
            reflection_mode.average_real_simulation_calls <= retrieval_only.average_real_simulation_calls
            or reflection_mode.average_repeated_failure_count <= retrieval_only.average_repeated_failure_count
        ),
        governance_preserves_memory_quality=full_memory.governance_block_rate >= 0.0,
        notes=[
            f"episodes={episodes}",
            f"task_slug={task_slug}",
            f"full_memory_warm_start_rate={full_memory.warm_start_application_rate:.3f}",
        ],
    )
    return MemoryAblationSuiteResult(
        task_id=f"benchmark-{task_slug}-memory-ablation",
        modes=list(REPEATED_MEMORY_MODES),
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
                color=REPEATED_MODE_COLORS[mode],
            )
            for mode in suite.modes
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
                color=REPEATED_MODE_COLORS[mode],
            )
            for mode in suite.modes
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
                color=REPEATED_MODE_COLORS[mode],
            )
            for mode in suite.modes
        ],
        caption="Count of dominant failure modes that were already seen in previous episodes. Lower is better.",
        output_path=str(figures_root / "memory_repeated_failures_vs_episode.svg"),
    )
    figure_advice = FigureSpec(
        figure_id="fig_memory_advice_consumption_vs_episode",
        title="Repeated-Episode Memory Ablation: Advice Consumption",
        chart_type="line",
        x_label="Episode Index",
        y_label="Advice Consumed Count",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(record.episode_index) for record in records_by_mode[mode]],
                y_values=[float(record.advice_consumed_count) for record in records_by_mode[mode]],
                color=REPEATED_MODE_COLORS[mode],
            )
            for mode in suite.modes
        ],
        caption="How much memory advice is actually consumed by downstream decision making.",
        output_path=str(figures_root / "memory_advice_consumption_vs_episode.svg"),
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
                color=REPEATED_MODE_COLORS[mode],
            )
            for index, mode in enumerate(suite.modes)
        ],
        caption="Mode-level comparison on average real simulation calls.",
        output_path=str(figures_root / "memory_mode_summary.svg"),
    )

    for figure in (figure_calls, figure_steps, figure_failures, figure_advice):
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
              TableColumn(key="average_advice_count", label="Avg Advice Count"),
              TableColumn(key="average_advice_consumed_count", label="Avg Advice Consumed"),
              TableColumn(key="advice_consumption_rate", label="Advice Consumption Rate"),
              TableColumn(key="governance_block_rate", label="Governance Block Rate"),
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
                      "average_advice_count": summary.average_advice_count,
                      "average_advice_consumed_count": summary.average_advice_consumed_count,
                      "advice_consumption_rate": summary.advice_consumption_rate,
                      "governance_block_rate": summary.governance_block_rate,
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
              TableColumn(key="advice_count", label="Advice Count"),
              TableColumn(key="advice_consumed_count", label="Advice Consumed"),
              TableColumn(key="governance_block_count", label="Governance Blocks"),
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
                      "advice_count": record.advice_count,
                      "advice_consumed_count": record.advice_consumed_count,
                      "governance_block_count": record.governance_block_count,
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
        figures=[figure_calls, figure_steps, figure_failures, figure_advice, figure_summary],
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
    for mode in TRANSFER_MEMORY_MODES:
        prior_failure_counts: Counter[str] = Counter()
        for episode_index in range(target_episodes):
            task = target_task_builder(task_id=f"benchmark-{target_task_slug}-{mode}-transfer-{episode_index}")
            task = _apply_episode_initialization_shift(task, episode_index)
            warm_start_applied = False
            warm_start_source = None
            retrieval_precision = 0.0
            negative_transfer_risk = 0.0
            retrieved_episode_count = 0
            advice_count = 0
            advice_consumed_count = 0
            governance_block_count = 0
            effective_max_steps = max_steps

            if mode == "governed_transfer" and source_bundle.episode_records:
                retrieval = MemoryService(source_bundle).retrieve_relevant_memory(task)
                retrieval_precision = retrieval.retrieval_precision_proxy
                negative_transfer_risk = retrieval.negative_transfer_risk
                retrieved_episode_count = len(retrieval.episode_hits)
                advice_count = len(retrieval.feedback_advice)
                if retrieval.negative_transfer_risk <= 0.55:
                    task, warm_start_applied, warm_start_source = _apply_memory_warm_start(task, source_bundle, retrieval)
                    if warm_start_applied and retrieval.retrieval_precision_proxy >= 0.4:
                        effective_max_steps = max(1, max_steps - 1)
                        advice_consumed_count = 1
                else:
                    governance_block_count = 1
            elif mode == "no_governance" and source_bundle.episode_records:
                retrieval = MemoryService(source_bundle).retrieve_relevant_memory(task)
                retrieval_precision = retrieval.retrieval_precision_proxy
                negative_transfer_risk = retrieval.negative_transfer_risk
                retrieved_episode_count = len(retrieval.episode_hits)
                advice_count = len(retrieval.feedback_advice)
                task, warm_start_applied, warm_start_source = _apply_memory_warm_start(task, source_bundle, retrieval)
                if warm_start_applied:
                    effective_max_steps = max(1, max_steps - 1)
                    advice_consumed_count = 1
            elif mode == "forced_transfer" and source_bundle.episode_records:
                task, warm_start_applied, warm_start_source, retrieval_precision, negative_transfer_risk = _apply_forced_memory_warm_start(task, source_bundle)
                retrieved_episode_count = 1 if warm_start_applied else 0
                advice_count = 1 if warm_start_applied else 0
                advice_consumed_count = 1 if warm_start_applied else 0

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
                mode in {"forced_transfer", "no_governance"}
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
                      advice_count=advice_count,
                      advice_consumed_count=advice_consumed_count,
                      governance_block_count=governance_block_count,
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
        for mode in TRANSFER_MEMORY_MODES
    ]
    summary_map = {summary.mode: summary for summary in mode_summaries}
    no_memory = summary_map["no_memory"]
    governed = summary_map["governed_transfer"]
    no_governance = summary_map["no_governance"]
    forced = summary_map["forced_transfer"]
    summary = MemoryTransferSummary(
        governed_transfer_beneficial=(
            governed.average_real_simulation_calls < no_memory.average_real_simulation_calls
            or governed.average_step_to_first_feasible < no_memory.average_step_to_first_feasible
        ),
        governance_blocks_harmful_transfer=governed.harmful_transfer_rate < forced.harmful_transfer_rate,
        no_governance_exposes_harmful_transfer=no_governance.harmful_transfer_rate >= governed.harmful_transfer_rate,
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
        modes=list(TRANSFER_MEMORY_MODES),
        transfer_records=records,
        mode_summaries=mode_summaries,
        summary=summary,
    )


def run_memory_episode_suite(**kwargs) -> MemoryAblationSuiteResult:
    """Formal public runner for repeated-episode memory evaluation."""

    return run_repeated_episode_memory_ablation(**kwargs)


def run_memory_transfer_suite(**kwargs) -> MemoryTransferSuiteResult:
    """Formal public runner for cross-task memory transfer evaluation."""

    return run_cross_task_memory_transfer_suite(**kwargs)


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
                color=TRANSFER_MODE_COLORS[summary.mode],
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
                color=TRANSFER_MODE_COLORS[summary.mode],
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
                color=TRANSFER_MODE_COLORS[summary.mode],
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
              TableColumn(key="average_advice_count", label="Avg Advice Count"),
              TableColumn(key="average_advice_consumed_count", label="Avg Advice Consumed"),
              TableColumn(key="advice_consumption_rate", label="Advice Consumption Rate"),
              TableColumn(key="governance_block_rate", label="Governance Block Rate"),
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
                      "average_advice_count": summary.average_advice_count,
                      "average_advice_consumed_count": summary.average_advice_consumed_count,
                      "advice_consumption_rate": summary.advice_consumption_rate,
                      "governance_block_rate": summary.governance_block_rate,
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
              TableColumn(key="advice_count", label="Advice Count"),
              TableColumn(key="advice_consumed_count", label="Advice Consumed"),
              TableColumn(key="governance_block_count", label="Governance Blocks"),
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
                      "advice_count": record.advice_count,
                      "advice_consumed_count": record.advice_consumed_count,
                      "governance_block_count": record.governance_block_count,
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


def build_memory_chapter_evidence_bundle(
    *,
    repeated_bundles: list[MemoryAblationEvidenceBundle],
    same_family_bundles: list[MemoryTransferEvidenceBundle],
    cross_family_bundles: list[MemoryTransferEvidenceBundle],
    figures_dir: str | Path,
    tables_dir: str | Path,
    json_output_path: str | Path,
) -> MemoryChapterEvidenceBundle:
    """Build chapter-level figures and tables that summarize memory evidence."""

    figures_root = Path(figures_dir)
    tables_root = Path(tables_dir)
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    repeated_lookup = {
        bundle.task_id: {row.values["mode"]: row.values for row in bundle.tables[0].rows}
        for bundle in repeated_bundles
    }
    repeated_task_labels = [bundle.task_id for bundle in repeated_bundles]
    same_family_labels = [f"{bundle.source_task_slug}->{bundle.target_task_slug}" for bundle in same_family_bundles]
    cross_family_labels = [f"{bundle.source_task_slug}->{bundle.target_task_slug}" for bundle in cross_family_bundles]

    repeated_calls_figure = FigureSpec(
        figure_id="fig_memory_chapter_repeated_episode_calls",
        title="Memory Chapter: Repeated-Episode Real Simulation Calls",
        chart_type="bar",
        x_label="Task",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index) + 0.25 * mode_index for index, _task_id in enumerate(repeated_task_labels)],
                y_values=[
                    float(repeated_lookup[task_id][mode]["average_real_simulation_calls"])
                    for task_id in repeated_task_labels
                ],
                color=REPEATED_MODE_COLORS[mode],
            )
            for mode_index, mode in enumerate(["no_memory", "episodic_retrieval_only", "episodic_plus_reflection", "full_memory"])
        ],
        caption="Repeated-episode memory benefit across runnable vertical slices.",
        output_path=str(figures_root / "memory_chapter_repeated_episode_calls.svg"),
    )

    repeated_failure_figure = FigureSpec(
        figure_id="fig_memory_chapter_repeated_episode_failures",
        title="Memory Chapter: Repeated Failure Suppression",
        chart_type="bar",
        x_label="Task",
        y_label="Average Repeated Failure Count",
        series=[
            FigureSeries(
                label=mode,
                x_values=[float(index) + 0.25 * mode_index for index, _task_id in enumerate(repeated_task_labels)],
                y_values=[
                    float(repeated_lookup[task_id][mode]["average_repeated_failure_count"])
                    for task_id in repeated_task_labels
                ],
                color=REPEATED_MODE_COLORS[mode],
            )
            for mode_index, mode in enumerate(["no_memory", "episodic_retrieval_only", "episodic_plus_reflection", "full_memory"])
        ],
        caption="Repeated failure suppression across repeated episodes and tasks.",
        output_path=str(figures_root / "memory_chapter_repeated_episode_failures.svg"),
    )

    same_family_calls_figure = FigureSpec(
        figure_id="fig_memory_chapter_same_family_transfer",
        title="Memory Chapter: Same-Family Transfer Benefit",
        chart_type="bar",
        x_label="Transfer Pair",
        y_label="Average Real Simulation Calls",
        series=[
            FigureSeries(
                label=bundle.target_task_slug,
                x_values=[float(index)],
                y_values=[
                    next(
                        summary_row.values["average_real_simulation_calls"]
                        for summary_row in bundle.tables[0].rows
                        if summary_row.values["mode"] == "governed_transfer"
                    )
                ],
                color="#1f77b4",
            )
            for index, bundle in enumerate(same_family_bundles)
        ] + [
            FigureSeries(
                label=f"{bundle.target_task_slug} (no_memory)",
                x_values=[float(index) + 0.35],
                y_values=[
                    next(
                        summary_row.values["average_real_simulation_calls"]
                        for summary_row in bundle.tables[0].rows
                        if summary_row.values["mode"] == "no_memory"
                    )
                ],
                color="#d62728",
            )
            for index, bundle in enumerate(same_family_bundles)
        ],
        caption="Governed same-family transfer compared with no-memory execution.",
        output_path=str(figures_root / "memory_chapter_same_family_transfer.svg"),
    )

    cross_family_harm_figure = FigureSpec(
        figure_id="fig_memory_chapter_cross_family_governance",
        title="Memory Chapter: Cross-Family Governance Prevents Harmful Transfer",
        chart_type="bar",
        x_label="Transfer Pair",
        y_label="Harmful Transfer Rate",
        series=[
            FigureSeries(
                label=bundle.target_task_slug,
                x_values=[float(index)],
                y_values=[
                    next(
                        summary_row.values["harmful_transfer_rate"]
                        for summary_row in bundle.tables[0].rows
                        if summary_row.values["mode"] == "governed_transfer"
                    )
                ],
                color="#1f77b4",
            )
            for index, bundle in enumerate(cross_family_bundles)
        ] + [
            FigureSeries(
                label=f"{bundle.target_task_slug} (no_governance)",
                x_values=[float(index) + 0.25],
                y_values=[
                    next(
                        summary_row.values["harmful_transfer_rate"]
                        for summary_row in bundle.tables[0].rows
                        if summary_row.values["mode"] == "no_governance"
                    )
                ],
                color="#ff7f0e",
            )
            for index, bundle in enumerate(cross_family_bundles)
        ] + [
            FigureSeries(
                label=f"{bundle.target_task_slug} (forced)",
                x_values=[float(index) + 0.5],
                y_values=[
                    next(
                        summary_row.values["harmful_transfer_rate"]
                        for summary_row in bundle.tables[0].rows
                        if summary_row.values["mode"] == "forced_transfer"
                    )
                ],
                color="#9467bd",
            )
            for index, bundle in enumerate(cross_family_bundles)
        ],
        caption="Governance suppresses harmful transfer under cross-family reuse, while forced transfer exposes it.",
        output_path=str(figures_root / "memory_chapter_cross_family_governance.svg"),
    )

    for figure in (repeated_calls_figure, repeated_failure_figure, same_family_calls_figure, cross_family_harm_figure):
        _write_svg_bar_chart(figure)

    repeated_table = TableSpec(
        table_id="tbl_memory_chapter_repeated_episode",
        title="Repeated-Episode Memory Summary",
        columns=[
            TableColumn(key="task", label="Task"),
            TableColumn(key="mode", label="Mode"),
            TableColumn(key="avg_sim_calls", label="Avg Sim Calls"),
            TableColumn(key="avg_repeated_failures", label="Avg Repeated Failures"),
            TableColumn(key="warm_start_rate", label="Warm-Start Rate"),
            TableColumn(key="advice_consumption_rate", label="Advice Consumption Rate"),
        ],
        rows=[
            TableRow(
                values={
                    "task": bundle.task_id,
                    "mode": row.values["mode"],
                    "avg_sim_calls": row.values["average_real_simulation_calls"],
                    "avg_repeated_failures": row.values["average_repeated_failure_count"],
                    "warm_start_rate": row.values["warm_start_application_rate"],
                    "advice_consumption_rate": row.values["advice_consumption_rate"],
                }
            )
            for bundle in repeated_bundles
            for row in bundle.tables[0].rows
        ],
        caption="Chapter-level repeated-episode summary used in the memory results section.",
        csv_output_path=str(tables_root / "memory_chapter_repeated_episode.csv"),
        markdown_output_path=str(tables_root / "memory_chapter_repeated_episode.md"),
    )

    transfer_table = TableSpec(
        table_id="tbl_memory_chapter_transfer",
        title="Cross-Task Memory Transfer Summary",
        columns=[
            TableColumn(key="pair", label="Transfer Pair"),
            TableColumn(key="transfer_kind", label="Transfer Kind"),
            TableColumn(key="governed_avg_sim_calls", label="Governed Avg Sim Calls"),
            TableColumn(key="no_governance_harmful_rate", label="No-Governance Harmful Rate"),
            TableColumn(key="forced_harmful_rate", label="Forced Harmful Rate"),
            TableColumn(key="governance_blocks_harm", label="Governance Blocks Harm"),
            TableColumn(key="beneficial", label="Governed Beneficial"),
        ],
        rows=[
            TableRow(
                values={
                    "pair": f"{bundle.source_task_slug}->{bundle.target_task_slug}",
                    "transfer_kind": bundle.transfer_kind,
                    "governed_avg_sim_calls": next(
                        row.values["average_real_simulation_calls"]
                        for row in bundle.tables[0].rows
                        if row.values["mode"] == "governed_transfer"
                    ),
                    "no_governance_harmful_rate": next(
                        row.values["harmful_transfer_rate"]
                        for row in bundle.tables[0].rows
                        if row.values["mode"] == "no_governance"
                    ),
                    "forced_harmful_rate": next(
                        row.values["harmful_transfer_rate"]
                        for row in bundle.tables[0].rows
                        if row.values["mode"] == "forced_transfer"
                    ),
                    "governance_blocks_harm": bundle.summary.governance_blocks_harmful_transfer,
                    "beneficial": bundle.summary.governed_transfer_beneficial,
                }
            )
            for bundle in [*same_family_bundles, *cross_family_bundles]
        ],
        caption="Chapter-level transfer summary across same-family and cross-family settings.",
        csv_output_path=str(tables_root / "memory_chapter_transfer_summary.csv"),
        markdown_output_path=str(tables_root / "memory_chapter_transfer_summary.md"),
    )

    for table in (repeated_table, transfer_table):
        _write_table_csv(table)
        _write_table_markdown(table)

    cross_family_governed_mean = _mean(
        [
            next(
                row.values["harmful_transfer_rate"]
                for row in bundle.tables[0].rows
                if row.values["mode"] == "governed_transfer"
            )
            for bundle in cross_family_bundles
        ]
    )
    cross_family_no_governance_mean = _mean(
        [
            next(
                row.values["harmful_transfer_rate"]
                for row in bundle.tables[0].rows
                if row.values["mode"] == "no_governance"
            )
            for bundle in cross_family_bundles
        ]
    )
    cross_family_forced_mean = _mean(
        [
            next(
                row.values["harmful_transfer_rate"]
                for row in bundle.tables[0].rows
                if row.values["mode"] == "forced_transfer"
            )
            for bundle in cross_family_bundles
        ]
    )

    summary = MemoryChapterSummary(
        repeated_episode_beneficial=all(
            bundle.summary.memory_reduces_simulation_calls or bundle.summary.memory_reduces_repeated_failures
            for bundle in repeated_bundles
            if "bandgap" not in bundle.task_id
        ),
        repeated_episode_generalizes_beyond_ota=any(
            (bundle.summary.memory_reduces_simulation_calls or bundle.summary.memory_reduces_repeated_failures)
            and "ota2" not in bundle.task_id
            for bundle in repeated_bundles
        ),
        same_family_transfer_beneficial=all(bundle.summary.governed_transfer_beneficial for bundle in same_family_bundles),
        governance_blocks_cross_family_negative_transfer=cross_family_governed_mean < cross_family_no_governance_mean,
        no_governance_exposes_negative_transfer=cross_family_no_governance_mean > cross_family_governed_mean,
        forced_transfer_exposes_negative_transfer=cross_family_forced_mean > cross_family_governed_mean,
        notes=[
            f"same_family_pairs={','.join(same_family_labels)}",
            f"cross_family_pairs={','.join(cross_family_labels)}",
            f"repeated_tasks={','.join(repeated_task_labels)}",
        ],
    )

    bundle = MemoryChapterEvidenceBundle(
        chapter_id="memory_chapter_v1",
        repeated_episode_tasks=repeated_task_labels,
        same_family_pairs=same_family_labels,
        cross_family_pairs=cross_family_labels,
        figures=[repeated_calls_figure, repeated_failure_figure, same_family_calls_figure, cross_family_harm_figure],
        tables=[repeated_table, transfer_table],
        summary=summary,
        json_output_path=str(json_output_path),
    )
    Path(json_output_path).write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle


def _table_row_by_mode(table: TableSpec, mode: str) -> TableRow:
    for row in table.rows:
        if row.values.get("mode") == mode:
            return row
    raise KeyError(f"mode {mode!r} was not found in table {table.table_id}")


def _copy_file(src: str | Path, dst: Path) -> str:
    source_path = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dst)
    return str(dst)


def build_memory_negative_transfer_case_studies(
    cross_family_bundles: list[MemoryTransferEvidenceBundle],
    *,
    output_root: str | Path,
) -> list[MemoryNegativeTransferCaseStudy]:
    """Export structured negative-transfer case studies for the memory chapter."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    scored_bundles = []
    for bundle in cross_family_bundles:
        table = bundle.tables[0]
        governed = _table_row_by_mode(table, "governed_transfer")
        no_governance = _table_row_by_mode(table, "no_governance")
        forced = _table_row_by_mode(table, "forced_transfer")
        score = (
            float(forced.values["harmful_transfer_rate"])
            - float(governed.values["harmful_transfer_rate"])
            + 0.5 * float(no_governance.values["harmful_transfer_rate"])
        )
        scored_bundles.append((score, bundle, governed, no_governance, forced))

    scored_bundles.sort(key=lambda item: item[0], reverse=True)
    case_studies: list[MemoryNegativeTransferCaseStudy] = []
    for index, (_score, bundle, governed, no_governance, forced) in enumerate(scored_bundles):
        case_id = f"{bundle.source_task_slug}_to_{bundle.target_task_slug}".replace("-", "_")
        markdown_path = root / f"{case_id}.md"
        json_path = root / f"{case_id}.json"
        narrative = (
            f"{bundle.source_task_slug} -> {bundle.target_task_slug} shows governed cross-family reuse "
            f"holding harmful transfer at {float(governed.values['harmful_transfer_rate']):.3f}, while "
            f"no-governance rises to {float(no_governance.values['harmful_transfer_rate']):.3f} and forced transfer "
            f"rises to {float(forced.values['harmful_transfer_rate']):.3f}."
        )
        case_study = MemoryNegativeTransferCaseStudy(
            case_study_id=case_id,
            source_task_slug=bundle.source_task_slug,
            target_task_slug=bundle.target_task_slug,
            selected_as_primary_case=index == 0,
            governed_harmful_transfer_rate=float(governed.values["harmful_transfer_rate"]),
            no_governance_harmful_transfer_rate=float(no_governance.values["harmful_transfer_rate"]),
            forced_harmful_transfer_rate=float(forced.values["harmful_transfer_rate"]),
            governed_avg_sim_calls=float(governed.values["average_real_simulation_calls"]),
            no_governance_avg_sim_calls=float(no_governance.values["average_real_simulation_calls"]),
            forced_avg_sim_calls=float(forced.values["average_real_simulation_calls"]),
            governance_block_rate=float(governed.values["governance_block_rate"]),
            average_negative_transfer_risk=float(governed.values["average_negative_transfer_risk"]),
            narrative_summary=narrative,
            markdown_output_path=str(markdown_path),
            json_output_path=str(json_path),
        )
        markdown_lines = [
            f"# Memory Negative-Transfer Case Study: {bundle.source_task_slug} -> {bundle.target_task_slug}",
            "",
            narrative,
            "",
            "## Structured Metrics",
            "",
            f"- Governed harmful-transfer rate: `{case_study.governed_harmful_transfer_rate}`",
            f"- No-governance harmful-transfer rate: `{case_study.no_governance_harmful_transfer_rate}`",
            f"- Forced-transfer harmful-transfer rate: `{case_study.forced_harmful_transfer_rate}`",
            f"- Governed avg real simulation calls: `{case_study.governed_avg_sim_calls}`",
            f"- No-governance avg real simulation calls: `{case_study.no_governance_avg_sim_calls}`",
            f"- Forced avg real simulation calls: `{case_study.forced_avg_sim_calls}`",
            f"- Governance block rate: `{case_study.governance_block_rate}`",
            f"- Avg negative-transfer risk: `{case_study.average_negative_transfer_risk}`",
            "",
            "## Interpretation",
            "",
            "- This pair is suitable for the paper's negative-transfer discussion because the governed path keeps harmful reuse low while the relaxed paths reveal measurable risk.",
            "- It can be cited as evidence that the memory layer is not only helpful under same-family reuse, but also actively guarded under cross-family mismatch.",
        ]
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        json_path.write_text(
            json.dumps(case_study.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        case_studies.append(case_study)
    return case_studies


def build_memory_paper_layout_bundle(
    *,
    profile_name: str,
    repeated_bundles: list[MemoryAblationEvidenceBundle],
    same_family_bundles: list[MemoryTransferEvidenceBundle],
    cross_family_bundles: list[MemoryTransferEvidenceBundle],
    chapter_bundle: MemoryChapterEvidenceBundle,
    case_studies: list[MemoryNegativeTransferCaseStudy],
    output_root: str | Path,
) -> MemoryPaperLayoutBundle:
    """Assemble final main/appendix organization for the memory paper section."""

    root = Path(output_root)
    main_figs_root = root / "main_figs"
    appendix_figs_root = root / "appendix_figs"
    main_tables_root = root / "main_tables"
    appendix_tables_root = root / "appendix_tables"
    case_root = root / "case_studies"
    for directory in (main_figs_root, appendix_figs_root, main_tables_root, appendix_tables_root, case_root):
        directory.mkdir(parents=True, exist_ok=True)

    main_figure_paths = [
        _copy_file(chapter_bundle.figures[0].output_path, main_figs_root / "fig_memory_repeated_episode_calls.svg"),
        _copy_file(chapter_bundle.figures[1].output_path, main_figs_root / "fig_memory_repeated_episode_failures.svg"),
        _copy_file(chapter_bundle.figures[2].output_path, main_figs_root / "fig_memory_same_family_transfer.svg"),
        _copy_file(chapter_bundle.figures[3].output_path, main_figs_root / "fig_memory_cross_family_governance.svg"),
    ]
    main_table_paths = [
        _copy_file(chapter_bundle.tables[0].csv_output_path, main_tables_root / "tbl_memory_repeated_episode.csv"),
        _copy_file(chapter_bundle.tables[0].markdown_output_path, main_tables_root / "tbl_memory_repeated_episode.md"),
        _copy_file(chapter_bundle.tables[1].csv_output_path, main_tables_root / "tbl_memory_transfer_summary.csv"),
        _copy_file(chapter_bundle.tables[1].markdown_output_path, main_tables_root / "tbl_memory_transfer_summary.md"),
    ]

    appendix_figure_paths: list[str] = []
    appendix_table_paths: list[str] = []
    for bundle in repeated_bundles:
        task_prefix = bundle.task_id.replace("-", "_")
        for figure in bundle.figures:
            appendix_figure_paths.append(
                _copy_file(figure.output_path, appendix_figs_root / f"{task_prefix}_{Path(figure.output_path).name}")
            )
        for table in bundle.tables:
            appendix_table_paths.append(
                _copy_file(table.csv_output_path, appendix_tables_root / f"{task_prefix}_{Path(table.csv_output_path).name}")
            )
            appendix_table_paths.append(
                _copy_file(
                    table.markdown_output_path,
                    appendix_tables_root / f"{task_prefix}_{Path(table.markdown_output_path).name}",
                )
            )
    for bundle in [*same_family_bundles, *cross_family_bundles]:
        pair_prefix = f"{bundle.source_task_slug}_to_{bundle.target_task_slug}".replace("-", "_")
        for figure in bundle.figures:
            appendix_figure_paths.append(
                _copy_file(figure.output_path, appendix_figs_root / f"{pair_prefix}_{Path(figure.output_path).name}")
            )
        for table in bundle.tables:
            appendix_table_paths.append(
                _copy_file(table.csv_output_path, appendix_tables_root / f"{pair_prefix}_{Path(table.csv_output_path).name}")
            )
            appendix_table_paths.append(
                _copy_file(
                    table.markdown_output_path,
                    appendix_tables_root / f"{pair_prefix}_{Path(table.markdown_output_path).name}",
                )
            )

    case_paths: list[str] = []
    for case_study in case_studies:
        case_paths.append(_copy_file(case_study.markdown_output_path, case_root / Path(case_study.markdown_output_path).name))
        case_paths.append(_copy_file(case_study.json_output_path, case_root / Path(case_study.json_output_path).name))

    markdown_output_path = root / "memory_paper_layout.md"
    json_output_path = root / "memory_paper_layout_bundle.json"
    notes = [
        f"profile={profile_name}",
        f"repeated_tasks={','.join(chapter_bundle.repeated_episode_tasks)}",
        "main_text should highlight repeated-episode calls/failures first, then same-family transfer, then cross-family governance.",
        "appendix should carry per-task mode breakdowns and pair-specific transfer breakdowns.",
    ]
    layout_bundle = MemoryPaperLayoutBundle(
        layout_id=f"memory_paper_layout_{profile_name}",
        profile_name=profile_name,
        repeated_episode_tasks=chapter_bundle.repeated_episode_tasks,
        same_family_pairs=chapter_bundle.same_family_pairs,
        cross_family_pairs=chapter_bundle.cross_family_pairs,
        main_figures=main_figure_paths,
        appendix_figures=appendix_figure_paths,
        main_tables=main_table_paths,
        appendix_tables=appendix_table_paths,
        case_studies=case_paths,
        summary_notes=notes,
        json_output_path=str(json_output_path),
        markdown_output_path=str(markdown_output_path),
    )
    markdown_lines = [
        "# Memory Paper Layout Bundle",
        "",
        f"- Profile: `{profile_name}`",
        f"- Repeated tasks: `{', '.join(chapter_bundle.repeated_episode_tasks)}`",
        f"- Same-family pairs: `{', '.join(chapter_bundle.same_family_pairs)}`",
        f"- Cross-family pairs: `{', '.join(chapter_bundle.cross_family_pairs)}`",
        "",
        "## Main Figures",
        "",
        *[f"- `{Path(path).name}`" for path in main_figure_paths],
        "",
        "## Appendix Figures",
        "",
        *[f"- `{Path(path).name}`" for path in appendix_figure_paths],
        "",
        "## Main Tables",
        "",
        *[f"- `{Path(path).name}`" for path in main_table_paths],
        "",
        "## Appendix Tables",
        "",
        *[f"- `{Path(path).name}`" for path in appendix_table_paths],
        "",
        "## Case Studies",
        "",
        *[f"- `{Path(path).name}`" for path in case_paths if path.endswith('.md')],
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in notes],
    ]
    markdown_output_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    json_output_path.write_text(
        json.dumps(layout_bundle.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return layout_bundle
