"""Shared benchmark protocol contracts for frozen runnable slices and baseline audits."""

from __future__ import annotations

from libs.schema.experiment import ExperimentBudget

BASELINE_BENCHMARK_MODES = [
    "full_simulation_baseline",
    "top_k_baseline",
    "random_search_baseline",
    "bayesopt_baseline",
    "cmaes_baseline",
    "rl_baseline",
    "no_world_model_baseline",
    "full_system",
]

METHODOLOGY_BENCHMARK_MODES = [
    "full_system",
    "no_world_model",
    "no_calibration",
    "no_fidelity_escalation",
]

PLANNER_ABLATION_BENCHMARK_MODES = [
    "full_system",
    "top_k_baseline",
    "no_fidelity_escalation",
    "no_phase_updates",
    "no_calibration_replanning",
    "no_rollout_planning",
]

DEFAULT_BENCHMARK_STEPS = 3
DEFAULT_BENCHMARK_REPEAT_RUNS = 5
DEFAULT_BENCHMARK_MAX_SIMULATIONS = 6
DEFAULT_BENCHMARK_MAX_CANDIDATES_PER_STEP = 3

BASELINE_MODE_NARRATIVES = {
    "full_simulation_baseline": "Upper-cost reference that evaluates every proposed candidate with real simulation rather than planner-side selectivity.",
    "top_k_baseline": "Simple rank-then-simulate heuristic that removes the richer planner control stack.",
    "random_search_baseline": "Deterministic random-search style baseline using the shared candidate and verification interfaces.",
    "bayesopt_baseline": "Lightweight internal BayesOpt-style baseline sharing the same benchmark budget and truth loop.",
    "cmaes_baseline": "Lightweight internal CMA-ES-style baseline sharing the same benchmark budget and truth loop.",
    "rl_baseline": "Lightweight internal RL-style policy baseline sharing the same benchmark budget and truth loop.",
    "no_world_model_baseline": "Ablative baseline that preserves the closed-loop search path while removing world-model guidance.",
    "full_system": "The complete analog-agent planner with world-model guidance, calibration, and selective truth escalation.",
}


def benchmark_modes_for_profile(profile: str) -> list[str]:
    if profile == "methodology":
        return list(METHODOLOGY_BENCHMARK_MODES)
    if profile == "planner_ablation":
        return list(PLANNER_ABLATION_BENCHMARK_MODES)
    return list(BASELINE_BENCHMARK_MODES)


def default_benchmark_budget() -> ExperimentBudget:
    return ExperimentBudget(
        max_simulations=DEFAULT_BENCHMARK_MAX_SIMULATIONS,
        max_candidates_per_step=DEFAULT_BENCHMARK_MAX_CANDIDATES_PER_STEP,
    )


def benchmark_protocol_contract() -> dict[str, object]:
    return {
        "default_steps": DEFAULT_BENCHMARK_STEPS,
        "default_repeat_runs": DEFAULT_BENCHMARK_REPEAT_RUNS,
        "default_budget": default_benchmark_budget().model_dump(),
        "baseline_modes": list(BASELINE_BENCHMARK_MODES),
        "methodology_modes": list(METHODOLOGY_BENCHMARK_MODES),
        "planner_ablation_modes": list(PLANNER_ABLATION_BENCHMARK_MODES),
        "baseline_mode_narratives": dict(BASELINE_MODE_NARRATIVES),
    }
