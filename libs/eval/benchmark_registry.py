"""Formal registry helpers for multi-task benchmark design."""

from __future__ import annotations

import json
from pathlib import Path

from libs.schema.benchmark import BenchmarkSuiteDefinition, BenchmarkTaskDefinition
from libs.vertical_slices.folded_cascode_spec import load_folded_cascode_v1_config
from libs.vertical_slices.ota2_spec import load_ota2_v1_config

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_ROOT = REPO_ROOT / "configs" / "benchmarks"
SUITE_CONFIG_PATH = BENCHMARK_CONFIG_ROOT / "multi_task_suite_v1.yaml"


def _load_structured_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_suite_definition() -> BenchmarkSuiteDefinition:
    """Load the formal multi-task benchmark-suite definition."""

    return BenchmarkSuiteDefinition.model_validate(_load_structured_payload(SUITE_CONFIG_PATH))


def _load_ota2_as_benchmark_task() -> BenchmarkTaskDefinition:
    config = load_ota2_v1_config()
    return BenchmarkTaskDefinition(
        benchmark_id=config.version,
        version=config.version,
        family=config.family,
        category="amplifier",
        benchmark_role="paper_primary",
        execution_readiness="frozen_runnable",
        vertical_slice_bound=True,
        physical_validity_target=config.defaults.model_binding.truth_level,
        freeze_policy=config.freeze_policy.model_dump(),
        task=config.task.model_dump(),
        measurement_contract={
            "primary_metrics": list(config.measurement_targets),
            "auxiliary_metrics": ["slew_rate_v_per_us"],
            "reporting_metrics": list(config.measurement_targets),
        },
        execution_defaults={
            "backend_preference": config.defaults.backend_preference,
            "default_fidelity": config.defaults.fidelity_policy.default_fidelity,
            "promoted_fidelity": config.defaults.fidelity_policy.promoted_fidelity,
            "truth_level": config.defaults.model_binding.truth_level,
            "model_type": config.defaults.model_binding.model_type,
        },
        intended_templates={
            "netlist_template": config.paths.netlist_template,
            "quick_truth_testbench": config.paths.quick_truth_testbench,
            "focused_truth_testbench": config.paths.focused_truth_testbench,
            "measurement_contract": config.paths.measurement_contract,
        },
        notes=[
            "canonical_paper_vertical_slice",
            "frozen_runnable_path",
            "submission_facing_benchmark",
        ],
    )


def _load_folded_cascode_as_benchmark_task() -> BenchmarkTaskDefinition:
    config = load_folded_cascode_v1_config()
    return BenchmarkTaskDefinition(
        benchmark_id=config.version,
        version=config.version,
        family=config.family,
        category="amplifier",
        benchmark_role="paper_secondary",
        execution_readiness="frozen_runnable",
        vertical_slice_bound=True,
        physical_validity_target=config.defaults.model_binding.truth_level,
        freeze_policy=config.freeze_policy.model_dump(),
        task=config.task.model_dump(),
        measurement_contract={
            "primary_metrics": list(config.measurement_targets),
            "auxiliary_metrics": ["slew_rate_v_per_us"],
            "reporting_metrics": list(config.measurement_targets),
        },
        execution_defaults={
            "backend_preference": config.defaults.backend_preference,
            "default_fidelity": config.defaults.fidelity_policy.default_fidelity,
            "promoted_fidelity": config.defaults.fidelity_policy.promoted_fidelity,
            "truth_level": config.defaults.model_binding.truth_level,
            "model_type": config.defaults.model_binding.model_type,
        },
        intended_templates={
            "netlist_template": config.paths.netlist_template,
            "quick_truth_testbench": config.paths.quick_truth_testbench,
            "focused_truth_testbench": config.paths.focused_truth_testbench,
            "measurement_contract": config.paths.measurement_contract,
        },
        notes=[
            "secondary_amplifier_generalization_task",
            "folded_cascode_v1_is_now_runnable",
            "shares_agentic_contract_with_ota_v1",
        ],
    )


def load_benchmark_task_definition(benchmark_name: str) -> BenchmarkTaskDefinition:
    """Load one benchmark-task definition by its public benchmark name."""

    if benchmark_name in {"ota2", "ota2_v1"}:
        return _load_ota2_as_benchmark_task()
    if benchmark_name in {"folded_cascode", "folded_cascode_v1"}:
        return _load_folded_cascode_as_benchmark_task()
    config_path = BENCHMARK_CONFIG_ROOT / f"{benchmark_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"unknown benchmark config: {benchmark_name}")
    return BenchmarkTaskDefinition.model_validate(_load_structured_payload(config_path))


def list_benchmark_definitions() -> list[BenchmarkTaskDefinition]:
    """Return all benchmark definitions referenced by the suite."""

    suite = load_benchmark_suite_definition()
    return [load_benchmark_task_definition(benchmark_id.replace("_v1", "")) if benchmark_id == "ota2_v1" else load_benchmark_task_definition(benchmark_id) for benchmark_id in suite.benchmark_ids]


def runnable_benchmark_ids() -> list[str]:
    """Return benchmark identifiers that are currently runnable in the main path."""

    return [
        benchmark.benchmark_id
        for benchmark in list_benchmark_definitions()
        if benchmark.execution_readiness == "frozen_runnable"
    ]
