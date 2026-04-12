"""Frozen folded-cascode OTA v1 vertical-slice configuration and task builders."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.design_task import DesignTask
from libs.tasking.compiler import compile_design_task

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "benchmarks" / "folded_cascode.yaml"


class FoldedCascodeVariableDefinition(BaseModel):
    """Frozen physical meaning for one folded-cascode v1 design variable."""

    model_config = ConfigDict(extra="forbid")

    units: str
    role: str
    description: str


class FoldedCascodePathConfig(BaseModel):
    """Canonical folded-cascode v1 path layout."""

    model_config = ConfigDict(extra="forbid")

    netlist_template: str
    quick_truth_testbench: str
    focused_truth_testbench: str
    measurement_contract: str


class FoldedCascodeFidelityThresholds(BaseModel):
    """Frozen folded-cascode v1 escalation thresholds."""

    model_config = ConfigDict(extra="forbid")

    predicted_feasible_probability_min: float
    near_feasible_margin_abs_max: float
    focused_truth_batch_limit: int
    measurement_recheck_enabled: bool = True


class FoldedCascodeFidelityDefaults(BaseModel):
    """Frozen folded-cascode v1 fidelity defaults."""

    model_config = ConfigDict(extra="forbid")

    default_fidelity: str
    promoted_fidelity: str
    quick_truth_analyses: list[str] = Field(default_factory=list)
    focused_truth_analyses: list[str] = Field(default_factory=list)
    escalation_thresholds: FoldedCascodeFidelityThresholds


class FoldedCascodeModelBindingDefaults(BaseModel):
    """Frozen folded-cascode v1 model-binding defaults."""

    model_config = ConfigDict(extra="forbid")

    model_type: str
    truth_level: str
    default_builtin_model: str
    external_model_card_path: str = ""


class FoldedCascodeTaskConfig(BaseModel):
    """Frozen DesignSpec-facing task block for folded-cascode v1."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    process_node: str
    supply_voltage_v: float
    objectives: dict[str, list[str]] = Field(default_factory=dict)
    hard_constraints: dict[str, dict[str, float]] = Field(default_factory=dict)
    environment: dict[str, object] = Field(default_factory=dict)
    testbench_plan: list[str] = Field(default_factory=list)
    design_variables: list[str] = Field(default_factory=list)


class FoldedCascodeDefaults(BaseModel):
    """Frozen folded-cascode v1 system defaults."""

    model_config = ConfigDict(extra="forbid")

    backend_preference: str
    fidelity_policy: FoldedCascodeFidelityDefaults
    model_binding: FoldedCascodeModelBindingDefaults


class FoldedCascodeVersionFreeze(BaseModel):
    """Version-freeze contract for folded-cascode v1."""

    model_config = ConfigDict(extra="forbid")

    frozen_version: str
    paper_track: bool
    change_rule: str


class FoldedCascodeVerticalSliceConfig(BaseModel):
    """Top-level folded-cascode v1 frozen vertical-slice configuration."""

    model_config = ConfigDict(extra="forbid")

    version: str
    family: str
    freeze_policy: FoldedCascodeVersionFreeze
    task: FoldedCascodeTaskConfig
    design_variables: dict[str, FoldedCascodeVariableDefinition] = Field(default_factory=dict)
    measurement_targets: list[str] = Field(default_factory=list)
    paths: FoldedCascodePathConfig
    defaults: FoldedCascodeDefaults


def _path_from_repo(relative_path: str) -> Path:
    return REPO_ROOT / Path(relative_path)


def _normalize_metric_key(metric: str) -> str:
    return metric.strip().lower().replace(" ", "_")


def load_folded_cascode_v1_config() -> FoldedCascodeVerticalSliceConfig:
    """Load the frozen folded-cascode v1 vertical-slice configuration."""

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return FoldedCascodeVerticalSliceConfig.model_validate(payload)


def folded_cascode_v1_netlist_template_path() -> Path:
    """Return the canonical folded-cascode v1 netlist template path."""

    return _path_from_repo(load_folded_cascode_v1_config().paths.netlist_template)


def folded_cascode_v1_testbench_path(fidelity_level: str) -> Path:
    """Return the canonical folded-cascode v1 testbench config path for one fidelity."""

    paths = load_folded_cascode_v1_config().paths
    if fidelity_level == "quick_truth":
        return _path_from_repo(paths.quick_truth_testbench)
    if fidelity_level == "focused_truth":
        return _path_from_repo(paths.focused_truth_testbench)
    raise ValueError(f"unsupported folded-cascode v1 fidelity path lookup: {fidelity_level}")


def folded_cascode_v1_measurement_contract_path() -> Path:
    """Return the canonical folded-cascode v1 measurement-contract path."""

    return _path_from_repo(load_folded_cascode_v1_config().paths.measurement_contract)


def build_folded_cascode_v1_design_spec(*, task_id: str | None = None) -> DesignSpec:
    """Build the frozen folded-cascode v1 DesignSpec."""

    config = load_folded_cascode_v1_config()
    hard_constraints = {
        _normalize_metric_key(metric): MetricRange(**values)
        for metric, values in config.task.hard_constraints.items()
    }
    return DesignSpec(
        task_id=task_id or config.task.task_id,
        circuit_family=config.family,
        process_node=config.task.process_node,
        supply_voltage_v=config.task.supply_voltage_v,
        objectives=Objectives(**config.task.objectives),
        hard_constraints=hard_constraints,
        environment=Environment.model_validate(config.task.environment),
        testbench_plan=list(config.task.testbench_plan),
        design_variables=list(config.task.design_variables),
        missing_information=[],
        notes=[
            f"vertical_slice={config.version}",
            "folded_cascode_v1_is_the_secondary_frozen_paper_path",
            f"default_truth_level={config.defaults.model_binding.truth_level}",
        ],
        compile_confidence=0.98,
    )


def build_folded_cascode_v1_design_task(*, task_id: str | None = None) -> DesignTask:
    """Compile the frozen folded-cascode v1 DesignTask."""

    response = compile_design_task(build_folded_cascode_v1_design_spec(task_id=task_id))
    if response.design_task is None:
        raise ValueError("failed to compile frozen folded-cascode v1 design task")
    return response.design_task

