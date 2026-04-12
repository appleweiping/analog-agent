"""Frozen OTA v1 vertical-slice configuration and task builders."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.design_task import DesignTask
from libs.tasking.compiler import compile_design_task

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "benchmarks" / "ota2.yaml"


class Ota2VariableDefinition(BaseModel):
    """Frozen physical meaning for one OTA v1 design variable."""

    model_config = ConfigDict(extra="forbid")

    units: str
    role: str
    description: str


class Ota2PathConfig(BaseModel):
    """Canonical OTA v1 path layout."""

    model_config = ConfigDict(extra="forbid")

    netlist_template: str
    quick_truth_testbench: str
    focused_truth_testbench: str
    measurement_contract: str


class Ota2FidelityThresholds(BaseModel):
    """Frozen OTA v1 escalation thresholds."""

    model_config = ConfigDict(extra="forbid")

    predicted_feasible_probability_min: float
    near_feasible_margin_abs_max: float
    focused_truth_batch_limit: int
    measurement_recheck_enabled: bool = True


class Ota2FidelityDefaults(BaseModel):
    """Frozen OTA v1 fidelity defaults."""

    model_config = ConfigDict(extra="forbid")

    default_fidelity: str
    promoted_fidelity: str
    quick_truth_analyses: list[str] = Field(default_factory=list)
    focused_truth_analyses: list[str] = Field(default_factory=list)
    escalation_thresholds: Ota2FidelityThresholds


class Ota2ModelBindingDefaults(BaseModel):
    """Frozen OTA v1 model-binding defaults."""

    model_config = ConfigDict(extra="forbid")

    model_type: str
    truth_level: str
    default_builtin_model: str
    external_model_card_path: str = ""


class Ota2TaskConfig(BaseModel):
    """Frozen DesignSpec-facing task block for OTA v1."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    process_node: str
    supply_voltage_v: float
    objectives: dict[str, list[str]] = Field(default_factory=dict)
    hard_constraints: dict[str, dict[str, float]] = Field(default_factory=dict)
    environment: dict[str, object] = Field(default_factory=dict)
    testbench_plan: list[str] = Field(default_factory=list)
    design_variables: list[str] = Field(default_factory=list)


class Ota2Defaults(BaseModel):
    """Frozen OTA v1 system defaults."""

    model_config = ConfigDict(extra="forbid")

    backend_preference: str
    fidelity_policy: Ota2FidelityDefaults
    model_binding: Ota2ModelBindingDefaults


class Ota2VersionFreeze(BaseModel):
    """Version-freeze contract for OTA v1."""

    model_config = ConfigDict(extra="forbid")

    frozen_version: str
    paper_track: bool
    change_rule: str


class Ota2VerticalSliceConfig(BaseModel):
    """Top-level OTA v1 frozen vertical-slice configuration."""

    model_config = ConfigDict(extra="forbid")

    version: str
    family: str
    freeze_policy: Ota2VersionFreeze
    task: Ota2TaskConfig
    design_variables: dict[str, Ota2VariableDefinition] = Field(default_factory=dict)
    measurement_targets: list[str] = Field(default_factory=list)
    paths: Ota2PathConfig
    defaults: Ota2Defaults


def _path_from_repo(relative_path: str) -> Path:
    return REPO_ROOT / Path(relative_path)


def _normalize_metric_key(metric: str) -> str:
    return metric.strip().lower().replace(" ", "_")


def load_ota2_v1_config() -> Ota2VerticalSliceConfig:
    """Load the frozen OTA v1 vertical-slice configuration."""

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return Ota2VerticalSliceConfig.model_validate(payload)


def ota2_v1_netlist_template_path() -> Path:
    """Return the canonical OTA v1 netlist template path."""

    return _path_from_repo(load_ota2_v1_config().paths.netlist_template)


def ota2_v1_testbench_path(fidelity_level: str) -> Path:
    """Return the canonical OTA v1 testbench config path for one fidelity."""

    paths = load_ota2_v1_config().paths
    if fidelity_level == "quick_truth":
        return _path_from_repo(paths.quick_truth_testbench)
    if fidelity_level == "focused_truth":
        return _path_from_repo(paths.focused_truth_testbench)
    raise ValueError(f"unsupported ota2 v1 fidelity path lookup: {fidelity_level}")


def ota2_v1_measurement_contract_path() -> Path:
    """Return the canonical OTA v1 measurement-contract path."""

    return _path_from_repo(load_ota2_v1_config().paths.measurement_contract)


def build_ota2_v1_design_spec(*, task_id: str | None = None) -> DesignSpec:
    """Build the frozen OTA v1 DesignSpec."""

    config = load_ota2_v1_config()
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
            "ota_v1_is_the_frozen_paper_path",
            f"default_truth_level={config.defaults.model_binding.truth_level}",
        ],
        compile_confidence=0.99,
    )


def build_ota2_v1_design_task(*, task_id: str | None = None) -> DesignTask:
    """Compile the frozen OTA v1 DesignTask."""

    response = compile_design_task(build_ota2_v1_design_spec(task_id=task_id))
    if response.design_task is None:
        raise ValueError("failed to compile frozen ota2 v1 design task")
    return response.design_task
