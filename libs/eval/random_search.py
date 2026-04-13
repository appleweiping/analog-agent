"""Deterministic random-search helpers for research baselines."""

from __future__ import annotations

import math
from typing import Any

from libs.schema.design_task import DesignTask, DesignVariable
from libs.utils.hashing import stable_hash


def deterministic_unit(seed: str) -> float:
    """Return a deterministic pseudo-random scalar in [0, 1]."""

    return int(stable_hash(seed)[:8], 16) / 0xFFFFFFFF


def _sample_numeric(variable: DesignVariable, ratio: float) -> float | int:
    lower = float(variable.domain.lower)  # type: ignore[arg-type]
    upper = float(variable.domain.upper)  # type: ignore[arg-type]
    if variable.scale == "log" and lower > 0.0 and upper > 0.0:
        value = math.exp(math.log(lower) + (math.log(upper) - math.log(lower)) * ratio)
    else:
        value = lower + (upper - lower) * ratio
    if variable.kind == "integer" or variable.dtype == "int":
        return int(round(value))
    return round(float(value), 12)


def _sample_choice(variable: DesignVariable, ratio: float) -> str | float | int | bool:
    choices = list(variable.domain.choices)
    if not choices:
        raise ValueError(f"variable '{variable.name}' is missing explicit choices")
    index = min(int(ratio * len(choices)), len(choices) - 1)
    value: Any = choices[index]
    if variable.dtype == "bool":
        return bool(value)
    if variable.dtype == "int":
        return int(value)
    if variable.dtype == "float":
        return float(value)
    return value


def sample_variable_value(variable: DesignVariable, *, seed: str) -> str | float | int | bool:
    """Sample one variable value deterministically from its formal domain."""

    ratio = deterministic_unit(seed)
    if variable.kind in {"continuous", "integer"}:
        return _sample_numeric(variable, ratio)
    return _sample_choice(variable, ratio)


def sample_parameter_values(
    task: DesignTask,
    *,
    run_label: str,
    step_index: int,
    sample_index: int,
) -> dict[str, str | float | int | bool]:
    """Sample a deterministic random-search parameter assignment from DesignTask."""

    values: dict[str, str | float | int | bool] = dict(task.initial_state.template_defaults)
    for variable in task.design_space.variables:
        seed = f"{run_label}|{task.task_id}|{step_index}|{sample_index}|{variable.name}"
        values[variable.name] = sample_variable_value(variable, seed=seed)
    return values
