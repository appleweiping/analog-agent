"""Experiment and baseline comparison routes."""

from __future__ import annotations

from fastapi import APIRouter

from libs.eval.experiment_runner import run_experiment, run_experiment_suite
from libs.schema.experiment import ExperimentResult, ExperimentRunRequest, ExperimentSuiteRequest, ExperimentSuiteResult

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/run", response_model=ExperimentResult)
def run_single_experiment(request: ExperimentRunRequest) -> ExperimentResult:
    """Run one structured experiment mode."""

    return run_experiment(
        request.design_task,
        request.mode,
        request.budget,
        request.steps,
        run_index=request.run_index,
        fidelity_level=request.fidelity_level,
        backend_preference=request.backend_preference,
    )


@router.post("/run-suite", response_model=ExperimentSuiteResult)
def run_suite(request: ExperimentSuiteRequest) -> ExperimentSuiteResult:
    """Run repeated experiments across multiple execution modes."""

    return run_experiment_suite(
        request.design_task,
        request.modes,
        request.budget,
        request.steps,
        repeat_runs=request.repeat_runs,
        fidelity_level=request.fidelity_level,
        backend_preference=request.backend_preference,
    )
