"""Regression and acceptance tests for the fifth layer."""

from __future__ import annotations

import unittest

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.testing import (
    SimulationAcceptanceCase,
    build_acceptance_summary,
    evaluate_case,
)
from libs.tasking.compiler import compile_design_task


def _task(task_id: str, family: str, *, constraints: dict[str, MetricRange], load_cap_f: float) -> object:
    spec = DesignSpec(
        task_id=task_id,
        circuit_family=family,
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints=constraints,
        environment=Environment(temperature_c=[-20.0, 27.0, 85.0], corners=["tt", "ss", "ff"], load_cap_f=load_cap_f, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran", "noise"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.9,
    )
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


class SimulationAcceptanceTests(unittest.TestCase):
    def test_acceptance_suite_covers_fidelity_and_backend_modes(self) -> None:
        cases = [
            SimulationAcceptanceCase(
                name="standard_verification",
                category="standard",
                design_task=_task("accept-standard", "two_stage_ota", constraints={"gbw_hz": MetricRange(min=8e7), "phase_margin_deg": MetricRange(min=55.0)}, load_cap_f=2e-12),
                fidelity_level="focused_validation",
                backend="ngspice",
            ),
            SimulationAcceptanceCase(
                name="robustness_certification",
                category="multi_environment",
                design_task=_task("accept-robust", "two_stage_ota", constraints={"phase_margin_deg": MetricRange(min=50.0)}, load_cap_f=2.5e-12),
                fidelity_level="full_robustness_certification",
                backend="xyce",
            ),
            SimulationAcceptanceCase(
                name="diagnosis_case",
                category="failure_diagnosis",
                design_task=_task("accept-diagnosis", "ldo", constraints={"power_w": MetricRange(max=2e-3), "output_swing_v": MetricRange(min=0.9)}, load_cap_f=4e-12),
                fidelity_level="targeted_failure_analysis",
                backend="spectre_compat",
            ),
        ]

        results = [evaluate_case(case) for case in cases]
        summary = build_acceptance_summary(results)

        self.assertEqual(summary.total_cases, 3)
        self.assertGreaterEqual(summary.schema_validity_rate, 1.0)
        self.assertGreaterEqual(summary.measurement_correctness_rate, 1.0)
        self.assertGreaterEqual(summary.constraint_accuracy_rate, 1.0)
        self.assertGreaterEqual(summary.feedback_utility_rate, 1.0)

    def test_regression_stability_is_deterministic(self) -> None:
        case = SimulationAcceptanceCase(
            name="regression",
            category="regression",
            design_task=_task("accept-regression", "two_stage_ota", constraints={"gbw_hz": MetricRange(min=8e7)}, load_cap_f=2e-12),
            fidelity_level="focused_validation",
            backend="ngspice",
        )
        first = evaluate_case(case)
        second = evaluate_case(case)

        self.assertEqual(first.result, second.result)
        self.assertEqual(first.feedback_usable, second.feedback_usable)
        self.assertEqual(first.diagnosis_correct, second.diagnosis_correct)
