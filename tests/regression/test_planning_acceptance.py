"""Regression and acceptance coverage for the planning layer."""

from __future__ import annotations

import unittest

from libs.planner.testing import PlanningAcceptanceCase, build_acceptance_summary, evaluate_case
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.tasking.compiler import compile_design_task


def compile_task(spec: DesignSpec):
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


class PlanningAcceptanceTests(unittest.TestCase):
    def test_acceptance_summary_tracks_planning_outcomes(self) -> None:
        ota_task = compile_task(
            DesignSpec(
                task_id="planning-accept-ota",
                circuit_family="two_stage_ota",
                process_node="65nm",
                supply_voltage_v=1.2,
                objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
                hard_constraints={"gbw_hz": MetricRange(min=1e8), "phase_margin_deg": MetricRange(min=60.0)},
                environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
                testbench_plan=["op", "ac", "tran"],
                design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                missing_information=[],
                notes=[],
                compile_confidence=0.92,
            )
        )
        folded_task = compile_task(
            DesignSpec(
                task_id="planning-accept-folded",
                circuit_family="folded_cascode_ota",
                process_node="65nm",
                supply_voltage_v=1.2,
                objectives=Objectives(maximize=["dc_gain_db"], minimize=["power_w"]),
                hard_constraints={"phase_margin_deg": MetricRange(min=60.0)},
                environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
                testbench_plan=["op", "ac"],
                design_variables=["w_in", "l_in", "w_cas", "l_cas", "ibias", "cc"],
                missing_information=[],
                notes=[],
                compile_confidence=0.9,
            )
        )
        ldo_task = compile_task(
            DesignSpec(
                task_id="planning-accept-ldo",
                circuit_family="ldo",
                process_node="180nm",
                supply_voltage_v=1.8,
                objectives=Objectives(minimize=["power_w"]),
                hard_constraints={"power_w": MetricRange(max=1.5e-3)},
                environment=Environment(temperature_c=[27.0, 125.0], corners=["tt", "ss"], supply_voltage_v=1.8),
                testbench_plan=["op", "ac"],
                design_variables=["w_pass", "l_pass", "w_err", "l_err", "ibias", "c_comp"],
                missing_information=[],
                notes=[],
                compile_confidence=0.89,
            )
        )
        cases = [
            PlanningAcceptanceCase(name="ota-standard", category="standard", design_task=ota_task, max_rounds=2),
            PlanningAcceptanceCase(name="folded-budget", category="budget", design_task=folded_task, max_rounds=2),
            PlanningAcceptanceCase(name="ldo-feedback", category="feedback", design_task=ldo_task, max_rounds=2, inject_feedback=True),
        ]

        results = [evaluate_case(case) for case in cases]
        summary = build_acceptance_summary(results)

        self.assertEqual(summary.total_cases, 3)
        self.assertGreaterEqual(summary.schema_validity_rate, 1.0)
        self.assertGreaterEqual(summary.decision_quality_rate, 1.0)
        self.assertGreaterEqual(summary.budget_efficiency_rate, 1.0)

    def test_budget_limited_planning_stays_within_explicit_budget(self) -> None:
        task = compile_task(
            DesignSpec(
                task_id="planning-accept-budget",
                circuit_family="two_stage_ota",
                process_node="65nm",
                supply_voltage_v=1.2,
                objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
                hard_constraints={"phase_margin_deg": MetricRange(min=60.0)},
                environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
                testbench_plan=["op", "ac"],
                design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                missing_information=[],
                notes=[],
                compile_confidence=0.91,
            )
        )
        result = evaluate_case(PlanningAcceptanceCase(name="budget-limited", category="budget", design_task=task, max_rounds=2))

        self.assertTrue(result.budget_efficient)

    def test_regression_stability_keeps_traceability_intact(self) -> None:
        task = compile_task(
            DesignSpec(
                task_id="planning-accept-regression",
                circuit_family="bandgap",
                process_node="180nm",
                supply_voltage_v=1.8,
                objectives=Objectives(minimize=["power_w"]),
                hard_constraints={"power_w": MetricRange(max=2e-4)},
                environment=Environment(temperature_c=[-40.0, 27.0, 125.0], corners=["tt"], supply_voltage_v=1.8),
                testbench_plan=["op"],
                design_variables=["area_ratio", "r1", "r2", "w_core", "l_core", "ibias"],
                missing_information=[],
                notes=[],
                compile_confidence=0.88,
            )
        )
        result = evaluate_case(PlanningAcceptanceCase(name="regression", category="stability", design_task=task, max_rounds=1))

        self.assertTrue(result.trace_complete)

