"""Regression-style acceptance coverage for the task formalization layer."""

from __future__ import annotations

import unittest

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.tasking.testing import TaskAcceptanceCase, build_acceptance_summary, evaluate_case


class TaskFormalizationAcceptanceTests(unittest.TestCase):
    def test_acceptance_summary_tracks_second_layer_outcomes(self) -> None:
        cases = [
            TaskAcceptanceCase(
                name="standard-ota",
                category="standard",
                design_spec=DesignSpec(
                    task_id="accept-standard-ota",
                    circuit_family="two_stage_ota",
                    process_node="65nm",
                    supply_voltage_v=1.2,
                    objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
                    hard_constraints={
                        "gbw_hz": MetricRange(min=1e8),
                        "phase_margin_deg": MetricRange(min=60.0),
                        "power_w": MetricRange(max=1e-3),
                    },
                    environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
                    testbench_plan=["op", "ac"],
                    design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                    missing_information=[],
                    notes=[],
                    compile_confidence=0.92,
                ),
            ),
            TaskAcceptanceCase(
                name="underspecified-ota",
                category="underspecified",
                design_spec=DesignSpec(
                    task_id="accept-underspecified-ota",
                    circuit_family="two_stage_ota",
                    process_node=None,
                    supply_voltage_v=None,
                    objectives=Objectives(maximize=["gbw_hz"]),
                    hard_constraints={"gbw_hz": MetricRange(target=1e8)},
                    environment=Environment(),
                    testbench_plan=["op"],
                    design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                    missing_information=["process_node", "load_cap_f"],
                    notes=[],
                    compile_confidence=0.5,
                ),
            ),
            TaskAcceptanceCase(
                name="ambiguous-amp",
                category="ambiguous",
                design_spec=DesignSpec(
                    task_id="accept-ambiguous",
                    circuit_family="unknown",
                    process_node=None,
                    supply_voltage_v=None,
                    objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
                    hard_constraints={},
                    environment=Environment(),
                    testbench_plan=["op", "ac"],
                    design_variables=[],
                    missing_information=["circuit_family", "process_node", "load_cap_f"],
                    notes=["high-speed low-power amplifier"],
                    compile_confidence=0.44,
                ),
            ),
            TaskAcceptanceCase(
                name="conflict-power",
                category="adversarial",
                design_spec=DesignSpec(
                    task_id="accept-conflict",
                    circuit_family="ldo",
                    process_node="180nm",
                    supply_voltage_v=1.8,
                    objectives=Objectives(minimize=["power_w"]),
                    hard_constraints={"power_w": MetricRange(min=1e-3)},
                    environment=Environment(temperature_c=[27.0], corners=["tt"], output_load_ohm=100.0, supply_voltage_v=1.8),
                    testbench_plan=["op"],
                    design_variables=["w_pass", "l_pass", "w_err", "l_err", "ibias", "c_comp"],
                    missing_information=[],
                    notes=[],
                    compile_confidence=0.9,
                ),
            ),
        ]

        results = [evaluate_case(case) for case in cases]
        summary = build_acceptance_summary(results)

        self.assertEqual(summary.total_cases, 4)
        self.assertEqual(summary.passed_cases, 3)
        self.assertGreaterEqual(summary.schema_validity_rate, 0.75)
        self.assertGreaterEqual(summary.problem_completeness_rate, 0.75)
        self.assertGreaterEqual(summary.unresolved_dependency_recall, 1.0)
        self.assertIn("constraint_direction_error", summary.error_type_distribution)
