"""Unit tests for the task formalization layer."""

from __future__ import annotations

import unittest

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.tasking.compiler import compile_design_task
from libs.tasking.testing import fake_planner_consume, fake_simulator_adapter, fake_world_model_adapter


def build_standard_ota_spec() -> DesignSpec:
    return DesignSpec(
        task_id="spec-standard-ota",
        circuit_family="two_stage_ota",
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={
            "gbw_hz": MetricRange(min=1e8),
            "phase_margin_deg": MetricRange(min=60.0),
            "power_w": MetricRange(max=1e-3),
        },
        environment=Environment(
            temperature_c=[27.0],
            corners=["tt"],
            load_cap_f=2e-12,
            supply_voltage_v=1.2,
        ),
        testbench_plan=["op", "ac"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.92,
    )


class TaskFormalizationLayerTests(unittest.TestCase):
    def test_standard_case_compiles_into_solver_ready_design_task(self) -> None:
        response = compile_design_task(build_standard_ota_spec())

        self.assertEqual(response.status, "compiled")
        self.assertIsNotNone(response.design_task)
        task = response.design_task
        assert task is not None

        self.assertEqual(task.task_type, "sizing")
        self.assertEqual(task.topology.topology_mode, "fixed")
        self.assertEqual(task.topology.template_id, "ota2_miller_basic_v1")
        self.assertEqual(task.objective.objective_mode, "multi_objective")
        self.assertTrue(task.validation_status.is_valid)
        self.assertGreaterEqual(len(task.design_space.variables), 6)
        self.assertEqual(fake_planner_consume(task)["task_type"], "sizing")
        self.assertTrue(fake_simulator_adapter(task))
        self.assertIn("feature_keys", fake_world_model_adapter(task))

    def test_underspecified_case_marks_unresolved_dependencies(self) -> None:
        spec = build_standard_ota_spec().model_copy(
            update={
                "task_id": "spec-underspecified-ota",
                "process_node": None,
                "supply_voltage_v": None,
                "environment": Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=None),
                "missing_information": ["process_node", "load_cap_f"],
                "compile_confidence": 0.55,
            }
        )

        response = compile_design_task(spec)

        self.assertEqual(response.status, "compiled_with_warnings")
        self.assertIsNotNone(response.design_task)
        task = response.design_task
        assert task is not None
        self.assertFalse(task.validation_status.is_valid)
        self.assertIn("process_node", task.validation_status.unresolved_dependencies)
        self.assertIn("load_cap_f", task.validation_status.unresolved_dependencies)
        self.assertIn("supply_voltage_v", task.validation_status.unresolved_dependencies)

    def test_ambiguous_case_becomes_multi_objective_without_invented_constraints(self) -> None:
        spec = DesignSpec(
            task_id="spec-ambiguous",
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
            compile_confidence=0.48,
        )

        response = compile_design_task(spec)

        self.assertEqual(response.status, "compiled_with_warnings")
        self.assertIsNotNone(response.design_task)
        task = response.design_task
        assert task is not None
        self.assertEqual(task.task_type, "topology_sizing")
        self.assertEqual(task.topology.topology_mode, "search_space")
        self.assertEqual(task.objective.objective_mode, "multi_objective")
        self.assertEqual(task.constraints.hard_constraints, [])
        self.assertIn("topology_template_choice", [variable.name for variable in task.design_space.variables])

    def test_conflicting_case_is_reported_as_invalid(self) -> None:
        spec = build_standard_ota_spec().model_copy(
            update={
                "task_id": "spec-conflict",
                "hard_constraints": {
                    "power_w": MetricRange(min=5e-3),
                    "phase_margin_deg": MetricRange(min=60.0),
                },
                "objectives": Objectives(minimize=["power_w"]),
                "testbench_plan": ["op", "ac"],
            }
        )

        response = compile_design_task(spec)

        self.assertEqual(response.status, "invalid")
        self.assertIsNone(response.design_task)
        self.assertIn("constraint_direction_error", [issue.code for issue in response.report.validation_errors])

    def test_topology_boundary_case_exposes_structural_variable(self) -> None:
        response = compile_design_task(build_standard_ota_spec(), task_type_hint="topology_sizing")

        self.assertEqual(response.status, "compiled")
        self.assertIsNotNone(response.design_task)
        task = response.design_task
        assert task is not None
        self.assertEqual(task.task_type, "topology_sizing")
        self.assertEqual(task.topology.topology_mode, "template_family")
        structural_variables = [variable for variable in task.design_space.variables if variable.role == "structural_template_choice"]
        self.assertEqual(len(structural_variables), 1)

    def test_cost_aware_case_builds_expensive_evaluation_plan(self) -> None:
        spec = DesignSpec(
            task_id="spec-bandgap",
            circuit_family="bandgap",
            process_node="180nm",
            supply_voltage_v=1.8,
            objectives=Objectives(minimize=["power_w"]),
            hard_constraints={"power_w": MetricRange(max=200e-6)},
            environment=Environment(
                temperature_c=[-40.0, 27.0, 125.0],
                corners=["tt", "ss", "ff"],
                supply_voltage_v=1.8,
            ),
            testbench_plan=["op"],
            design_variables=["area_ratio", "r1", "r2", "w_core", "l_core", "ibias"],
            missing_information=[],
            notes=["must remain stable across temperature"],
            compile_confidence=0.88,
        )

        response = compile_design_task(spec)

        self.assertEqual(response.status, "compiled")
        self.assertIsNotNone(response.design_task)
        task = response.design_task
        assert task is not None
        self.assertEqual(task.evaluation_plan.simulation_budget_class, "expensive")
        self.assertEqual(task.evaluation_plan.fidelity_policy, "staged_fidelity")
        self.assertIn("pvt_sweep", [analysis.analysis_type for analysis in task.evaluation_plan.analyses])

    def test_regression_stability_keeps_canonical_problem_shape(self) -> None:
        first = compile_design_task(build_standard_ota_spec())
        second = compile_design_task(build_standard_ota_spec())
        assert first.design_task is not None
        assert second.design_task is not None

        self.assertEqual(first.design_task.task_id, second.design_task.task_id)
        self.assertEqual(
            [variable.name for variable in first.design_task.design_space.variables],
            [variable.name for variable in second.design_task.design_space.variables],
        )
        self.assertEqual(
            [constraint.name for constraint in first.design_task.constraints.hard_constraints],
            [constraint.name for constraint in second.design_task.constraints.hard_constraints],
        )
        self.assertEqual(
            [analysis.analysis_type for analysis in first.design_task.evaluation_plan.analyses],
            [analysis.analysis_type for analysis in second.design_task.evaluation_plan.analyses],
        )
