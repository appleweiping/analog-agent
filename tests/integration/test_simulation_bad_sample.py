"""Integration tests for bad-sample handling on the real ngspice path."""

from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.service import SimulationService
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class SimulationBadSampleIntegrationTests(unittest.TestCase):
    def _context(self):
        spec = DesignSpec(
            task_id="integration-bad-sample-ota",
            circuit_family="two_stage_ota",
            process_node="65nm",
            supply_voltage_v=1.2,
            objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
            hard_constraints={
                "gbw_hz": MetricRange(min=8e7),
                "phase_margin_deg": MetricRange(min=55.0),
                "power_w": MetricRange(max=1.5e-3),
            },
            environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
            testbench_plan=["op", "ac"],
            design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
            missing_information=[],
            notes=[],
            compile_confidence=0.95,
        )
        task = compile_design_task(spec).design_task
        assert task is not None
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
        candidate = search_state.candidate_pool_state.candidates[0]
        for parameter in candidate.world_state_snapshot.parameter_state:
            if parameter.variable_name == "cc":
                parameter.value = 1e-15
            elif parameter.variable_name == "ibias":
                parameter.value = 0.02
            elif parameter.variable_name in {"w_in", "w_tail"}:
                parameter.value = 2e-4
        return task, planning_bundle, search_state, candidate.candidate_id

    def test_real_ngspice_bad_sample_is_structured_not_misjudged(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context()
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="quick_truth",
            backend_preference="ngspice",
        )

        self.assertEqual(execution.simulation_bundle.backend_binding.invocation_mode, "native")
        self.assertEqual(execution.verification_result.feasibility_status, "measurement_invalid")
        self.assertEqual(execution.verification_result.failure_attribution.primary_failure_class, "measurement_failure")
        self.assertEqual(execution.verification_result.planner_feedback.feedback_basis, "measurement_failure")
        self.assertEqual(execution.verification_result.planner_feedback.lifecycle_update, "needs_more_simulation")
        statuses = {item.metric: (item.status.status, item.failure_reason.code) for item in execution.verification_result.measurement_report.measurement_results}
        self.assertEqual(statuses["gbw_hz"], ("indeterminate", "curve_exists_but_no_crossing"))
        self.assertEqual(statuses["phase_margin_deg"], ("indeterminate", "curve_exists_but_no_crossing"))
        self.assertEqual(statuses["power_w"][0], "measured")
