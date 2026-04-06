"""Unit tests for the simulation and verification layer."""

from __future__ import annotations

import unittest

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.compiler import compile_simulation_bundle
from libs.simulation.service import SimulationService
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


def build_standard_ota_task():
    spec = DesignSpec(
        task_id="sim-standard-ota",
        circuit_family="two_stage_ota",
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={
            "gbw_hz": MetricRange(min=8e7),
            "phase_margin_deg": MetricRange(min=55.0),
            "power_w": MetricRange(max=1.5e-3),
        },
        environment=Environment(temperature_c=[27.0, 85.0], corners=["tt", "ss", "ff"], load_cap_f=2e-12, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran", "noise"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.95,
    )
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


class SimulationLayerTests(unittest.TestCase):
    def _context(self):
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
        candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
        return task, planning_bundle, search_state, candidate_id

    def test_compile_simulation_bundle(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context()
        compiled = compile_simulation_bundle(task, planning_bundle, search_state, candidate_id, fidelity_level="focused_validation")

        self.assertIn(compiled.status, {"compiled", "compiled_with_warnings"})
        self.assertIsNotNone(compiled.simulation_bundle)
        assert compiled.simulation_bundle is not None
        self.assertTrue(compiled.simulation_bundle.analysis_plan.ordered_analyses)
        self.assertTrue(compiled.simulation_bundle.measurement_contract.metric_definitions)

    def test_execute_standard_validation_pipeline(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context()
        execution = SimulationService(task, planning_bundle, search_state).execute(candidate_id, fidelity_level="focused_validation")

        self.assertTrue(execution.verification_result.measurement_report.measured_metrics)
        self.assertTrue(execution.verification_result.constraint_assessment)
        self.assertIn(execution.verification_result.feasibility_status, {"feasible_nominal", "constraint_fail", "near_feasible"})
        self.assertTrue(execution.verification_result.calibration_payload.truth_record.metrics)

    def test_full_robustness_certification_emits_certificate(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context()
        execution = SimulationService(task, planning_bundle, search_state).execute(candidate_id, fidelity_level="full_robustness_certification")

        self.assertGreaterEqual(len(execution.verification_result.robustness_summary.evaluated_conditions), 1)
        self.assertIn(
            execution.verification_result.robustness_summary.certification_status,
            {"partial_robust", "robust_certified", "robustness_failed"},
        )

    def test_backend_consistency_across_ngspice_and_xyce(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context()
        service = SimulationService(task, planning_bundle, search_state)
        ng = service.execute(candidate_id, fidelity_level="focused_validation", backend_preference="ngspice")
        xy = service.execute(candidate_id, fidelity_level="focused_validation", backend_preference="xyce")

        ng_metrics = {metric.metric: metric.value for metric in ng.verification_result.measurement_report.measured_metrics}
        xy_metrics = {metric.metric: metric.value for metric in xy.verification_result.measurement_report.measured_metrics}
        self.assertAlmostEqual(ng_metrics["gbw_hz"], xy_metrics["gbw_hz"], delta=max(1.0, ng_metrics["gbw_hz"] * 0.01))
        self.assertAlmostEqual(ng_metrics["phase_margin_deg"], xy_metrics["phase_margin_deg"], delta=1.0)

    def test_targeted_failure_analysis_surfaces_diagnosis(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context()
        candidate = search_state.candidate_pool_state.candidates[0]
        for index, parameter in enumerate(candidate.world_state_snapshot.parameter_state):
            if parameter.variable_name == "cc":
                candidate.world_state_snapshot.parameter_state[index].value = 1e-13
        execution = SimulationService(task, planning_bundle, search_state).execute(candidate_id, fidelity_level="targeted_failure_analysis")

        self.assertNotEqual(execution.verification_result.failure_attribution.primary_failure_class, "measurement_failure")
        self.assertTrue(execution.verification_result.failure_attribution.evidence)
