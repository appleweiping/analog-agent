"""Focused tests for physical-claim-scope metadata and validation."""

from __future__ import annotations

import unittest

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.compiler import compile_simulation_bundle
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


def _context(*, single_point: bool = False):
    environment = (
        Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2)
        if single_point
        else Environment(temperature_c=[27.0, 85.0], corners=["tt", "ff"], load_cap_f=2e-12, supply_voltage_v=1.2)
    )
    spec = DesignSpec(
        task_id="unit-claim-scope-ota",
        circuit_family="two_stage_ota",
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={"gbw_hz": MetricRange(min=8e7), "phase_margin_deg": MetricRange(min=55.0)},
        environment=environment,
        testbench_plan=["op", "ac", "tran"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.92,
    )
    task = compile_design_task(spec).design_task
    assert task is not None
    world_model_bundle = compile_world_model_bundle(task).world_model_bundle
    assert world_model_bundle is not None
    planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
    assert planning_bundle is not None
    search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
    candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
    return task, planning_bundle, search_state, candidate_id


class SimulationClaimScopeTests(unittest.TestCase):
    def test_single_point_nominal_scope_is_labeled_and_warned(self) -> None:
        task, planning_bundle, search_state, candidate_id = _context(single_point=True)
        compiled = compile_simulation_bundle(
            task,
            planning_bundle,
            search_state,
            candidate_id,
            backend_preference="xyce",
        )

        self.assertIsNotNone(compiled.simulation_bundle)
        bundle = compiled.simulation_bundle
        assert bundle is not None
        self.assertEqual(bundle.metadata.physical_claim_scope.nominal_profile, "single_point_nominal")
        self.assertEqual(bundle.metadata.physical_claim_scope.truth_claim_tier, "demonstrator_only")
        self.assertEqual(compiled.report.acceptance_summary["claim_profile"], "single_point_nominal")
        self.assertTrue(
            any(
                warning.path == "metadata.physical_claim_scope"
                and "single-point nominal" in warning.message
                for warning in bundle.validation_status.warnings
            )
        )

    def test_configured_truth_candidate_scope_is_structured(self) -> None:
        task, planning_bundle, search_state, candidate_id = _context(single_point=False)
        compiled = compile_simulation_bundle(
            task,
            planning_bundle,
            search_state,
            candidate_id,
            backend_preference="ngspice",
            model_binding_overrides={"configured_truth_mode": "external_pdk_root", "pdk_root": "mock://pdks/sky130A"},
        )

        self.assertIsNotNone(compiled.simulation_bundle)
        bundle = compiled.simulation_bundle
        assert bundle is not None
        self.assertEqual(bundle.metadata.physical_claim_scope.nominal_profile, "multi_condition_nominal")
        self.assertEqual(bundle.metadata.physical_claim_scope.truth_claim_tier, "configured_candidate")
        self.assertEqual(compiled.report.acceptance_summary["truth_claim_tier"], "configured_candidate")


if __name__ == "__main__":
    unittest.main()
