"""Unit tests for Day-7 model binding and physical-validity validation."""

from __future__ import annotations

import unittest

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.compiler import compile_simulation_bundle
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


def _context():
    spec = DesignSpec(
        task_id="unit-model-binding-ota",
        circuit_family="two_stage_ota",
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={"gbw_hz": MetricRange(min=8e7), "phase_margin_deg": MetricRange(min=55.0)},
        environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.9,
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


class ModelBindingValidationTests(unittest.TestCase):
    def test_missing_external_model_binding_marks_bundle_as_invalid_strength(self) -> None:
        task, planning_bundle, search_state, candidate_id = _context()
        compiled = compile_simulation_bundle(
            task,
            planning_bundle,
            search_state,
            candidate_id,
            backend_preference="ngspice",
            model_binding_overrides={"model_type": "external"},
        )

        self.assertIsNone(compiled.simulation_bundle)
        self.assertEqual(compiled.status, "invalid")
        self.assertTrue(
            any(
                issue.path == "model_binding.model_source"
                and "configured_truth was requested" in issue.message
                for issue in compiled.report.validation_errors
            )
        )
        self.assertEqual(compiled.report.acceptance_summary["validation_state"], "invalid")

    def test_configured_truth_pdk_candidate_path_is_structurally_compiled(self) -> None:
        task, planning_bundle, search_state, candidate_id = _context()
        compiled = compile_simulation_bundle(
            task,
            planning_bundle,
            search_state,
            candidate_id,
            backend_preference="ngspice",
            model_binding_overrides={
                "configured_truth_mode": "external_pdk_root",
                "pdk_root": "mock://pdks/sky130A",
            },
        )

        self.assertIsNotNone(compiled.simulation_bundle)
        bundle = compiled.simulation_bundle
        assert bundle is not None
        self.assertEqual(bundle.model_binding.model_type, "external")
        self.assertEqual(bundle.model_binding.validity_level.truth_level, "configured_truth")
        self.assertIn("external_pdk_root_candidate", bundle.model_binding.validity_level.detail)
        self.assertEqual(bundle.validation_status.is_valid, True)
        self.assertTrue(
            any(
                "configured_truth candidate path via external PDK root" in warning.message
                for warning in bundle.validation_status.warnings
            )
        )
        self.assertEqual(compiled.report.acceptance_summary["configured_truth_path"], "external_pdk_candidate")
        self.assertIn("configured_truth_path=external_pdk_candidate", bundle.metadata.assumptions)

    def test_configured_truth_without_any_external_source_is_invalid(self) -> None:
        task, planning_bundle, search_state, candidate_id = _context()
        compiled = compile_simulation_bundle(
            task,
            planning_bundle,
            search_state,
            candidate_id,
            backend_preference="ngspice",
            model_binding_overrides={
                "configured_truth_mode": "external_pdk_root",
            },
        )

        self.assertIsNone(compiled.simulation_bundle)
        self.assertEqual(compiled.status, "invalid")
        self.assertEqual(compiled.report.acceptance_summary["configured_truth_path"], "configured_truth_missing_source")
        self.assertEqual(compiled.report.acceptance_summary["validation_state"], "invalid")


if __name__ == "__main__":
    unittest.main()
