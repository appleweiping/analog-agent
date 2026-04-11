"""Integration coverage for Day-7 model binding and physical-validity governance."""

from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.memory.compiler import compile_memory_bundle
from libs.memory.service import MemoryService
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.service import SimulationService
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class ModelBindingProvenanceIntegrationTests(unittest.TestCase):
    def _context(self, task_id: str = "model-binding-ota"):
        spec = DesignSpec(
            task_id=task_id,
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
            testbench_plan=["op", "ac", "tran"],
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
        planning_service = PlanningService(planning_bundle, task, world_model_bundle)
        search_state = planning_service.initialize_search().search_state
        candidate = search_state.candidate_pool_state.candidates[0]
        return task, world_model_bundle, planning_bundle, planning_service, search_state, candidate

    def test_demonstrator_mode_marks_result_as_demonstrator_truth(self) -> None:
        task, _, planning_bundle, _, search_state, candidate = self._context("demo-truth-ota")
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate.candidate_id,
            fidelity_level="quick_truth",
            backend_preference="ngspice",
        )

        self.assertEqual(execution.simulation_bundle.model_binding.model_type, "builtin")
        self.assertEqual(execution.verification_result.validation_status.truth_level, "demonstrator_truth")
        self.assertEqual(execution.verification_result.validation_status.validity_state, "weak")
        self.assertEqual(execution.verification_result.simulation_provenance.backend, "ngspice")

    def test_external_model_override_promotes_to_configured_truth(self) -> None:
        task, _, planning_bundle, _, search_state, candidate = self._context("configured-truth-ota")
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate.candidate_id,
            fidelity_level="focused_truth",
            backend_preference="ngspice",
            model_binding_overrides={
                "model_type": "external",
                "external_model_card_path": "mock://pdk/ota2_model_card.sp",
            },
        )

        self.assertEqual(execution.simulation_bundle.model_binding.model_type, "external")
        self.assertEqual(execution.verification_result.validation_status.truth_level, "configured_truth")
        self.assertEqual(execution.verification_result.validation_status.validity_state, "strong")
        self.assertEqual(
            execution.verification_result.simulation_provenance.model_binding.model_source.locator,
            "mock://pdk/ota2_model_card.sp",
        )

    def test_validation_status_propagates_into_l3_l4_l6(self) -> None:
        task, world_model_bundle, planning_bundle, planning_service, search_state, candidate = self._context("propagation-ota")
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate.candidate_id,
            fidelity_level="quick_truth",
            backend_preference="ngspice",
        )

        world_model_service = WorldModelService(world_model_bundle, task)
        calibration = world_model_service.calibrate_with_truth(
            candidate.world_state_snapshot,
            execution.verification_result.calibration_payload.truth_record,
        )
        self.assertIn("demonstrator_truth_calibration_boundary", calibration.trust_assessment.reasons)

        feedback = planning_service.ingest_simulation_feedback(
            search_state,
            candidate.candidate_id,
            execution.verification_result.calibration_payload.truth_record,
            execution.verification_result.planner_feedback,
        )
        self.assertTrue(feedback.search_state.risk_context.calibration_required)
        self.assertIn("truth_level=demonstrator_truth", feedback.search_state.risk_context.active_failure_modes)

        memory_bundle = compile_memory_bundle().memory_bundle
        assert memory_bundle is not None
        ingestion = MemoryService(memory_bundle).ingest_episode(task, feedback.search_state, execution.verification_result)
        self.assertEqual(ingestion.episode_record.final_outcome.truth_level, "demonstrator_truth")
        self.assertEqual(ingestion.episode_record.final_outcome.validation_status, "weak")
        self.assertTrue(
            any(
                reference.truth_level == "demonstrator_truth" and reference.validation_status == "weak"
                for reference in ingestion.episode_record.evidence_refs
            )
        )


if __name__ == "__main__":
    unittest.main()
