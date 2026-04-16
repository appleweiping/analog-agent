"""Regression matrix for truth-boundary semantics across paper and non-paper modes."""

from __future__ import annotations

import unittest

from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.compiler import compile_simulation_bundle
from libs.simulation.service import SimulationService
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


def _truth_policy_task(task_id: str = "truth-policy-ota"):
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
    return task


class TruthPolicyRegressionTests(unittest.TestCase):
    def _context(self, task_id: str = "truth-policy-ota"):
        task = _truth_policy_task(task_id)
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
        candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
        return task, planning_bundle, search_state, candidate_id

    def test_non_paper_mock_truth_compile_remains_allowed_but_labeled(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context("truth-matrix-mock-compile")
        compiled = compile_simulation_bundle(
            task,
            planning_bundle,
            search_state,
            candidate_id,
            fidelity_level="focused_validation",
            backend_preference="xyce",
            paper_mode=False,
        )

        self.assertEqual(compiled.status, "compiled_with_warnings")
        self.assertIsNotNone(compiled.simulation_bundle)
        assert compiled.simulation_bundle is not None
        self.assertEqual(compiled.simulation_bundle.backend_binding.invocation_mode, "mock_truth")
        self.assertFalse(compiled.simulation_bundle.metadata.paper_truth_policy.paper_mode)
        self.assertFalse(compiled.simulation_bundle.simulation_provenance.paper_mode)
        self.assertFalse(compiled.simulation_bundle.simulation_provenance.paper_safe)
        self.assertTrue(
            any(
                issue.path == "backend_binding.invocation_mode"
                and "mock_truth mode" in issue.message
                for issue in compiled.report.validation_warnings
            )
        )

    def test_non_paper_mock_truth_verify_remains_executable_but_not_paper_safe(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context("truth-matrix-mock-verify")
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="focused_validation",
            backend_preference="xyce",
            paper_mode=False,
        )

        self.assertEqual(execution.simulation_bundle.backend_binding.invocation_mode, "mock_truth")
        self.assertFalse(execution.simulation_bundle.metadata.paper_truth_policy.paper_mode)
        self.assertFalse(execution.verification_result.simulation_provenance.paper_mode)
        self.assertFalse(execution.verification_result.simulation_provenance.paper_safe)
        self.assertEqual(execution.verification_result.validation_status.truth_level, "demonstrator_truth")
        self.assertEqual(execution.verification_result.validation_status.validity_state, "weak")

    @unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
    def test_paper_mode_native_demonstrator_truth_is_allowed_and_marked_paper_safe(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context("truth-matrix-native-demo-paper")
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="focused_validation",
            backend_preference="ngspice",
            paper_mode=True,
        )

        self.assertEqual(execution.simulation_bundle.backend_binding.invocation_mode, "native")
        self.assertTrue(execution.simulation_bundle.metadata.paper_truth_policy.paper_mode)
        self.assertTrue(execution.verification_result.simulation_provenance.paper_mode)
        self.assertTrue(execution.verification_result.simulation_provenance.paper_safe)
        self.assertEqual(execution.verification_result.validation_status.truth_level, "demonstrator_truth")
        self.assertEqual(execution.verification_result.validation_status.validity_state, "weak")
        self.assertTrue(all(record.execution_context and record.execution_context.paper_safe for record in execution.simulation_bundle.artifact_registry.records))

    @unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
    def test_paper_mode_native_configured_truth_is_strong_and_paper_safe(self) -> None:
        task, planning_bundle, search_state, candidate_id = self._context("truth-matrix-native-configured-paper")
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="focused_truth",
            backend_preference="ngspice",
            paper_mode=True,
            model_binding_overrides={
                "model_type": "external",
                "external_model_card_path": "mock://pdk/ota2_model_card.sp",
            },
        )

        self.assertEqual(execution.simulation_bundle.backend_binding.invocation_mode, "native")
        self.assertTrue(execution.verification_result.simulation_provenance.paper_mode)
        self.assertTrue(execution.verification_result.simulation_provenance.paper_safe)
        self.assertEqual(execution.verification_result.validation_status.truth_level, "configured_truth")
        self.assertEqual(execution.verification_result.validation_status.validity_state, "strong")
        self.assertEqual(execution.simulation_bundle.model_binding.model_type, "external")


if __name__ == "__main__":
    unittest.main()
