"""Integration tests for Day-6 fidelity-aware truth execution."""

from __future__ import annotations

import unittest

from apps.orchestrator.job_runner import run_planning_truth_loop
from apps.worker_simulator.ngspice_runner import native_ngspice_available
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.service import SimulationService
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


@unittest.skipUnless(native_ngspice_available(), "native ngspice is not available in this environment")
class FidelityExecutionIntegrationTests(unittest.TestCase):
    def _context(self):
        spec = DesignSpec(
            task_id="integration-fidelity-ota",
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
        candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
        return task, planning_bundle, search_state, planning_service, candidate_id

    def test_quick_truth_path_executes_quick_truth_profile(self) -> None:
        task, planning_bundle, search_state, _, candidate_id = self._context()
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="quick_truth",
            backend_preference="ngspice",
        )

        analyses = [record.analysis_type for record in execution.verification_result.measurement_report.executed_analyses]
        self.assertEqual(execution.verification_result.executed_fidelity, "quick_truth")
        self.assertEqual(analyses, ["op", "ac"])
        self.assertNotIn("tran", analyses)

    def test_focused_truth_path_executes_stronger_profile(self) -> None:
        task, planning_bundle, search_state, _, candidate_id = self._context()
        execution = SimulationService(task, planning_bundle, search_state).verify_candidate(
            candidate_id,
            fidelity_level="focused_truth",
            backend_preference="ngspice",
        )

        analyses = [record.analysis_type for record in execution.verification_result.measurement_report.executed_analyses]
        measured = {item.metric for item in execution.verification_result.measurement_report.measurement_results}
        self.assertEqual(execution.verification_result.executed_fidelity, "focused_truth")
        self.assertIn("tran", analyses)
        self.assertIn("slew_rate_v_per_us", measured)

    def test_planner_escalation_requests_focused_truth_for_high_value_candidate(self) -> None:
        task, planning_bundle, _, planning_service, _ = self._context()
        state = planning_service.evaluate_candidates(planning_service.propose_candidates(planning_service.initialize_search().search_state).search_state).search_state
        selection = planning_service.select_for_simulation(state)

        self.assertTrue(selection.selected_candidates)
        self.assertIn("focused_truth", set(selection.requested_fidelity_map.values()))

        candidate = selection.selected_candidates[0]
        requested_fidelity = selection.requested_fidelity_map[candidate.candidate_id]
        execution = SimulationService(task, planning_bundle, selection.search_state).verify_candidate(
            candidate.candidate_id,
            fidelity_level=requested_fidelity,
            backend_preference="ngspice",
            escalation_reason=f"planner_selected:{requested_fidelity}",
        )
        self.assertEqual(execution.verification_result.executed_fidelity, "focused_truth")
        self.assertEqual(execution.verification_result.execution_profile.resolved_fidelity, "focused_truth")

    def test_budget_aware_loop_does_not_upgrade_every_selectable_candidate(self) -> None:
        task, _, _, planning_service, _ = self._context()
        state = planning_service.evaluate_candidates(planning_service.propose_candidates(planning_service.initialize_search().search_state).search_state).search_state
        selectable_count = sum(
            1
            for candidate in state.candidate_pool_state.candidates
            if candidate.lifecycle_status in {"frontier", "best_feasible", "best_infeasible"}
            and candidate.predicted_uncertainty is not None
            and candidate.simulation_value_estimate is not None
        )
        selection = planning_service.select_for_simulation(state)
        focused_count = sum(1 for fidelity in selection.requested_fidelity_map.values() if fidelity == "focused_truth")

        self.assertGreater(selectable_count, 1)
        self.assertLessEqual(focused_count, planning_service.planning_bundle.escalation_policy.focused_truth_batch_limit)
        self.assertLess(focused_count, selectable_count)

        loop = run_planning_truth_loop(task, max_steps=3, fidelity_level="quick_truth", backend_preference="ngspice")
        self.assertLess(loop.comparison_summary.selective_simulation_calls, loop.comparison_summary.baseline_full_simulation_calls)
