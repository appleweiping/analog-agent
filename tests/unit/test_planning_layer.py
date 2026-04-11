"""Unit tests for the planning layer."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.world_model import TruthCalibrationRecord, TruthConstraint, TruthMetric
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


def build_standard_ota_task():
    spec = DesignSpec(
        task_id="plan-standard-ota",
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
        testbench_plan=["op", "ac", "tran"],
        design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
        missing_information=[],
        notes=[],
        compile_confidence=0.93,
    )
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


class PlanningLayerTests(unittest.TestCase):
    def test_compile_and_initialize_search(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        compiled = compile_planning_bundle(task, world_model_bundle)

        self.assertIn(compiled.status, {"compiled", "compiled_with_warnings"})
        self.assertIsNotNone(compiled.planning_bundle)
        planning_bundle = compiled.planning_bundle
        assert planning_bundle is not None

        service = PlanningService(planning_bundle, task, world_model_bundle)
        initialized = service.initialize_search()

        self.assertTrue(initialized.search_state.candidate_pool_state.candidates)
        self.assertTrue(initialized.search_state.trace_log)
        self.assertTrue(service.validate_search_state(initialized.search_state).is_valid)

    def test_proposal_and_evaluation_expand_frontier(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        service = PlanningService(planning_bundle, task, world_model_bundle)

        state = service.initialize_search().search_state
        initial_count = len(state.candidate_pool_state.candidates)
        proposed = service.propose_candidates(state)
        evaluated = service.evaluate_candidates(proposed.search_state)

        self.assertGreater(len(evaluated.search_state.candidate_pool_state.candidates), initial_count)
        self.assertGreaterEqual(len(evaluated.search_state.trace_log), 3)
        self.assertTrue(all(candidate.priority_score >= -5.0 for candidate in evaluated.candidates))

    def test_select_for_simulation_respects_budget(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        service = PlanningService(planning_bundle, task, world_model_bundle)

        state = service.evaluate_candidates(service.propose_candidates(service.initialize_search().search_state).search_state).search_state
        selection = service.select_for_simulation(state)

        self.assertLessEqual(selection.search_state.budget_state.simulations_used, selection.search_state.budget_state.simulation_budget)
        self.assertTrue(all(candidate.lifecycle_status == "queued_for_simulation" for candidate in selection.selected_candidates))
        selectable_count = sum(
            1
            for candidate in state.candidate_pool_state.candidates
            if candidate.lifecycle_status in {"frontier", "best_feasible", "best_infeasible"}
            and candidate.predicted_uncertainty is not None
            and candidate.simulation_value_estimate is not None
        )
        if selectable_count > 1:
            self.assertLess(len(selection.selected_candidates), selectable_count)

    def test_rank_candidates_returns_explicit_priority_order(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        service = PlanningService(planning_bundle, task, world_model_bundle)

        state = service.evaluate_candidates(service.propose_candidates(service.initialize_search().search_state).search_state).search_state
        frontier = [
            candidate
            for candidate in state.candidate_pool_state.candidates
            if candidate.lifecycle_status in {"frontier", "best_feasible", "best_infeasible"}
        ]
        ranked = service.rank_candidates(frontier)

        self.assertTrue(ranked)
        self.assertTrue(all(ranked[index].priority_score >= ranked[index + 1].priority_score for index in range(len(ranked) - 1)))
        self.assertTrue(all(candidate.predicted_feasibility is not None for candidate in ranked))
        self.assertTrue(all(candidate.predicted_uncertainty is not None for candidate in ranked))
        self.assertTrue(all(candidate.simulation_value_estimate is not None for candidate in ranked))

    def test_world_model_mismatch_triggers_calibration_recovery_signal(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        service = PlanningService(planning_bundle, task, world_model_bundle)

        state = service.evaluate_candidates(service.propose_candidates(service.initialize_search().search_state).search_state).search_state
        candidate = max(state.candidate_pool_state.candidates, key=lambda item: item.priority_score)
        feedback = TruthCalibrationRecord(
            simulator_signature="fake-ngspice",
            analysis_fidelity="full_ground_truth",
            truth_level="demonstrator_truth",
            validation_status="weak",
            metrics=[TruthMetric(metric="gbw_hz", value=1e6)],
            constraints=[TruthConstraint(constraint_name="gbw_hz_min", constraint_group="bandwidth", is_satisfied=False, margin=-1.0)],
            artifact_refs=["artifact://feedback/ota"],
            provenance_tags=["planning_fixture"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        updated = service.ingest_simulation_feedback(state, candidate.candidate_id, feedback)

        self.assertTrue(updated.search_state.risk_context.calibration_required)

    def test_advance_phase_progresses_when_feasible_candidate_exists(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None
        service = PlanningService(planning_bundle, task, world_model_bundle)

        state = service.initialize_search().search_state
        advanced = service.advance_phase(state).search_state

        self.assertIn(advanced.phase_state.current_phase, {"feasibility_bootstrapping", "performance_refinement"})

    def test_regression_stability_keeps_initial_search_deterministic(self) -> None:
        task = build_standard_ota_task()
        world_model_bundle = compile_world_model_bundle(task).world_model_bundle
        assert world_model_bundle is not None
        planning_bundle = compile_planning_bundle(task, world_model_bundle).planning_bundle
        assert planning_bundle is not None

        first = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
        second = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state

        self.assertEqual(
            [candidate.candidate_id for candidate in first.candidate_pool_state.candidates],
            [candidate.candidate_id for candidate in second.candidate_pool_state.candidates],
        )
        self.assertEqual(first.frontier_state.frontier_candidate_ids, second.frontier_state.frontier_candidate_ids)
