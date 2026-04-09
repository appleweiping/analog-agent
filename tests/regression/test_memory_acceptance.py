"""Regression and acceptance tests for the sixth layer."""

from __future__ import annotations

import unittest

from libs.memory.compiler import compile_memory_bundle
from libs.memory.service import MemoryService
from libs.memory.testing import (
    MemoryAcceptanceCase,
    build_acceptance_summary,
    evaluate_case,
)
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.simulation.service import SimulationService


def _case(task_id: str, family: str, *, load_cap: float):
    spec = DesignSpec(
        task_id=task_id,
        circuit_family=family,
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={"gbw_hz": MetricRange(min=8e7), "phase_margin_deg": MetricRange(min=55.0)},
        environment=Environment(temperature_c=[-20.0, 27.0, 85.0], corners=["tt", "ss", "ff"], load_cap_f=load_cap, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran", "noise"],
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
    planner = PlanningService(planning_bundle, task, world_model_bundle)
    search_state = planner.initialize_search().search_state
    search_state = planner.propose_candidates(search_state).search_state
    search_state = planner.evaluate_candidates(search_state).search_state
    verification = SimulationService(task, planning_bundle, search_state).verify_candidate(search_state.candidate_pool_state.candidates[0].candidate_id).verification_result
    return task, search_state, verification


class MemoryAcceptanceTests(unittest.TestCase):
    def test_acceptance_suite_covers_memory_failure_modes(self) -> None:
        cases = []
        for name, family, load_cap in [
            ("episodic", "two_stage_ota", 2e-12),
            ("transfer", "ldo", 4e-12),
            ("robustness", "bandgap", 1e-12),
        ]:
            task, search_state, verification = _case(f"accept-{name}", family, load_cap=load_cap)
            cases.append(
                MemoryAcceptanceCase(
                    name=name,
                    category=name,
                    design_task=task,
                    search_state=search_state,
                    verification_result=verification,
                )
            )

        results = [evaluate_case(case) for case in cases]
        summary = build_acceptance_summary(results)

        self.assertEqual(summary.total_cases, 3)
        self.assertGreaterEqual(summary.schema_validity_rate, 1.0)
        self.assertGreaterEqual(summary.knowledge_validity_rate, 1.0)
        self.assertGreaterEqual(summary.feedback_adoption_rate, 1.0)
        self.assertLessEqual(summary.negative_transfer_ratio, 1.0)

    def test_regression_stability_is_deterministic(self) -> None:
        task, search_state, verification = _case("accept-regression", "two_stage_ota", load_cap=2e-12)
        case = MemoryAcceptanceCase(
            name="regression",
            category="regression",
            design_task=task,
            search_state=search_state,
            verification_result=verification,
        )
        first = evaluate_case(case)
        second = evaluate_case(case)

        self.assertEqual(first.result, second.result)
        self.assertEqual(first.feedback_useful, second.feedback_useful)
        self.assertEqual(first.retrieval_relevant, second.retrieval_relevant)

    def test_negative_transfer_and_governance_controls(self) -> None:
        bundle = compile_memory_bundle().memory_bundle
        assert bundle is not None
        bundle = bundle.model_copy(
            update={
                "forgetting_policy": bundle.forgetting_policy.model_copy(update={"max_episode_records": 2, "max_pattern_records": 2}),
            }
        )
        service = MemoryService(bundle)
        first = service.ingest_episode(*_case("accept-govern-a", "two_stage_ota", load_cap=2e-12)).memory_bundle
        second = MemoryService(first).ingest_episode(*_case("accept-govern-b", "two_stage_ota", load_cap=2.2e-12)).memory_bundle
        retrieval = MemoryService(second).retrieve_relevant_memory(_case("accept-govern-c", "bandgap", load_cap=1e-12)[0])

        self.assertLessEqual(retrieval.negative_transfer_risk, 1.0)
        self.assertLessEqual(len(second.episode_records), 2)
        self.assertLessEqual(len(second.pattern_records), 2)


if __name__ == "__main__":
    unittest.main()
