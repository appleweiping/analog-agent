"""Unit tests for the sixth-layer memory and reflection system."""

from __future__ import annotations

import unittest

from libs.memory.compiler import compile_memory_bundle
from libs.memory.quality_governor import apply_quality_governance
from libs.memory.service import MemoryService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle
from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.simulation.service import SimulationService


def _build_pipeline(task_id: str, family: str = "two_stage_ota"):
    spec = DesignSpec(
        task_id=task_id,
        circuit_family=family,
        process_node="65nm",
        supply_voltage_v=1.2,
        objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
        hard_constraints={
            "gbw_hz": MetricRange(min=8e7),
            "phase_margin_deg": MetricRange(min=55.0),
        },
        environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
        testbench_plan=["op", "ac", "tran", "noise"],
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
    planner = PlanningService(planning_bundle, task, world_model_bundle)
    search_state = planner.initialize_search().search_state
    search_state = planner.propose_candidates(search_state).search_state
    search_state = planner.evaluate_candidates(search_state).search_state
    candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
    execution = SimulationService(task, planning_bundle, search_state).execute(candidate_id)
    return task, search_state, execution.verification_result


class MemoryLayerTests(unittest.TestCase):
    def test_compile_and_ingest_produce_formal_memory_objects(self) -> None:
        bundle = compile_memory_bundle().memory_bundle
        assert bundle is not None
        task, search_state, verification_result = _build_pipeline("memory-standard")

        ingestion = MemoryService(bundle).ingest_episode(task, search_state, verification_result)

        self.assertTrue(ingestion.memory_bundle.validation_status.is_valid)
        self.assertEqual(len(ingestion.memory_bundle.episode_records), 1)
        self.assertTrue(ingestion.episode_record.evidence_refs)
        self.assertIsNotNone(ingestion.reflection_report)
        self.assertTrue(ingestion.emitted_feedback)

    def test_pattern_mining_requires_multiple_supporting_episodes(self) -> None:
        bundle = compile_memory_bundle().memory_bundle
        assert bundle is not None
        service = MemoryService(bundle)
        task_a, search_a, verification_a = _build_pipeline("memory-pattern-a")
        first = service.ingest_episode(task_a, search_a, verification_a)
        task_b, search_b, verification_b = _build_pipeline("memory-pattern-b")
        second = MemoryService(first.memory_bundle).ingest_episode(task_b, search_b, verification_b)

        self.assertTrue(second.memory_bundle.pattern_records)
        self.assertTrue(all(pattern.supporting_evidence_count >= 2 for pattern in second.memory_bundle.pattern_records))

    def test_retrieval_is_task_signature_conditioned(self) -> None:
        bundle = compile_memory_bundle().memory_bundle
        assert bundle is not None
        task_a, search_a, verification_a = _build_pipeline("memory-retrieve-a", "two_stage_ota")
        updated = MemoryService(bundle).ingest_episode(task_a, search_a, verification_a).memory_bundle

        same_family_task, _, _ = _build_pipeline("memory-retrieve-b", "two_stage_ota")
        unrelated_task, _, _ = _build_pipeline("memory-retrieve-c", "bandgap")
        same_result = MemoryService(updated).retrieve_relevant_memory(same_family_task)
        different_result = MemoryService(updated).retrieve_relevant_memory(unrelated_task)

        self.assertGreaterEqual(same_result.retrieval_precision_proxy, different_result.retrieval_precision_proxy)
        self.assertLessEqual(same_result.negative_transfer_risk, different_result.negative_transfer_risk)

    def test_governance_and_forgetting_bound_growth(self) -> None:
        bundle = compile_memory_bundle().memory_bundle
        assert bundle is not None
        bundle = bundle.model_copy(
            update={
                "forgetting_policy": bundle.forgetting_policy.model_copy(update={"max_episode_records": 1, "max_pattern_records": 1}),
            }
        )
        service = MemoryService(bundle)
        first = service.ingest_episode(*_build_pipeline("memory-forget-a")).memory_bundle
        second = MemoryService(first).ingest_episode(*_build_pipeline("memory-forget-b")).memory_bundle
        governed = apply_quality_governance(second)

        self.assertLessEqual(len(governed.episode_records), 1)
        self.assertLessEqual(len(governed.pattern_records), 1)
        self.assertTrue(all(pattern.governance_state in {"active", "candidate", "conflicted", "deprecated", "forgotten"} for pattern in governed.pattern_records))


if __name__ == "__main__":
    unittest.main()
