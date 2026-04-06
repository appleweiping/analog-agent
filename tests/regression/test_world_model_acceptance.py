"""Regression and acceptance coverage for the world model layer."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.schema.world_model import TruthCalibrationRecord, TruthMetric
from libs.tasking.compiler import compile_design_task
from libs.world_model.action_builder import build_design_action
from libs.world_model.compiler import compile_world_model_bundle
from libs.world_model.service import WorldModelService
from libs.world_model.state_builder import build_world_state
from libs.world_model.testing import WorldModelAcceptanceCase, build_acceptance_summary, evaluate_case


def compile_task(spec: DesignSpec):
    compiled = compile_design_task(spec)
    assert compiled.design_task is not None
    return compiled.design_task


class WorldModelAcceptanceTests(unittest.TestCase):
    def test_acceptance_summary_tracks_third_layer_outcomes(self) -> None:
        ota_task = compile_task(
            DesignSpec(
                task_id="accept-ota",
                circuit_family="two_stage_ota",
                process_node="65nm",
                supply_voltage_v=1.2,
                objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
                hard_constraints={"gbw_hz": MetricRange(min=1e8), "phase_margin_deg": MetricRange(min=60.0)},
                environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
                testbench_plan=["op", "ac"],
                design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                missing_information=[],
                notes=[],
                compile_confidence=0.9,
            )
        )
        folded_task = compile_task(
            DesignSpec(
                task_id="accept-folded",
                circuit_family="folded_cascode_ota",
                process_node="65nm",
                supply_voltage_v=1.2,
                objectives=Objectives(maximize=["dc_gain_db"], minimize=["power_w"]),
                hard_constraints={"phase_margin_deg": MetricRange(min=60.0)},
                environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=1.5e-12, supply_voltage_v=1.2),
                testbench_plan=["op", "ac"],
                design_variables=["w_in", "l_in", "w_cas", "l_cas", "ibias", "cc"],
                missing_information=[],
                notes=[],
                compile_confidence=0.89,
            )
        )
        ldo_task = compile_task(
            DesignSpec(
                task_id="accept-ldo",
                circuit_family="ldo",
                process_node="180nm",
                supply_voltage_v=1.8,
                objectives=Objectives(minimize=["power_w"]),
                hard_constraints={"power_w": MetricRange(max=1.5e-3)},
                environment=Environment(temperature_c=[27.0, 125.0], corners=["tt", "ss"], supply_voltage_v=1.8),
                testbench_plan=["op", "ac"],
                design_variables=["w_pass", "l_pass", "w_err", "l_err", "ibias", "c_comp"],
                missing_information=[],
                notes=[],
                compile_confidence=0.9,
            )
        )

        cases = [
            WorldModelAcceptanceCase(
                name="ota-static",
                category="standard_static",
                design_task=ota_task,
                initial_state=build_world_state(ota_task),
                actions=[
                    build_design_action(
                        ota_task,
                        action_family="parameter_update",
                        target_kind="variable",
                        variable_names=["ibias"],
                        operator="scale",
                        payload={"factor": 1.15},
                        expected_scope=["operating_point", "power"],
                        source="planner",
                    )
                ],
            ),
            WorldModelAcceptanceCase(
                name="folded-boundary",
                category="boundary",
                design_task=folded_task,
                initial_state=build_world_state(folded_task, parameter_values={"cc": 0.05e-12, "ibias": 3e-3}),
                actions=[],
            ),
            WorldModelAcceptanceCase(
                name="ldo-calibration",
                category="calibration",
                design_task=ldo_task,
                initial_state=build_world_state(ldo_task, parameter_values={"w_pass": 400e-6, "ibias": 200e-6}),
                actions=[],
                truth_record=TruthCalibrationRecord(
                    simulator_signature="ngspice-calibration",
                    analysis_fidelity="full_ground_truth",
                    metrics=[TruthMetric(metric="power_w", value=1.0e-3)],
                    constraints=[],
                    artifact_refs=["artifact://ldo-calibration"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            ),
        ]

        results = [evaluate_case(case) for case in cases]
        summary = build_acceptance_summary(results)

        self.assertEqual(summary.total_cases, 3)
        self.assertGreaterEqual(summary.schema_validity_rate, 1.0)
        self.assertGreaterEqual(summary.predictive_validity_rate, 1.0)
        self.assertGreaterEqual(summary.feasibility_reliability_rate, 1.0)

    def test_planning_in_the_loop_prefers_feasible_candidate_over_metric_only_baseline(self) -> None:
        task = compile_task(
            DesignSpec(
                task_id="accept-planning-ota",
                circuit_family="two_stage_ota",
                process_node="65nm",
                supply_voltage_v=1.2,
                objectives=Objectives(maximize=["gbw_hz"], minimize=["power_w"]),
                hard_constraints={"phase_margin_deg": MetricRange(min=60.0), "power_w": MetricRange(max=1e-3)},
                environment=Environment(temperature_c=[27.0], corners=["tt"], load_cap_f=2e-12, supply_voltage_v=1.2),
                testbench_plan=["op", "ac", "tran"],
                design_variables=["w_in", "l_in", "w_tail", "l_tail", "ibias", "cc"],
                missing_information=[],
                notes=[],
                compile_confidence=0.94,
            )
        )
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None
        service = WorldModelService(bundle, task)
        initial = build_world_state(task)
        actions = [
            build_design_action(task, action_family="parameter_update", target_kind="variable", variable_names=["cc"], operator="scale", payload={"factor": 1.6}, expected_scope=["ac_stability"], source="planner"),
            build_design_action(task, action_family="parameter_update", target_kind="variable", variable_names=["ibias"], operator="scale", payload={"factor": 1.8}, expected_scope=["operating_point", "power"], source="planner"),
            build_design_action(task, action_family="parameter_update", target_kind="variable", variable_names=["ibias"], operator="set", payload={"value": 1.0}, expected_scope=["operating_point"], source="planner"),
        ]
        candidate_states = [service.predict_transition(initial, action).next_state for action in actions]
        full_ranking = service.rank_candidates(candidate_states)
        metric_only_best = max(
            candidate_states,
            key=lambda state: next(metric.value for metric in service.predict_metrics(state).metrics if metric.metric == "gbw_hz"),
        )
        full_best = next(state for state in candidate_states if state.state_id == full_ranking.ranked_candidates[0].state_id)
        full_feasibility = service.predict_feasibility(full_best).overall_feasibility
        baseline_feasibility = service.predict_feasibility(metric_only_best).overall_feasibility

        self.assertGreaterEqual(full_feasibility, baseline_feasibility)

    def test_regression_stability_keeps_deterministic_predictions(self) -> None:
        task = compile_task(
            DesignSpec(
                task_id="accept-regression-bandgap",
                circuit_family="bandgap",
                process_node="180nm",
                supply_voltage_v=1.8,
                objectives=Objectives(minimize=["power_w"]),
                hard_constraints={"power_w": MetricRange(max=2e-4)},
                environment=Environment(temperature_c=[-40.0, 27.0, 125.0], corners=["tt"], supply_voltage_v=1.8),
                testbench_plan=["op"],
                design_variables=["area_ratio", "r1", "r2", "w_core", "l_core", "ibias"],
                missing_information=[],
                notes=[],
                compile_confidence=0.88,
            )
        )
        bundle = compile_world_model_bundle(task).world_model_bundle
        assert bundle is not None
        service = WorldModelService(bundle, task)
        state = build_world_state(task)

        first = service.predict_metrics(state)
        second = service.predict_metrics(state)

        self.assertEqual([metric.metric for metric in first.metrics], [metric.metric for metric in second.metrics])
        self.assertEqual([metric.value for metric in first.metrics], [metric.value for metric in second.metrics])
