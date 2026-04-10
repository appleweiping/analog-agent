"""Unit tests for fifth-layer measurement contract robustness."""

from __future__ import annotations

import unittest

from libs.planner.compiler import compile_planning_bundle
from libs.planner.service import PlanningService
from libs.schema.design_spec import DesignSpec, Environment, MetricRange, Objectives
from libs.simulation.compiler import compile_simulation_bundle
from libs.simulation.constraint_verifier import verify_constraints
from libs.simulation.measurement_extractors import extract_measurement_report
from libs.tasking.compiler import compile_design_task
from libs.world_model.compiler import compile_world_model_bundle


def _bundle():
    spec = DesignSpec(
        task_id="measurement-contract-ota",
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
        testbench_plan=["op", "ac"],
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
    search_state = PlanningService(planning_bundle, task, world_model_bundle).initialize_search().search_state
    candidate_id = search_state.candidate_pool_state.candidates[0].candidate_id
    compiled = compile_simulation_bundle(task, planning_bundle, search_state, candidate_id, fidelity_level="focused_validation")
    assert compiled.simulation_bundle is not None
    return task, compiled.simulation_bundle, candidate_id


def _base_outputs():
    return [
        {
            "status": "ok",
            "analysis_type": "op",
            "metrics": {},
            "op_diagnostics": {"supply_currents": [-1.0e-4], "supply_voltage_v": 1.2},
            "runtime_ms": 10,
            "artifact_ref": "art_op",
        },
        {
            "status": "ok",
            "analysis_type": "ac",
            "metrics": {},
            "ac_curve": [
                {"frequency_hz": 1.0, "gain_db": 60.0, "phase_deg": -90.0},
                {"frequency_hz": 1.0e6, "gain_db": 20.0, "phase_deg": -120.0},
                {"frequency_hz": 1.0e8, "gain_db": -5.0, "phase_deg": -135.0},
            ],
            "op_diagnostics": {},
            "runtime_ms": 11,
            "artifact_ref": "art_ac",
        },
    ]


class MeasurementContractTests(unittest.TestCase):
    def test_normal_curve_extracts_all_core_metrics(self) -> None:
        _, bundle, candidate_id = _bundle()
        report = extract_measurement_report(bundle, _base_outputs(), candidate_id=candidate_id)

        statuses = {item.metric: item.status.status for item in report.measurement_results}
        self.assertEqual(statuses["dc_gain_db"], "measured")
        self.assertEqual(statuses["gbw_hz"], "measured")
        self.assertEqual(statuses["phase_margin_deg"], "measured")
        self.assertEqual(statuses["power_w"], "measured")

    def test_no_unity_crossing_returns_structured_indeterminate_status(self) -> None:
        _, bundle, candidate_id = _bundle()
        outputs = _base_outputs()
        outputs[1]["ac_curve"] = [
            {"frequency_hz": 1.0, "gain_db": 80.0, "phase_deg": -90.0},
            {"frequency_hz": 1.0e6, "gain_db": 40.0, "phase_deg": -110.0},
            {"frequency_hz": 1.0e8, "gain_db": 10.0, "phase_deg": -130.0},
        ]

        report = extract_measurement_report(bundle, outputs, candidate_id=candidate_id)
        gbw = next(item for item in report.measurement_results if item.metric == "gbw_hz")
        phase = next(item for item in report.measurement_results if item.metric == "phase_margin_deg")

        self.assertEqual(gbw.status.status, "indeterminate")
        self.assertEqual(gbw.failure_reason.code, "curve_exists_but_no_crossing")
        self.assertEqual(phase.failure_reason.code, "curve_exists_but_no_crossing")

    def test_multiple_crossings_use_structured_selection_rule(self) -> None:
        _, bundle, candidate_id = _bundle()
        outputs = _base_outputs()
        outputs[1]["ac_curve"] = [
            {"frequency_hz": 1.0, "gain_db": 10.0, "phase_deg": -100.0},
            {"frequency_hz": 10.0, "gain_db": -2.0, "phase_deg": -120.0},
            {"frequency_hz": 100.0, "gain_db": 4.0, "phase_deg": -150.0},
            {"frequency_hz": 1000.0, "gain_db": -3.0, "phase_deg": -170.0},
        ]

        report = extract_measurement_report(bundle, outputs, candidate_id=candidate_id)
        gbw = next(item for item in report.measurement_results if item.metric == "gbw_hz")
        phase = next(item for item in report.measurement_results if item.metric == "phase_margin_deg")

        self.assertEqual(gbw.status.status, "measured")
        self.assertEqual(gbw.failure_reason.code, "multiple_crossings")
        self.assertEqual(phase.failure_reason.code, "multiple_crossings")

    def test_invalid_phase_readout_is_not_treated_as_design_value(self) -> None:
        _, bundle, candidate_id = _bundle()
        outputs = _base_outputs()
        outputs[1]["ac_curve"] = [
            {"frequency_hz": 1.0, "gain_db": 20.0, "phase_deg": -100.0},
            {"frequency_hz": 100.0, "gain_db": -2.0, "phase_deg": None},
            {"frequency_hz": 1000.0, "gain_db": -10.0, "phase_deg": None},
        ]

        report = extract_measurement_report(bundle, outputs, candidate_id=candidate_id)
        phase = next(item for item in report.measurement_results if item.metric == "phase_margin_deg")

        self.assertEqual(phase.status.status, "indeterminate")
        self.assertEqual(phase.failure_reason.code, "invalid_phase_readout")

    def test_power_direction_anomaly_is_structured_not_zeroed(self) -> None:
        _, bundle, candidate_id = _bundle()
        outputs = _base_outputs()
        outputs[0]["op_diagnostics"] = {"supply_currents": [-1.0e-4, 2.0e-4], "supply_voltage_v": 1.2}

        report = extract_measurement_report(bundle, outputs, candidate_id=candidate_id)
        power = next(item for item in report.measurement_results if item.metric == "power_w")

        self.assertEqual(power.status.status, "indeterminate")
        self.assertEqual(power.failure_reason.code, "current_direction_ambiguous")

    def test_half_failed_sample_preserves_partial_measurement_state(self) -> None:
        _, bundle, candidate_id = _bundle()
        outputs = _base_outputs()
        outputs[1] = {
            "status": "error",
            "analysis_type": "ac",
            "metrics": {},
            "runtime_ms": 11,
            "artifact_ref": "art_ac",
        }

        report = extract_measurement_report(bundle, outputs, candidate_id=candidate_id)
        power = next(item for item in report.measurement_results if item.metric == "power_w")
        gbw = next(item for item in report.measurement_results if item.metric == "gbw_hz")

        self.assertEqual(power.status.status, "measured")
        self.assertEqual(gbw.status.status, "analysis_failed")

    def test_constraint_adjudication_marks_measurement_unavailable_without_design_semantics(self) -> None:
        task, bundle, candidate_id = _bundle()
        outputs = _base_outputs()
        outputs[1]["ac_curve"] = [
            {"frequency_hz": 1.0, "gain_db": 80.0, "phase_deg": -90.0},
            {"frequency_hz": 1.0e6, "gain_db": 40.0, "phase_deg": -110.0},
            {"frequency_hz": 1.0e8, "gain_db": 10.0, "phase_deg": -130.0},
        ]

        report = extract_measurement_report(bundle, outputs, candidate_id=candidate_id)
        assessments = verify_constraints(task, report, bundle.verification_policy)
        gbw_assessment = next(item for item in assessments if item.metric == "gbw_hz")

        self.assertEqual(gbw_assessment.assessment_basis, "measurement_unavailable")
        self.assertEqual(gbw_assessment.measurement_failure_reason, "curve_exists_but_no_crossing")
