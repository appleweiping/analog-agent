"""Unit tests for the interaction layer."""

from __future__ import annotations

import unittest

from libs.interaction.repair_loop import run_repair_loop
from libs.interaction.spec_compiler import compile_spec
from libs.interaction.validator import validate_design_spec
from libs.schema.design_spec import DESIGN_VARIABLES_BY_FAMILY, MetricRange


class InteractionLayerTests(unittest.TestCase):
    def test_standard_case_compiles_into_design_spec(self) -> None:
        response = compile_spec(
            "設計一個兩級 OTA，GBW > 100MHz，PM > 60°，功耗 < 1mW，1.2V 供電，65nm 工藝"
        )

        self.assertEqual(response.status, "compiled")
        self.assertIsNotNone(response.design_spec)
        spec = response.design_spec
        assert spec is not None

        self.assertEqual(spec.circuit_family, "two_stage_ota")
        self.assertEqual(spec.process_node, "65nm")
        self.assertAlmostEqual(spec.supply_voltage_v or 0.0, 1.2)
        self.assertAlmostEqual(spec.hard_constraints["gbw_hz"].min or 0.0, 1e8)
        self.assertAlmostEqual(spec.hard_constraints["phase_margin_deg"].min or 0.0, 60.0)
        self.assertAlmostEqual(spec.hard_constraints["power_w"].max or 0.0, 1e-3)
        self.assertIn("op", spec.testbench_plan)
        self.assertIn("ac", spec.testbench_plan)
        self.assertIn("load_cap_f", spec.missing_information)

    def test_underspecified_case_marks_missing_information(self) -> None:
        response = compile_spec("設計一個 OTA，GBW 100MHz")

        self.assertEqual(response.status, "compiled")
        self.assertIsNotNone(response.design_spec)
        spec = response.design_spec
        assert spec is not None

        self.assertEqual(spec.circuit_family, "unknown")
        self.assertAlmostEqual(spec.hard_constraints["gbw_hz"].target or 0.0, 1e8)
        self.assertIn("circuit_family", spec.missing_information)
        self.assertIn("process_node", spec.missing_information)
        self.assertIn("load_cap_f", spec.missing_information)

    def test_ambiguous_case_becomes_qualitative_objectives(self) -> None:
        response = compile_spec("我要一個高速低功耗放大器")

        self.assertEqual(response.status, "compiled")
        self.assertIsNotNone(response.design_spec)
        spec = response.design_spec
        assert spec is not None

        self.assertEqual(spec.circuit_family, "unknown")
        self.assertEqual(spec.hard_constraints, {})
        self.assertIn("gbw_hz", spec.objectives.maximize)
        self.assertIn("power_w", spec.objectives.minimize)
        self.assertIn("circuit_family", spec.missing_information)
        self.assertIn("process_node", spec.missing_information)

    def test_adversarial_case_is_reported_as_invalid(self) -> None:
        response = compile_spec("設計 OTA，功耗小於 -1mW，帶寬 10Hz 以上")

        self.assertEqual(response.status, "invalid")
        self.assertIsNone(response.design_spec)
        self.assertTrue(response.report.validation_issues)
        self.assertIn("parser_error", [issue.code for issue in response.report.validation_issues])

    def test_interactive_mode_returns_clarification_request(self) -> None:
        response = compile_spec("Design an OTA with GBW 100MHz", mode="interactive")

        self.assertEqual(response.status, "clarification_required")
        self.assertIsNone(response.design_spec)
        self.assertIsNotNone(response.clarification_request)
        clarification = response.clarification_request
        assert clarification is not None
        self.assertIn("process_node", clarification.missing_information)
        self.assertTrue(clarification.suggested_questions)

    def test_repair_loop_restores_a_broken_spec(self) -> None:
        base_response = compile_spec(
            "設計一個兩級 OTA，GBW > 100MHz，PM > 60°，功耗 < 1mW，1.2V 供電，65nm 工藝"
        )
        base_spec = base_response.design_spec
        assert base_spec is not None

        broken_spec = base_spec.model_copy(
            update={
                "testbench_plan": [],
                "design_variables": [],
                "compile_confidence": 1.4,
                "hard_constraints": {
                    "gbw_hz": MetricRange.model_construct(min=2e8, max=1e8, target=None, priority="hard"),
                    "phase_margin_deg": base_spec.hard_constraints["phase_margin_deg"],
                    "power_w": base_spec.hard_constraints["power_w"],
                },
            }
        )

        repaired_spec, attempts = run_repair_loop(broken_spec)
        validation = validate_design_spec(repaired_spec)

        self.assertGreater(attempts, 0)
        self.assertTrue(validation.valid)
        self.assertEqual(repaired_spec.design_variables, DESIGN_VARIABLES_BY_FAMILY["two_stage_ota"])
        self.assertEqual(repaired_spec.testbench_plan, ["op", "ac"])
        self.assertLessEqual(repaired_spec.compile_confidence, 1.0)
        self.assertLessEqual(repaired_spec.hard_constraints["gbw_hz"].min or 0.0, repaired_spec.hard_constraints["gbw_hz"].max or 0.0)
