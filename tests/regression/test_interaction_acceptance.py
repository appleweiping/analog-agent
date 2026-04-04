"""Regression-style acceptance coverage for the interaction layer."""

from __future__ import annotations

import unittest

from libs.interaction.testing import AcceptanceCase, build_acceptance_summary, evaluate_case


class InteractionAcceptanceTests(unittest.TestCase):
    def test_acceptance_summary_tracks_case_outcomes(self) -> None:
        cases = [
            AcceptanceCase(
                name="standard-ota",
                category="standard",
                prompt="設計一個兩級 OTA，GBW > 100MHz，PM > 60°，功耗 < 1mW，1.2V 供電，65nm 工藝",
            ),
            AcceptanceCase(
                name="underspecified-ota",
                category="underspecified",
                prompt="設計一個 OTA，GBW 100MHz",
            ),
            AcceptanceCase(
                name="ambiguous-amp",
                category="ambiguous",
                prompt="我要一個高速低功耗放大器",
            ),
            AcceptanceCase(
                name="adversarial-power",
                category="adversarial",
                prompt="設計 OTA，功耗小於 -1mW，帶寬 10Hz 以上",
            ),
        ]

        results = [evaluate_case(case) for case in cases]
        summary = build_acceptance_summary(results)

        self.assertEqual(summary.total_cases, 4)
        self.assertEqual(summary.passed_cases, 3)
        self.assertGreaterEqual(summary.schema_validity_rate, 0.5)
        self.assertIn("parser_error", summary.error_type_distribution)
