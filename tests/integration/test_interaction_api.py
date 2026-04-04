"""Integration tests for interaction-layer API routes."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "fastapi/httpx are not installed in this environment",
)
class InteractionApiTests(unittest.TestCase):
    def test_compile_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app

        client = TestClient(app)
        response = client.post(
            "/interaction/compile",
            json={
                "text": "設計一個兩級 OTA，GBW > 100MHz，PM > 60°，功耗 < 1mW，1.2V 供電，65nm 工藝",
                "mode": "strict",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "compiled")
        self.assertEqual(payload["design_spec"]["circuit_family"], "two_stage_ota")

    def test_validate_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from apps.api_server.main import app
        from libs.interaction.spec_compiler import compile_spec

        compiled = compile_spec("設計一個兩級 OTA，GBW > 100MHz，PM > 60°，功耗 < 1mW，1.2V 供電，65nm 工藝")
        assert compiled.design_spec is not None

        client = TestClient(app)
        response = client.post(
            "/interaction/validate",
            json={"design_spec": compiled.design_spec.model_dump()},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("valid", payload)
