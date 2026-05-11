from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from apps.worker_llm.critic_agent import CriticAgent
from apps.worker_llm.parser_agent import ParserAgent
from apps.worker_llm.planner_agent import PlannerAgent
from apps.worker_llm.provider_adapter import LLMProviderAdapter, LLMRequest, LLMRoleConfig, load_role_config


class LLMProviderAdapterTests(unittest.TestCase):
    def test_codex_local_returns_structured_handoff(self) -> None:
        config = LLMRoleConfig(role="planner", model="codex-local", provider="codex_local")
        response = LLMProviderAdapter(config).complete(
            LLMRequest(role="planner", task="plan", payload={"spec": {"family": "ota2"}})
        )

        self.assertEqual(response.status, "local_handoff")
        self.assertEqual(response.provider, "codex_local")
        self.assertIn("route_through_codex_conversation", response.text)

    def test_missing_remote_api_key_is_structured(self) -> None:
        config = LLMRoleConfig(role="critic", model="gpt-test", provider="openai")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            response = LLMProviderAdapter(config).complete(
                LLMRequest(role="critic", task="critique", payload={"run": "x"})
            )

        self.assertEqual(response.status, "missing_api_key")
        self.assertEqual(response.provider, "openai")
        self.assertIn("OPENAI_API_KEY", response.error or "")

    def test_role_wrappers_no_longer_return_stub_status(self) -> None:
        self.assertNotEqual(PlannerAgent().plan({"family": "ota2"})["status"], "stub")
        self.assertNotEqual(ParserAgent().parse("design an OTA")["status"], "stub")
        self.assertNotEqual(CriticAgent().critique({"status": "failed"})["status"], "stub")

    def test_configs_are_versioned_and_default_local(self) -> None:
        with patch.dict(os.environ, {"ANALOG_AGENT_LLM_PROVIDER": "", "ANALOG_AGENT_LLM_MODEL": ""}, clear=False):
            os.environ.pop("ANALOG_AGENT_LLM_PROVIDER", None)
            os.environ.pop("ANALOG_AGENT_LLM_MODEL", None)
            config = load_role_config("planner")

        self.assertEqual(config.provider, "codex_local")
        self.assertEqual(config.prompt_contract_version, "v1")
        self.assertEqual(config.tool_schema_version, "v1")
        self.assertEqual(config.response_schema_version, "v1")

    def test_env_provider_and_model_override_tracked_role_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ANALOG_AGENT_LLM_PROVIDER": "openai",
                "ANALOG_AGENT_LLM_MODEL": "gpt-test",
            },
            clear=False,
        ):
            config = load_role_config("planner")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-test")

    def test_role_specific_env_override_wins_over_global_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ANALOG_AGENT_LLM_PROVIDER": "openai",
                "ANALOG_AGENT_LLM_PROVIDER_PLANNER": "anthropic",
                "ANALOG_AGENT_LLM_MODEL": "gpt-test",
                "ANALOG_AGENT_LLM_MODEL_PLANNER": "claude-test",
            },
            clear=False,
        ):
            config = load_role_config("planner")

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.model, "claude-test")


if __name__ == "__main__":
    unittest.main()
