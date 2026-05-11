"""Provider-neutral LLM adapter for worker roles."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from libs.utils.io import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "llm"

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/responses",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


@dataclass(frozen=True)
class LLMRoleConfig:
    """Runtime configuration for one worker role."""

    role: str
    model: str
    provider: str = "codex_local"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    system_goal: str = ""
    prompt_contract_version: str = "v1"
    tool_schema_version: str = "v1"
    response_schema_version: str = "v1"


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral request object."""

    role: str
    task: str
    payload: dict[str, Any]
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral response object."""

    provider: str
    model: str
    role: str
    status: str
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def load_role_config(role: str) -> LLMRoleConfig:
    """Load a role config from `configs/llm`, defaulting to local Codex handoff."""

    config_path = CONFIG_ROOT / f"{role}.yaml"
    data = read_yaml(config_path) if config_path.exists() else {}
    role_key = role.upper().replace("-", "_")
    provider = str(
        os.getenv(f"ANALOG_AGENT_LLM_PROVIDER_{role_key}")
        or os.getenv("ANALOG_AGENT_LLM_PROVIDER")
        or data.get("provider")
        or "codex_local"
    ).strip()
    model = str(
        os.getenv(f"ANALOG_AGENT_LLM_MODEL_{role_key}")
        or os.getenv("ANALOG_AGENT_LLM_MODEL")
        or data.get("model")
        or "codex-local"
    )
    return LLMRoleConfig(
        role=str(data.get("role", role)),
        model=model,
        provider=provider or "codex_local",
        temperature=float(data.get("temperature", 0.0)),
        max_output_tokens=int(data.get("max_output_tokens", 2048)),
        system_goal=str(data.get("system_goal", "")),
        prompt_contract_version=str(data.get("prompt_contract_version", "v1")),
        tool_schema_version=str(data.get("tool_schema_version", "v1")),
        response_schema_version=str(data.get("response_schema_version", "v1")),
    )


def _messages(config: LLMRoleConfig, llm_request: LLMRequest) -> list[dict[str, str]]:
    system_prompt = llm_request.system_prompt or config.system_goal
    user_content = json.dumps(
        {
            "task": llm_request.task,
            "payload": llm_request.payload,
            "metadata": llm_request.metadata,
            "prompt_contract_version": config.prompt_contract_version,
            "tool_schema_version": config.tool_schema_version,
            "response_schema_version": config.response_schema_version,
        },
        sort_keys=True,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def _codex_local_response(config: LLMRoleConfig, llm_request: LLMRequest) -> LLMResponse:
    payload = {
        "provider": "codex_local",
        "role": config.role,
        "task": llm_request.task,
        "payload": llm_request.payload,
        "metadata": {
            **llm_request.metadata,
            "prompt_contract_version": config.prompt_contract_version,
            "tool_schema_version": config.tool_schema_version,
            "response_schema_version": config.response_schema_version,
        },
        "next_action": "route_through_codex_conversation_or_configure_remote_provider",
    }
    return LLMResponse(
        provider="codex_local",
        model=config.model,
        role=config.role,
        status="local_handoff",
        text=json.dumps(payload, sort_keys=True),
        raw=payload,
    )


def _headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    if provider == "gemini":
        return {"content-type": "application/json"}
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }


def _provider_payload(config: LLMRoleConfig, llm_request: LLMRequest) -> tuple[str, dict[str, Any]]:
    provider = config.provider
    messages = _messages(config, llm_request)
    endpoint = PROVIDER_ENDPOINTS[provider]
    if provider == "openai":
        return endpoint, {
            "model": config.model,
            "input": messages,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
        }
    if provider == "anthropic":
        system = next((item["content"] for item in messages if item["role"] == "system"), "")
        user_messages = [item for item in messages if item["role"] != "system"]
        return endpoint, {
            "model": config.model,
            "system": system,
            "messages": user_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
    if provider == "gemini":
        endpoint = endpoint.format(model=config.model)
        return endpoint, {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "\n\n".join(item["content"] for item in messages)}],
                }
            ],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_output_tokens,
            },
        }
    if provider == "openrouter":
        return endpoint, {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
    raise ValueError(f"unsupported LLM provider: {provider}")


def _extract_text(provider: str, raw: dict[str, Any]) -> str:
    if provider == "openai":
        if isinstance(raw.get("output_text"), str):
            return str(raw["output_text"])
        for item in raw.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    return str(content.get("text", ""))
    if provider == "anthropic":
        return "\n".join(str(item.get("text", "")) for item in raw.get("content", []) if item.get("type") == "text")
    if provider == "gemini":
        candidates = raw.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return "\n".join(str(part.get("text", "")) for part in parts)
    if provider == "openrouter":
        choices = raw.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
    return ""


class LLMProviderAdapter:
    """Small standard-library adapter for role-based LLM calls."""

    def __init__(self, config: LLMRoleConfig) -> None:
        self.config = config

    def complete(self, llm_request: LLMRequest) -> LLMResponse:
        provider = self.config.provider
        if provider == "codex_local":
            return _codex_local_response(self.config, llm_request)
        if provider not in PROVIDER_ENV_KEYS:
            raise ValueError(f"unsupported LLM provider: {provider}")
        api_key = os.getenv(PROVIDER_ENV_KEYS[provider], "").strip()
        if not api_key:
            return LLMResponse(
                provider=provider,
                model=self.config.model,
                role=self.config.role,
                status="missing_api_key",
                text="",
                error=f"missing {PROVIDER_ENV_KEYS[provider]}",
            )
        endpoint, payload = _provider_payload(self.config, llm_request)
        if provider == "gemini":
            separator = "&" if "?" in endpoint else "?"
            endpoint = f"{endpoint}{separator}key={api_key}"
        request = urlrequest.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=_headers(provider, api_key),
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=60) as response:  # noqa: S310 - user-configured API endpoints are fixed constants.
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            return LLMResponse(
                provider=provider,
                model=self.config.model,
                role=self.config.role,
                status="request_failed",
                text="",
                error=str(exc),
            )
        return LLMResponse(
            provider=provider,
            model=self.config.model,
            role=self.config.role,
            status="completed",
            text=_extract_text(provider, raw),
            raw=raw,
        )


def complete_role(role: str, task: str, payload: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> LLMResponse:
    """Complete one role task through its configured provider."""

    config = load_role_config(role)
    return LLMProviderAdapter(config).complete(
        LLMRequest(
            role=config.role,
            task=task,
            payload=payload,
            metadata=metadata or {},
        )
    )
