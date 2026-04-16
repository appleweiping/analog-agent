"""Validate the container-first ngspice runtime contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised via fallback path
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from apps.worker_simulator.ngspice_runner import _load_ngspice_config


def _parse_compose_without_yaml(text: str) -> dict[str, object]:
    """Parse the narrow docker-compose shape used by this repo without PyYAML."""

    services: dict[str, object] = {}
    current_service: str | None = None
    current_section: str | None = None
    in_command = False
    command_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent <= 1 and stripped == "services:":
            current_section = None
            continue
        if indent == 2 and stripped.endswith(":"):
            current_service = stripped[:-1]
            services[current_service] = {}
            current_section = None
            in_command = False
            command_lines = []
            continue
        if not current_service:
            continue

        service = services[current_service]
        assert isinstance(service, dict)

        if in_command:
            if indent >= 6:
                command_lines.append(stripped)
                service["command"] = " ".join(command_lines).strip()
                continue
            in_command = False

        if indent == 4 and stripped.startswith("command:"):
            in_command = True
            command_lines = []
            service["command"] = ""
            current_section = None
            continue

        if indent == 4 and stripped.endswith(":"):
            key = stripped[:-1]
            current_section = key
            if key in {"environment", "build"}:
                service[key] = {}
            elif key in {"volumes", "ports"}:
                service[key] = []
            else:
                service[key] = {}
            continue

        if indent == 4 and ":" in stripped and current_section is None:
            key, value = stripped.split(":", 1)
            service[key.strip()] = value.strip().strip('"')
            continue

        if indent >= 6 and current_section == "environment" and ":" in stripped:
            key, value = stripped.split(":", 1)
            env = service.setdefault("environment", {})
            assert isinstance(env, dict)
            env[key.strip()] = value.strip().strip('"')
            continue

        if indent >= 6 and current_section == "build" and ":" in stripped:
            key, value = stripped.split(":", 1)
            build = service.setdefault("build", {})
            assert isinstance(build, dict)
            build[key.strip()] = value.strip().strip('"')
            continue

        if indent >= 6 and current_section in {"volumes", "ports"} and stripped.startswith("- "):
            items = service.setdefault(current_section, [])
            assert isinstance(items, list)
            items.append(stripped[2:].strip().strip('"'))
            continue

    return {"services": services}


def _load_compose(compose_path: Path) -> dict[str, object]:
    text = compose_path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, dict):
            return loaded
    return _parse_compose_without_yaml(text)


def _docker_available() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if not docker:
        return False, ""
    return True, docker


def _probe_compose_services(repo_root: Path) -> dict[str, object]:
    available, docker_bin = _docker_available()
    if not available:
        return {"compose_available": False, "services": [], "error": "docker_not_installed"}
    try:
        result = subprocess.run(
            [docker_bin, "compose", "config", "--services"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"compose_available": False, "services": [], "error": "docker_compose_probe_failed"}
    services = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "compose_available": result.returncode == 0,
        "services": services,
        "error": "" if result.returncode == 0 else (result.stderr.strip() or "docker_compose_config_failed"),
    }


def build_status() -> dict[str, object]:
    config = _load_ngspice_config()
    compose_path = REPO_ROOT / "docker-compose.yml"
    dockerfile_path = REPO_ROOT / "infra" / "docker" / "Dockerfile.ngspice"
    compose = _load_compose(compose_path)
    services = compose.get("services", {})
    service_name = str(config.get("container_service_name", "api"))
    service = services.get(service_name, {})
    environment = service.get("environment", {}) or {}
    volumes = service.get("volumes", []) or []
    command = str(service.get("command", "")).strip()
    dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
    docker_probe = _probe_compose_services(REPO_ROOT)

    required_env = {
        "ANALOG_AGENT_NGSPICE_BIN": str(config.get("container_ngspice_bin", "/usr/bin/ngspice")),
        "ANALOG_AGENT_PDK_ROOT": str(config.get("container_pdk_root", "/pdk/sky130A")),
    }
    env_checks = {
        key: str(environment.get(key, "")).strip() == expected
        for key, expected in required_env.items()
    }
    volume_checks = {
        "workspace_mount_present": any(str(item).startswith(".:/workspace") for item in volumes),
        "pdk_mount_present": any("/pdk/sky130A" in str(item) for item in volumes),
    }
    dockerfile_checks = {
        "installs_ngspice": " ngspice " in f" {dockerfile_text} ",
        "sets_ngspice_env": "ANALOG_AGENT_NGSPICE_BIN=/usr/bin/ngspice" in dockerfile_text,
        "workdir_matches": f"WORKDIR {config.get('container_workdir', '/workspace')}" in dockerfile_text,
    }
    command_checks = {
        "installs_project_editable": "pip install -e ." in command,
        "launches_uvicorn": "uvicorn apps.api_server.main:app" in command,
    }
    expected_service_present = service_name in services
    ready = (
        expected_service_present
        and all(env_checks.values())
        and all(volume_checks.values())
        and all(dockerfile_checks.values())
        and all(command_checks.values())
    )

    return {
        "ready": ready,
        "compose_path": str(compose_path),
        "dockerfile_path": str(dockerfile_path),
        "service_name": service_name,
        "service_present": expected_service_present,
        "docker_cli_available": docker_probe["compose_available"] or docker_probe["error"] != "docker_not_installed",
        "docker_probe": docker_probe,
        "env_checks": env_checks,
        "volume_checks": volume_checks,
        "dockerfile_checks": dockerfile_checks,
        "command_checks": command_checks,
        "container_contract": {
            "workdir": str(config.get("container_workdir", "/workspace")),
            "ngspice_bin": str(config.get("container_ngspice_bin", "/usr/bin/ngspice")),
            "pdk_root": str(config.get("container_pdk_root", "/pdk/sky130A")),
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_status(), indent=2))
