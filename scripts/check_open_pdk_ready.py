"""Check whether the local open-PDK layout is ready for configured-truth work."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised via fallback path
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.worker_simulator.ngspice_runner import configured_pdk_root, external_model_card_path


def _parse_contract_without_yaml(text: str) -> dict[str, object]:
    """Parse the narrow SKY130 contract file shape without PyYAML."""

    data: dict[str, object] = {}
    current_list_key: str | None = None
    current_multiline_key: str | None = None
    multiline_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if current_multiline_key is not None:
            if indent >= 2:
                multiline_lines.append(line[2:] if line.startswith("  ") else stripped)
                data[current_multiline_key] = "\n".join(multiline_lines).rstrip()
                continue
            current_multiline_key = None
            multiline_lines = []

        if stripped.endswith(":") and ":" not in stripped[:-1]:
            current_list_key = stripped[:-1]
            data[current_list_key] = []
            continue

        if current_list_key and stripped.startswith("- "):
            items = data.setdefault(current_list_key, [])
            assert isinstance(items, list)
            items.append(stripped[2:].strip().strip('"'))
            continue
        current_list_key = None

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "|":
                current_multiline_key = key
                multiline_lines = []
                data[key] = ""
                continue
            lowered = value.lower()
            if lowered in {"true", "false"}:
                data[key] = lowered == "true"
            else:
                data[key] = value.strip('"')

    return data


def _read_contract() -> dict[str, object]:
    contract_path = Path(__file__).resolve().parents[1] / "configs" / "pdk" / "sky130_open.yaml"
    text = contract_path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = _parse_contract_without_yaml(text)
    if not isinstance(loaded, dict):
        raise ValueError("sky130_open.yaml must deserialize to a mapping")
    return loaded


def _readiness_state(*, root_present: bool, all_present: bool, any_present: bool) -> str:
    if not root_present:
        return "missing_root"
    if all_present:
        return "ready"
    if any_present:
        return "partial"
    return "missing_required_subpaths"


def build_status() -> dict[str, object]:
    root = configured_pdk_root()
    contract = _read_contract()
    required = [str(item) for item in contract.get("required_subpaths", [])]
    checks = []
    for relpath in required:
        target = root / relpath if root else None
        checks.append(
            {
                "path": relpath,
                "present": bool(target and target.exists()),
                "resolved_path": str(target) if target else "",
            }
        )
    model_card = external_model_card_path()
    root_present = bool(root and root.exists())
    all_present = bool(root_present and checks and all(item["present"] for item in checks))
    any_present = any(item["present"] for item in checks)
    readiness_state = _readiness_state(root_present=root_present, all_present=all_present, any_present=any_present)
    missing = [item["path"] for item in checks if not item["present"]]
    recommended_actions: list[str] = []
    if not root_present:
        recommended_actions.append(
            f"stage_or_mount_pdk_root_at:{contract.get('expected_root_hint', '/pdk/sky130A')}"
        )
    if missing:
        recommended_actions.extend(f"populate_required_subpath:{path}" for path in missing)
    if bool(contract.get("version_lock_required", False)):
        recommended_actions.append(f"record_version_lock:{contract.get('version', 'user_managed')}")
        recommended_actions.append(f"record_checksum_manifest:{contract.get('checksum_manifest_hint', '.pdk/sky130A.sha256')}")
    if model_card is None:
        recommended_actions.append("external_model_card_optional_but_recommended")
    else:
        recommended_actions.append("external_model_card_present")

    return {
        "ready": all_present,
        "readiness_state": readiness_state,
        "contract_name": contract.get("name", "sky130_open"),
        "distribution": contract.get("distribution", ""),
        "version": contract.get("version", ""),
        "version_lock_required": bool(contract.get("version_lock_required", False)),
        "checksum_policy": contract.get("checksum_policy", ""),
        "checksum_manifest_hint": contract.get("checksum_manifest_hint", ""),
        "readiness_policy": contract.get("readiness_policy", ""),
        "pdk_root": str(root) if root else "",
        "pdk_root_present": root_present,
        "expected_root_hint": contract.get("expected_root_hint", "/pdk/sky130A"),
        "required_checks": checks,
        "missing_required_subpaths": missing,
        "external_model_card": str(model_card) if model_card else "",
        "external_model_card_present": bool(model_card and model_card.exists()),
        "truth_level_when_ready": contract.get("truth_level_when_present", "configured_truth_candidate"),
        "validation_state_when_ready": contract.get("validation_state_when_present", "strong_if_external_model_present"),
        "validation_state_when_partial": contract.get("validation_state_when_partial", "weak"),
        "validation_state_when_missing": contract.get("validation_state_when_missing", "invalid"),
        "recommended_mount_hint": contract.get("recommended_mount_hint", ""),
        "recommended_actions": recommended_actions,
    }


if __name__ == "__main__":
    print(json.dumps(build_status(), indent=2))
