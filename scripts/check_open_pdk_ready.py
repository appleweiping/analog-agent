"""Check whether the local open-PDK layout is ready for configured-truth work."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.worker_simulator.ngspice_runner import configured_pdk_root, external_model_card_path


def _read_contract() -> dict[str, object]:
    contract_path = Path(__file__).resolve().parents[1] / "configs" / "pdk" / "sky130_open.yaml"
    contract: dict[str, object] = {}
    for line in contract_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if value:
            contract[key.strip()] = value
    return contract


def build_status() -> dict[str, object]:
    root = configured_pdk_root()
    contract = _read_contract()
    required = [
        "libs.tech/ngspice",
        "libs.ref/sky130_fd_pr/spice",
    ]
    checks = []
    for relpath in required:
        target = root / relpath if root else None
        checks.append(
            {
                "path": relpath,
                "present": bool(target and target.exists()),
            }
        )
    model_card = external_model_card_path()
    return {
        "ready": bool(root and all(item["present"] for item in checks)),
        "pdk_root": str(root) if root else "",
        "expected_root_hint": contract.get("expected_root_hint", "/pdk/sky130A"),
        "required_checks": checks,
        "external_model_card": str(model_card) if model_card else "",
    }


if __name__ == "__main__":
    print(json.dumps(build_status(), indent=2))
