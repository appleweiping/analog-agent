"""Minimal batch-mode ngspice smoke test with timeout and structured output."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apps.worker_simulator.ngspice_runner import run_ngspice_batch

    netlist_path = repo_root / "scripts" / "smoke" / "test_ngspice_min.cir"
    log_path = repo_root / ".artifacts" / "ngspice_logs" / "test_ngspice_min.log"
    result = run_ngspice_batch(netlist_path, log_path, timeout_sec=20)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
