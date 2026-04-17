"""Audit whether the frozen benchmark suite exposes a parity-aligned baseline roster."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.benchmark_protocol import BASELINE_BENCHMARK_MODES, BASELINE_MODE_NARRATIVES
from libs.eval.benchmark_registry import list_benchmark_definitions, load_benchmark_suite_definition


def build_status() -> dict[str, object]:
    suite = load_benchmark_suite_definition()
    benchmarks = list_benchmark_definitions()
    supported_modes = set(suite.supported_modes)
    expected_modes = set(BASELINE_BENCHMARK_MODES)
    missing_modes = [mode for mode in BASELINE_BENCHMARK_MODES if mode not in supported_modes]
    all_frozen_runnable = all(item.execution_readiness == "frozen_runnable" for item in benchmarks)
    narrative_ready = all(mode in BASELINE_MODE_NARRATIVES for mode in BASELINE_BENCHMARK_MODES)
    parity_ready = all_frozen_runnable and not missing_modes and narrative_ready
    return {
        "stage": "Stage E",
        "audit": "baseline_parity",
        "parity_status": "baseline_parity_ready" if parity_ready else "baseline_parity_incomplete",
        "all_benchmarks_frozen_runnable": all_frozen_runnable,
        "expected_baseline_modes": list(BASELINE_BENCHMARK_MODES),
        "supported_modes": suite.supported_modes,
        "missing_supported_modes": missing_modes,
        "baseline_mode_narratives_ready": narrative_ready,
        "benchmark_ids": [item.benchmark_id for item in benchmarks],
        "ready_for_common_protocol_confirmation": parity_ready,
        "notes": [
            "Baseline parity here means the frozen benchmark suite advertises the same baseline roster that the runnable vertical slices are expected to support.",
            "This parity audit checks contract-level roster fairness and does not claim that every baseline has already been rerun for every benchmark in the current environment.",
        ],
    }


def main() -> None:
    print(json.dumps(build_status(), indent=2))


if __name__ == "__main__":
    main()
