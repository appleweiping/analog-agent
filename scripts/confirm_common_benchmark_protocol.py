"""Confirm that frozen runnable benchmarks share one explicit common execution protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.benchmark_protocol import benchmark_protocol_contract
from libs.eval.benchmark_registry import list_benchmark_definitions, load_benchmark_suite_definition


def build_status() -> dict[str, object]:
    suite = load_benchmark_suite_definition()
    benchmarks = list_benchmark_definitions()
    protocol = benchmark_protocol_contract()
    backend_values = sorted({item.execution_defaults.backend_preference for item in benchmarks})
    truth_values = sorted({item.execution_defaults.truth_level for item in benchmarks})
    model_values = sorted({item.execution_defaults.model_type for item in benchmarks})
    promoted_fidelity_values = sorted({item.execution_defaults.promoted_fidelity for item in benchmarks})
    required_reporting_axes = {"task_family", "truth_level", "fidelity_level", "simulation_budget", "failure_mode_distribution"}
    required_template_keys = {"netlist_template", "quick_truth_testbench", "focused_truth_testbench", "measurement_contract"}
    reporting_axes_ready = required_reporting_axes.issubset(set(suite.reporting_axes))
    template_contract_ready = all(required_template_keys.issubset(set(item.intended_templates)) for item in benchmarks)
    protocol_ready = (
        len(backend_values) == 1
        and len(truth_values) == 1
        and len(model_values) == 1
        and len(promoted_fidelity_values) == 1
        and reporting_axes_ready
        and template_contract_ready
    )
    return {
        "stage": "Stage E",
        "audit": "common_protocol",
        "protocol_status": "common_protocol_confirmed" if protocol_ready else "common_protocol_incomplete",
        "backend_preference_values": backend_values,
        "truth_level_values": truth_values,
        "model_type_values": model_values,
        "promoted_fidelity_values": promoted_fidelity_values,
        "reporting_axes_ready": reporting_axes_ready,
        "template_contract_ready": template_contract_ready,
        "default_protocol": protocol,
        "ready_for_baseline_narrative": protocol_ready,
        "notes": [
            "Common protocol means the benchmark suite shares one explicit execution contract for backend, truth level, promoted fidelity, reporting axes, and template expectations.",
            "This script confirms contract-level consistency and does not substitute for later benchmark result synthesis.",
        ],
    }


def main() -> None:
    print(json.dumps(build_status(), indent=2))


if __name__ == "__main__":
    main()
