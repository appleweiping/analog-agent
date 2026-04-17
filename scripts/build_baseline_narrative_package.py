"""Build a submission-facing baseline narrative package from the benchmark contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.benchmark_protocol import BASELINE_BENCHMARK_MODES, BASELINE_MODE_NARRATIVES, benchmark_protocol_contract
from libs.eval.benchmark_registry import list_benchmark_definitions, load_benchmark_suite_definition


def build_package() -> dict[str, object]:
    suite = load_benchmark_suite_definition()
    benchmarks = list_benchmark_definitions()
    benchmark_roles = {
        item.benchmark_id: {
            "family": item.family,
            "role": item.benchmark_role,
            "truth_level": item.execution_defaults.truth_level,
        }
        for item in benchmarks
    }
    narrative_sections = {
        mode: {
            "mode": mode,
            "narrative": BASELINE_MODE_NARRATIVES[mode],
            "fairness_note": "Uses the shared frozen benchmark contract and common simulation budget.",
        }
        for mode in BASELINE_BENCHMARK_MODES
    }
    summary_notes = [
        f"primary_benchmark={suite.primary_benchmark_id}",
        "full_simulation_baseline should be framed as an upper-cost reference, not as a scalable search policy.",
        "bayesopt/cmaes/rl baselines should be described honestly as lightweight internal baselines rather than production-strength external reimplementations.",
        "top_k and no_world_model baselines are important because they isolate whether planner/world-model structure beats simpler in-house alternatives.",
    ]
    return {
        "package_id": "baseline_narrative_package_v1",
        "suite_id": suite.suite_id,
        "baseline_modes": list(BASELINE_BENCHMARK_MODES),
        "benchmark_roles": benchmark_roles,
        "protocol": benchmark_protocol_contract(),
        "narrative_sections": narrative_sections,
        "summary_notes": summary_notes,
    }


def export_package(output_root: str | Path) -> dict[str, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = build_package()
    json_path = root / "baseline_narrative_package.json"
    markdown_path = root / "baseline_narrative_package.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Baseline Narrative Package",
        "",
        f"- Suite: `{payload['suite_id']}`",
        f"- Baseline modes: `{', '.join(payload['baseline_modes'])}`",
        "",
        "## Baseline Narratives",
        "",
    ]
    for mode in payload["baseline_modes"]:
        section = payload["narrative_sections"][mode]
        lines.extend(
            [
                f"### `{mode}`",
                "",
                f"- Narrative: {section['narrative']}",
                f"- Fairness note: {section['fairness_note']}",
                "",
            ]
        )
    lines.extend(["## Notes", "", *[f"- {note}" for note in payload["summary_notes"]]])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json_output_path": str(json_path), "markdown_output_path": str(markdown_path)}


def main() -> None:
    outputs = export_package(Path("research/papers/baseline_narrative"))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
