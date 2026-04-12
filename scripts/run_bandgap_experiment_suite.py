"""Run the frozen bandgap v1 experiment suite."""

from __future__ import annotations

import json
from pathlib import Path

from libs.vertical_slices.bandgap import run_bandgap_experiment_suite


def main() -> None:
    suite = run_bandgap_experiment_suite(export_directory=Path("research/benchmarks"))
    print(json.dumps(suite.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
