"""Build submission-facing multi-task benchmark rollup tables."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.eval.benchmark_reporting import build_multitask_rollup_bundle


def main() -> None:
    bundle = build_multitask_rollup_bundle(output_root=Path("research/papers/benchmark_rollup"))
    print(json.dumps(bundle.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
