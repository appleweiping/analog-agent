"""Run the frozen OTA v1 acceptance path."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.vertical_slices.ota2 import run_ota_acceptance


def main() -> None:
    result = run_ota_acceptance()
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
