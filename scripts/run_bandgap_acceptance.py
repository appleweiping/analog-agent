"""Run the frozen bandgap v1 acceptance path."""

from __future__ import annotations

import json

from libs.vertical_slices.bandgap import run_bandgap_acceptance


def main() -> None:
    result = run_bandgap_acceptance()
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
