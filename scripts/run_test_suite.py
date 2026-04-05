"""Run repository test suites with optional dependency gating."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

REQUIRED_API_DEPENDENCIES = ("fastapi", "httpx")


def _missing_dependencies() -> list[str]:
    return [name for name in REQUIRED_API_DEPENDENCIES if importlib.util.find_spec(name) is None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unittest suites for analog-agent.")
    parser.add_argument(
        "--require-api-deps",
        action="store_true",
        help="Fail fast when FastAPI/httpx are unavailable instead of letting tests skip.",
    )
    args = parser.parse_args()

    missing = _missing_dependencies()
    if args.require_api_deps and missing:
        print(
            "Missing required API test dependencies: "
            + ", ".join(missing)
            + ". Activate .venv and install with `pip install -e \".[dev]\"`.",
            file=sys.stderr,
        )
        return 1

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
