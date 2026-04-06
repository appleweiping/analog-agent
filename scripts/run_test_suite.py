"""Run repository test suites with optional dependency gating."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REQUIRED_API_DEPENDENCIES = ("fastapi", "httpx")


def _missing_dependencies() -> list[str]:
    return [name for name in REQUIRED_API_DEPENDENCIES if importlib.util.find_spec(name) is None]


def _is_venv_python() -> bool:
    return Path(sys.executable).resolve().parts[-3:] == (".venv", "Scripts", "python.exe") or ".venv" in Path(sys.executable).resolve().parts


def _print_environment_summary() -> None:
    print(f"Python executable: {sys.executable}")
    pip_version = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_version.returncode == 0 and pip_version.stdout.strip():
        print(f"pip target: {pip_version.stdout.strip()}")
    else:
        print("pip target: unavailable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unittest suites for analog-agent.")
    parser.add_argument(
        "--require-api-deps",
        action="store_true",
        help="Fail fast when FastAPI/httpx are unavailable instead of letting tests skip.",
    )
    parser.add_argument(
        "--require-venv",
        action="store_true",
        help="Fail fast unless tests are executed from the repository .venv interpreter.",
    )
    args = parser.parse_args()

    _print_environment_summary()

    if args.require_venv and not _is_venv_python():
        print(
            "Expected to run from the repository .venv interpreter. "
            "Activate .venv or use `.\\scripts\\run_test_suite.ps1 -UseVenv`.",
            file=sys.stderr,
        )
        return 1

    missing = _missing_dependencies()
    if args.require_api_deps and missing:
        print(
            "Missing required API test dependencies: "
            + ", ".join(missing)
            + ". Activate .venv and install with `py -3.12 -m pip install -e \".[dev]\"` "
            + "or run `.\\scripts\\bootstrap_dev_env.ps1`.",
            file=sys.stderr,
        )
        return 1

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
