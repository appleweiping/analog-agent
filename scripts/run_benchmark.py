"""Run a benchmark experiment stub."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="ota2")
    args = parser.parse_args()
    print(f"benchmark run stub for {args.benchmark}")


if __name__ == "__main__":
    main()
