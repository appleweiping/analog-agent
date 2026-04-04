"""Replay a logged design trace stub."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="research/datasets/online_traces/example.json")
    args = parser.parse_args()
    print(f"trace replay stub for {args.trace}")


if __name__ == "__main__":
    main()
