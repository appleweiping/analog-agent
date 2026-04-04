"""Build a dataset artifact for surrogate training."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="research/datasets/offline_lhs")
    args = parser.parse_args()
    print(f"dataset build stub from {args.source}")


if __name__ == "__main__":
    main()
