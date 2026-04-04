"""Export benchmark results into paper tables."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/papers/tables")
    args = parser.parse_args()
    print(f"paper table export stub into {args.output}")


if __name__ == "__main__":
    main()
