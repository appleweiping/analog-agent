"""Train a world model from a prepared dataset."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/world_model/tabular_surrogate.yaml")
    args = parser.parse_args()
    print(f"world model training stub with {args.config}")


if __name__ == "__main__":
    main()
