#!/usr/bin/env python3
"""Gera sete variações sutis por classe para um experimento por categoria."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.offline_variants import generate_category_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/landmarks")
    parser.add_argument("--output", default="data/augmented/escola_10")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--categories", nargs="+", default=["escola"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = generate_category_variants(
        args.source, args.output, args.annotations, args.categories, seed=args.seed,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "class_counts"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
