#!/usr/bin/env python
"""Generate phase-1 synthetic state-tracking data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physical_state_tracking.dataset import generate_dataset, write_jsonl
from physical_state_tracking.shells import SHELLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--z", type=int, default=0, help="latent abstraction index")
    parser.add_argument("--d", type=int, default=4, help="transition depth")
    parser.add_argument("--k", type=int, default=3, help="number of state variables")
    parser.add_argument("--w", type=int, default=1, help="distractor width")
    parser.add_argument(
        "--shells",
        nargs="+",
        choices=SHELLS,
        default=list(SHELLS),
        help="shells to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase1/synthetic_state_tracking.jsonl"),
        help="output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = generate_dataset(
        n=args.n,
        shells=args.shells,
        seed=args.seed,
        z=args.z,
        d=args.d,
        k=args.k,
        w=args.w,
    )
    write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
