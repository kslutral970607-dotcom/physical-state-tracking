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
from physical_state_tracking.shells import PROMPT_VARIANTS, SHELLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--z", type=int, default=None, help="initial position, 0 through 10")
    parser.add_argument("--d", type=int, choices=(-1, 1), default=None, help="initial direction")
    parser.add_argument("--k", type=int, default=0, help="initial bounce count")
    parser.add_argument("--w", type=str, default=None, help="dummy variable, such as a color")
    parser.add_argument("--num-steps", type=int, default=6, help="number of transition steps")
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="metadata_json",
        help="prompt interface to use",
    )
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
        num_steps=args.num_steps,
        prompt_variant=args.prompt_variant,
    )
    write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
