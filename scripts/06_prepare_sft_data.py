#!/usr/bin/env python
"""Convert generated state-tracking JSONL samples into chat SFT JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physical_state_tracking.behavior import read_jsonl, write_jsonl
from physical_state_tracking.sft_data import SFT_FORMATS, format_sft_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input generated JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="output SFT JSONL path")
    parser.add_argument(
        "--format",
        choices=SFT_FORMATS,
        default="final_only",
        help="assistant target format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input)
    examples = format_sft_examples(samples, output_format=args.format)
    write_jsonl(examples, args.output)
    print(f"Wrote {len(examples)} SFT examples to {args.output}")


if __name__ == "__main__":
    main()
