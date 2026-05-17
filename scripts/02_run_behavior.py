#!/usr/bin/env python
"""Run Phase 2 behavior evaluation with a HuggingFace model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physical_state_tracking.behavior import (
    compute_metrics,
    evaluate_dataset,
    read_jsonl,
    write_jsonl,
    write_metrics_csv,
)
from physical_state_tracking.hf_runner import HuggingFaceRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/dev.jsonl"),
        help="input dataset JSONL path",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/behavior_predictions.jsonl"),
        help="raw prediction output JSONL path",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/behavior_metrics.csv"),
        help="metrics CSV output path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0+ for CUDA device")
    parser.add_argument("--limit", type=int, default=None, help="optional sample limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input)
    if args.limit is not None:
        samples = samples[: args.limit]

    runner = HuggingFaceRunner(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    rows = evaluate_dataset(samples, runner.generate)
    metrics = compute_metrics(rows)

    write_jsonl(rows, args.predictions)
    write_metrics_csv(metrics, args.metrics)

    print(f"Wrote {len(rows)} predictions to {args.predictions}")
    print(f"Wrote metrics to {args.metrics}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")


if __name__ == "__main__":
    main()
