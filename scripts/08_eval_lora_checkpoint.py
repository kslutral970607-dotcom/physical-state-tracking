#!/usr/bin/env python
"""Evaluate a LoRA adapter checkpoint with the existing behavior metrics."""

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
from physical_state_tracking.hf_runner import SYSTEM_MESSAGE


class LoraCheckpointRunner:
    def __init__(self, model_name: str, checkpoint: Path, max_new_tokens: int, device: int) -> None:
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LoRA evaluation dependencies are missing. Install requirements.txt, including transformers, torch, peft, and accelerate."
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self._model = PeftModel.from_pretrained(base_model, checkpoint)
        if device >= 0:
            self._model.to(f"cuda:{device}")
        self._model.eval()

    def generate(self, prompt: str) -> str:
        request = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(request, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        input_token_count = inputs["input_ids"].shape[-1]
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        completion_ids = output_ids[0][input_token_count:]
        return self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="base model name")
    parser.add_argument("--checkpoint", type=Path, required=True, help="LoRA adapter checkpoint directory")
    parser.add_argument("--input", type=Path, required=True, help="generated behavior JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="prediction JSONL output path")
    parser.add_argument("--metrics", type=Path, required=True, help="metrics CSV output path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0+ for CUDA device")
    parser.add_argument("--limit", type=int, default=None, help="optional sample limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input)
    if args.limit is not None:
        samples = samples[: args.limit]

    runner = LoraCheckpointRunner(
        model_name=args.model,
        checkpoint=args.checkpoint,
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
