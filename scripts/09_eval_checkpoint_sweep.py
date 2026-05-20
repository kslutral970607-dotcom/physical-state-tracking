#!/usr/bin/env python
"""Evaluate LoRA checkpoints across datasets and summarize shortcut patterns."""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physical_state_tracking.behavior import compute_metrics, evaluate_dataset, read_jsonl, write_jsonl
from physical_state_tracking.error_patterns import compute_error_patterns
from physical_state_tracking.hf_runner import SYSTEM_MESSAGE


class LoraCheckpointRunner:
    def __init__(self, model_name: str, checkpoint: Path, max_new_tokens: int, device: int) -> None:
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LoRA sweep dependencies are missing. Install requirements.txt, including transformers, torch, peft, and accelerate."
            ) from exc

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
    parser.add_argument("--model", required=True, help="base model name or local path")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True, help="LoRA checkpoint directories")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="dataset bindings as name=path, such as neg_ood=data/processed/neg.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV summary output path")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0+ for CUDA device")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="optional per-dataset sample limit")
    parser.add_argument("--save-predictions", action="store_true", help="save per-example JSONL prediction files")
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/sweeps"))
    return parser.parse_args()


def parse_datasets(bindings: List[str]) -> List[Tuple[str, Path]]:
    datasets = []
    for binding in bindings:
        if "=" not in binding:
            raise ValueError(f"dataset must be name=path, got: {binding}")
        name, path = binding.split("=", 1)
        if not name:
            raise ValueError(f"dataset name must be non-empty: {binding}")
        datasets.append((name, Path(path)))
    return datasets


def write_summary(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def checkpoint_label(path: Path) -> str:
    label = path.name or str(path)
    parent = path.parent.name
    if label.startswith("checkpoint-") and parent:
        label = f"{parent}__{label}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def maybe_empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    datasets = parse_datasets(args.datasets)
    loaded_datasets: Dict[str, List[dict]] = {}
    for dataset_name, dataset_path in datasets:
        samples = read_jsonl(dataset_path)
        loaded_datasets[dataset_name] = samples[: args.limit] if args.limit is not None else samples

    summary_rows = []
    for checkpoint in args.checkpoints:
        runner = LoraCheckpointRunner(
            model_name=args.model,
            checkpoint=checkpoint,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        ckpt_label = checkpoint_label(checkpoint)
        for dataset_name, dataset_path in datasets:
            samples = loaded_datasets[dataset_name]
            prediction_rows = evaluate_dataset(samples, runner.generate)
            metrics = compute_metrics(prediction_rows)
            patterns = compute_error_patterns(prediction_rows)
            row = {
                "checkpoint": ckpt_label,
                "checkpoint_path": str(checkpoint),
                "dataset": dataset_name,
                "dataset_path": str(dataset_path),
                "n_examples": len(samples),
            }
            row.update(metrics)
            row.update(patterns)
            summary_rows.append(row)

            if args.save_predictions:
                prediction_path = args.predictions_dir / ckpt_label / f"{dataset_name}.jsonl"
                write_jsonl(prediction_rows, prediction_path)
                print(f"Wrote predictions to {prediction_path}")

        del runner
        gc.collect()
        maybe_empty_cuda_cache()

    write_summary(summary_rows, args.output)
    print(f"Wrote sweep summary to {args.output}")


if __name__ == "__main__":
    main()
