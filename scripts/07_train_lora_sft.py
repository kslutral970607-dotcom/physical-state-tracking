#!/usr/bin/env python
"""Train a PEFT LoRA adapter on chat-formatted state-tracking SFT data."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="base model name")
    parser.add_argument("--train-data", type=Path, required=True, help="chat SFT train JSONL")
    parser.add_argument("--eval-data", type=Path, default=None, help="optional chat SFT eval JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="adapter output directory")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="steps")
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default=None)
    parser.add_argument(
        "--save-at-steps",
        default=None,
        help="comma-separated custom adapter checkpoint steps, such as 5,10,20,50,75,100,150,200",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ChatSftDataset:
    def __init__(self, rows: List[dict], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        messages = self.rows[index]["messages"]
        prompt_messages = messages[:-1]
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }


@dataclass
class DataCollator:
    tokenizer: object

    def __call__(self, features: List[dict]) -> dict:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class SaveAtStepsCallbackBase:
    def __init__(self, steps: Set[int], output_dir: Path, tokenizer) -> None:
        self.steps = steps
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.saved_steps: Set[int] = set()

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step not in self.steps or step in self.saved_steps:
            return control
        model = kwargs["model"]
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        self.saved_steps.add(step)
        print(f"Saved custom LoRA checkpoint at step {step}: {checkpoint_dir}")
        return control


def parse_save_at_steps(value: str | None) -> Set[int]:
    if not value:
        return set()
    steps = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        step = int(part)
        if step < 1:
            raise ValueError("--save-at-steps values must be positive integers")
        steps.add(step)
    return steps


def main() -> None:
    args = parse_args()
    protected_output = Path("outputs/lora_qwen15b_shortcut_dpos_only")
    if args.output_dir == protected_output or args.output_dir.resolve() == (REPO_ROOT / protected_output).resolve():
        raise RuntimeError(
            "Refusing to train into outputs/lora_qwen15b_shortcut_dpos_only. "
            "Keep the shortcut checkpoint preserved and choose a new --output-dir."
        )
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "LoRA SFT dependencies are missing. Install requirements.txt, including transformers, torch, peft, and accelerate."
        ) from exc

    class SaveAtStepsCallback(SaveAtStepsCallbackBase, TrainerCallback):
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSftDataset(read_jsonl(args.train_data), tokenizer, args.max_length)
    eval_dataset = ChatSftDataset(read_jsonl(args.eval_data), tokenizer, args.max_length) if args.eval_data else None

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
        "save_total_limit": args.save_total_limit,
        "save_strategy": args.save_strategy,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": [],
        "remove_unused_columns": False,
    }
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    eval_strategy = args.eval_strategy if args.eval_strategy is not None else ("steps" if eval_dataset is not None else "no")
    if eval_dataset is None and eval_strategy != "no":
        raise ValueError("--eval-strategy requires --eval-data unless it is 'no'")
    training_kwargs[strategy_key] = eval_strategy
    training_args = TrainingArguments(**training_kwargs)
    callbacks = []
    custom_save_steps = parse_save_at_steps(args.save_at_steps)
    if custom_save_steps:
        callbacks.append(SaveAtStepsCallback(custom_save_steps, args.output_dir, tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
