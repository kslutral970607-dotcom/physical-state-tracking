"""Supervised fine-tuning data formatting for state-tracking samples."""

from __future__ import annotations

import json
from typing import Iterable, List, Mapping


SYSTEM_MESSAGE = "You are a precise deterministic state-machine simulator. Compute exactly."

SFT_FORMATS = ("final_only", "trajectory_then_final")


def format_sft_examples(samples: Iterable[dict], output_format: str = "final_only") -> List[dict]:
    if output_format not in SFT_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(SFT_FORMATS)}")
    return [format_sft_example(sample, output_format=output_format) for sample in samples]


def format_sft_example(sample: Mapping[str, object], output_format: str = "final_only") -> dict:
    if output_format not in SFT_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(SFT_FORMATS)}")

    prompt = str(sample["prompt"])
    final_state = _ordered_state(sample["final_ground_truth_state"])
    assistant_content = _assistant_content(sample, final_state, output_format)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "shell": sample.get("shell"),
            "steps": sample.get("steps"),
            "initial_state": sample.get("initial_state"),
            "final_ground_truth_state": final_state,
            "trajectory": sample.get("trajectory"),
            "prompt_variant": sample.get("prompt_variant"),
        },
    }


def final_state_json(state: Mapping[str, object]) -> str:
    return json.dumps(_ordered_state(state), ensure_ascii=False)


def _assistant_content(sample: Mapping[str, object], final_state: dict, output_format: str) -> str:
    final_json = final_state_json(final_state)
    if output_format == "final_only":
        return final_json

    lines = ["Trajectory:"]
    for index, state in enumerate(sample.get("trajectory", []), start=1):
        lines.append(f"Step {index}: {final_state_json(state)}")
    lines.extend(["Final:", final_json])
    return "\n".join(lines)


def _ordered_state(state: object) -> dict:
    if not isinstance(state, Mapping):
        raise ValueError("state must be a mapping")
    return {key: state[key] for key in ("z", "d", "k", "w")}
