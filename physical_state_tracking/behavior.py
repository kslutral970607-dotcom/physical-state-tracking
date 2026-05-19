"""Behavior evaluation utilities for state-tracking model outputs."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional


PredictionFn = Callable[[str], str]


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_json_object(text: str) -> Optional[dict]:
    """Extract the final object-like JSON payload from model text."""

    decoder = json.JSONDecoder()
    parsed_objects = []
    for start in _candidate_starts(text):
        candidate = text[start:].strip()
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            parsed = _parse_python_literal(candidate)
        if isinstance(parsed, dict):
            parsed_objects.append(parsed)
    return parsed_objects[-1] if parsed_objects else None


def evaluate_dataset(samples: Iterable[dict], predict: PredictionFn) -> List[dict]:
    rows = []
    for index, sample in enumerate(samples):
        prompt = sample["prompt"]
        raw_output = predict(prompt)
        parsed = parse_json_object(raw_output)
        expected = sample["final_ground_truth_state"]
        prediction = _normalize_state(parsed) if parsed is not None else None

        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "steps": sample["steps"],
                "trajectory": sample["trajectory"],
                "transition_rule_metadata": sample["transition_rule_metadata"],
                "prompt_variant": sample.get("prompt_variant"),
                "shell": sample["shell"],
                "z": sample["z"],
                "d": sample["d"],
                "k": sample["k"],
                "w": sample["w"],
                "initial_state": sample["initial_state"],
                "final_ground_truth_state": expected,
                "raw_output": raw_output,
                "parsed_output": _json_safe(parsed),
                "prediction": prediction,
                "json_valid": parsed is not None,
                "exact_match": prediction == expected,
            }
        )
    return rows


def compute_metrics(prediction_rows: Iterable[dict]) -> Dict[str, float]:
    rows = list(prediction_rows)
    return {
        "exact_match": _mean(row["exact_match"] for row in rows),
        "z_accuracy": _field_accuracy(rows, "z"),
        "d_accuracy": _field_accuracy(rows, "d"),
        "k_accuracy": _field_accuracy(rows, "k"),
        "w_preservation": _field_accuracy(rows, "w"),
        "json_validity": _mean(row["json_valid"] for row in rows),
    }


def write_metrics_csv(metrics: Mapping[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow({"metric": metric, "value": f"{value:.6f}"})


def _candidate_starts(text: str) -> List[int]:
    starts = [index for index, char in enumerate(text) if char == "{"]
    return starts if starts else [0]


def _parse_python_literal(candidate: str) -> Optional[dict]:
    end = _matching_brace_end(candidate)
    if end is None:
        return None
    try:
        parsed = ast.literal_eval(candidate[:end])
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _matching_brace_end(text: str) -> Optional[int]:
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _normalize_state(parsed: dict) -> dict:
    for wrapper_key in ("final_state", "final_ground_truth_state", "state", "answer"):
        if wrapper_key in parsed and isinstance(parsed[wrapper_key], dict):
            parsed = parsed[wrapper_key]
            break
    return {str(key): _normalize_value(value) for key, value in parsed.items()}


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _normalize_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _mean(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _field_accuracy(rows: List[dict], key: str) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        prediction = row.get("prediction")
        expected = row.get("final_ground_truth_state", {})
        if isinstance(prediction, dict) and prediction.get(key) == expected.get(key):
            correct += 1
    return correct / len(rows)
