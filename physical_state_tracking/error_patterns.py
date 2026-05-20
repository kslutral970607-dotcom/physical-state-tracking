"""Shortcut and error-pattern metrics for state-tracking predictions."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping


REQUIRED_STATE_KEYS = {"z", "d", "k", "w"}


def compute_error_patterns(prediction_rows: Iterable[dict]) -> Dict[str, float]:
    rows = list(prediction_rows)
    if not rows:
        return {
            "pred_d_is_1_rate": 0.0,
            "pred_d_is_minus1_rate": 0.0,
            "z_plus_step_rate": 0.0,
            "z_minus_step_rate": 0.0,
            "z_noop_rate": 0.0,
            "d_copy_rate": 0.0,
            "d_flip_rate": 0.0,
            "k_copy_rate": 0.0,
            "w_copy_rate": 0.0,
            "null_or_invalid_rate": 0.0,
            "extra_keys_rate": 0.0,
        }

    return {
        "pred_d_is_1_rate": _mean(_prediction(row).get("d") == 1 for row in rows),
        "pred_d_is_minus1_rate": _mean(_prediction(row).get("d") == -1 for row in rows),
        "z_plus_step_rate": _mean(_prediction(row).get("z") == _initial(row).get("z") + _first_step_size(row) for row in rows),
        "z_minus_step_rate": _mean(_prediction(row).get("z") == _initial(row).get("z") - _first_step_size(row) for row in rows),
        "z_noop_rate": _mean(_prediction(row).get("z") == _initial(row).get("z") for row in rows),
        "d_copy_rate": _mean(_prediction(row).get("d") == _initial(row).get("d") for row in rows),
        "d_flip_rate": _mean(_prediction(row).get("d") == -_initial(row).get("d") for row in rows),
        "k_copy_rate": _mean(_prediction(row).get("k") == _initial(row).get("k") for row in rows),
        "w_copy_rate": _mean(_prediction(row).get("w") == _initial(row).get("w") for row in rows),
        "null_or_invalid_rate": _mean(_is_null_or_invalid(row) for row in rows),
        "extra_keys_rate": _mean(_has_extra_keys(row) for row in rows),
    }


def _prediction(row: Mapping[str, object]) -> dict:
    prediction = row.get("prediction")
    return prediction if isinstance(prediction, dict) else {}


def _initial(row: Mapping[str, object]) -> dict:
    initial_state = row.get("initial_state")
    if isinstance(initial_state, dict):
        return initial_state
    return {"z": row.get("z"), "d": row.get("d"), "k": row.get("k"), "w": row.get("w")}


def _first_step_size(row: Mapping[str, object]) -> int:
    steps = row.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], dict):
        return int(steps[0].get("step_size", 0))
    return 0


def _is_null_or_invalid(row: Mapping[str, object]) -> bool:
    prediction = row.get("prediction")
    if not isinstance(prediction, dict):
        return True
    return not REQUIRED_STATE_KEYS.issubset(prediction.keys())


def _has_extra_keys(row: Mapping[str, object]) -> bool:
    parsed = row.get("parsed_output")
    if not isinstance(parsed, dict):
        return False
    return bool(set(parsed.keys()) - REQUIRED_STATE_KEYS)


def _mean(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0
