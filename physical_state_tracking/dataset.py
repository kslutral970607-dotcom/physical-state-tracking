"""Dataset generation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence

from .shells import SHELLS, make_sample


def generate_dataset(
    n: int,
    shells: Sequence[str] = SHELLS,
    seed: int = 0,
    z: int = 0,
    d: int = 4,
    k: int = 3,
    w: int = 1,
) -> List[dict]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not shells:
        raise ValueError("shells must not be empty")
    if d < 1:
        raise ValueError("d must be at least 1")
    if k < 1 or k > 4:
        raise ValueError("k must be between 1 and 4")
    if w < 0:
        raise ValueError("w must be non-negative")

    rng = Random(seed)
    samples = []
    for index in range(n):
        shell = shells[index % len(shells)]
        samples.append(make_sample(shell=shell, rng=rng, z=z, d=d, k=k, w=w).to_dict())
    return samples


def write_jsonl(samples: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
