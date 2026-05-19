"""Dataset generation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence

from .shells import PROMPT_VARIANTS, SHELLS, make_sample


def generate_dataset(
    n: int,
    shells: Sequence[str] = SHELLS,
    seed: int = 0,
    z: int | None = None,
    d: int | None = None,
    k: int = 0,
    w: str | None = None,
    num_steps: int = 6,
    prompt_variant: str = "metadata_json",
) -> List[dict]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not shells:
        raise ValueError("shells must not be empty")
    if z is not None and not 0 <= z <= 10:
        raise ValueError("z must be between 0 and 10")
    if d is not None and d not in (-1, 1):
        raise ValueError("d must be either -1 or 1")
    if k < 0:
        raise ValueError("k must be non-negative")
    if w is not None and not w:
        raise ValueError("w must be a non-empty string")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant must be one of: {', '.join(PROMPT_VARIANTS)}")

    rng = Random(seed)
    samples = []
    for index in range(n):
        shell = shells[index % len(shells)]
        samples.append(
            make_sample(
                shell=shell,
                rng=rng,
                z=z,
                d=d,
                k=k,
                w=w,
                num_steps=num_steps,
                prompt_variant=prompt_variant,
            ).to_dict()
        )
    return samples


def write_jsonl(samples: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
