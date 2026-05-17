"""Prompt shells for bouncing state-tracking samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from random import Random
from typing import Dict, List

from .state_machine import BounceStep, State, StateMachine


SHELLS = (
    "symbolic",
    "explicit_physics",
    "implicit_physics",
    "finance",
    "game",
    "adversarial",
)

COLORS = ("blue", "red", "green", "yellow")


@dataclass(frozen=True)
class Sample:
    prompt: str
    shell: str
    steps: List[dict]
    initial_state: State
    final_ground_truth_state: State
    trajectory: List[State]
    transition_rule_metadata: dict
    z: int
    d: int
    k: int
    w: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "shell": self.shell,
            "steps": self.steps,
            "initial_state": self.initial_state,
            "final_ground_truth_state": self.final_ground_truth_state,
            "trajectory": self.trajectory,
            "transition_rule_metadata": self.transition_rule_metadata,
            "z": self.z,
            "d": self.d,
            "k": self.k,
            "w": self.w,
        }


def make_sample(
    shell: str,
    rng: Random,
    z: int | None,
    d: int | None,
    k: int,
    w: str | None,
    num_steps: int,
) -> Sample:
    if shell not in SHELLS:
        raise ValueError(f"unknown shell: {shell}")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    initial_state = {
        "z": rng.randint(1, 9) if z is None else z,
        "d": rng.choice((-1, 1)) if d is None else d,
        "k": k,
        "w": rng.choice(COLORS) if w is None else w,
    }
    step_sizes = [rng.randint(1, 4) for _ in range(num_steps)]
    bounce_steps = [BounceStep(step_size=value) for value in step_sizes]
    machine = StateMachine(initial_state)
    final_state, trajectory = machine.run(bounce_steps)
    metadata = machine.metadata()
    metadata["step_sizes"] = step_sizes
    steps = [_step(shell, index, step_size) for index, step_size in enumerate(step_sizes)]

    return Sample(
        prompt=_prompt(shell, initial_state, steps, metadata),
        shell=shell,
        steps=steps,
        initial_state=initial_state,
        final_ground_truth_state=final_state,
        trajectory=trajectory,
        transition_rule_metadata=metadata,
        z=int(initial_state["z"]),
        d=int(initial_state["d"]),
        k=int(initial_state["k"]),
        w=str(initial_state["w"]),
    )


def _step(shell: str, index: int, step_size: int) -> dict:
    return {
        "index": index + 1,
        "step_size": step_size,
        "text": _describe(shell, step_size),
    }


def _describe(shell: str, step_size: int) -> str:
    if shell == "symbolic":
        return f"Apply one transition with step_size {step_size}."
    if shell == "explicit_physics":
        return f"A puck moves {step_size} units in its current direction and bounces if it reaches or crosses a wall."
    if shell == "implicit_physics":
        return f"The marker drifts {step_size} ticks along its current tendency; edge overflow turns it around."
    if shell == "finance":
        return f"The account index shifts by {step_size} risk bands; crossing a limit reflects the position and flips direction."
    if shell == "game":
        return f"The token advances {step_size} cells; overshooting an arena edge bounces it back and reverses direction."
    if shell == "adversarial":
        return (
            f"A misleading note claims color changes after moving {step_size}, "
            "but the real rule keeps w unchanged and only updates z, d, and k."
        )
    raise ValueError(f"unknown shell: {shell}")


def _prompt(shell: str, initial_state: State, steps: List[dict], metadata: dict) -> str:
    lines = [
        f"Track the bouncing state machine for the {shell} shell.",
        "State keys are exactly z, d, k, w.",
        f"Initial state JSON: {json.dumps(initial_state, sort_keys=True)}",
        f"Transition rule metadata JSON: {json.dumps(metadata, sort_keys=True)}",
        "Steps JSON:",
        json.dumps(steps, sort_keys=True),
        "Use the transition rule. Boundaries are 0 and 10. Reaching or crossing a boundary causes a bounce. The dummy variable w must remain unchanged.",
        "Return only one valid JSON object.",
        'The object must contain exactly these keys: "z", "d", "k", "w".',
        '"z" must be the computed final integer position.',
        '"d" must be the computed final direction, either 1 or -1.',
        '"k" must be the computed final bounce count.',
        '"w" must be copied unchanged from the initial state.',
        "Do not include Markdown fences.",
        "Do not include explanation.",
    ]
    return "\n".join(lines)
