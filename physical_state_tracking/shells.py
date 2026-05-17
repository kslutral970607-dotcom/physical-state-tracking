"""Prompt shells for synthetic state-tracking samples."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Sequence

from .state_machine import Operation, StateMachine, State


SHELLS = (
    "symbolic",
    "explicit_physics",
    "implicit_physics",
    "finance",
    "game",
    "adversarial",
)


@dataclass(frozen=True)
class Sample:
    prompt: str
    shell: str
    steps: List[str]
    initial_state: State
    final_ground_truth_state: State
    z: int
    d: int
    k: int
    w: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "shell": self.shell,
            "steps": self.steps,
            "initial_state": self.initial_state,
            "final_ground_truth_state": self.final_ground_truth_state,
            "z": self.z,
            "d": self.d,
            "k": self.k,
            "w": self.w,
        }


def make_sample(shell: str, rng: Random, z: int, d: int, k: int, w: int) -> Sample:
    if shell not in SHELLS:
        raise ValueError(f"unknown shell: {shell}")

    variables = _variables(shell)
    initial_state = {name: rng.randint(1, 9) for name in variables[:k]}
    operations = _make_operations(shell, list(initial_state), rng, d, w)
    final_state = StateMachine(initial_state).apply(operations)
    steps = [operation.description for operation in operations]

    return Sample(
        prompt=_prompt(shell, initial_state, steps),
        shell=shell,
        steps=steps,
        initial_state=initial_state,
        final_ground_truth_state=final_state,
        z=z,
        d=d,
        k=k,
        w=w,
    )


def _variables(shell: str) -> Sequence[str]:
    return {
        "symbolic": ("A", "B", "C", "D"),
        "explicit_physics": ("cart", "block", "tray", "bin"),
        "implicit_physics": ("cup", "bowl", "box", "bag"),
        "finance": ("checking", "savings", "brokerage", "cash"),
        "game": ("health", "mana", "coins", "keys"),
        "adversarial": ("red", "blue", "green", "yellow"),
    }[shell]


def _make_operations(
    shell: str, variables: List[str], rng: Random, depth: int, noise_width: int
) -> List[Operation]:
    operations: List[Operation] = []
    for step_index in range(depth):
        variable = variables[step_index % len(variables)]
        delta = rng.choice((-3, -2, -1, 1, 2, 3))
        if shell == "adversarial" and step_index % 2 == 1:
            delta *= -1
        operations.append(
            Operation(
                variable=variable,
                delta=delta,
                description=_describe(shell, variable, delta, step_index, noise_width),
            )
        )
    return operations


def _describe(shell: str, variable: str, delta: int, step_index: int, noise_width: int) -> str:
    magnitude = abs(delta)
    direction = "increases" if delta > 0 else "decreases"
    distractor = f" Ignore note {step_index % max(1, noise_width)}." if noise_width else ""

    if shell == "symbolic":
        return f"{variable} {direction} by {magnitude}.{distractor}"
    if shell == "explicit_physics":
        verb = "moves forward" if delta > 0 else "moves backward"
        return f"The {variable} {verb} by {magnitude} units.{distractor}"
    if shell == "implicit_physics":
        verb = "receives" if delta > 0 else "loses"
        return f"The {variable} quietly {verb} {magnitude} tokens.{distractor}"
    if shell == "finance":
        verb = "deposit" if delta > 0 else "withdrawal"
        return f"A {verb} of {magnitude} posts to {variable}.{distractor}"
    if shell == "game":
        verb = "gains" if delta > 0 else "spends"
        return f"The player {verb} {magnitude} {variable}.{distractor}"
    if shell == "adversarial":
        apparent = "down" if delta > 0 else "up"
        true_direction = "up" if delta > 0 else "down"
        return (
            f"A misleading note says {variable} goes {apparent}, "
            f"but the rule says it actually goes {true_direction} by {magnitude}.{distractor}"
        )
    raise ValueError(f"unknown shell: {shell}")


def _prompt(shell: str, initial_state: State, steps: List[str]) -> str:
    lines = [
        f"Track the final state for the {shell} scenario.",
        f"Initial state: {initial_state}",
        "Steps:",
    ]
    lines.extend(f"{index + 1}. {step}" for index, step in enumerate(steps))
    lines.append("Return only the final state.")
    return "\n".join(lines)
