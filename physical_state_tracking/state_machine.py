"""Bouncing state machine used by all dataset shells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


State = Dict[str, object]

LOWER_BOUNDARY = 0
UPPER_BOUNDARY = 10


@dataclass(frozen=True)
class BounceStep:
    step_size: int


class StateMachine:
    """Apply bounded one-dimensional motion with reflection."""

    def __init__(
        self,
        initial_state: Mapping[str, object],
        lower_boundary: int = LOWER_BOUNDARY,
        upper_boundary: int = UPPER_BOUNDARY,
    ) -> None:
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.initial_state = self._validate_state(initial_state)

    def apply(self, steps: List[BounceStep]) -> State:
        state, _ = self.run(steps)
        return state

    def run(self, steps: List[BounceStep]) -> tuple[State, List[State]]:
        state = dict(self.initial_state)
        trajectory = []
        for step in steps:
            state = self.transition(state, step.step_size)
            trajectory.append(dict(state))
        return state, trajectory

    def transition(self, state: Mapping[str, object], step_size: int) -> State:
        if step_size < 1:
            raise ValueError("step_size must be positive")

        current = self._validate_state(state)
        proposed_z = int(current["z"]) + step_size * int(current["d"])
        next_z = proposed_z
        next_d = int(current["d"])
        next_k = int(current["k"])

        if proposed_z <= self.lower_boundary:
            next_z = self.lower_boundary + (self.lower_boundary - proposed_z)
            next_d *= -1
            next_k += 1
        elif proposed_z >= self.upper_boundary:
            next_z = self.upper_boundary - (proposed_z - self.upper_boundary)
            next_d *= -1
            next_k += 1

        if not self.lower_boundary <= next_z <= self.upper_boundary:
            raise ValueError("reflected position escaped the boundaries")

        return {"z": next_z, "d": next_d, "k": next_k, "w": current["w"]}

    def metadata(self) -> dict:
        return {
            "state_variables": ["z", "d", "k", "w"],
            "boundaries": {"lower": self.lower_boundary, "upper": self.upper_boundary},
            "rule": (
                "Each step proposes z_next = z + step_size * d. "
                "If z_next reaches or crosses 0 or 10, reflect it back into range, "
                "reverse d, and increment k. Otherwise update z normally. "
                "w is causally irrelevant and remains unchanged."
            ),
        }

    def _validate_state(self, state: Mapping[str, object]) -> State:
        required = {"z", "d", "k", "w"}
        if set(state) != required:
            raise ValueError("state must contain exactly z, d, k, and w")

        z = state["z"]
        d = state["d"]
        k = state["k"]
        if not isinstance(z, int) or not self.lower_boundary <= z <= self.upper_boundary:
            raise ValueError("z must be an integer between 0 and 10")
        if d not in (-1, 1):
            raise ValueError("d must be either -1 or 1")
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        if not isinstance(state["w"], str):
            raise ValueError("w must be a string dummy variable")
        return {"z": z, "d": d, "k": k, "w": state["w"]}
