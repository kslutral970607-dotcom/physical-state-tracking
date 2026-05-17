"""Small deterministic state machine used by all dataset shells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


State = Dict[str, int]


@dataclass(frozen=True)
class Operation:
    """A single transition on one named state variable."""

    variable: str
    delta: int
    description: str


class StateMachine:
    """Apply additive transitions while preserving the initial state."""

    def __init__(self, initial_state: Mapping[str, int]) -> None:
        if not initial_state:
            raise ValueError("initial_state must not be empty")
        self.initial_state: State = dict(initial_state)

    def apply(self, operations: Iterable[Operation]) -> State:
        state = dict(self.initial_state)
        for operation in operations:
            if operation.variable not in state:
                raise KeyError(f"unknown state variable: {operation.variable}")
            state[operation.variable] += operation.delta
        return state
