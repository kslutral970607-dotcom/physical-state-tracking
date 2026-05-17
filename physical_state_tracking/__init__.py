"""Synthetic state-tracking datasets for causal abstraction experiments."""

from .dataset import generate_dataset
from .state_machine import Operation, StateMachine

__all__ = ["Operation", "StateMachine", "generate_dataset"]
