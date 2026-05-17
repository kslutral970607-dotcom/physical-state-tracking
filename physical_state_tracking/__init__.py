"""Synthetic state-tracking datasets for causal abstraction experiments."""

from .dataset import generate_dataset
from .state_machine import BounceStep, StateMachine

__all__ = ["BounceStep", "StateMachine", "generate_dataset"]
