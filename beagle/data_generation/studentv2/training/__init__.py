"""
Training modules for the Semi-Markov student model.

This package contains:
- train_metacog_transitions: Train P(next_metacog | prev2_metacog, prev1_metacog)
- train_action_emissions: Train P(next_action | metacog, prev_action)
- load_scores: Utility to load student performance data
"""

from .load_scores import load_scores
from .train_metacog_transitions import train_metacog_transitions
from .train_action_emissions import train_action_emissions

__all__ = [
    'load_scores',
    'train_metacog_transitions',
    'train_action_emissions',
]
