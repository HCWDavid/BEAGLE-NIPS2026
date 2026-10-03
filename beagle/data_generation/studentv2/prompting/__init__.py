# Epistemic Gating Module
# Handles the "trap" mechanism for BEAGLE student simulation

from .gating import (
    EpistemicGating,
    NeuroSymbolicPromptGenerator,  # Legacy alias
    PromptBlocks,  # Legacy, for tests
    get_generator,
    set_sparse_comments,
)

__all__ = [
    "EpistemicGating",
    "NeuroSymbolicPromptGenerator",  # Legacy alias
    "PromptBlocks",  # Legacy
    "get_generator",
    "set_sparse_comments",
]
