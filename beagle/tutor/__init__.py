"""
BEAGLE Tutor Module

Modular tutoring strategies for the BEAGLE simulated student system.
Implements tutor types based on educational research:

1. Rule-Based (Hint Factory style) - Barnes & Stamper, 2010
2. ML-Based (ITAP style) - Rivers & Koedinger, 2017
3. ZPD (Zone of Proximal Development Adaptive Scaffolding) - Cohn et al., 2025

Usage:
    from beagle.tutor import TutorFactory, TutorType

    tutor = TutorFactory.create(TutorType.ZPD, llm_client=client)
    hint = await tutor.generate_hint(context)
"""

from beagle.tutor.base import TutorStrategy, TutorContext, TutorResponse, ScaffoldLevel, HintHistoryEntry
from beagle.tutor.rule_based import RuleBasedTutor
from beagle.tutor.ml_based import MLBasedTutor
from beagle.tutor.zpd_tutor import ZPDTutor
from beagle.tutor.factory import TutorFactory, TutorType

# Backwards compatibility alias
LLMBasedTutor = ZPDTutor

__all__ = [
    # Base classes
    "TutorStrategy",
    "TutorContext",
    "TutorResponse",
    "ScaffoldLevel",
    "HintHistoryEntry",
    # Implementations
    "RuleBasedTutor",
    "MLBasedTutor",
    "ZPDTutor",
    "LLMBasedTutor",  # Alias for backwards compat
    # Factory
    "TutorFactory",
    "TutorType",
]
