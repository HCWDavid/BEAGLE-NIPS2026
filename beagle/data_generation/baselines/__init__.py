"""
Baseline Methods for BEAGLE Evaluation

This module contains baseline implementations for comparison with BEAGLE:
- VanillaStudent: Pure LLM prompting approach (no Markov control, no BKT)
- CoTStudent: Chain-of-Thought prompting (explicit reasoning about mistakes)
- FewShotStudent: In-context learning with real student trace exemplars

These baselines are used to demonstrate the "Competence Bias" problem
where pure LLMs solve problems too efficiently without realistic struggle.
"""

from beagle.data_generation.baselines.vanilla_student import VanillaStudent
from beagle.data_generation.baselines.cot_student import CoTStudent
from beagle.data_generation.baselines.fewshot_student import FewShotStudent

__all__ = ["VanillaStudent", "CoTStudent", "FewShotStudent"]
