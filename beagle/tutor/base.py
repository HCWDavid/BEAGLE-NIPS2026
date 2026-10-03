"""
Base classes for the BEAGLE Tutor Module.

Defines the abstract interface that all tutor strategies must implement,
along with context and response data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from beagle.data_generation.studentv2.bkt.bkt import BKT, MasteryLevel


class ScaffoldLevel(Enum):
    """
    Graduated scaffolding levels based on Vygotsky's ZPD.

    Based on Cohn et al. (2025) "A Theory of Adaptive Scaffolding
    for LLM-Based Pedagogical Agents"

    Levels:
        NONE: Student is in comfort zone, no hint needed
        MINIMAL: Upper ZPD - redirect attention with a question
        GUIDING: Mid ZPD - name the concept but don't solve
        EXPLICIT: Lower ZPD - tell what to do (not exact code)
        EXAMPLE: Near boundary - show worked example of similar problem
    """
    NONE = 0      # Comfort zone - no scaffolding needed
    MINIMAL = 1   # Upper ZPD - minimal prompt, redirect attention
    GUIDING = 2   # Mid ZPD - guiding question, name concept
    EXPLICIT = 3  # Lower ZPD - explicit hint, tell what to do
    EXAMPLE = 4   # Near boundary - worked example


@dataclass
class HintHistoryEntry:
    """Record of a previous hint given to the student."""
    step: int
    kc_id: Optional[str]
    scaffold_level: ScaffoldLevel
    hint_text: str
    was_helpful: Optional[bool] = None  # Set later based on student progress


@dataclass
class TutorContext:
    """
    Context passed to tutor for generating hints.

    Contains all information needed to generate an appropriate hint:
    - Problem and code state
    - BKT knowledge tracking
    - Error information
    - Hint history for escalation
    """
    # Problem context
    problem_description: str
    problem_id: str = ""

    # Current code state
    current_code: str = ""
    last_output: str = ""
    last_error: str = ""
    error_type: str = ""  # e.g., "SyntaxError", "NameError"

    # Student question (if assistance was requested)
    student_question: str = ""

    # BKT knowledge state
    bkt: Optional[BKT] = None
    required_kcs: List[str] = field(default_factory=list)

    # Performance context
    performance_level: str = "low"  # "low" or "high"
    current_step: int = 0
    tests_passed: int = 0
    tests_total: int = 0

    # Hint history for escalation tracking
    hint_history: List[HintHistoryEntry] = field(default_factory=list)

    # Additional context
    current_metacognitive_state: str = ""
    current_cognitive_state: str = ""

    def get_mastery_for_kc(self, kc_id: str) -> float:
        """Get BKT mastery level (P(L)) for a specific KC."""
        if self.bkt is None:
            return 0.5  # Default to mid-level if no BKT

        state = self.bkt.get_kc_state(kc_id)
        if state is None:
            return 0.5

        return state.p_known

    def get_lowest_mastery_kc(self) -> Optional[str]:
        """Find the KC with lowest mastery (most likely struggle point)."""
        if self.bkt is None or not self.required_kcs:
            return None

        lowest_kc = None
        lowest_mastery = 1.0

        for kc_id in self.required_kcs:
            mastery = self.get_mastery_for_kc(kc_id)
            if mastery < lowest_mastery:
                lowest_mastery = mastery
                lowest_kc = kc_id

        return lowest_kc

    def get_recent_hint_count_for_kc(self, kc_id: str, window: int = 3) -> int:
        """Count recent hints for a specific KC (for escalation)."""
        recent = self.hint_history[-window:] if self.hint_history else []
        return sum(1 for h in recent if h.kc_id == kc_id)

    def get_knowledge_state_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge state for all required KCs."""
        if self.bkt is None:
            return {"available": False}

        summary = {
            "available": True,
            "kcs": {}
        }

        for kc_id in self.required_kcs:
            state = self.bkt.get_kc_state(kc_id)
            if state:
                summary["kcs"][kc_id] = {
                    "p_known": state.p_known,
                    "mastery_level": state.mastery_level.value,
                    "efi_active": state.efi_active,
                    "demonstrated": state.demonstrated
                }

        return summary


@dataclass
class TutorResponse:
    """
    Response from a tutor strategy.

    Contains the hint text and metadata about the hint generation.
    """
    # The hint text to show the student
    hint: str

    # Metadata about the hint
    scaffold_level: ScaffoldLevel = ScaffoldLevel.GUIDING
    target_kc: Optional[str] = None  # Which KC this hint addresses

    # For tracking and analysis
    tutor_type: str = ""  # "rule_based", "ml_based", "llm_based"
    reasoning: str = ""   # Why this hint was chosen (for debugging)

    # Whether this is a "no hint needed" response
    no_hint_needed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hint": self.hint,
            "scaffold_level": self.scaffold_level.value,
            "target_kc": self.target_kc,
            "tutor_type": self.tutor_type,
            "reasoning": self.reasoning,
            "no_hint_needed": self.no_hint_needed
        }


class TutorStrategy(ABC):
    """
    Abstract base class for tutor strategies.

    All tutor implementations must inherit from this class and implement
    the generate_hint method.

    Tutor Types:
        1. Rule-Based: Template hints based on error patterns
        2. ML-Based: BKT-informed hints based on mastery state
        3. LLM-Based: ZPD adaptive scaffolding with LLM generation
    """

    @property
    @abstractmethod
    def tutor_type(self) -> str:
        """Return the tutor type identifier."""
        pass

    @abstractmethod
    async def generate_hint(self, context: TutorContext) -> TutorResponse:
        """
        Generate a hint for the student based on context.

        Args:
            context: TutorContext with problem, code, BKT state, etc.

        Returns:
            TutorResponse with hint text and metadata
        """
        pass

    def determine_scaffold_level(self, context: TutorContext) -> ScaffoldLevel:
        """
        Determine appropriate scaffold level based on ZPD position.

        Uses BKT mastery to place student in ZPD zones:
        - Mastery > 0.7: Comfort zone (no scaffold)
        - Mastery 0.5-0.7: Upper ZPD (minimal)
        - Mastery 0.3-0.5: Mid ZPD (guiding)
        - Mastery < 0.3: Lower ZPD (explicit)

        Escalates based on repeated failures on same KC.

        Args:
            context: TutorContext with BKT state

        Returns:
            Appropriate ScaffoldLevel
        """
        # Find the KC most likely causing struggle
        target_kc = context.get_lowest_mastery_kc()

        if target_kc is None:
            # No BKT tracking, default to guiding
            return ScaffoldLevel.GUIDING

        mastery = context.get_mastery_for_kc(target_kc)

        # Base level from mastery (ZPD position)
        if mastery > 0.7:
            base_level = ScaffoldLevel.NONE
        elif mastery > 0.5:
            base_level = ScaffoldLevel.MINIMAL
        elif mastery > 0.3:
            base_level = ScaffoldLevel.GUIDING
        else:
            base_level = ScaffoldLevel.EXPLICIT

        # Escalate based on repeated failures (reverse fading)
        recent_hints = context.get_recent_hint_count_for_kc(target_kc)
        escalation = min(recent_hints, 2)  # Max 2 levels of escalation

        final_level = min(
            base_level.value + escalation,
            ScaffoldLevel.EXAMPLE.value
        )

        return ScaffoldLevel(final_level)

    def format_kc_name(self, kc_id: str) -> str:
        """Convert KC ID to human-readable name."""
        return kc_id.replace("KC_", "").replace("_", " ").lower()
