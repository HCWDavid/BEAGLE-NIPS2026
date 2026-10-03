"""
Tutor Factory

Factory pattern for creating and managing tutor instances.
Allows easy switching between tutor types for experiments.
"""

from enum import Enum
from typing import Optional

from beagle.tutor.base import TutorStrategy
from beagle.tutor.rule_based import RuleBasedTutor
from beagle.tutor.ml_based import MLBasedTutor
from beagle.tutor.zpd_tutor import ZPDTutor
from beagle.tutor.default_tutor import DefaultLLMTutor
from beagle.utils.llm_client import LLMClient


class TutorType(Enum):
    """Available tutor types."""
    NONE = "none"                # No tutor (baseline)
    RULE_BASED = "rule_based"    # Template hints (Hint Factory style)
    ML_BASED = "ml_based"        # BKT-informed hints (ITAP style)
    ZPD = "zpd"                  # ZPD adaptive scaffolding (BKT-informed)
    SIMPLE_LLM = "simple_llm"    # Simple LLM prompts (no BKT awareness)
    LLM_BASED = "llm_based"      # Alias for simple_llm (legacy)


class NoTutor(TutorStrategy):
    """
    Null tutor that provides no hints.
    Used as baseline in controlled experiments.
    """

    @property
    def tutor_type(self) -> str:
        return "none"

    async def generate_hint(self, context):
        from beagle.tutor.base import TutorResponse, ScaffoldLevel
        return TutorResponse(
            hint="",
            scaffold_level=ScaffoldLevel.NONE,
            target_kc=None,
            tutor_type=self.tutor_type,
            reasoning="No tutor enabled (baseline condition)",
            no_hint_needed=True,
        )


class TutorFactory:
    """
    Factory for creating tutor instances.

    Usage:
        # Create a specific tutor type
        tutor = TutorFactory.create(TutorType.LLM_BASED, llm_client=client)

        # Get all available types
        types = TutorFactory.available_types()

        # Create from string name
        tutor = TutorFactory.from_string("ml_based")
    """

    _registry = {
        TutorType.NONE: NoTutor,
        TutorType.RULE_BASED: RuleBasedTutor,
        TutorType.ML_BASED: MLBasedTutor,
        TutorType.ZPD: ZPDTutor,
        TutorType.SIMPLE_LLM: DefaultLLMTutor,
        TutorType.LLM_BASED: DefaultLLMTutor,  # Legacy alias for simple_llm
    }

    @classmethod
    def create(
        cls,
        tutor_type: TutorType,
        llm_client: Optional[LLMClient] = None,
        **kwargs
    ) -> TutorStrategy:
        """
        Create a tutor instance of the specified type.

        Args:
            tutor_type: The type of tutor to create
            llm_client: LLMClient instance (required for LLM_BASED)
            **kwargs: Additional arguments passed to tutor constructor

        Returns:
            TutorStrategy instance

        Raises:
            ValueError: If tutor_type is not supported
        """
        if tutor_type not in cls._registry:
            raise ValueError(
                f"Unknown tutor type: {tutor_type}. "
                f"Available types: {[t.value for t in cls._registry.keys()]}"
            )

        tutor_class = cls._registry[tutor_type]

        # LLM-based tutors require llm_client
        if tutor_type in [TutorType.LLM_BASED, TutorType.ZPD, TutorType.SIMPLE_LLM]:
            return tutor_class(llm_client=llm_client, **kwargs)

        return tutor_class(**kwargs)

    @classmethod
    def from_string(
        cls,
        type_string: str,
        llm_client: Optional[LLMClient] = None,
        **kwargs
    ) -> TutorStrategy:
        """
        Create a tutor from a string type name.

        Args:
            type_string: String name of tutor type (e.g., "ml_based")
            llm_client: LLMClient instance (required for "llm_based")
            **kwargs: Additional arguments

        Returns:
            TutorStrategy instance

        Raises:
            ValueError: If type_string is not recognized
        """
        # Normalize string
        type_string = type_string.lower().strip()

        # Find matching TutorType
        for tutor_type in TutorType:
            if tutor_type.value == type_string:
                return cls.create(tutor_type, llm_client=llm_client, **kwargs)

        # Not found
        available = [t.value for t in TutorType]
        raise ValueError(
            f"Unknown tutor type: '{type_string}'. "
            f"Available types: {available}"
        )

    @classmethod
    def available_types(cls) -> list[str]:
        """Get list of available tutor type names."""
        return [t.value for t in TutorType]

    @classmethod
    def get_description(cls, tutor_type: TutorType) -> str:
        """Get description of a tutor type."""
        descriptions = {
            TutorType.NONE: (
                "No tutor (baseline). No hints are provided."
            ),
            TutorType.RULE_BASED: (
                "Rule-based tutor (Hint Factory style). "
                "Uses template hints triggered by error patterns. "
                "Based on Barnes & Stamper (2010)."
            ),
            TutorType.ML_BASED: (
                "ML-based tutor (ITAP style). "
                "Uses BKT knowledge state to generate personalized hints "
                "targeting the student's weakest knowledge components. "
                "Based on Rivers & Koedinger (2017)."
            ),
            TutorType.LLM_BASED: (
                "LLM-based tutor (ZPD Adaptive Scaffolding). "
                "Uses BKT mastery to determine ZPD position and generates "
                "graduated scaffolding hints via LLM. "
                "Based on Cohn et al. (2025)."
            ),
        }
        return descriptions.get(tutor_type, "Unknown tutor type")

    @classmethod
    def print_info(cls) -> None:
        """Print information about all available tutor types."""
        print("\n" + "=" * 60)
        print("BEAGLE Tutor Types")
        print("=" * 60)

        for tutor_type in TutorType:
            print(f"\n{tutor_type.value}:")
            print(f"  {cls.get_description(tutor_type)}")

        print("\n" + "=" * 60)


# Convenience function for quick access
def create_tutor(
    tutor_type: str = "llm_based",
    llm_client: Optional[LLMClient] = None,
    **kwargs
) -> TutorStrategy:
    """
    Convenience function to create a tutor.

    Args:
        tutor_type: String name of tutor type
        llm_client: LLMClient instance (required for llm_based)
        **kwargs: Additional arguments

    Returns:
        TutorStrategy instance

    Example:
        tutor = create_tutor("ml_based")
        hint = await tutor.generate_hint(context)
    """
    return TutorFactory.from_string(tutor_type, llm_client=llm_client, **kwargs)
