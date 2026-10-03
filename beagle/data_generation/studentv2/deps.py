from dataclasses import dataclass, field
from typing import Literal, Optional, List, TYPE_CHECKING

from beagle.data_generation.studentv2.semi_markov_model import SemiMarkovModel
from beagle.utils.llm_client import LLMClient
from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.data_generation.studentv2.bkt.bkt import BKT
from beagle.data_generation.classroom.assessment_oracle import AutomatedAssessmentOracle

if TYPE_CHECKING:
    from beagle.tutor.base import TutorStrategy


@dataclass
class StudentDeps:
    """Dependencies injected into the graph nodes."""
    markov_model: SemiMarkovModel
    llm_client: LLMClient
    performance_level: Literal["low", "high"]  # ρ_behavior: Controls Semi-Markov transitions
    bkt: BKT
    assessment_oracle: AutomatedAssessmentOracle
    persona_type: Optional[Literal["low_performer", "high_performer"]] = None  # ρ_persona: Controls language style
    ide_oracle: Optional[IDEOracle] = None
    duration_multiplier: float = 1.0  # Multiplier for segment durations (e.g., 0.5 = half)
    cache_bkt: bool = True  # If True, sample BKT once per metacog phase; if False, sample per cognitive step

    # Tutor configuration (for controlled experiments)
    tutor_strategy: Optional['TutorStrategy'] = None  # Pluggable tutor (None = use default LLM tutor)
    controlled_assistance_steps: List[int] = field(default_factory=list)  # Steps where assistance is forced

    # Performance vs Competence mode (for transfer test case study)
    # When True, BKT updates are skipped when student uses tutor help
    # This models "copying without learning" - student can perform but doesn't internalize
    skip_bkt_on_assisted: bool = False

    # Incremental checkpoint saving
    # If set, save simulation state to this path after each cognitive step
    checkpoint_path: Optional[str] = None

    def __post_init__(self):
        # Default persona_type based on performance_level if not explicitly set
        if self.persona_type is None:
            self.persona_type = f"{self.performance_level}_performer"

