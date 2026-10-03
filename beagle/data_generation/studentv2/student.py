"""
Student V2 - Markov-Driven Student Model

This module defines the Student class that uses a Markov model to drive its behavior.
The Markov model determines the sequence of (Cognitive State, Metacognitive State) tuples.
The Cognitive State (CONSTRUCTING, DEBUGGING, ASSESSING) acts as the high-level "action"
that guides the LLM's behavior.

Usage:
    student = Student(performance_level="low")
    student.load_model("path/to/markov_model.npy")
    
    # Start a session
    history = student.solve_problem("Write a function to calculate projectile range")
"""

import logging
from typing import List, Tuple, Optional, Literal, Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from beagle.data_generation.studentv2.sequence_generator import MarkovSequence
    from beagle.tutor.base import TutorStrategy

from beagle.data_generation.studentv2.semi_markov_model import SemiMarkovModel
from beagle.data_generation.studentv2.graph import student_graph
from beagle.data_generation.studentv2.state import StudentState
from beagle.data_generation.studentv2.deps import StudentDeps
from beagle.data_generation.studentv2.nodes import MarkovNode
from beagle.utils.llm_client import LLMClient
from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.data_generation.studentv2.bkt.bkt import BKT
from beagle.data_generation.studentv2.bkt.student_profiles import initialize_student_kcs, StudentProfile
from beagle.data_generation.classroom.assessment_oracle import AutomatedAssessmentOracle

logger = logging.getLogger(__name__)


class Student:
    """
    Represents a student whose behavior is driven by a Markov model and LLM agents via a Graph.
    """

    def __init__(
        self,
        performance_level: Literal["low", "high"] = "low",
        profile_override: Optional[Literal["low", "high"]] = None,  # Override persona (for ablation)
        model_path: Optional[str] = None,
        llm_model: str = "google-gla:gemini-2.5-flash",
        ide_oracle: Optional[IDEOracle] = None,
        duration_multiplier: float = 1.0,
        cache_bkt: bool = True,
        efi_kcs: Optional[List[str]] = None,
        force_assistance_steps: Optional[List[int]] = None,
        tutor_strategy: Optional['TutorStrategy'] = None,
        skip_bkt_on_assisted: bool = False,
        bkt_profile: Optional[str] = None,
        bkt_seed: Optional[int] = None
    ):
        """
        Initialize the student.

        Args:
            performance_level: "low" or "high" - Controls Semi-Markov model behavior
            profile_override: Optional override for persona type (linguistic style).
                             If None, derives from performance_level. Use for ablation experiments
                             to decouple Semi-Markov behavior from linguistic persona.
            model_path: Path to the trained Markov model .npy file
            llm_model: LLM model string to use for agents
            ide_oracle: Optional IDE Oracle for code execution
            duration_multiplier: Multiplier for metacognitive segment durations.
                                 Use < 1.0 for shorter simulations (e.g., 0.5 = half duration)
            cache_bkt: If True, sample BKT once per metacog phase (new behavior).
                       If False, sample BKT per cognitive step (old behavior).
            efi_kcs: Optional list of Knowledge Components to apply EFI (Explicit Flaw Injection).
                     When EFI is active on a KC, the student "doesn't know it exists" and
                     cannot use that concept in their code.
            force_assistance_steps: Optional list of step numbers to force tutor assistance.
            tutor_strategy: Optional pluggable tutor strategy for controlled experiments.
                           If None, uses default LLM-based tutor. Options:
                           - TutorFactory.create(TutorType.RULE_BASED)
                           - TutorFactory.create(TutorType.ML_BASED)
                           - TutorFactory.create(TutorType.LLM_BASED, llm_client=...)
            skip_bkt_on_assisted: If True, skip BKT updates when student uses tutor help.
                                  This enables "Performance != Competence" mode where
                                  copying tutor's answer doesn't count as learning.
                                  Used for transfer test case studies.
            bkt_profile: Override BKT profile for all students. Options: 'zero', 'low', 'average', 'high'.
                         If not set, defaults to BELOW_AVERAGE (decoupled from performance level).
            bkt_seed: Seed for BKT initialization. If set, all runs use identical P(L) values.
        """
        self.performance_level: Literal["low", "high"] = performance_level
        # Derive persona_type from profile_override or performance_level
        persona_base = profile_override or performance_level
        self.persona_type: Literal["low_performer", "high_performer"] = f"{persona_base}_performer"
        self.markov_model: Optional[SemiMarkovModel] = None
        self.llm_client = LLMClient(model=llm_model)
        self.ide_oracle = ide_oracle
        self.duration_multiplier = duration_multiplier
        self.cache_bkt = cache_bkt
        self.efi_kcs = efi_kcs or []
        self.force_assistance_steps = force_assistance_steps or []
        self.tutor_strategy = tutor_strategy
        self.skip_bkt_on_assisted = skip_bkt_on_assisted
        self.bkt_profile = bkt_profile
        self.bkt_seed = bkt_seed

        # Initialize BKT and Assessment Oracle
        self.bkt = BKT()
        self.assessment_oracle = AutomatedAssessmentOracle()

        if model_path:
            self.load_model(model_path)

        self.state: Optional[StudentState] = None


    def load_model(self, model_path: str):
        """Load the Markov model from file."""
        try:
            self.markov_model = SemiMarkovModel.load(model_path)
            logger.info(f"Loaded Semi-Markov model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def solve_problem(self,
                      problem_description: str,
                      problem_id: Optional[str] = None,
                      required_kcs: List[str] = None,
                      max_steps: int = 50,
                      starting_code: str = "",
                      checkpoint_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run the student simulation on a problem.

        Args:
            problem_description: The problem to solve.
            problem_id: The ID of the problem (for IDE Oracle).
            required_kcs: List of Knowledge Components required for the problem.
            max_steps: Maximum number of steps to simulate.
            starting_code: Initial code template to start with (e.g., Bielefeld template).
            checkpoint_path: Optional path to save incremental progress after each step.

        Returns:
            The history of the simulation.
        """
        if not self.markov_model:
            raise RuntimeError(
                "Markov model not loaded. Call load_model() first."
            )

        # Initialize state
        self.state = StudentState(
            problem_description=problem_description,
            problem_id=problem_id or "",
            max_steps=max_steps,
            required_kcs=required_kcs or [],
            force_assistance_steps=self.force_assistance_steps,
            current_code=starting_code
        )
        
        # Initialize KCs in BKT based on profile (skip if BKT is disabled)
        if required_kcs and self.bkt is not None:
            # Determine BKT profile with priority: bkt_profile > default (BELOW_AVERAGE)
            profile_map = {
                "zero": StudentProfile.ZERO,  # Controlled experiment: all start at 0
                "low": StudentProfile.BELOW_AVERAGE,
                "average": StudentProfile.AVERAGE,
                "high": StudentProfile.ABOVE_AVERAGE
            }
            
            if self.bkt_profile:
                # Explicit override: use specified profile
                profile = profile_map.get(self.bkt_profile, StudentProfile.AVERAGE)
                profile_mode = f" [OVERRIDE: {self.bkt_profile}]"
            else:
                # Default: use BELOW_AVERAGE for all (decoupled from performance level)
                profile = StudentProfile.BELOW_AVERAGE
                profile_mode = ""

            logger.info(f"Initializing BKT for {self.performance_level} performer (Profile: {profile.name}){profile_mode}")

            # Initialize KCs
            initialize_student_kcs(
                self.bkt,
                required_kcs,
                profile,
                random_state=self.bkt_seed
            )

            # Log initial mastery for debugging/verification
            for kc in required_kcs:
                state = self.bkt.get_kc_state(kc)
                if state:
                    logger.info(f"  {kc}: P(L)={state.p_known:.2f}")

            # Apply EFI (Explicit Flaw Injection) to specified KCs
            for efi_kc in self.efi_kcs:
                if efi_kc in required_kcs:
                    self.bkt.set_efi(efi_kc, True)
                    logger.info(f"  {efi_kc}: EFI ACTIVATED (student doesn't know this exists)")
        elif self.bkt is None:
            logger.info("BKT disabled - skipping knowledge state initialization")

        # Initialize deps
        deps = StudentDeps(
            markov_model=self.markov_model,
            llm_client=self.llm_client,
            performance_level=self.performance_level,
            ide_oracle=self.ide_oracle,
            bkt=self.bkt,
            assessment_oracle=self.assessment_oracle,
            persona_type=self.persona_type,  # ρ_persona: controls language style
            duration_multiplier=self.duration_multiplier,
            cache_bkt=self.cache_bkt,
            tutor_strategy=self.tutor_strategy,
            controlled_assistance_steps=self.force_assistance_steps,
            skip_bkt_on_assisted=self.skip_bkt_on_assisted,
            checkpoint_path=checkpoint_path
        )


        # Run graph
        tutor_info = f", tutor={type(self.tutor_strategy).__name__}" if self.tutor_strategy else ""
        logger.info(
            f"Starting simulation for {self.performance_level} performer (cache_bkt={self.cache_bkt}{tutor_info})..."
        )

        # We start with MarkovNode
        # run_sync will execute until End is returned
        student_graph.run_sync(MarkovNode(), state=self.state, deps=deps)

        logger.info("Simulation complete.")
        return self.state.history

    def solve_problem_with_sequence(
        self,
        problem_description: str,
        sequence: 'MarkovSequence',
        problem_id: Optional[str] = None,
        required_kcs: List[str] = None,
        starting_code: str = "",
        checkpoint_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the student simulation using a pre-generated Markov sequence.

        This method uses a pre-generated Markov chain (metacog states, cognitive
        actions) while keeping interrupt decisions (assistance, off-topic) real-time
        based on actual test progress.

        Separates:
        - Markov chain (pre-generated, deterministic for given seed)
        - Interrupt decisions (real-time, based on tests_passed/tests_total)
        - Content generation (LLM, stochastic)

        Benefits:
        - Reproducible Markov behavioral patterns
        - Analyze Markov dynamics before LLM execution
        - Batch processing with known behavioral profiles

        Args:
            problem_description: The problem to solve.
            sequence: A MarkovSequence object from SequenceGenerator.
            problem_id: The ID of the problem (for IDE Oracle).
            required_kcs: List of Knowledge Components required for the problem.
            starting_code: Initial code template to start with (e.g., Bielefeld template).

        Returns:
            The history of the simulation.

        Example:
            from beagle.data_generation.studentv2.sequence_generator import SequenceGenerator

            # Pre-generate a Markov sequence
            gen = SequenceGenerator()
            sequence = gen.generate(performance_level="low", max_steps=20, seed=42)

            # Use the sequence in simulation (interrupts still real-time)
            student = Student(performance_level="low", model_path="...")
            history = student.solve_problem_with_sequence(
                problem_description="Write a function...",
                sequence=sequence,
                problem_id="projectile_motion",
                required_kcs=["KC_C1_FUNCTION_DEF", ...]
            )
        """
        if not self.markov_model:
            raise RuntimeError(
                "Markov model not loaded. Call load_model() first."
            )

        # Initialize state with pre-generated sequence
        self.state = StudentState(
            problem_description=problem_description,
            problem_id=problem_id or "",
            max_steps=len(sequence.steps),  # Use sequence length as max
            required_kcs=required_kcs or [],
            pregenerated_sequence=sequence,  # Attach the sequence
            sequence_index=0,
            force_assistance_steps=self.force_assistance_steps,
            current_code=starting_code
        )

        # Initialize KCs in BKT based on profile (skip if BKT is disabled)
        if required_kcs and self.bkt is not None:
            # Determine BKT profile with priority: bkt_profile > default (BELOW_AVERAGE)
            profile_map = {
                "low": StudentProfile.BELOW_AVERAGE,
                "average": StudentProfile.AVERAGE,
                "high": StudentProfile.ABOVE_AVERAGE
            }
            
            if self.bkt_profile:
                # Explicit override: use specified profile
                profile = profile_map.get(self.bkt_profile, StudentProfile.AVERAGE)
                profile_mode = f" [OVERRIDE: {self.bkt_profile}]"
            else:
                # Default: use BELOW_AVERAGE for all (decoupled from performance level)
                profile = StudentProfile.BELOW_AVERAGE
                profile_mode = ""

            logger.info(f"Initializing BKT for {self.performance_level} performer (Profile: {profile.name}){profile_mode}")

            initialize_student_kcs(
                self.bkt,
                required_kcs,
                profile,
                random_state=self.bkt_seed
            )

            for kc in required_kcs:
                state = self.bkt.get_kc_state(kc)
                if state:
                    logger.info(f"  {kc}: P(L)={state.p_known:.2f}")

            # Apply EFI (Explicit Flaw Injection) to specified KCs
            for efi_kc in self.efi_kcs:
                if efi_kc in required_kcs:
                    self.bkt.set_efi(efi_kc, True)
                    logger.info(f"  {efi_kc}: EFI ACTIVATED (student doesn't know this exists)")
        elif self.bkt is None:
            logger.info("BKT disabled - skipping knowledge state initialization")

        # Initialize deps
        deps = StudentDeps(
            markov_model=self.markov_model,
            llm_client=self.llm_client,
            performance_level=self.performance_level,
            ide_oracle=self.ide_oracle,
            bkt=self.bkt,
            assessment_oracle=self.assessment_oracle,
            persona_type=self.persona_type,  # ρ_persona: controls language style
            duration_multiplier=self.duration_multiplier,
            cache_bkt=self.cache_bkt,
            tutor_strategy=self.tutor_strategy,
            controlled_assistance_steps=self.force_assistance_steps,
            skip_bkt_on_assisted=self.skip_bkt_on_assisted,
            checkpoint_path=checkpoint_path
        )


        # Run graph with pre-generated sequence
        tutor_info = f", tutor={type(self.tutor_strategy).__name__}" if self.tutor_strategy else ""
        logger.info(
            f"Starting simulation with pre-generated sequence "
            f"(seed={sequence.seed}, {len(sequence.steps)} steps, cache_bkt={self.cache_bkt}{tutor_info})..."
        )

        student_graph.run_sync(MarkovNode(), state=self.state, deps=deps)

        logger.info("Simulation complete.")
        return self.state.history

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full history of behaviors."""
        return self.state.history if self.state else []
