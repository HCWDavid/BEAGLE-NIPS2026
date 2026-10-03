"""
Bayesian Knowledge Tracing (BKT) for Knowledge Component Tracking

This module implements standard Bayesian Knowledge Tracing (BKT) for tracking
student knowledge acquisition over time.

References:
- Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the 
  acquisition of procedural knowledge.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MasteryLevel(Enum):
    """Qualitative mastery labels derived from BKT P(L)."""
    UNKNOWN = "UNKNOWN"  # P(L) < 0.3
    PARTIAL = "PARTIAL"  # 0.3 ≤ P(L) < 0.7
    MASTERED = "MASTERED"  # P(L) ≥ 0.7


class Observation(Enum):
    """BKT observation type - did the student get it correct or wrong."""
    CORRECT = "CORRECT"
    WRONG = "WRONG"


# =============================================================================
# VALID KC REGISTRY
# =============================================================================
# All valid Knowledge Component IDs. Any KC used in a problem must be listed here.
# Format: KC_<domain><number>_<NAME>
# Domains: C=Coding, P=Physics, M=Math, B=Biology
VALID_KC_IDS = {
    # Coding KCs (C1-C12) - used by projectile_motion, particle_simulator
    'KC_C1_FUNCTION_DEF_RETURN',  # Function definition with return
    'KC_C2_MATH_LIBRARY',         # import math, numpy
    'KC_C3_FUNCTION_USAGE_ASSIGNMENT',  # calling functions, assigning to vars
    'KC_C4_ARITHMETIC_IMPLEMENTATION',  # +, -, *, /, **
    'KC_C9_CLASS_DEFINITION',     # class keyword
    'KC_C10_INIT_METHOD',         # __init__ method
    'KC_C11_INSTANCE_VARIABLES',  # self.x = ...
    'KC_C12_METHOD_DEFINITION',   # methods in classes

    # Additional Coding KCs (C13-C16) - used by gradient_descent
    'KC_C13_LOOP_CONSTRUCT',      # for/while loops
    'KC_C14_CONDITIONAL_LOGIC',   # if/else statements
    'KC_C15_VARIABLE_UPDATE',     # x = x + 1, x += 1
    'KC_C16_FUNCTION_CALL',       # calling functions with args

    # Physics KCs (P1-P11)
    'KC_P1_VECTOR_DECOMPOSITION', # v_x, v_y components
    'KC_P2_TRIG_APPLICATION',     # sin/cos for physics
    'KC_P3_TIME_OF_FLIGHT',       # t = 2*v_y/g
    'KC_P4_RANGE_FORMULA',        # R = v_x * t
    'KC_P5_UNIT_RADIANS',         # degree to radian conversion
    'KC_P9_NUMERICAL_INTEGRATION',  # Euler method, dt updates
    'KC_P10_FORCE_ACCELERATION',  # F = ma
    'KC_P11_KINETIC_ENERGY',      # KE = 0.5*m*v^2

    # Additional Physics KCs (P12-P16) - new problems
    'KC_P12_POTENTIAL_ENERGY',    # PE = 0.5*k*x² (spring) or m*g*h
    'KC_P13_HOOKES_LAW',          # F = -kx spring force
    'KC_P14_COLLISION_REFLECTION', # velocity reversal on bounce
    'KC_P15_RELATIVE_VELOCITY',   # vector addition of velocities
    'KC_P16_FRICTION',            # μmg cos(θ) friction force

    # Math KCs (M1-M5) - for gradient descent, calculus
    'KC_M1_DERIVATIVE_CONCEPT',       # understanding gradients
    'KC_M2_NUMERICAL_DIFFERENTIATION',  # (f(x+h) - f(x-h)) / 2h
    'KC_M3_ITERATIVE_ALGORITHM',      # loops that converge
    'KC_M4_CONVERGENCE_CRITERIA',     # stopping conditions
    'KC_M5_LEARNING_RATE',            # step size / eta

    # Biology KCs (B1-B3) - for population_growth (logistic Verhulst dynamics)
    'KC_B1_LOGISTIC_GROWTH_LAW',      # dN/dt = r*N*(1 - N/K)
    'KC_B2_CARRYING_CAPACITY',        # K as environmental upper bound; clamping
    'KC_B3_DENSITY_DEPENDENCE',       # (1 - N/K) per-capita scaling factor
}


def validate_kc_id(kc_id: str) -> None:
    """
    Validate that a KC ID is in the valid registry.

    Raises:
        ValueError: If KC ID is not registered
    """
    if kc_id not in VALID_KC_IDS:
        raise ValueError(
            f"Unknown KC '{kc_id}' is not registered in VALID_KC_IDS. "
            f"Please add it to the registry in bkt.py and ensure it has "
            f"an assessment method in assessment_oracle.py. "
            f"Valid KCs: {sorted(VALID_KC_IDS)}"
        )


# Human-readable KC descriptions for EFI constraints
# These must be explicit about what the student CANNOT use
KC_EFI_DESCRIPTIONS = {
    'KC_C1_FUNCTION_DEF_RETURN': 'how to define functions with return values',
    'KC_C2_MATH_LIBRARY': 'the Python math library (import math, math.sin, math.cos, math.radians, math.pi, etc.)',
    'KC_C3_FUNCTION_USAGE_ASSIGNMENT': 'how to call functions and assign results to variables',
    'KC_C4_ARITHMETIC_IMPLEMENTATION': 'how to implement arithmetic operations in code',
    'KC_C9_CLASS_DEFINITION': 'how to define Python classes (you do not know the "class" keyword or object-oriented programming)',
    'KC_C10_INIT_METHOD': 'how to define __init__ methods in classes',
    'KC_C11_INSTANCE_VARIABLES': 'how to use instance variables (self.x)',
    'KC_C12_METHOD_DEFINITION': 'how to define methods in classes',
    'KC_C13_LOOP_CONSTRUCT': 'how to use loops (for, while)',
    'KC_C14_CONDITIONAL_LOGIC': 'conditional statements (if/else)',
    'KC_C15_VARIABLE_UPDATE': 'how to update variables (x = x + 1)',
    'KC_C16_FUNCTION_CALL': 'how to call functions with arguments',
    'KC_P1_VECTOR_DECOMPOSITION': 'how to decompose vectors into components',
    'KC_P2_TRIG_APPLICATION': 'trigonometric functions for physics problems',
    'KC_P3_TIME_OF_FLIGHT': 'time of flight calculations',
    'KC_P4_RANGE_FORMULA': 'projectile range formula',
    'KC_P5_UNIT_RADIANS': 'the math.radians() function or radian conversion (you do not know that angles need to be converted from degrees to radians)',
    'KC_P9_NUMERICAL_INTEGRATION': 'numerical integration (Euler method)',
    'KC_P10_FORCE_ACCELERATION': 'F = ma calculations',
    'KC_P11_KINETIC_ENERGY': 'kinetic energy formula (0.5*m*v^2)',
    'KC_P12_POTENTIAL_ENERGY': 'potential energy formulas (spring PE = 0.5*k*x² or gravitational PE = m*g*h)',
    'KC_P13_HOOKES_LAW': "Hooke's Law (F = -kx) for spring force",
    'KC_P14_COLLISION_REFLECTION': 'velocity reversal during collisions/bounces',
    'KC_P15_RELATIVE_VELOCITY': 'vector addition of velocities (boat + current)',
    'KC_P16_FRICTION': 'friction force calculations (μ*N or μ*mg*cos(θ))',
    'KC_M1_DERIVATIVE_CONCEPT': 'the concept of derivatives/gradients',
    'KC_M2_NUMERICAL_DIFFERENTIATION': 'numerical differentiation formulas',
    'KC_M3_ITERATIVE_ALGORITHM': 'iterative algorithms that converge',
    'KC_M4_CONVERGENCE_CRITERIA': 'stopping conditions for algorithms',
    'KC_M5_LEARNING_RATE': 'learning rate / step size in optimization',
}


@dataclass
class BKTParameters:
    """
    Standard BKT parameters for a Knowledge Component with validated constraints.
    
    Parameter Ranges (based on BKT literature):
    - Conservative: p(T)=0.1-0.2, p(S)=0.02-0.1, p(G)=0.1-0.3, p(L₀)=0.2-0.4
    - Standard: p(T)=0.15-0.25, p(S)=0.05-0.15, p(G)=0.1-0.25, p(L₀)=0.25-0.5
    - Liberal: p(T)≤0.6, p(S)≤0.3 (0.5 max), p(G)≤0.3 (0.5 max), p(L₀)≤1.0
    
    Constraints:
    - p(T) > 0 (must have positive learning rate)
    - p(S) < (1 - p(G)) (slip and guess probabilities must be complementary)
    - p(G) < (1 - p(S)) (slip and guess probabilities must be complementary)
    - All parameters in [0, 1]
    
    Attributes:
        p_init: Initial probability of knowing (P(L₀)) - range [0, 1]
        p_learn: Probability of learning (P(T)) - must be > 0, recommended ≤ 0.6
        p_slip: Probability of slip (P(S)) - recommended ≤ 0.3, must be < (1 - p(G))
        p_guess: Probability of guess (P(G)) - recommended ≤ 0.3, must be < (1 - p(S))
    """
    p_init: float = 0.1  # P(L₀)
    p_learn: float = 0.25  # P(T) - Increased from 0.15 to prevent BKT collapse
    p_slip: float = 0.05  # P(S) - Decreased from 0.1 to be less punishing
    p_guess: float = 0.2  # P(G)

    def __post_init__(self):
        """
        Validate BKT parameters against standard constraints.
        
        Raises:
            ValueError: If any parameter violates constraints
        """
        errors = []

        # Basic range constraints [0, 1]
        for name, value in [
            ("p_init", self.p_init), ("p_learn", self.p_learn),
            ("p_slip", self.p_slip), ("p_guess", self.p_guess)
        ]:
            if not 0 <= value <= 1:
                errors.append(f"{name}={value:.3f} must be in [0, 1]")

        # Constraint: p(T) > 0 (positive learning rate)
        if self.p_learn <= 0:
            errors.append(
                f"p_learn={self.p_learn:.3f} must be > 0 "
                "(learning rate must be positive)"
            )

        # Constraint: p(T) ≤ 0.6 (liberal upper bound warning)
        if self.p_learn > 0.6:
            logger.warning(
                f"p_learn={self.p_learn:.3f} exceeds liberal upper bound of 0.6. "
                "This may indicate unrealistically fast learning."
            )

        # Constraint: p(S) < (1 - p(G))
        if self.p_slip >= (1 - self.p_guess):
            errors.append(
                f"p_slip={self.p_slip:.3f} must be < (1 - p_guess)="
                f"{1 - self.p_guess:.3f}. "
                "Slip and guess probabilities must satisfy: p(S) + p(G) < 1"
            )

        # Constraint: p(G) < (1 - p(S))
        if self.p_guess >= (1 - self.p_slip):
            errors.append(
                f"p_guess={self.p_guess:.3f} must be < (1 - p_slip)="
                f"{1 - self.p_slip:.3f}. "
                "Slip and guess probabilities must satisfy: p(S) + p(G) < 1"
            )

        # Warning: p(S) > 0.3 (liberal upper bound)
        if self.p_slip > 0.3:
            logger.warning(
                f"p_slip={self.p_slip:.3f} exceeds recommended upper bound of 0.3. "
                "High slip rates may indicate poor KC definition."
            )

        # Warning: p(G) > 0.3 (liberal upper bound)
        if self.p_guess > 0.3:
            logger.warning(
                f"p_guess={self.p_guess:.3f} exceeds recommended upper bound of 0.3. "
                "High guess rates may indicate poor assessment design."
            )

        # Maximum bounds (0.5) for slip and guess
        if self.p_slip > 0.5:
            errors.append(
                f"p_slip={self.p_slip:.3f} must be ≤ 0.5 (absolute maximum)"
            )

        if self.p_guess > 0.5:
            errors.append(
                f"p_guess={self.p_guess:.3f} must be ≤ 0.5 (absolute maximum)"
            )

        # Raise all errors together
        if errors:
            raise ValueError(
                "BKT parameter validation failed:\n  - " +
                "\n  - ".join(errors)
            )


@dataclass
class BKTState:
    """
    Current BKT state for a Knowledge Component.

    Attributes:
        kc_id: Knowledge Component identifier
        p_known: Current probability of knowing P(L)
        mastery_level: Qualitative label derived from P(L)
        parameters: BKT parameters
        observation_count: Number of observations processed
        correct_count: Number of correct observations
        efi_active: Explicit Flaw Injection flag - when True, student has a misconception
        demonstrated: Whether this KC has been demonstrated through passing tests
                     Once demonstrated=True, student cannot "slip" on this KC anymore
    """
    kc_id: str
    p_known: float
    mastery_level: MasteryLevel
    parameters: BKTParameters
    observation_count: int = 0
    correct_count: int = 0
    efi_active: bool = False
    demonstrated: bool = False  # Locked once tested successfully

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "kc_id": self.kc_id,
            "p_known": self.p_known,
            "mastery_level": self.mastery_level.value,
            "parameters": {
                "p_init": self.parameters.p_init,
                "p_learn": self.parameters.p_learn,
                "p_slip": self.parameters.p_slip,
                "p_guess": self.parameters.p_guess
            },
            "observation_count": self.observation_count,
            "correct_count": self.correct_count,
            "efi_active": self.efi_active,
            "demonstrated": self.demonstrated
        }


class BKTEngine:
    """
    Core BKT engine implementing standard equations.
    
    Standard BKT Equations:
    (1) P(L_n | correct) = [P(L_n)(1 - P(S))] / [P(L_n)(1 - P(S)) + (1 - P(L_n))P(G)]
    (2) P(L_n | wrong) = [P(L_n)P(S)] / [P(L_n)P(S) + (1 - P(L_n))(1 - P(G))]
    (3) P(L_n+1) = P(L_n | obs) + (1 - P(L_n | obs))P(T)
    
    Where:
    - P(L) = Probability of knowing the skill
    - P(S) = Probability of slip (knew but got wrong)
    - P(G) = Probability of guess (didn't know but got right)
    - P(T) = Probability of learning from the opportunity
    """

    # Thresholds for mastery level classification
    MASTERY_THRESHOLDS = {
        MasteryLevel.UNKNOWN: (0.0, 0.3),
        MasteryLevel.PARTIAL: (0.3, 0.7),
        MasteryLevel.MASTERED: (0.7, 1.0)
    }

    @staticmethod
    def update(state: BKTState, observation: Observation) -> BKTState:
        """
        Update BKT state given an observation.
        
        Args:
            state: Current BKT state
            observation: Observed outcome (CORRECT/WRONG)
            
        Returns:
            Updated BKT state
        """
        # Get parameters
        p_s = state.parameters.p_slip
        p_g = state.parameters.p_guess
        p_t = state.parameters.p_learn
        p_l = state.p_known

        # Apply Equation (1) or (2) to update belief based on observation
        if observation == Observation.CORRECT:
            # Equation (1): P(L_n | obs_n = correct)
            numerator = p_l * (1 - p_s)
            denominator = p_l * (1 - p_s) + (1 - p_l) * p_g
            p_l_given_obs = numerator / denominator if denominator > 0 else p_l
        else:  # WRONG
            # Equation (2): P(L_n | obs_n = wrong)
            numerator = p_l * p_s
            denominator = p_l * p_s + (1 - p_l) * (1 - p_g)
            p_l_given_obs = numerator / denominator if denominator > 0 else p_l

        # Apply Equation (3): Incorporate learning opportunity
        # P(L_n+1) = P(L_n | obs_n) + (1 - P(L_n | obs_n))P(T)
        p_l_new = p_l_given_obs + (1 - p_l_given_obs) * p_t

        # Determine new mastery level
        mastery = BKTEngine._classify_mastery(p_l_new)

        # Update counts
        new_obs_count = state.observation_count + 1
        new_correct_count = state.correct_count + (
            1 if observation == Observation.CORRECT else 0
        )

        logger.debug(
            f"BKT Update [{state.kc_id}]: "
            f"P(L): {p_l:.3f} → {p_l_new:.3f}, "
            f"Obs: {observation.value}, "
            f"Mastery: {state.mastery_level.value} → {mastery.value}"
        )

        return BKTState(
            kc_id=state.kc_id,
            p_known=p_l_new,
            mastery_level=mastery,
            parameters=state.parameters,
            observation_count=new_obs_count,
            correct_count=new_correct_count,
            efi_active=state.efi_active,  # Preserve EFI flag
            demonstrated=state.demonstrated  # Preserve demonstrated flag
        )

    @staticmethod
    def _classify_mastery(p_known: float) -> MasteryLevel:
        """Classify P(L) into qualitative mastery level."""
        for level, (low, high) in BKTEngine.MASTERY_THRESHOLDS.items():
            if low <= p_known < high:
                return level
        return MasteryLevel.MASTERED  # p_known == 1.0


class BKT:
    """
    Main interface for Bayesian Knowledge Tracing.
    
    Usage:
        bkt = BKT()
        bkt.initialize_kc("KC_PYTHON_LOOPS")
        bkt.update("KC_PYTHON_LOOPS", Observation.CORRECT)
        state = bkt.get_kc_state("KC_PYTHON_LOOPS")
    """

    def __init__(
        self, kc_parameters: Optional[Dict[str, BKTParameters]] = None
    ):
        """
        Initialize BKT system.
        
        Args:
            kc_parameters: Optional dict mapping KC IDs to custom BKT parameters
        """
        self.kc_states: Dict[str, BKTState] = {}
        self.default_parameters = BKTParameters()
        self.kc_parameters = kc_parameters or {}

        logger.info("Initialized BKT system")

    def initialize_kc(
        self,
        kc_id: str,
        parameters: Optional[BKTParameters] = None,
        efi_active: bool = False
    ) -> BKTState:
        """
        Initialize BKT state for a Knowledge Component.
        
        Args:
            kc_id: Knowledge Component identifier
            parameters: Optional custom BKT parameters for this KC
            efi_active: Explicit Flaw Injection - if True, student has a misconception
                       that causes consistent errors until corrected
            
        Returns:
            Initial BKT state

        Raises:
            ValueError: If kc_id is not in VALID_KC_IDS registry
        """
        # Validate KC is registered
        validate_kc_id(kc_id)

        params = parameters or self.kc_parameters.get(
            kc_id, self.default_parameters
        )
        state = BKTState(
            kc_id=kc_id,
            p_known=params.p_init,
            mastery_level=BKTEngine._classify_mastery(params.p_init),
            parameters=params,
            efi_active=efi_active
        )
        self.kc_states[kc_id] = state
        logger.info(f"Initialized KC: {kc_id} with P(L₀)={params.p_init:.3f}")
        return state

    def update(self, kc_id: str, observation: Observation) -> BKTState:
        """
        Update BKT state for a KC given an observation.
        
        EFI Behavior: 
        - If EFI is active and observation is WRONG: Freeze BKT (no update)
          The error is due to misconception, not lack of knowledge
        - If EFI is active and observation is CORRECT: Deactivate EFI and update BKT
          This represents corrective feedback fixing the misconception
        
        Args:
            kc_id: Knowledge Component identifier
            observation: CORRECT or WRONG observation
            
        Returns:
            Updated BKT state (or unchanged state if EFI freezes update)
        """
        # Initialize KC if needed
        if kc_id not in self.kc_states:
            self.initialize_kc(kc_id)

        current_state = self.kc_states[kc_id]

        # EFI: Freeze BKT updates when misconception causes errors
        if current_state.efi_active and observation == Observation.WRONG:
            logger.debug(
                f"EFI active for {kc_id}: freezing BKT update "
                "(error due to misconception, not lack of knowledge)"
            )
            # Return current state without updating P(L)
            # Just increment observation count
            frozen_state = BKTState(
                kc_id=current_state.kc_id,
                p_known=current_state.p_known,  # Keep P(L) frozen
                mastery_level=current_state.mastery_level,
                parameters=current_state.parameters,
                observation_count=current_state.observation_count + 1,
                correct_count=current_state.correct_count,
                efi_active=True  # Keep EFI active
            )
            self.kc_states[kc_id] = frozen_state
            return frozen_state

        # EFI: Deactivate explicit flaw when correct observation received
        if current_state.efi_active and observation == Observation.CORRECT:
            logger.info(
                f"EFI deactivated for {kc_id}: correct observation received "
                "(misconception corrected)"
            )
            current_state.efi_active = False

        # Update using BKT engine (normal update or after EFI deactivation)
        updated_state = BKTEngine.update(current_state, observation)

        # Preserve EFI state in updated state
        updated_state.efi_active = current_state.efi_active

        self.kc_states[kc_id] = updated_state

        return updated_state

    def sample(self, kc_id: str) -> Observation:
        """
        Sample whether student would get this KC correct or incorrect.

        Uses BKT model to probabilistically determine performance based on:
        - P(L): Current probability student knows the skill
        - P(S): Probability of slip (knows but gets wrong)
        - P(G): Probability of guess (doesn't know but gets right)
        - EFI: Explicit Flaw Injection - if active, always returns WRONG
        - Demonstrated: If KC has been demonstrated through testing, no slipping

        Formula (when EFI is inactive and not demonstrated):
        P(correct) = P(L) × (1 - P(S)) + (1 - P(L)) × P(G)

        Formula (when demonstrated=True, no slipping allowed):
        P(correct) = P(L) + (1 - P(L)) × P(G)

        Args:
            kc_id: Knowledge Component identifier

        Returns:
            Observation.CORRECT or Observation.WRONG based on sampling
            If EFI is active, always returns Observation.WRONG
            If demonstrated=True, slipping is disabled

        Example:
            >>> bkt = BKT()
            >>> bkt.initialize_kc("KC_LOOPS")
            >>> observation = bkt.sample("KC_LOOPS")
            >>> # observation is CORRECT or WRONG based on current mastery
        """
        import random

        # Initialize KC if needed
        if kc_id not in self.kc_states:
            self.initialize_kc(kc_id)

        state = self.kc_states[kc_id]

        # EFI: If explicit flaw is active, always return WRONG
        if state.efi_active:
            logger.debug(
                f"EFI active for {kc_id}: sampling WRONG (misconception)"
            )
            return Observation.WRONG

        # If KC has been demonstrated through testing, no slipping allowed
        # P(correct) = P(L) + (1 - P(L)) × P(G)  (no slip term)
        if state.demonstrated:
            p_l = state.p_known
            p_g = state.parameters.p_guess
            # Student either knows it (P(L)) or might guess correctly ((1-P(L))*P(G))
            p_correct = p_l + (1 - p_l) * p_g
            logger.debug(
                f"KC {kc_id} demonstrated: P(correct)={p_correct:.3f} (no slip)"
            )
        else:
            # Standard BKT formula with slipping
            # P(correct) = P(L) × (1 - P(S)) + (1 - P(L)) × P(G)
            p_l = state.p_known
            p_s = state.parameters.p_slip
            p_g = state.parameters.p_guess
            p_correct = p_l * (1 - p_s) + (1 - p_l) * p_g

        # Sample from Bernoulli distribution
        if random.random() < p_correct:
            return Observation.CORRECT
        else:
            return Observation.WRONG

    def set_efi(self, kc_id: str, active: bool) -> None:
        """
        Set Explicit Flaw Injection status for a KC.
        
        Args:
            kc_id: Knowledge Component identifier
            active: True to activate EFI (inject flaw), False to deactivate
        """
        if kc_id not in self.kc_states:
            self.initialize_kc(kc_id, efi_active=active)
        else:
            self.kc_states[kc_id].efi_active = active
            logger.info(
                f"EFI {'activated' if active else 'deactivated'} for {kc_id}"
            )

    def has_efi(self, kc_id: str) -> bool:
        """
        Check if EFI is active for a KC.

        Args:
            kc_id: Knowledge Component identifier

        Returns:
            True if EFI is active, False otherwise
        """
        if kc_id not in self.kc_states:
            return False
        return self.kc_states[kc_id].efi_active

    def mark_demonstrated(self, kc_id: str) -> None:
        """
        Mark a KC as demonstrated through successful testing.

        Once a KC is demonstrated, the student cannot "slip" on it anymore.
        This represents locking in knowledge that has been proven through tests.

        Args:
            kc_id: Knowledge Component identifier
        """
        if kc_id not in self.kc_states:
            logger.warning(f"Cannot mark {kc_id} as demonstrated: not initialized")
            return

        if not self.kc_states[kc_id].demonstrated:
            self.kc_states[kc_id].demonstrated = True
            logger.info(f"KC {kc_id} marked as demonstrated (no more slipping)")

    def unblock_efi(self, kc_id: str, source: str = "tutor") -> bool:
        """
        Tutor-mediated unblocking of an EFI-active KC.

        Without this path an EFI-active KC is mathematically unrecoverable:
          - sample() always returns WRONG when EFI is active (line ~548).
          - update() FREEZES P(L) when EFI-active + WRONG (lines ~470-488).
        So under self-practice alone, P(L) stays pinned at its initial value
        (typically 0.10) forever — never crossing the 0.30 unblock threshold.
        This contradicts the SRL framing where students learn unknown
        concepts when explained by someone (the tutor).

        Calling this method:
          1. Deactivates EFI on the KC (sample() resumes normal behavior).
          2. Credits one CORRECT observation (one-shot Bayesian bump,
             which under standard parameters lifts P(L) from 0.10 to ~0.51,
             clearly above the unblock threshold).

        No-op if the KC is unknown or EFI is not currently active.

        Args:
            kc_id: Knowledge Component identifier
            source: Provenance string for logging (e.g., "tutor", "scaffold")

        Returns:
            True if EFI was deactivated, False if no-op.
        """
        if kc_id not in self.kc_states:
            return False
        state = self.kc_states[kc_id]
        if not state.efi_active:
            return False
        state.efi_active = False
        # Apply one positive observation to break the 0.267 fixed point.
        self.update(kc_id, Observation.CORRECT)
        logger.info(
            f"EFI unblocked on {kc_id} via {source}; "
            f"one CORRECT observation credited (P(L)={self.kc_states[kc_id].p_known:.3f})"
        )
        return True

    def is_demonstrated(self, kc_id: str) -> bool:
        """
        Check if a KC has been demonstrated through testing.

        Args:
            kc_id: Knowledge Component identifier

        Returns:
            True if KC has been demonstrated, False otherwise
        """
        if kc_id not in self.kc_states:
            return False
        return self.kc_states[kc_id].demonstrated

    def sample_and_feedback(
        self, kc_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample correct/incorrect for each KC and generate template-based feedback.

        This is the main method for generating BKT-driven feedback for the student.
        For each KC:
        1. Sample from BKT using P(correct) = P(L)*(1-P(S)) + (1-P(L))*P(G)
        2. Generate feedback based on state:
           - EFI active: "You do not know about [KC]" (student doesn't know it exists)
           - Sampled WRONG: "You incorrectly applied [KC]" (tried but failed)
           - Sampled CORRECT: "You correctly applied [KC]"
        3. Return the sampled observation for later BKT update

        Args:
            kc_ids: List of KC identifiers to sample

        Returns:
            Dict mapping kc_id to {
                'observation': Observation.CORRECT or WRONG,
                'feedback': feedback string based on state,
                'p_correct': The probability that was used for sampling,
                'p_known': Current P(L) for this KC,
                'efi_active': Whether EFI is blocking this KC
            }

        Example:
            >>> results = bkt.sample_and_feedback(["KC_RETURN", "KC_MATH_LIBRARY"])
            >>> # If KC_MATH_LIBRARY has EFI active:
            >>> # "You do not know about c2 math library"
            >>> # If KC_RETURN sampled CORRECT:
            >>> # "You correctly applied return statements"
        """
        results = {}

        for kc_id in kc_ids:
            # Initialize if needed
            if kc_id not in self.kc_states:
                self.initialize_kc(kc_id)

            state = self.kc_states[kc_id]

            # Sample observation
            observation = self.sample(kc_id)

            # Calculate p_correct for logging
            p_l = state.p_known
            p_s = state.parameters.p_slip
            p_g = state.parameters.p_guess

            if state.demonstrated:
                p_correct = p_l + (1 - p_l) * p_g
            else:
                p_correct = p_l * (1 - p_s) + (1 - p_l) * p_g

            # Generate human-readable KC name
            kc_name = kc_id.replace("KC_", "").replace("_", " ").lower()

            # Generate feedback based on state
            # EFI = student doesn't know this concept exists (different from "tried but failed")
            if state.efi_active:
                feedback = f"You do not know about {kc_name}"
            elif observation == Observation.CORRECT:
                feedback = f"You correctly applied {kc_name}"
            else:
                feedback = f"You incorrectly applied {kc_name}"

            results[kc_id] = {
                'observation': observation,
                'feedback': feedback,
                'p_correct': p_correct,
                'p_known': p_l,
                'demonstrated': state.demonstrated,
                'efi_active': state.efi_active
            }

        return results

    def get_sampled_knowledge_state(
        self, kc_ids: Optional[List[str]] = None
    ) -> str:
        """
        Generate knowledge state description using BKT sampling.

        Groups KCs into three categories:
        1. EFI active: "You do not know about: [KCs]" (student doesn't know these exist)
        2. Sampled WRONG: "You incorrectly applied: [KCs]" (tried but failed)
        3. Sampled CORRECT: "You correctly applied: [KCs]"

        Args:
            kc_ids: Optional list of KCs to include. If None, uses all KCs.

        Returns:
            Template-based feedback string like:
            "You do not know about: math library.
             You correctly applied: function definition.
             You incorrectly applied: arithmetic operations."
        """
        target_kcs = kc_ids if kc_ids else list(self.kc_states.keys())

        if not target_kcs:
            return "No knowledge components tracked."

        # Sample all KCs
        results = self.sample_and_feedback(target_kcs)

        # Group by: unknown (EFI), correct, incorrect
        unknown_kcs = []  # EFI active - student doesn't know these exist
        correct_kcs = []  # Sampled correct
        incorrect_kcs = []  # Sampled incorrect (but knows it exists)

        for kc_id, info in results.items():
            # Use explicit description for EFI, fallback to formatted name
            if info['efi_active']:
                kc_desc = KC_EFI_DESCRIPTIONS.get(kc_id, kc_id.replace("KC_", "").replace("_", " ").lower())
                unknown_kcs.append(kc_desc)
            else:
                kc_name = kc_id.replace("KC_", "").replace("_", " ").lower()
                if info['observation'] == Observation.CORRECT:
                    correct_kcs.append(kc_name)
                else:
                    incorrect_kcs.append(kc_name)

        # Build feedback string
        parts = []
        if unknown_kcs:
            # Strong constraint for EFI - student CANNOT use these concepts
            parts.append(
                f"CRITICAL CONSTRAINT - You have NEVER heard of and CANNOT use: {', '.join(unknown_kcs)}. "
                f"These concepts do not exist in your knowledge. You must solve the problem WITHOUT using them. "
                f"If the code requires '{', '.join(unknown_kcs)}', you will be stuck and confused."
            )
        if correct_kcs:
            parts.append(f"You correctly applied: {', '.join(correct_kcs)}")
        if incorrect_kcs:
            parts.append(f"You incorrectly applied: {', '.join(incorrect_kcs)}")

        return " ".join(parts) if parts else "No feedback available."

    def get_kc_state(self, kc_id: str) -> Optional[BKTState]:
        """Get current BKT state for a KC."""
        return self.kc_states.get(kc_id)

    def get_all_states(self) -> Dict[str, BKTState]:
        """Get all KC states."""
        return self.kc_states.copy()

    def get_mastery_summary(self) -> Dict[str, Any]:
        """
        Get summary of mastery across all KCs.
        
        Returns:
            Dict with counts and details for each mastery level
        """
        summary = {
            "total_kcs": len(self.kc_states),
            "mastered": [],
            "partial": [],
            "unknown": []
        }

        for kc_id, state in self.kc_states.items():
            kc_info = {
                "kc_id": kc_id,
                "p_known": state.p_known,
                "observations": state.observation_count
            }

            if state.mastery_level == MasteryLevel.MASTERED:
                summary["mastered"].append(kc_info)
            elif state.mastery_level == MasteryLevel.PARTIAL:
                summary["partial"].append(kc_info)
            else:
                summary["unknown"].append(kc_info)

        return summary

    def get_mastered_kcs(self) -> List[str]:
        """
        Get list of mastered KC IDs.
        
        Returns:
            List of KC IDs with mastery level MASTERED (P(L) >= 0.7)
        """
        return [
            kc_id for kc_id, state in self.kc_states.items()
            if state.mastery_level == MasteryLevel.MASTERED
        ]

    def get_knowledge_description(
        self,
        kc_ids: Optional[List[str]] = None,
        include_mastered: bool = False
    ) -> str:
        """
        Convert BKT state to natural language description of student's knowledge.
        
        This is used to inform LLMs about what the student knows/doesn't know,
        allowing them to naturally generate realistic student behavior based on
        knowledge gaps rather than hardcoded misconceptions.
        
        Args:
            kc_ids: Optional list of specific KCs to describe. If None, describes all KCs.
            include_mastered: Whether to include mastered KCs (default: False, only gaps)
        
        Returns:
            Natural language description of knowledge state
            
        Example:
            >>> bkt.get_knowledge_description()
            "Knowledge gaps: The student doesn't know about importing the math library 
            (P(L)=0.15). They have partial understanding of angle conversion (P(L)=0.45)."
        """
        if not self.kc_states:
            return "No knowledge components tracked."

        # Determine which KCs to describe
        target_kcs = kc_ids if kc_ids else list(self.kc_states.keys())

        # Group KCs by mastery level
        unknown = []  # P(L) < 0.3
        partial = []  # 0.3 <= P(L) < 0.7
        mastered = []  # P(L) >= 0.7

        for kc_id in target_kcs:
            if kc_id not in self.kc_states:
                continue

            state = self.kc_states[kc_id]
            p_l = state.p_known

            # Clean up KC name for natural language
            kc_name = kc_id.replace("KC_", "").replace("_", " ").lower()

            kc_desc = f"{kc_name} (P(L)={p_l:.2f})"

            if p_l < 0.3:
                unknown.append(kc_desc)
            elif p_l < 0.7:
                partial.append(kc_desc)
            else:
                mastered.append(kc_desc)

        # Build natural language description
        parts = []

        if unknown:
            parts.append(
                f"The student doesn't understand: {', '.join(unknown)}"
            )

        if partial:
            parts.append(
                f"The student has partial understanding of: {', '.join(partial)}"
            )

        if include_mastered and mastered:
            parts.append(f"The student has mastered: {', '.join(mastered)}")

        if not parts:
            if include_mastered:
                return "The student has mastered all knowledge components."
            else:
                return "The student has no significant knowledge gaps."

        return ". ".join(parts) + "."

    def print_kc_progress(
        self,
        sort_by: str = "mastery",
        show_stats: bool = True,
        use_color: bool = True
    ) -> None:
        """
        Print progress bars for all Knowledge Components.
        
        Args:
            sort_by: Sort KCs by 'mastery', 'name', or 'observations'
            show_stats: Show observation counts and accuracy
            use_color: Use ANSI color codes (set False for plain text)
        """
        if not self.kc_states:
            print("No Knowledge Components tracked yet.")
            return

        # ANSI color codes
        colors = {
            "red": "\033[91m",
            "yellow": "\033[93m",
            "green": "\033[92m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }

        def color_text(text: str, color: str) -> str:
            """Apply color to text if color is enabled."""
            if not use_color:
                return text
            return f"{colors.get(color, '')}{text}{colors['reset']}"

        # Sort KCs
        kc_list = list(self.kc_states.items())
        if sort_by == "mastery":
            kc_list.sort(key=lambda x: x[1].p_known, reverse=True)
        elif sort_by == "name":
            kc_list.sort(key=lambda x: x[0])
        elif sort_by == "observations":
            kc_list.sort(key=lambda x: x[1].observation_count, reverse=True)

        # Print header
        print("\n" + color_text("=" * 70, "bold"))
        print(color_text("Knowledge Component Progress", "bold"))
        print(color_text("=" * 70, "bold"))

        # Print each KC
        for kc_id, state in kc_list:
            mastery = state.p_known

            # Create progress bar (20 characters wide)
            bar_length = int(mastery * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            # Determine color based on mastery level
            if mastery >= 0.7:
                bar_color = "green"
                level = "MASTERED"
            elif mastery >= 0.3:
                bar_color = "yellow"
                level = "PARTIAL"
            else:
                bar_color = "red"
                level = "UNKNOWN"

            # Build the line
            colored_bar = color_text(bar, bar_color)
            mastery_text = color_text(f"{mastery:.3f}", bar_color)

            line = f"{kc_id:30s} {colored_bar} {mastery_text}"

            # Add EFI indicator if active
            if state.efi_active:
                efi_indicator = color_text(" [EFI]", "red")
                line += efi_indicator

            # Add stats if requested
            if show_stats:
                accuracy = (
                    state.correct_count / state.observation_count
                    if state.observation_count > 0 else 0.0
                )
                stats = f" [{state.observation_count:3d} obs, {accuracy:.1%} acc]"
                line += stats

            print(line)

        # Print summary
        summary = self.get_mastery_summary()
        print(color_text("-" * 70, "bold"))

        mastered_text = color_text(
            f"Mastered: {len(summary['mastered'])}", 'green'
        )
        partial_text = color_text(
            f"Partial: {len(summary['partial'])}", 'yellow'
        )
        unknown_text = color_text(f"Unknown: {len(summary['unknown'])}", 'red')

        print(
            f"Total KCs: {summary['total_kcs']} | {mastered_text} | {partial_text} | {unknown_text}"
        )
        print(color_text("=" * 70, "bold") + "\n")

    def get_kc_progress_data(self) -> list[Dict[str, Any]]:
        """
        Get progress data for all KCs (for use with external visualization).
        
        Returns:
            List of dicts with KC progress information
        """
        progress_data = []

        for kc_id, state in self.kc_states.items():
            accuracy = (
                state.correct_count /
                state.observation_count if state.observation_count > 0 else 0.0
            )

            # Determine color/level
            if state.p_known >= 0.7:
                color = "green"
                level = "MASTERED"
            elif state.p_known >= 0.3:
                color = "yellow"
                level = "PARTIAL"
            else:
                color = "red"
                level = "UNKNOWN"

            progress_data.append(
                {
                    "kc_id": kc_id,
                    "p_known": state.p_known,
                    "mastery_level": level,
                    "color": color,
                    "observation_count": state.observation_count,
                    "correct_count": state.correct_count,
                    "accuracy": accuracy
                }
            )

        return progress_data


if __name__ == "__main__":
    """Demo showcasing BKT progress tracking with visual progress bars."""
    import random

    print("\n" + "=" * 70)
    print("BKT Progress Bar Demo")
    print("=" * 70)
    print(
        "\nSimulating student learning across multiple Knowledge Components..."
    )

    # Initialize BKT
    bkt = BKT()

    # Define some Python KCs with different difficulty levels
    kcs = {
        "KC_PYTHON_VARIABLES": BKTParameters(p_init=0.3, p_learn=0.25),
        "KC_PYTHON_LOOPS": BKTParameters(p_init=0.1, p_learn=0.2),
        "KC_PYTHON_FUNCTIONS": BKTParameters(p_init=0.05, p_learn=0.15),
        "KC_PYTHON_CONDITIONALS": BKTParameters(p_init=0.2, p_learn=0.2),
        "KC_PYTHON_LISTS": BKTParameters(p_init=0.1, p_learn=0.18),
        "KC_PYTHON_DICTS": BKTParameters(p_init=0.05, p_learn=0.12),
        "KC_PYTHON_STRING_OPS": BKTParameters(p_init=0.15, p_learn=0.22),
        "KC_PYTHON_FILE_IO": BKTParameters(p_init=0.02, p_learn=0.1),
    }

    # Initialize all KCs
    for kc_id, params in kcs.items():
        bkt.initialize_kc(kc_id, params)

    print("\n📊 Initial State:")
    bkt.print_kc_progress(sort_by="name", show_stats=False)

    # Simulate learning over multiple rounds
    print("\n🎓 Simulating 5 practice rounds...\n")

    for round_num in range(1, 6):
        print(f"Round {round_num}: Practicing problems...")

        # Each round, student practices each KC
        for kc_id in kcs.keys():
            # Use the new sample() method to determine if student gets it right
            observation = bkt.sample(kc_id)
            bkt.update(kc_id, observation)

        # Show progress after each round
        if round_num % 2 == 0:
            print(f"\n📊 Progress after Round {round_num}:")
            bkt.print_kc_progress(sort_by="mastery")

    # Final summary
    print("\n" + "=" * 70)
    print("🎯 FINAL LEARNING SUMMARY")
    print("=" * 70)
    bkt.print_kc_progress(sort_by="mastery")

    # Show some statistics
    summary = bkt.get_mastery_summary()
    print("\n📈 Learning Analytics:")
    print(f"  • Started with {len(kcs)} Knowledge Components")
    print(f"  • Mastered: {len(summary['mastered'])} KCs")
    print(f"  • Partial Mastery: {len(summary['partial'])} KCs")
    print(f"  • Still Learning: {len(summary['unknown'])} KCs")

    # Show most/least mastered
    if summary['mastered']:
        best = max(summary['mastered'], key=lambda x: x['p_known'])
        print(
            f"\n  🏆 Best Performance: {best['kc_id']} (P(L)={best['p_known']:.3f})"
        )

    if summary['unknown']:
        needs_work = min(summary['unknown'], key=lambda x: x['p_known'])
        print(
            f"  📚 Needs More Practice: {needs_work['kc_id']} (P(L)={needs_work['p_known']:.3f})"
        )

    print("\n" + "=" * 70)
    print(
        "Demo complete! Try running: python -m beagle.data_generation.bkt.bkt"
    )
    print("=" * 70 + "\n")
