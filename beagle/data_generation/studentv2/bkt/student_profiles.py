"""
Student profiles for realistic BKT initialization.

This module provides pre-defined student performance profiles using Beta distributions
to model realistic initial knowledge states. These profiles are based on educational
research showing that student populations typically follow non-uniform distributions
rather than all starting at the same mastery level.

Usage:
    from beagle.data_generation.bkt.student_profiles import StudentProfile, sample_initial_mastery
    
    # Create a below-average student
    p_init = sample_initial_mastery(StudentProfile.BELOW_AVERAGE)
    
    # Or sample multiple KCs for one student
    from beagle.data_generation.bkt.student_profiles import initialize_student_kcs
    bkt = BKT()
    initialize_student_kcs(bkt, ["KC1", "KC2", "KC3"], StudentProfile.AVERAGE)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class StudentProfile(Enum):
    """
    Pre-defined student performance profiles based on typical classroom distributions.
    
    Each profile uses a Beta distribution Beta(α, β) to model the initial mastery
    probability P(L₀) for knowledge components. The Beta distribution provides
    realistic variation within each profile's expected range.
    
    Attributes:
        ZERO: No initial knowledge (fixed at 0.0 for all KCs)
        STRUGGLING: Very low mastery (mean ~0.10, range 0.0-0.25)
        BELOW_AVERAGE: Low-moderate mastery (mean ~0.29, range 0.10-0.50)
        AVERAGE: Moderate mastery (mean ~0.50, range 0.20-0.80)
        ABOVE_AVERAGE: Moderate-high mastery (mean ~0.71, range 0.50-0.90)
        ADVANCED: High mastery (mean ~0.90, range 0.75-1.0)
    """
    ZERO = "zero"
    STRUGGLING = "struggling"
    BELOW_AVERAGE = "below_average"
    AVERAGE = "average"
    ABOVE_AVERAGE = "above_average"
    ADVANCED = "advanced"


@dataclass
class ProfileConfig:
    """
    Configuration for a student profile's Beta distribution.
    
    Attributes:
        alpha: Beta distribution α parameter (shape parameter 1)
        beta: Beta distribution β parameter (shape parameter 2)
        description: Human-readable description of the profile
        mean: Expected mean of P(L₀) for this profile
        typical_range: Typical range (10th-90th percentile) of P(L₀) values
    """
    alpha: float
    beta: float
    description: str
    mean: float
    typical_range: tuple[float, float]


# Profile definitions with Beta distribution parameters
PROFILE_CONFIGS: Dict[StudentProfile, ProfileConfig] = {
    StudentProfile.ZERO:
    ProfileConfig(
        alpha=0.0,  # Special case: fixed at 0
        beta=0.0,   # Not used (fixed value)
        description="Zero initial knowledge for controlled experiments",
        mean=0.0,
        typical_range=(0.0, 0.0)
    ),
    StudentProfile.STRUGGLING:
    ProfileConfig(
        alpha=1.0,
        beta=9.0,
        description="Struggling student with minimal prior knowledge",
        mean=0.10,
        typical_range=(0.01, 0.25)
    ),
    StudentProfile.BELOW_AVERAGE:
    ProfileConfig(
        alpha=2.0,
        beta=5.0,
        description="Below-average student with developing knowledge",
        mean=0.29,
        typical_range=(0.10, 0.50)
    ),
    StudentProfile.AVERAGE:
    ProfileConfig(
        alpha=2.0,
        beta=2.0,
        description="Average student with moderate knowledge",
        mean=0.50,
        typical_range=(0.20, 0.80)
    ),
    StudentProfile.ABOVE_AVERAGE:
    ProfileConfig(
        alpha=5.0,
        beta=2.0,
        description="Above-average student with strong knowledge",
        mean=0.71,
        typical_range=(0.50, 0.90)
    ),
    StudentProfile.ADVANCED:
    ProfileConfig(
        alpha=9.0,
        beta=1.0,
        description="Advanced student with excellent prior knowledge",
        mean=0.90,
        typical_range=(0.75, 0.99)
    )
}


def sample_initial_mastery(
    profile: StudentProfile, random_state: Optional[int] = None
) -> float:
    """
    Sample an initial mastery probability P(L₀) from a student profile.
    
    Uses the Beta distribution specified by the profile to generate a realistic
    initial mastery level. The distribution ensures natural variation while
    staying within the expected range for that student type.
    
    Args:
        profile: The student profile to sample from
        random_state: Optional random seed for reproducibility
        
    Returns:
        Initial mastery probability P(L₀) in range [0, 1]
        
    Example:
        >>> p_init = sample_initial_mastery(StudentProfile.AVERAGE)
        >>> print(f"Initial mastery: {p_init:.3f}")
        Initial mastery: 0.523
        
        >>> # Reproducible sampling
        >>> p_init = sample_initial_mastery(StudentProfile.BELOW_AVERAGE, random_state=42)
        >>> print(f"Initial mastery: {p_init:.3f}")
        Initial mastery: 0.354
    """
    config = PROFILE_CONFIGS[profile]

    # Special case: ZERO profile always returns 0.0
    if profile == StudentProfile.ZERO:
        return 0.0

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return rng.beta(config.alpha, config.beta)
    else:
        return np.random.beta(config.alpha, config.beta)


def sample_multiple_mastery(
    profile: StudentProfile,
    n_kcs: int,
    random_state: Optional[int] = None
) -> List[float]:
    """
    Sample initial mastery probabilities for multiple KCs from a student profile.
    
    Args:
        profile: The student profile to sample from
        n_kcs: Number of knowledge components to sample for
        random_state: Optional random seed for reproducibility
        
    Returns:
        List of initial mastery probabilities, one per KC
        
    Example:
        >>> p_inits = sample_multiple_mastery(StudentProfile.AVERAGE, n_kcs=3, random_state=42)
        >>> for i, p in enumerate(p_inits, 1):
        ...     print(f"KC{i}: P(L₀) = {p:.3f}")
        KC1: P(L₀) = 0.569
        KC2: P(L₀) = 0.566
        KC3: P(L₀) = 0.581
    """
    config = PROFILE_CONFIGS[profile]

    # Special case: ZERO profile always returns 0.0 for all KCs
    if profile == StudentProfile.ZERO:
        return [0.0] * n_kcs

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return rng.beta(config.alpha, config.beta, size=n_kcs).tolist()
    else:
        return np.random.beta(config.alpha, config.beta, size=n_kcs).tolist()


def initialize_student_kcs(
    bkt,
    kc_names: List[str],
    profile: StudentProfile,
    base_params: Optional[Dict] = None,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Initialize multiple KCs for a student using a profile-based approach.
    
    This is a convenience function that samples initial mastery levels from
    a student profile and initializes all KCs with those values. Each KC
    gets a unique P(L₀) sampled from the profile's distribution.
    
    Args:
        bkt: BKT instance to initialize
        kc_names: List of KC names to initialize
        profile: Student profile to use for sampling
        base_params: Optional dict of base BKT parameters (p_learn, p_slip, p_guess).
                    If not provided, uses BKT defaults.
        random_state: Optional random seed for reproducibility
        
    Returns:
        Dictionary mapping KC names to their initialized P(L₀) values
        
    Example:
        >>> from beagle.data_generation.bkt.bkt import BKT
        >>> bkt = BKT()
        >>> kcs = ["VECTORS", "KINEMATICS", "FORCES"]
        >>> init_values = initialize_student_kcs(
        ...     bkt, kcs, StudentProfile.BELOW_AVERAGE, random_state=42
        ... )
        >>> for kc, p_init in init_values.items():
        ...     print(f"{kc}: P(L₀) = {p_init:.3f}")
        VECTORS: P(L₀) = 0.354
        KINEMATICS: P(L₀) = 0.249
        FORCES: P(L₀) = 0.416
    """
    from .bkt import BKTParameters

    # Sample P(L₀) values for all KCs
    p_inits = sample_multiple_mastery(profile, len(kc_names), random_state)

    # Prepare base parameters
    if base_params is None:
        base_params = {}

    # Initialize each KC
    init_values = {}
    for kc_name, p_init in zip(kc_names, p_inits):
        params = BKTParameters(
            p_init=p_init,
            p_learn=base_params.get(
                'p_learn', 0.25
            ),  # Updated from 0.15 to prevent BKT collapse
            p_slip=base_params.get('p_slip', 0.05
                                   ),  # Updated from 0.1 to be less punishing
            p_guess=base_params.get('p_guess', 0.2)
        )
        bkt.initialize_kc(kc_name, params)
        init_values[kc_name] = p_init

    return init_values


def get_profile_info(profile: StudentProfile) -> ProfileConfig:
    """
    Get detailed information about a student profile.
    
    Args:
        profile: The student profile to query
        
    Returns:
        ProfileConfig with distribution parameters and statistics
        
    Example:
        >>> info = get_profile_info(StudentProfile.AVERAGE)
        >>> print(f"{info.description}")
        Average student with moderate knowledge
        >>> print(f"Mean: {info.mean:.2f}")
        Mean: 0.50
        >>> print(f"Typical range: {info.typical_range}")
        Typical range: (0.2, 0.8)
    """
    return PROFILE_CONFIGS[profile]


def print_all_profiles():
    """
    Print a summary of all available student profiles.
    
    Useful for understanding the characteristics of each profile type.
    
    Example:
        >>> print_all_profiles()
        Student Performance Profiles
        ============================
        
        STRUGGLING (Beta α=1.0, β=9.0)
          Description: Struggling student with minimal prior knowledge
          Mean P(L₀): 0.10
          Typical range: 0.01 - 0.25
        ...
    """
    print("\nStudent Performance Profiles")
    print("=" * 70)

    for profile in StudentProfile:
        config = PROFILE_CONFIGS[profile]
        print(f"\n{profile.name} (Beta α={config.alpha}, β={config.beta})")
        print(f"  Description: {config.description}")
        print(f"  Mean P(L₀): {config.mean:.2f}")
        print(
            f"  Typical range: {config.typical_range[0]:.2f} - {config.typical_range[1]:.2f}"
        )


def sample_student_population(
    profile_distribution: Dict[StudentProfile, float],
    n_students: int,
    random_state: Optional[int] = None
) -> List[StudentProfile]:
    """
    Sample a population of students with specified profile distribution.
    
    Useful for generating realistic classroom populations where students
    have varying performance levels.
    
    Args:
        profile_distribution: Dict mapping profiles to their proportions
                             (must sum to 1.0)
        n_students: Number of students to generate
        random_state: Optional random seed for reproducibility
        
    Returns:
        List of StudentProfile assignments, one per student
        
    Example:
        >>> # Create a realistic classroom: 10% struggling, 25% below avg,
        >>> # 30% average, 25% above avg, 10% advanced
        >>> distribution = {
        ...     StudentProfile.STRUGGLING: 0.10,
        ...     StudentProfile.BELOW_AVERAGE: 0.25,
        ...     StudentProfile.AVERAGE: 0.30,
        ...     StudentProfile.ABOVE_AVERAGE: 0.25,
        ...     StudentProfile.ADVANCED: 0.10
        ... }
        >>> students = sample_student_population(distribution, n_students=20, random_state=42)
        >>> from collections import Counter
        >>> Counter(students)
        Counter({<StudentProfile.AVERAGE: 'average'>: 7,
                 <StudentProfile.BELOW_AVERAGE: 'below_average'>: 5, ...})
    """
    # Validate distribution
    total = sum(profile_distribution.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Profile distribution must sum to 1.0, got {total}")

    # Convert to arrays for sampling
    profiles = list(profile_distribution.keys())
    probabilities = [profile_distribution[p] for p in profiles]

    # Sample using indices then map back to profiles
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(profiles), size=n_students, p=probabilities)
    else:
        indices = np.random.choice(
            len(profiles), size=n_students, p=probabilities
        )

    return [profiles[i] for i in indices]


# Convenience functions for common use cases
def create_struggling_student(
    bkt, kc_names: List[str], random_state: Optional[int] = None
):
    """Initialize a struggling student."""
    return initialize_student_kcs(
        bkt, kc_names, StudentProfile.STRUGGLING, random_state=random_state
    )


def create_below_average_student(
    bkt, kc_names: List[str], random_state: Optional[int] = None
):
    """Initialize a below-average student."""
    return initialize_student_kcs(
        bkt, kc_names, StudentProfile.BELOW_AVERAGE, random_state=random_state
    )


def create_average_student(
    bkt, kc_names: List[str], random_state: Optional[int] = None
):
    """Initialize an average student."""
    return initialize_student_kcs(
        bkt, kc_names, StudentProfile.AVERAGE, random_state=random_state
    )


def create_above_average_student(
    bkt, kc_names: List[str], random_state: Optional[int] = None
):
    """Initialize an above-average student."""
    return initialize_student_kcs(
        bkt, kc_names, StudentProfile.ABOVE_AVERAGE, random_state=random_state
    )


def create_advanced_student(
    bkt, kc_names: List[str], random_state: Optional[int] = None
):
    """Initialize an advanced student."""
    return initialize_student_kcs(
        bkt, kc_names, StudentProfile.ADVANCED, random_state=random_state
    )
