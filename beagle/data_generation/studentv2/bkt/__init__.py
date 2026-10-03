"""
Bayesian Knowledge Tracing (BKT) for Knowledge Component Tracking

This module implements standard Bayesian Knowledge Tracing (BKT) for tracking
student knowledge acquisition over time.

Key Components:
- BKT: Main interface for knowledge tracing
- BKTEngine: Core BKT equations (deterministic)
- BKTParameters: Configuration for BKT parameters
- BKTState: Current state of a Knowledge Component
- Observation: CORRECT or WRONG observations
- MasteryLevel: Qualitative mastery categories
- StudentProfile: Pre-defined student performance profiles
"""

from .bkt import (
    BKT, BKTEngine, BKTParameters, BKTState, MasteryLevel, Observation
)
from .student_profiles import (
    StudentProfile,
    sample_initial_mastery,
    sample_multiple_mastery,
    initialize_student_kcs,
    get_profile_info,
    print_all_profiles,
    sample_student_population,
    create_struggling_student,
    create_below_average_student,
    create_average_student,
    create_above_average_student,
    create_advanced_student,
)

__all__ = [
    'BKT',
    'BKTEngine',
    'BKTParameters',
    'BKTState',
    'MasteryLevel',
    'Observation',
    'StudentProfile',
    'sample_initial_mastery',
    'sample_multiple_mastery',
    'initialize_student_kcs',
    'get_profile_info',
    'print_all_profiles',
    'sample_student_population',
    'create_struggling_student',
    'create_below_average_student',
    'create_average_student',
    'create_above_average_student',
    'create_advanced_student',
]
