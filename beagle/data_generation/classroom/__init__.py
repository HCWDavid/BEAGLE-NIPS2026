"""
Classroom - Environment for student-tutor-problem interaction

This module contains:
- AssessmentOracle: Evaluates KC mastery
- TutorAgent: Provides pedagogical hints (if available)

Note: The old Classroom class is deprecated. Use studentv2 with IDEOracle instead.
"""

# Only import the assessment oracle which is still used
from beagle.data_generation.classroom.assessment_oracle import AutomatedAssessmentOracle

__all__ = [
    'AutomatedAssessmentOracle'
]
