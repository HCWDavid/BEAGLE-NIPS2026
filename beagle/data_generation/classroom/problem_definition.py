"""
Problem Definition - Specification of a coding problem

Defines what a student needs to solve, including:
- Problem description
- Function signature
- Required knowledge components (KCs)

Tests are now in separate pytest files (test_*.py) in the problem directory.

Does NOT include:
- Solution information (that's for evaluation/tutor)
- Student-specific attributes (misconceptions are set when creating Student with EFI)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ProblemDefinition:
    """
    Definition of a coding problem.

    Contains only the problem specification - what the student needs to solve.
    Does NOT contain solution information or student-specific attributes.
    Tests are in separate pytest files.

    Attributes:
        problem_id: Unique identifier for the problem
        title: Human-readable problem title
        description: Full problem description (what to implement)
        function_name: Name of the function to implement
        required_kcs: Knowledge components needed to solve this problem
        starting_code: Optional initial code template students start with
    """
    problem_id: str
    title: str
    description: str
    function_name: str
    required_kcs: List[str]  # Knowledge components needed to solve this problem
    starting_code: str = ""  # Optional initial code template (e.g., Bielefeld template)
