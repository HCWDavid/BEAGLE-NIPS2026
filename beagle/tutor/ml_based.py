"""
ML-Based Tutor (ITAP Style)

Based on Rivers & Koedinger (2017) "Data-Driven Hint Generation in
Vast Solution Spaces: a Self-Improving Python Programming Tutor"

This tutor uses BKT knowledge state to generate personalized hints
targeting the student's weakest knowledge components.

Reference:
    Rivers, K., & Koedinger, K. R. (2017). Data-driven hint generation
    in vast solution spaces: A self-improving python programming tutor.
    International Journal of Artificial Intelligence in Education, 27(1), 37-64.
"""

from typing import Dict, List, Optional, Tuple

from beagle.tutor.base import (
    TutorStrategy,
    TutorContext,
    TutorResponse,
    ScaffoldLevel,
)
from beagle.data_generation.studentv2.bkt.bkt import MasteryLevel


class MLBasedTutor(TutorStrategy):
    """
    ML-based tutor using BKT knowledge state for personalized hints.

    Implements the ITAP-style approach where hints are:
    1. Targeted at the lowest-mastery KC
    2. Personalized based on current knowledge state
    3. Graduated based on mastery level

    Key differences from Rule-Based:
        - Uses BKT mastery to personalize hints
        - Adapts hint specificity based on student's understanding level
        - Considers hint history for escalation
    """

    # KC -> Hint templates at different scaffold levels
    # Format: (MINIMAL, GUIDING, EXPLICIT, EXAMPLE)
    KC_HINT_TEMPLATES: Dict[str, Tuple[str, str, str, str]] = {
        "KC_P5_UNIT_RADIANS": (
            "Think about what type of units the math.sin() function expects.",
            "Trigonometric functions expect angles in a specific unit. What unit is that?",
            "You need to convert your angle from degrees to radians using math.radians().",
            "Example: math.sin(math.radians(90)) gives 1.0 because 90 degrees = pi/2 radians.",
        ),
        "KC_C2_MATH_LIBRARY": (
            "Consider what libraries might help with mathematical calculations.",
            "Python has a built-in library for math operations. Have you imported it?",
            "Add 'import math' at the top of your code to use functions like math.sin().",
            "Example: import math; result = math.sqrt(16)  # returns 4.0",
        ),
        "KC_C1_FUNCTION_DEF": (
            "Think about how you structure reusable code in Python.",
            "You can create a function to organize your code. What keyword defines a function?",
            "Define a function using: def function_name(parameter): followed by indented code.",
            "Example: def calculate_area(radius): return 3.14 * radius * radius",
        ),
        "KC_C1_FUNCTION_DEF_RETURN": (
            "Think about how your function communicates its result back.",
            "Functions can give back values. What keyword sends a value back to the caller?",
            "Use the 'return' statement to send a value back from your function.",
            "Example: def double(x): return x * 2; result = double(5)  # result is 10",
        ),
        "KC_C3_VARIABLE": (
            "Consider how you're storing and accessing values in your program.",
            "Variables hold values. Are you using the right variable names?",
            "Assign values to variables using =, like: velocity = 10",
            "Example: x = 5; y = x + 3  # y is now 8",
        ),
        "KC_C4_ARITHMETIC_IMPLEMENTATION": (
            "Review the mathematical operations in your code.",
            "Check the order of operations. Are parentheses needed anywhere?",
            "Make sure your arithmetic follows the correct formula. Check operator precedence.",
            "Example: distance = 0.5 * acceleration * time**2  # note the ** for power",
        ),
        "KC_P2_EQUATION_PROJECTILE": (
            "Think about the physics formula for projectile range.",
            "The range depends on velocity, angle, and gravity. What's the relationship?",
            "Range formula: R = (v^2 * sin(2*theta)) / g. Make sure angle is in radians.",
            "Example: range = (velocity**2 * math.sin(2*math.radians(angle))) / 9.8",
        ),
        "KC_C5_ARITHMETIC": (
            "Think about the basic math operations you need.",
            "Which arithmetic operators do you need? (+, -, *, /, **)",
            "Use the correct operators: + add, - subtract, * multiply, / divide, ** power.",
            "Example: result = (a + b) * c / d  # parentheses control order",
        ),
        "KC_C9_CLASS_DEFINITION": (
            "Consider how you would group related data and functions together.",
            "Python uses classes to create custom types. What keyword creates a class?",
            "Define a class using: class ClassName: followed by methods using def.",
            "Example: class Point: def __init__(self, x, y): self.x = x; self.y = y",
        ),
    }

    # Default templates for unknown KCs
    DEFAULT_TEMPLATES: Tuple[str, str, str, str] = (
        "Think carefully about what this part of the code should do.",
        "Review the concept you're trying to apply. What are the key steps?",
        "Check your understanding of this concept. You may need to review it.",
        "Try breaking down the problem into smaller steps and tackle each one.",
    )

    # Mastery-based encouragement prefixes
    MASTERY_PREFIXES: Dict[str, str] = {
        "unknown": "I see you're still learning this concept. ",
        "partial": "You're making progress on this! ",
        "mastered": "You know this well, just double-check: ",
    }

    @property
    def tutor_type(self) -> str:
        return "ml_based"

    async def generate_hint(self, context: TutorContext) -> TutorResponse:
        """
        Generate a BKT-informed hint targeting weakest knowledge component.

        Process:
        1. Identify lowest-mastery KC
        2. Determine scaffold level based on mastery + history
        3. Select appropriate hint template
        4. Personalize based on mastery level

        Args:
            context: TutorContext with BKT state

        Returns:
            TutorResponse with personalized hint
        """
        # 1. Find the weakest KC
        target_kc = context.get_lowest_mastery_kc()

        if target_kc is None:
            # No BKT tracking, fall back to generic hint
            return TutorResponse(
                hint="Review your code carefully and think about what each part should do.",
                scaffold_level=ScaffoldLevel.GUIDING,
                target_kc=None,
                tutor_type=self.tutor_type,
                reasoning="No BKT tracking available, used generic hint",
            )

        # 2. Get mastery level and determine scaffold
        mastery = context.get_mastery_for_kc(target_kc)
        scaffold_level = self.determine_scaffold_level(context)

        # If comfort zone, minimal hint
        if scaffold_level == ScaffoldLevel.NONE:
            return TutorResponse(
                hint="You're doing well! Keep going with your current approach.",
                scaffold_level=ScaffoldLevel.NONE,
                target_kc=target_kc,
                tutor_type=self.tutor_type,
                reasoning=f"KC {target_kc} mastery {mastery:.2f} is high, minimal intervention",
                no_hint_needed=True,
            )

        # 3. Get hint templates for this KC
        templates = self.KC_HINT_TEMPLATES.get(target_kc, self.DEFAULT_TEMPLATES)

        # Map scaffold level to template index (1-4 maps to 0-3)
        template_idx = min(scaffold_level.value - 1, len(templates) - 1)
        template_idx = max(0, template_idx)  # Ensure non-negative
        hint_template = templates[template_idx]

        # 4. Add mastery-based prefix for personalization
        if mastery < 0.3:
            prefix = self.MASTERY_PREFIXES["unknown"]
        elif mastery < 0.7:
            prefix = self.MASTERY_PREFIXES["partial"]
        else:
            prefix = self.MASTERY_PREFIXES["mastered"]

        hint = prefix + hint_template

        # 5. If student asked a question, acknowledge it
        if context.student_question:
            hint = f"To help with your question: {hint}"

        # Build reasoning
        reasoning = (
            f"Target KC: {target_kc} (mastery={mastery:.2f}), "
            f"Scaffold level: {scaffold_level.name}, "
            f"Recent hints on KC: {context.get_recent_hint_count_for_kc(target_kc)}"
        )

        return TutorResponse(
            hint=hint,
            scaffold_level=scaffold_level,
            target_kc=target_kc,
            tutor_type=self.tutor_type,
            reasoning=reasoning,
        )

    def get_kc_priority_order(self, context: TutorContext) -> List[Tuple[str, float]]:
        """
        Get KCs sorted by priority (lowest mastery first).

        Args:
            context: TutorContext with BKT state

        Returns:
            List of (kc_id, mastery) tuples sorted by mastery ascending
        """
        if context.bkt is None:
            return []

        kc_masteries = []
        for kc_id in context.required_kcs:
            mastery = context.get_mastery_for_kc(kc_id)
            kc_masteries.append((kc_id, mastery))

        # Sort by mastery (lowest first = highest priority)
        kc_masteries.sort(key=lambda x: x[1])

        return kc_masteries

    def add_kc_templates(
        self,
        kc_id: str,
        minimal: str,
        guiding: str,
        explicit: str,
        example: str
    ) -> None:
        """
        Add or update hint templates for a KC.

        Args:
            kc_id: Knowledge component identifier
            minimal: Level 1 hint (redirect attention)
            guiding: Level 2 hint (name concept)
            explicit: Level 3 hint (tell what to do)
            example: Level 4 hint (worked example)
        """
        self.KC_HINT_TEMPLATES[kc_id] = (minimal, guiding, explicit, example)
