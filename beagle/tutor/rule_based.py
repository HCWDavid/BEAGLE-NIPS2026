"""
Rule-Based Tutor (Hint Factory Style)

Based on Barnes & Stamper (2010) "Using Hint Factory to Compare Different
Hint Policies"

This tutor uses template-based hints triggered by error patterns.
Simple pattern matching with variable substitution.

Reference:
    Barnes, T., & Stamper, J. (2010). Automatic hint generation for logic
    proof tutoring using historical data. Journal of Educational Technology
    & Society, 13(1), 3-12.
"""

import re
from typing import Dict, Optional

from beagle.tutor.base import (
    TutorStrategy,
    TutorContext,
    TutorResponse,
    ScaffoldLevel,
)


class RuleBasedTutor(TutorStrategy):
    """
    Rule-based tutor using template hints triggered by error patterns.

    Implements the Hint Factory approach where hints are pre-authored
    templates matched to specific error types or code patterns.

    Limitations:
        - Cannot adapt to context beyond pattern matching
        - Same error type always gets same hint
        - No escalation based on student progress
    """

    # Error type -> hint template mappings
    ERROR_HINTS: Dict[str, str] = {
        # Python syntax errors
        "SyntaxError": "Check for missing colons, parentheses, or quotes in your code.",
        "IndentationError": "Python uses indentation for code blocks. Check your spacing - use 4 spaces for each level.",

        # Name and type errors
        "NameError": "Make sure you've defined the variable before using it. Check for typos in variable names.",
        "TypeError": "Check your data types - you might be mixing strings and numbers, or passing wrong argument types.",
        "AttributeError": "The object doesn't have that attribute or method. Check the object type and spelling.",

        # Value and index errors
        "ValueError": "The value you're using isn't valid for this operation. Check your input data.",
        "IndexError": "Your list index is out of range. Check your loop bounds and list length.",
        "KeyError": "That key doesn't exist in the dictionary. Check the key spelling or use .get() method.",

        # Math errors
        "ZeroDivisionError": "You're dividing by zero. Add a check before division to handle this case.",
        "OverflowError": "The number is too large. Check your calculations for potential overflow.",

        # Import errors
        "ImportError": "The module couldn't be imported. Check the module name and if it's installed.",
        "ModuleNotFoundError": "The module wasn't found. Make sure it's installed or check the name spelling.",

        # File errors
        "FileNotFoundError": "The file wasn't found. Check the file path and make sure the file exists.",
        "PermissionError": "You don't have permission to access this file. Check file permissions.",

        # Runtime errors
        "RecursionError": "Too much recursion - your function is calling itself too many times. Add a base case.",
        "MemoryError": "Out of memory. Your program is using too much memory - check for infinite loops.",
        "TimeoutError": "The operation timed out. Check for infinite loops or slow operations.",
    }

    # Code pattern -> hint mappings (regex patterns)
    CODE_PATTERN_HINTS: Dict[str, str] = {
        # Loop issues
        r"while\s+True\s*:(?!.*break)": "Your while True loop might run forever. Make sure you have a break condition.",
        r"for\s+\w+\s+in\s+range\s*\(\s*\)": "Your range() is empty. Specify start, stop values.",

        # Common Python mistakes
        r"=\s*=": "Use == for comparison, not = =. The spaces might be a typo.",
        r"print\s+[^(]": "In Python 3, print is a function. Use print() with parentheses.",
        r"except\s*:": "Bare except catches all exceptions. Consider catching specific exceptions.",

        # Math-related
        r"math\.sin\s*\([^)]*\*?\s*\d+[^)]*\)": "Remember: math.sin() expects radians, not degrees. Use math.radians() to convert.",
        r"degrees?\s*[/\*]": "Make sure you're converting between degrees and radians correctly.",

        # Variable issues
        r"(\w+)\s*=\s*\1\s*[+\-*/]": "Check if you're updating the variable correctly. Did you mean to use a different variable?",
    }

    # KC-specific hints (for when we know which KC is weak)
    KC_HINTS: Dict[str, str] = {
        "KC_P5_UNIT_RADIANS": "Remember: trigonometric functions like sin() and cos() expect angles in radians, not degrees.",
        "KC_C2_MATH_LIBRARY": "You might need to import the math library to use mathematical functions.",
        "KC_C1_FUNCTION_DEF": "To define a function, use: def function_name(parameters):",
        "KC_C1_FUNCTION_DEF_RETURN": "Functions can return values using the return statement.",
        "KC_C3_VARIABLE": "Variables store values. Use meaningful names and assign with =.",
        "KC_C4_ARITHMETIC_IMPLEMENTATION": "Check your arithmetic operations. Make sure the order of operations is correct.",
        "KC_P2_EQUATION_PROJECTILE": "For projectile motion, remember the relationship between angle, velocity, and distance.",
        "KC_C9_CLASS_DEFINITION": "To define a class, use: class ClassName:",
    }

    # Generic fallback hints by category
    FALLBACK_HINTS: Dict[str, str] = {
        "syntax": "Check your syntax carefully. Look for missing punctuation or incorrect indentation.",
        "logic": "Think through your logic step by step. What should happen at each step?",
        "math": "Double-check your mathematical operations and formulas.",
        "default": "Review your code carefully. Try adding print statements to debug.",
    }

    @property
    def tutor_type(self) -> str:
        return "rule_based"

    async def generate_hint(self, context: TutorContext) -> TutorResponse:
        """
        Generate a hint based on error pattern matching.

        Priority order:
        1. Match error type (most specific)
        2. Match code patterns
        3. Match weak KC
        4. Fallback hint

        Args:
            context: TutorContext with error and code info

        Returns:
            TutorResponse with template-based hint
        """
        hint = None
        reasoning = ""
        target_kc = None

        # 1. Try to match error type
        if context.error_type and context.error_type in self.ERROR_HINTS:
            hint = self.ERROR_HINTS[context.error_type]
            reasoning = f"Matched error type: {context.error_type}"

        # 2. Try to match code patterns
        if hint is None and context.current_code:
            for pattern, pattern_hint in self.CODE_PATTERN_HINTS.items():
                if re.search(pattern, context.current_code):
                    hint = pattern_hint
                    reasoning = f"Matched code pattern: {pattern}"
                    break

        # 3. Try to use KC-specific hint based on lowest mastery
        if hint is None:
            target_kc = context.get_lowest_mastery_kc()
            if target_kc and target_kc in self.KC_HINTS:
                hint = self.KC_HINTS[target_kc]
                reasoning = f"Used KC hint for lowest mastery: {target_kc}"

        # 4. Fallback based on error category or generic
        if hint is None:
            if context.error_type:
                if "Syntax" in context.error_type or "Indentation" in context.error_type:
                    hint = self.FALLBACK_HINTS["syntax"]
                elif "Math" in context.last_error or "Arithmetic" in context.last_error:
                    hint = self.FALLBACK_HINTS["math"]
                else:
                    hint = self.FALLBACK_HINTS["default"]
                reasoning = f"Used fallback hint for error category"
            else:
                hint = self.FALLBACK_HINTS["default"]
                reasoning = "Used generic fallback hint"

        # Respond to student question if one was asked
        if context.student_question:
            # Prepend acknowledgment of the question
            hint = f"Regarding your question: {hint}"

        return TutorResponse(
            hint=hint,
            scaffold_level=ScaffoldLevel.GUIDING,  # Rule-based is always at guiding level
            target_kc=target_kc,
            tutor_type=self.tutor_type,
            reasoning=reasoning,
        )

    def add_error_hint(self, error_type: str, hint: str) -> None:
        """Add or update an error type hint."""
        self.ERROR_HINTS[error_type] = hint

    def add_pattern_hint(self, pattern: str, hint: str) -> None:
        """Add or update a code pattern hint."""
        self.CODE_PATTERN_HINTS[pattern] = hint

    def add_kc_hint(self, kc_id: str, hint: str) -> None:
        """Add or update a KC-specific hint."""
        self.KC_HINTS[kc_id] = hint
