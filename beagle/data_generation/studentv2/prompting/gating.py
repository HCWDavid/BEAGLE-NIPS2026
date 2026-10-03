"""
Epistemic Gating Module

This module implements the "Epistemic Blindness" mechanism for BEAGLE.
When a student is in the "Enacting" (impulsive) metacognitive state,
they cannot see detailed error messages - simulating a student who
glances at output without reading carefully.

This is the "Trap" that prevents LLMs from being "Smart Guessers".
"""

from typing import Optional

# ==============================================================================
# EPISTEMIC BLINDNESS (THE TRAP)
# ==============================================================================

class EpistemicGating:
    """
    Applies epistemic blindness based on metacognitive state.
    
    When in "Enacting" state, the student only sees that something failed,
    not the specific error details. This prevents LLMs from using traceback
    information to make "smart" debugging decisions that a novice wouldn't make.
    """
    
    VALID_METACOG_STATES = {'Planning', 'Enacting', 'Monitoring', 'Reflecting'}
    
    def apply_blindness(self, metacog_state: str, raw_output: str) -> str:
        """
        Apply epistemic blindness filter based on metacognitive state.
        
        Args:
            metacog_state: Current metacognitive state (Planning, Enacting, etc.)
            raw_output: Raw execution output (may contain tracebacks)
            
        Returns:
            Filtered output based on epistemic visibility rules
        """
        if not raw_output:
            return "(No output yet)"

        # Only censor if we are in the "Enacting" state
        if metacog_state == "Enacting":
            if "Error" in raw_output or "Traceback" in raw_output:
                # V34: Extract error TYPE so LLM doesn't guess wrong
                # A student glancing at red text would at least see "TypeError" or "AttributeError"
                import re
                error_type = "Error"  # default
                # Try to extract specific error type
                match = re.search(r'(TypeError|AttributeError|NameError|SyntaxError|IndentationError|ImportError|ValueError|KeyError)', raw_output)
                if match:
                    error_type = match.group(1)
                
                return (
                    f"[SYSTEM OUTPUT]: The program crashed with a {error_type}.\n"
                    "(OBSERVATION: You glanced at the red text and saw the error type, "
                    "but did NOT read the full message. You're acting impulsively.)"
                )
            elif "FAIL" in raw_output.upper():
                return (
                    "[SYSTEM OUTPUT]: Tests Failed.\n"
                    "(OBSERVATION: You saw it didn't work, but didn't analyze the specific numbers.)"
                )
        
        # If Monitoring/Planning/Reflecting, they see everything (Full Traceback)
        return raw_output


# ==============================================================================
# LEGACY EXPORTS (for backward compatibility)
# ==============================================================================

# These are kept for backward compatibility with existing code.
# The set_sparse_comments is now handled directly in nodes.py

USE_SPARSE_COMMENTS = False

def set_sparse_comments(enabled: bool) -> None:
    """Set the comment style. Now handled in nodes.py, kept for compatibility."""
    global USE_SPARSE_COMMENTS
    USE_SPARSE_COMMENTS = enabled


# Legacy PromptBlocks - kept for tests, but not used in production
class PromptBlocks:
    """Legacy class - kept for backward compatibility with tests."""
    
    MANDATES = {
        "Enacting": "*** METACOGNITIVE STATE: ENACTING ***",
        "Monitoring": "*** METACOGNITIVE STATE: MONITORING ***",
        "Planning": "*** METACOGNITIVE STATE: PLANNING ***",
        "Reflecting": "*** METACOGNITIVE STATE: REFLECTING ***",
    }
    
    TASKS = {
        "CONSTRUCTING": "TASK: DRAFTING CODE",
        "DEBUGGING": "TASK: FIXING ERRORS",
        "ASSESSING": "TASK: ASSESSING OUTPUT",
    }
    
    @property
    def AGENTS(self):
        return {
            "strategist": "You are the Brain.",
            "executor": "You are the Hands.",
        }


# Legacy class name alias
NeuroSymbolicPromptGenerator = EpistemicGating


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

_gating_instance: Optional[EpistemicGating] = None

def get_generator() -> EpistemicGating:
    """Get singleton instance of the epistemic gating module."""
    global _gating_instance
    if _gating_instance is None:
        _gating_instance = EpistemicGating()
    return _gating_instance
