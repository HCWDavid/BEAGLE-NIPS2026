import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class EpisodicMemory:
    """
    A single episodic memory representing a lesson learned from debugging.

    Used to prevent amnesia - when student encounters similar error again,
    they should recall having fixed it before.
    """
    error_pattern: str      # Key part of error message (e.g., "unsupported operand type(s) for ^")
    realization: str        # What the student learned (e.g., "Python uses ** not ^ for power")
    step_learned: int       # When this was learned
    fix_applied: str = ""   # Optional: the fix that worked


def p_assistance(session_progress: float, performance_level: str) -> float:
    """
    Calculate probability of student seeking assistance.

    Args:
        session_progress: Progress through session (0.0 to 1.0)
        performance_level: "low" or "high"

    Returns:
        Probability of seeking assistance at this point

    Based on LAK24 data analysis:
    - Peak at mid-session (μ=0.5)
    - High performers seek help more often (15% vs 11.7% peak)
    """
    mu = 0.5      # Peak at mid-session
    sigma = 0.25
    gaussian = math.exp(-((session_progress - mu) ** 2) / (2 * sigma ** 2))
    peak_rate = 0.150 if performance_level == 'high' else 0.117
    return gaussian * peak_rate


def p_offtopic(session_progress: float, performance_level: str) -> float:
    """
    Calculate probability of student going off-topic.

    Args:
        session_progress: Progress through session (0.0 to 1.0)
        performance_level: "low" or "high"

    Returns:
        Probability of going off-topic at this point

    Based on LAK24 data analysis:
    - Peak late in session (μ=0.73)
    - Low performers go off-topic more often (9.2% vs 3.7% peak)
    """
    mu = 0.73     # Peak late in session
    sigma = 0.20
    gaussian = math.exp(-((session_progress - mu) ** 2) / (2 * sigma ** 2))
    peak_rate = 0.037 if performance_level == 'high' else 0.092
    return gaussian * peak_rate


@dataclass
class StudentState:
    """
    State of the student simulation.
    """
    # Problem Context
    problem_description: str = ""
    problem_id: str = ""  # ID for IDE Oracle (e.g., "projectile_motion")
    required_kcs: List[str] = field(default_factory=list)

    # Current Execution State
    current_code: str = ""
    last_output: str = ""  # Full output from last DEBUGGING/ASSESSING execution
    last_error_type: str = ""  # V36: Parsed error type (e.g., "TypeError") from last execution
    last_error_message: str = ""  # V36: Primary error message from last execution
    execution_success: bool = False

    # Markov State (Driver)
    current_cognitive_state: str = ""  # e.g., "CONSTRUCTING"
    current_metacognitive_state: str = ""  # e.g., "Planning"
    metacognitive_history: List[str] = field(
        default_factory=list
    )  # Track history for 2nd order Markov

    # Strategist Output
    current_goal: str = ""
    current_mindset: str = ""
    current_directive: str = ""

    # Executor Output
    current_monologue: str = ""
    current_error_seen: str = ""  # V36: Error the student stated seeing (from DebuggingOutput)

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Internal
    step_count: int = 0
    max_steps: int = 50

    # Semi-Markov State
    segment_duration: int = 0  # Total steps for current metacog phase
    segment_step: int = 0  # Current step within phase

    # Interrupt State Tracking
    tutor_response: str = ""  # Tutor's hint (available in next turn after assistance)
    just_received_tutor_help: bool = False  # Flag to skip interrupt checks on next turn
    pending_tutor_question: str = ""  # Question being passed to TutorNode
    assistance_count: int = 0  # How many times student asked for help
    offtopic_count: int = 0  # How many times student went off-topic

    # Test Progress Tracking (for session_progress calculation)
    tests_passed: int = 0  # Number of unit tests currently passing
    tests_total: int = 0  # Total number of unit tests

    # Pre-generated Sequence Mode
    # When set, MarkovNode consumes from this sequence instead of sampling
    pregenerated_sequence: Optional[Any] = None  # BehaviorSequence object
    sequence_index: int = 0  # Current position in the pre-generated sequence

    # BKT Caching (sample once per metacog phase)
    cached_knowledge_state: Optional[str] = None  # Cached BKT feedback for current metacog phase

    # Reflection Carry-Forward (from ASSESSING to next cognitive turn)
    # When ASSESSING, student observes output and reflects but doesn't modify code.
    # This reflection is passed to the next turn (e.g., DEBUGGING) to inform fixes.
    pending_reflection: str = ""  # Reflection from previous ASSESSING turn

    # Execution result cache (set by ExecutorNode, consumed by EnvironmentNode)
    # For DEBUGGING/ASSESSING, ExecutorNode runs code BEFORE LLM call to prevent
    # psychic debugging. EnvironmentNode uses this cached result for BKT updates.
    _cached_execution_result: Optional[Any] = None

    # V22: Thought Buffer (Anti-Repetition)
    # Stores recent strategic thoughts to prevent consecutive identical realizations
    thought_buffer: List[str] = field(default_factory=list)
    THOUGHT_BUFFER_SIZE: int = 3  # Keep last 3 thoughts

    # V22: Episodic Memory (Anti-Amnesia)
    # Stores key lessons learned from debugging to enable recall on repeated errors
    episodic_memories: List[EpisodicMemory] = field(default_factory=list)

    # V23: Monologue Buffer (Executor-Level Anti-Repetition)
    # Stores recent monologues/reflections to prevent verbatim copying at executor level
    # This complements the thought_buffer which operates at strategist level
    monologue_buffer: List[str] = field(default_factory=list)
    MONOLOGUE_BUFFER_SIZE: int = 3  # Keep last 3 monologues

    # Force assistance at specific steps (for evaluation)
    force_assistance_steps: List[int] = field(default_factory=list)

    # V34.5: Agent Memory Notes (LLM-authored error memory)
    # Stores short notes the agent writes to itself during DEBUGGING
    # Format: List of (step_number, note_text) tuples
    agent_notes: List[tuple] = field(default_factory=list)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save current simulation state to JSON incrementally.
        
        Called after each cognitive step to enable:
        - Real-time progress monitoring
        - Crash recovery (partial results saved)
        - Debugging mid-simulation
        """
        import json
        from pathlib import Path
        
        checkpoint_data = {
            'status': 'in_progress',
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'tests_passed': self.tests_passed,
            'tests_total': self.tests_total,
            'current_cognitive_state': self.current_cognitive_state,
            'current_metacognitive_state': self.current_metacognitive_state,
            'current_code': self.current_code,
            'current_goal': self.current_goal,
            'current_mindset': self.current_mindset,
            'current_monologue': self.current_monologue,
            'last_output': self.last_output,
            'cached_knowledge_state': self.cached_knowledge_state,
            'history': self.history
        }
        
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)