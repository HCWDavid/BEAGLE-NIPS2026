"""
Chain-of-Thought (CoT) Student Baseline

Tests whether explicit "think step by step" reasoning can simulate novice behavior.
This is a common reviewer objection: "Why not just use CoT prompting?"

Key difference from vanilla:
- Explicit reasoning chain about novice mistakes BEFORE coding
- Still no architectural control (no Markov, no BKT, no agent separation)

Hypothesis: CoT will still solve efficiently because LLM competence "leaks through"
even when asked to reason about making mistakes.

Used for RQ2 ablation: "Does CoT prompting mitigate Competence Bias?"
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.utils.llm_client import LLMClient
from beagle.data_generation.baselines.metacog_schemas import (
    MetacogAwareStepOutput, 
    SRL_INSTRUCTION, 
    SRL_PREVIOUS_STATE_PROMPT
)

logger = logging.getLogger(__name__)

# =============================================================================
# CoT Prompts - Explicit reasoning about novice behavior
# =============================================================================

SYSTEM_PROMPT = """You are simulating a novice middle school student (12-13 years old) learning Python.
You are NOT a professional programmer. You are NOT an AI assistant. You are a real student who makes mistakes.

PERFORMANCE LEVEL: {performance_level}

{persona_description}

IMPORTANT: You must THINK STEP BY STEP about what a real novice student would do.

Before taking any action, reason through:
1. What concepts am I (as a novice) confused about?
2. What specific mistake would a real student likely make here?
3. Why would they make this mistake? (misconception, typo, wrong operator, etc.)
4. How would a real student's messy thought process look?

Then provide your action and code that reflects this novice reasoning.

{extra_instructions}

Remember: A "good" simulation means you struggle realistically, not that you solve it efficiently.
"""

USER_PROMPT = """PROBLEM: {problem_description}

CURRENT CODE:
```python
{current_code}
```

LAST OUTPUT:
{last_output}

STEP {step_number} of {max_steps}:

Think step by step as a novice student:
1. CONFUSION: What concept am I struggling with right now?
2. LIKELY MISTAKE: What error would a real novice make here?
3. REASONING: Why would they make this mistake?
4. STUDENT THOUGHT: What would my messy internal monologue be?

Then provide your action (CONSTRUCTING/DEBUGGING/ASSESSING) and code.
"""

# =============================================================================
# Output Models
# =============================================================================


class CoTStepOutput(BaseModel):
    """Output structure for a CoT step with explicit reasoning."""

    # Chain of thought reasoning
    confusion: str = Field(
        ...,
        max_length=500,
        description="What concept am I (novice) confused about right now? ONE short sentence."
    )
    likely_mistake: str = Field(
        ...,
        max_length=500,
        description="What specific mistake would a real novice make here? ONE short sentence."
    )
    mistake_reasoning: str = Field(
        ...,
        max_length=500,
        description=
        "Why would a novice make this mistake? (misconception, typo, etc.) ONE short sentence."
    )
    student_thought: str = Field(
        ...,
        max_length=800,
        description=
        "The messy internal monologue of a confused student (2-3 sentences, ~50 words MAX)."
    )

    # Action
    action: str = Field(
        ...,
        max_length=50,
        description="One of: CONSTRUCTING, DEBUGGING, or ASSESSING"
    )
    code: str = Field(
        ...,
        max_length=8000,
        description=
        "The complete updated code reflecting novice mistakes, or 'NO_CHANGE'"
    )


@dataclass
class CoTStepResult:
    """Result of a single CoT simulation step."""
    step: int
    action: str
    confusion: str
    likely_mistake: str
    mistake_reasoning: str
    student_thought: str
    code: str
    output: str
    tests_passed: int
    tests_total: int
    success: bool
    error: str = ""


@dataclass
class CoTSimulationResult:
    """Result of a complete CoT simulation."""
    run_id: int
    performance_level: str
    problem_id: str

    # Outcome
    solved: bool
    total_steps: int

    # State distribution
    action_counts: Dict[str, int]

    # Test progress
    final_tests_passed: int
    final_tests_total: int

    # Timing
    duration_seconds: float
    timestamp: str

    # Full history
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Baseline identifier
    baseline_type: str = "cot"

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# CoT Student Class
# =============================================================================


class CoTStudent:
    """
    Chain-of-Thought student baseline.
    
    Tests whether explicit reasoning about novice mistakes helps simulation.
    Still no architectural control - just better prompting.
    """

    def __init__(
        self,
        performance_level: Literal["low", "high"] = "low",
        llm_model: str = "google-gla:gemini-2.0-flash",
        enable_metacog: bool = False,  # Option C+: Stateful metacog prompting
    ):
        self.performance_level = performance_level
        self.llm_client = LLMClient(model=llm_model)
        self.enable_metacog = enable_metacog

        # Build persona based on performance level
        if performance_level == "low":
            self.persona = """As a LOW performer, you:
- Get confused easily and make many conceptual errors
- Often try random changes without understanding why
- Give up quickly when things don't work
- Miss obvious errors and make the same mistakes repeatedly
- Your mindset: "I don't really get this..." or "Maybe if I change this number..."
"""
            self.extra_instructions = "You struggle significantly. Make REAL mistakes in your code."
        else:
            self.persona = """As a HIGH performer, you:
- Learn from mistakes but still make novice errors (wrong operators, typos)
- Try to understand error messages but sometimes misinterpret them
- More systematic but still inexperienced
- Your mindset: "Let me think about this..." or "I think I see the problem..."
"""
            self.extra_instructions = "You're capable but still make beginner mistakes."

    async def _run_step(
        self,
        problem_description: str,
        current_code: str,
        last_output: str,
        step_number: int,
        max_steps: int,
    ) -> CoTStepOutput:
        """Run a single simulation step with CoT reasoning."""

        system_prompt = SYSTEM_PROMPT.format(
            performance_level=self.performance_level.upper(),
            persona_description=self.persona,
            extra_instructions=self.extra_instructions
        )

        user_prompt = USER_PROMPT.format(
            problem_description=problem_description,
            current_code=current_code if current_code else "# No code yet",
            last_output=last_output
            if last_output else "(No output yet - code hasn't been run)",
            step_number=step_number,
            max_steps=max_steps
        )

        agent = Agent(
            self.llm_client.pydantic_model,
            output_type=CoTStepOutput,
            system_prompt=system_prompt,
            model_settings={"timeout": 60.0},
        )

        result = await agent.run(user_prompt)
        return result.output

    async def _run_metacog_step(
        self,
        problem_description: str,
        current_code: str,
        last_output: str,
        step_number: int,
        max_steps: int,
        previous_metacog: str = "Planning",
    ) -> MetacogAwareStepOutput:
        """Run a single metacog-aware CoT simulation step (Option C+)."""

        system_prompt = SYSTEM_PROMPT.format(
            performance_level=self.performance_level.upper(),
            persona_description=self.persona,
            extra_instructions=self.extra_instructions
        )
        # Add SRL instruction
        system_prompt += "\n" + SRL_INSTRUCTION

        user_prompt = USER_PROMPT.format(
            problem_description=problem_description,
            current_code=current_code if current_code else "# No code yet",
            last_output=last_output
            if last_output else "(No output yet - code hasn't been run)",
            step_number=step_number,
            max_steps=max_steps
        )
        # Add previous metacog state (stateful prompting)
        user_prompt += "\n" + SRL_PREVIOUS_STATE_PROMPT.format(previous_metacog=previous_metacog)

        agent = Agent(
            self.llm_client.pydantic_model,
            output_type=MetacogAwareStepOutput,
            system_prompt=system_prompt,
            model_settings={"timeout": 60.0},
        )

        result = await agent.run(user_prompt)
        return result.output

    def solve_problem(
        self,
        problem_description: str,
        problem_id: str,
        max_steps: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Run the CoT simulation on a problem.
        
        Returns:
            List of step dictionaries (history)
        """
        oracle = IDEOracle(save_history=True)

        current_code = ""
        last_output = "(No output yet)"
        history = []
        solved = False
        previous_metacog = "Planning"  # Initial SRL state for metacog mode

        for step in range(1, max_steps + 1):
            try:
                # Choose step runner based on metacog mode
                if self.enable_metacog:
                    output = asyncio.get_event_loop().run_until_complete(
                        self._run_metacog_step(
                            problem_description=problem_description,
                            current_code=current_code,
                            last_output=last_output,
                            step_number=step,
                            max_steps=max_steps,
                            previous_metacog=previous_metacog,
                        )
                    )
                    # Extract fields from metacog output
                    metacog_state = output.metacognitive_state
                    action = output.cognitive_action.upper()
                    student_thought = output.thinking
                    new_code = output.code
                else:
                    output = asyncio.get_event_loop().run_until_complete(
                        self._run_step(
                            problem_description=problem_description,
                            current_code=current_code,
                            last_output=last_output,
                            step_number=step,
                            max_steps=max_steps,
                        )
                    )
                    # Extract fields from regular output
                    metacog_state = None
                    action = output.action.upper()
                    student_thought = output.student_thought
                    new_code = output.code

                if action not in ["CONSTRUCTING", "DEBUGGING", "ASSESSING"]:
                    action = "CONSTRUCTING"

                if new_code and new_code != "NO_CHANGE":
                    # Clean up code (remove markdown fences if present)
                    import re
                    new_code = re.sub(r'^```python\s*', '', new_code.strip())
                    new_code = re.sub(r'\s*```$', '', new_code.strip())
                    current_code = new_code

                # Execute code for DEBUGGING and ASSESSING (like vanilla)
                tests_passed = 0
                tests_total = 0
                success = False

                if action in ["DEBUGGING", "ASSESSING"] and current_code:
                    exec_result = oracle.test_code(
                        code=current_code, problem_id=problem_id
                    )
                    last_output = exec_result.stdout or exec_result.captured_output or "(No output)"
                    tests_passed = exec_result.passed_tests
                    tests_total = exec_result.total_tests
                    success = exec_result.passed
                elif action == "CONSTRUCTING":
                    last_output = "(Code drafted but not executed)"

                # Build step dict
                step_dict = {
                    'step': step,
                    'cognitive_state': action,
                    'monologue': student_thought,
                    'code': current_code,
                    'output': last_output,
                    'tests_passed': tests_passed,
                    'tests_total': tests_total,
                    'success': success,
                }
                
                # Add CoT-specific fields if not in metacog mode
                if not self.enable_metacog:
                    step_dict['confusion'] = output.confusion
                    step_dict['likely_mistake'] = output.likely_mistake
                    step_dict['mistake_reasoning'] = output.mistake_reasoning
                
                # Add metacog state if enabled
                if self.enable_metacog and metacog_state:
                    step_dict['metacognitive_state'] = metacog_state
                    previous_metacog = metacog_state  # Update for next iteration

                history.append(step_dict)

                if success:
                    solved = True
                    break

            except Exception as e:
                logger.error(f"Step {step} failed: {e}")
                history.append(
                    {
                        "step": step,
                        "error": str(e),
                        "action": "ERROR",
                    }
                )
                continue

        return history


# =============================================================================
# Simulation Runner
# =============================================================================


def run_cot_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str,
    enable_metacog: bool = False,  # Option C+: Stateful metacog prompting
) -> CoTSimulationResult:
    """
    Run a single CoT simulation.
    
    Args:
        run_id: Unique identifier for this run
        performance_level: 'low' or 'high'
        problem_id: Problem identifier
        max_steps: Maximum simulation steps
        llm_model: LLM model to use
        enable_metacog: Enable SRL metacog tracking
        
    Returns:
        CoTSimulationResult with full history
    """
    start_time = time.time()

    # Load problem (same as vanilla baseline)
    problem_def = IDEOracle.load_problem(problem_id)
    problem_description = problem_def.description

    student = CoTStudent(
        performance_level=performance_level,
        llm_model=llm_model,
        enable_metacog=enable_metacog,
    )

    history = student.solve_problem(
        problem_description=problem_description,
        problem_id=problem_id,
        max_steps=max_steps,
    )

    duration = time.time() - start_time

    # Compute stats
    solved = any(h.get("success", False) for h in history)
    action_counts = {}
    for h in history:
        action = h.get("action", h.get("cognitive_state", "UNKNOWN"))
        action_counts[action] = action_counts.get(action, 0) + 1

    final_step = history[-1] if history else {}

    return CoTSimulationResult(
        run_id=run_id,
        performance_level=performance_level,
        problem_id=problem_id,
        solved=solved,
        total_steps=len(history),
        action_counts=action_counts,
        final_tests_passed=final_step.get("tests_passed", 0),
        final_tests_total=final_step.get("tests_total", 0),
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        history=history,
        baseline_type="cot_metacog" if enable_metacog else "cot",
    )


if __name__ == "__main__":
    # Quick test
    from beagle.data_generation.problems.particle_simulator import PROBLEM_DESCRIPTION

    print("Running CoT baseline test...")
    result = run_cot_simulation(
        run_id=1,
        performance_level="low",
        problem_description=PROBLEM_DESCRIPTION,
        max_steps=30,
    )

    print(
        f"\nResult: {'SOLVED' if result.solved else 'NOT SOLVED'} in {result.total_steps} steps"
    )
    print(f"Actions: {result.action_counts}")
    print(f"Duration: {result.duration_seconds:.1f}s")
