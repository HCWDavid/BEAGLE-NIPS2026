"""
Vanilla Student Baseline

A pure LLM-based student simulation with NO architectural control:
- No Semi-Markov model (LLM decides its own state transitions)
- No BKT constraints (no "forbidden knowledge")
- No separate Strategist/Executor (single LLM call per step)

This baseline demonstrates "Competence Bias" - the tendency of LLMs to solve
problems too efficiently, failing to simulate realistic novice struggle.

Used for RQ2: "Does the Neuro-Symbolic framework mitigate Competence Bias?"
"""

import asyncio
import logging
import re
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
# Prompts - These use the SAME persona as BEAGLE for fair comparison
# =============================================================================

SYSTEM_PROMPT = """You are simulating a novice middle school student (12-13 years old) learning Python.
You are NOT a professional programmer. You are NOT an AI assistant. You are a real student who makes mistakes.

PERFORMANCE LEVEL: {performance_level}

{persona_description}

At each step, you must:
1. THINK about what to do next (show your reasoning as a student would)
2. DECIDE your next action: CONSTRUCTING (write new code), DEBUGGING (run and fix), or ASSESSING (run and reflect)
3. Write the actual code if your action involves coding

CRITICAL RULES FOR REALISM:
- You are a BEGINNER. Make realistic mistakes (wrong operators, typos, confused logic)
- Use ^ for exponents (even though Python uses **) - you're used to math class
- Write messy code: no spaces around operators (x=y+5), single-letter variables
- Add emotional comments: # idk, # hope this works, # ???
- When stuck, try random things - you don't have perfect debugging skills

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
Think through what you should do next as a novice student. Then provide your action and code.
"""

# =============================================================================
# Output Models
# =============================================================================


class VanillaStepOutput(BaseModel):
    """Output structure for a single vanilla step."""
    thinking: str = Field(
        ...,
        max_length=800,
        description=
        "Your internal monologue as a confused/learning student (2-3 sentences, ~50 words MAX)."
    )
    action: str = Field(
        ...,
        max_length=50,
        description="One of: CONSTRUCTING, DEBUGGING, or ASSESSING"
    )
    code: str = Field(
        ...,
        max_length=8000,
        description="The complete updated code, or 'NO_CHANGE' if just thinking"
    )


@dataclass
class VanillaStepResult:
    """Result of a single simulation step."""
    step: int
    action: str
    thinking: str
    code: str
    output: str
    tests_passed: int
    tests_total: int
    success: bool
    error: str = ""


@dataclass
class VanillaSimulationResult:
    """Result of a complete vanilla simulation."""
    run_id: int
    performance_level: str
    problem_id: str

    # Outcome
    solved: bool
    total_steps: int

    # State distribution (for comparison with BEAGLE)
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
    baseline_type: str = "vanilla"

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Vanilla Student Class
# =============================================================================


class VanillaStudent:
    """
    Pure LLM student baseline with no architectural control.
    
    Key differences from BEAGLE:
    - LLM decides its own cognitive state (no Semi-Markov model)
    - No BKT tracking (no "forbidden knowledge" constraints)
    - Single LLM call per step (no Strategist/Executor separation)
    - No interrupt modeling (assistance/off-topic)
    
    This demonstrates "Competence Bias" - LLMs solve too efficiently.
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

        # Build persona based on performance level (matches BEAGLE profiles)
        if performance_level == "low":
            self.persona = """As a LOW performer, you:
- Get confused easily and make many conceptual errors
- Often try random changes without understanding why
- Give up quickly when things don't work
- Miss obvious errors and make the same mistakes repeatedly
- Your mindset: "I don't really get this..." or "Maybe if I change this number..."
"""
            self.extra_instructions = "You struggle significantly and often feel lost."
        else:
            self.persona = """As a HIGH performer, you:
- Learn from mistakes but still make novice errors (wrong operators, typos)
- Try to understand error messages but sometimes misinterpret them
- More systematic but still inexperienced
- Your mindset: "Let me think about this..." or "I think I see the problem..."
"""
            self.extra_instructions = "You're capable but still a beginner who makes mistakes."

    async def _run_step(
        self,
        problem_description: str,
        current_code: str,
        last_output: str,
        step_number: int,
        max_steps: int,
    ) -> VanillaStepOutput:
        """Run a single simulation step."""

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
            output_type=VanillaStepOutput,
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
        """Run a single metacog-aware simulation step (Option C+)."""

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
        Run the vanilla simulation on a problem.
        
        Returns:
            List of step dictionaries (history)
        """
        # Initialize oracle
        oracle = IDEOracle(save_history=True)

        # State
        current_code = ""
        last_output = "(No output yet)"
        history = []
        solved = False
        previous_metacog = "Planning"  # Option C+: Track previous metacog for stateful prompting

        for step in range(1, max_steps + 1):
            try:
                # Run step - choose method based on enable_metacog
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
                    metacog_state = output.metacognitive_state
                    action = output.cognitive_action.upper()
                    thinking = output.thinking
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
                    metacog_state = "N/A"
                    action = output.action.upper()
                    thinking = output.thinking
                # Normalize action names
                if action not in ["CONSTRUCTING", "DEBUGGING", "ASSESSING"]:
                    if "DEBUG" in action:
                        action = "DEBUGGING"
                    elif "ASSESS" in action or "REFLECT" in action:
                        action = "ASSESSING"
                    else:
                        action = "CONSTRUCTING"

                # Update code if provided
                new_code = output.code
                if new_code and "NO_CHANGE" not in new_code.upper(
                ) and "NO CHANGE" not in new_code.upper():
                    # Clean up code (remove markdown fences if present)
                    new_code = re.sub(r'^```python\s*', '', new_code.strip())
                    new_code = re.sub(r'\s*```$', '', new_code.strip())
                    current_code = new_code

                # Execute code for DEBUGGING and ASSESSING
                tests_passed = 0
                tests_total = 0

                if action in ["DEBUGGING", "ASSESSING"] and current_code:
                    exec_result = oracle.test_code(
                        code=current_code, problem_id=problem_id
                    )
                    last_output = exec_result.stdout or exec_result.captured_output or "(No output)"
                    tests_passed = exec_result.passed_tests
                    tests_total = exec_result.total_tests
                    solved = exec_result.passed
                elif action == "CONSTRUCTING":
                    last_output = "(Code drafted but not executed)"

                # Record step
                history.append(
                    {
                        "step": step,
                        "cognitive_state": action,
                        "metacognitive_state": metacog_state,
                        "thinking": thinking,
                        "monologue": thinking,  # For evaluator compatibility
                        "code": current_code,
                        "output": last_output,
                        "tests_passed": tests_passed,
                        "tests_total": tests_total,
                        "success": solved,
                    }
                )
                
                # Option C+: Update previous metacog for next iteration
                if self.enable_metacog:
                    previous_metacog = metacog_state

                logger.info(
                    f"  Step {step}: {action} - {'SOLVED!' if solved else f'{tests_passed}/{tests_total} tests'}"
                )

                if solved:
                    break

            except Exception as e:
                logger.error(f"  Step {step} failed: {e}")
                history.append(
                    {
                        "step": step,
                        "cognitive_state": "ERROR",
                        "metacognitive_state": "N/A",
                        "error": str(e),
                        "code": current_code,
                        "output": last_output,
                        "success": False,
                    }
                )

        return history


def run_vanilla_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str,
    enable_metacog: bool = False,  # Option C+: Enable SRL metacog prompting
) -> VanillaSimulationResult:
    """
    Run a complete vanilla baseline simulation.
    
    This is the main entry point for the evaluation script.
    
    Args:
        enable_metacog: If True, add SRL metacog tracking (Planning, Enacting, 
                       Monitoring, Reflecting) with stateful prompting.
    """
    baseline_name = "vanilla_metacog" if enable_metacog else "vanilla"
    logger.info(
        f"Starting {baseline_name.upper()} run {run_id} ({performance_level} performer)"
    )

    start_time = time.time()
    timestamp = datetime.now().isoformat()

    # Load problem
    problem_def = IDEOracle.load_problem(problem_id)

    # Create student
    student = VanillaStudent(
        performance_level=performance_level, 
        llm_model=llm_model,
        enable_metacog=enable_metacog
    )

    # Run simulation
    history = student.solve_problem(
        problem_description=problem_def.description,
        problem_id=problem_id,
        max_steps=max_steps
    )

    # Extract metrics
    solved = history[-1].get("success", False) if history else False

    action_counts = {
        "CONSTRUCTING": 0,
        "DEBUGGING": 0,
        "ASSESSING": 0
    }
    for step in history:
        action = step.get("cognitive_state", "")
        if action in action_counts:
            action_counts[action] += 1

    tests_passed = history[-1].get("tests_passed", 0) if history else 0
    tests_total = history[-1].get("tests_total", 0) if history else 0

    duration = time.time() - start_time

    result = VanillaSimulationResult(
        run_id=run_id,
        performance_level=performance_level,
        problem_id=problem_id,
        solved=solved,
        total_steps=len(history),
        action_counts=action_counts,
        final_tests_passed=tests_passed,
        final_tests_total=tests_total,
        duration_seconds=duration,
        timestamp=timestamp,
        history=history,
        baseline_type="vanilla"
    )

    logger.info(
        f"  Run {run_id} complete: {'SOLVED' if solved else 'NOT SOLVED'} in {result.total_steps} steps ({duration:.1f}s)"
    )

    return result
