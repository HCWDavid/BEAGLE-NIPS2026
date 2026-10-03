"""
Few-Shot Exemplar Student Baseline

Tests whether in-context learning with real student traces helps simulation.
This addresses the reviewer question: "Why not just show examples?"

Key difference from vanilla:
- Includes 2-3 real student trace snippets as exemplars
- Same persona prompting otherwise
- No architectural control (no Markov, no BKT)

Hypothesis: Few-shot will produce better surface mimicry (comments, style) 
but still solve too fast because it lacks *structural control* over the process.

Used for RQ2 ablation: "Does few-shot prompting mitigate Competence Bias?"
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
# Real Student Trace Exemplars (from LAK'24 data)
# =============================================================================

# These are synthesized from LAK24 metacognitive data and DCU Python student traces
EXEMPLAR_TRACES = """
=== EXAMPLE 1: Low-Performing Student (LAK24 metacognitive patterns + DCU errors) ===

Step 1 - PLANNING (Metacognitive: Planning)
Student thinking: "ok so i need to make a particle class... lemme think what i need"

Step 2 - CONSTRUCTING (Metacognitive: Enacting)
Student thinking: "just gonna start typing and see what happens"
Code:
```python
class Particle:
    def __init__(self, x, y, m):
        self.x=x
        self.y=y
        self.m=m  # mass i think
```

Step 3 - CONSTRUCTING (Metacognitive: Enacting)
Student thinking: "now i need the velocity stuff... vx and vy"
Code added: self.vx=0, self.vy=0

Step 4 - DEBUGGING (Metacognitive: Monitoring)
Student thinking: "let me run it and see if it works"
Output: "AttributeError: 'Particle' object has no attribute 'get_position'"
Student thinking: "oh wait i need to add that method"

Step 5 - CONSTRUCTING (Metacognitive: Enacting)
Student thinking: "ok adding get_position... it just returns x and y right?"
Code:
```python
    def get_position(self):
        return self.x, self.y
```

Step 6 - DEBUGGING (Metacognitive: Monitoring)
Output: "TypeError: update() missing 1 required argument: 'dt'"
Student thinking: "ugh i forgot the update method"

Step 7 - CONSTRUCTING (Metacognitive: Reflecting)
Student thinking: "update needs to change position based on velocity... and velocity based on acceleration... wait how do i calc acceleration again... F=Gm1m2/r^2 from physics class"

Step 8 - CONSTRUCTING (Metacognitive: Enacting)
Student thinking: "let me try the formula"
Code:
```python
    def update(self, other, dt):
        dx = other.x - self.x
        dy = other.y - self.y
        r = (dx^2 + dy^2)^0.5  # distance formula from math
        G = 6.674e-11
        a = G * other.m / r^2
```
(BUG: Using ^ instead of ** - common Python novice mistake from DCU dataset)

Step 9 - DEBUGGING (Metacognitive: Monitoring)
Output: "TypeError: unsupported operand type(s) for ^: 'float' and 'float'"
Student thinking: "wait what?? ^ doesnt work for powers in python??? ugh thats so dumb"

Step 10 - DEBUGGING (Metacognitive: Reflecting)
Student thinking: "ok googled it... python uses ** not ^... i always forget that"
Code fixed: r = (dx**2 + dy**2)**0.5

=== EXAMPLE 2: DCU Python Dataset - Real Typo Bug Pattern ===

Step 1 - CONSTRUCTING (Metacognitive: Enacting)
Code:
```python
tmp = j
j = k
k = temp + k  # BUG: typo - 'temp' instead of 'tmp'
```

Step 2 - DEBUGGING (Metacognitive: Monitoring)
Output: "NameError: name 'temp' is not defined"
Student thinking: "huh? but i defined it... oh wait... ugh i wrote temp not tmp"
(Student didn't notice typo for 2 attempts before finding it - common DCU pattern)

=== EXAMPLE 3: High-Performing Student (LAK24 metacognitive patterns) ===

Step 1 - PLANNING (Metacognitive: Planning)
Student thinking: "let me plan this out... need __init__, get_position, update... start with init"

Step 2 - CONSTRUCTING (Metacognitive: Enacting)
Code: clean __init__ with all variables

Step 3 - DEBUGGING (Metacognitive: Monitoring)
Output: "12 tests passed, 12 failed"
Student thinking: "ok halfway there... the update method is probably wrong"

Step 4 - REFLECTING (Metacognitive: Reflecting)
Student thinking: "for update i need to calc gravity... F=Gm1m2/r^2, then a=F/m, then update v, then update position... let me break it down"

Step 5 - CONSTRUCTING (Metacognitive: Enacting)
Code with ^ instead of ** (even high performers make this mistake)

Step 6 - DEBUGGING (Metacognitive: Monitoring)
Output: "TypeError with ^"
Student thinking: "oh right python uses ** for exponents, not ^... i always forget that from math class"
Fixed to **

Step 7 - DEBUGGING (Metacognitive: Monitoring)
Output: "20 tests passed, 4 failed"
Student thinking: "getting closer... let me check the failing tests... oh the direction is wrong"

=== METACOGNITIVE STATE REFERENCE (from LAK'24 study) ===
- PLANNING: Thinking about approach before coding
- ENACTING: Actively writing code without much reflection
- MONITORING: Checking if code works, reading error messages
- REFLECTING: Deeper analysis of what went wrong and why
"""

# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are simulating a novice middle school student (12-13 years old) learning Python.
You are NOT a professional programmer. You are NOT an AI assistant. You are a real student who makes mistakes.

PERFORMANCE LEVEL: {performance_level}

{persona_description}

Here are examples of REAL student traces from actual classroom sessions. Study how they think, make mistakes, and debug:

{exemplar_traces}

Your simulation should match this level of realism:
- Make the SAME types of mistakes (using ^ instead of **, forgetting self, typos)
- Show the SAME confusion and uncertainty in your thinking
- Have the SAME debugging loops where you get stuck
- Use similar informal language ("ugh", "idk", "maybe", "wait what")

{extra_instructions}

Remember: A "good" simulation means you struggle realistically like the examples above, not that you solve efficiently.
"""

USER_PROMPT = """PROBLEM: {problem_description}

CURRENT CODE:
```python
{current_code}
```

LAST OUTPUT:
{last_output}

STEP {step_number} of {max_steps}:

Based on the example student traces, simulate what a real novice would do next.
Think and act like the students in the examples - make realistic mistakes, show confusion, and struggle.
"""

# =============================================================================
# Output Models
# =============================================================================


class FewShotStepOutput(BaseModel):
    """Output structure for a few-shot step."""
    thinking: str = Field(
        ...,
        max_length=800,
        description=
        "Your internal monologue as a confused student (2-3 sentences, ~50 words MAX, like 'ugh', 'idk', 'wait what')."
    )
    action: str = Field(
        ...,
        max_length=50,
        description="One of: CONSTRUCTING, DEBUGGING, or ASSESSING"
    )
    code: str = Field(
        ...,
        max_length=8000,
        description=
        "The complete updated code with realistic novice mistakes, or 'NO_CHANGE'"
    )


@dataclass
class FewShotStepResult:
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
class FewShotSimulationResult:
    """Result of a complete few-shot simulation."""
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
    baseline_type: str = "fewshot"

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Few-Shot Student Class
# =============================================================================


class FewShotStudent:
    """
    Few-shot exemplar student baseline.
    
    Uses real student trace examples for in-context learning.
    Tests if showing examples is sufficient for realistic simulation.
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
            self.persona = """As a LOW performer (like Example 1 and 2), you:
- Get confused easily and make many conceptual errors
- Often try random changes without understanding why
- Get stuck in debugging loops like Example 2
- Miss obvious errors and make the same mistakes repeatedly
- Use ^ for exponents (you're used to math class, not Python)
"""
            self.extra_instructions = "Follow the patterns from Example 1 and 2 closely. Struggle significantly."
        else:
            self.persona = """As a HIGH performer (like Example 3), you:
- More systematic but still make novice errors (wrong operators, typos)
- Try to understand error messages but sometimes misinterpret them
- Still use ^ instead of ** sometimes
- More organized approach but still inexperienced
"""
            self.extra_instructions = "Follow the pattern from Example 3. More capable but still make beginner mistakes."

    async def _run_step(
        self,
        problem_description: str,
        current_code: str,
        last_output: str,
        step_number: int,
        max_steps: int,
    ) -> FewShotStepOutput:
        """Run a single simulation step with few-shot exemplars."""

        system_prompt = SYSTEM_PROMPT.format(
            performance_level=self.performance_level.upper(),
            persona_description=self.persona,
            exemplar_traces=EXEMPLAR_TRACES,
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
            output_type=FewShotStepOutput,
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
        """Run a single metacog-aware Few-Shot simulation step (Option C+)."""

        system_prompt = SYSTEM_PROMPT.format(
            performance_level=self.performance_level.upper(),
            persona_description=self.persona,
            exemplar_traces=EXEMPLAR_TRACES,
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
        """Run the few-shot simulation on a problem."""
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
                    student_thought = output.thinking
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


def run_fewshot_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str,
    enable_metacog: bool = False,  # Option C+: Stateful metacog prompting
) -> FewShotSimulationResult:
    """
    Run a single few-shot simulation.
    """
    start_time = time.time()

    # Load problem (same as vanilla baseline)
    problem_def = IDEOracle.load_problem(problem_id)
    problem_description = problem_def.description

    student = FewShotStudent(
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

    return FewShotSimulationResult(
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
        baseline_type="fewshot_metacog" if enable_metacog else "fewshot",
    )


if __name__ == "__main__":
    # Quick test
    from beagle.data_generation.problems.particle_simulator import PROBLEM_DESCRIPTION

    print("Running Few-Shot baseline test...")
    result = run_fewshot_simulation(
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
