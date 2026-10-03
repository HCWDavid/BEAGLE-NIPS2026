"""
LLM-SS (LLM for Student Synthesis) Inspired Baseline

Based on Markel et al. (EDM 2024) "Large Language Models for In-Context Student Modeling"
and informed by LLM-itation (ACE 2024) "Generating Synthetic Buggy Code Submissions"

Key approach:
- Uses REAL student traces from Dublin City University (DCU) Python dataset as exemplars
- Conditions generation via in-context learning (show real student debugging trajectories)
- Follows LLM-SS framework: observe student behavior → play the role of the student

This is a principled baseline because:
1. It uses actual student data (not synthetic) for conditioning
2. It follows published methodology from peer-reviewed work
3. It tests whether in-context exemplars can match BEAGLE's architectural approach

References:
- Markel et al. (2024): https://educationaldatamining.org/edm2024/proceedings/2024.EDM-short-papers.31/
- LLM-itation (2024): https://arxiv.org/abs/2411.10455
- DCU Dataset: https://figshare.com/articles/dataset/12610958

Used for RQ2: "Does exemplar-conditioning match neuro-symbolic control?"
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

logger = logging.getLogger(__name__)

# =============================================================================
# REAL Student Traces from DCU Python Dataset (591,707 submissions)
# These are ACTUAL student debugging trajectories, not synthetic
# =============================================================================

REAL_STUDENT_TRACES = """
=== REAL TRACE 1: Fibonacci (from DCU Python dataset) ===
Task: Print the first n Fibonacci numbers

Attempt 1 [FAILED]:
```python
n = input()

i = 1
j = 0
k = 1
print k
while i < n:
   tmp = j
   j = k
   k = temp + k  # BUG: typo - 'temp' instead of 'tmp'
   print k
   i = i + 1
```
(Student used wrong variable name - classic typo bug)

Attempt 2 [FAILED]:
```python
n = input()

i = 1
j = 0
k = 1
print k
while i < n:
   tmp = j
   j = k
   k = temp + k  # Still has the typo
   print k
   i = i + 1
```
(Student didn't notice the typo, submitted same code)

Attempt 3 [PASSED]:
```python
n = input()

prev = 1
curr = 1
i = 0

while i < n:
   print prev
   curr = prev + curr
   prev = curr - prev
   i = i + 1
```
(Student rewrote with different approach - gave up debugging the typo)

=== REAL TRACE 2: Reverse Groups (from DCU Python dataset) ===
Task: Read two groups of lines, print second group then first group

Attempt 1 [FAILED]:
```python
lines = []

line = raw_input()
while line != "end":
   lines.append(line)
   line = raw_input()

i = 0
while i < len(lines):
   print lines[len(lines) - i - 1]
   i += 1
```
(Wrong approach - reversing within group, not swapping groups)

Attempt 4 [FAILED]:
```python
#!\\usr\\bin\\env python
2
3 g1 = []
4 g2 = []
5
6 line = raw_input()
7 while line != "end":
8    g1.append(line)
9    line = raw_input()
```
(Student added line numbers by accident - copy-paste error)

Attempt 5 [FAILED]:
```python
#!\\usr\\bin\\env python

g1 = []
g2 = []

line = raw_input()
while line != "end":
    g1.append(line)
   line = raw_input()  # IndentationError - inconsistent spacing
```
(Indentation is all messed up - tabs vs spaces)

Attempt 7 [PASSED]:
```python
g1 = []
g2 = []

line = raw_input()
while line != "end":
    g1.append(line)
    line = raw_input()

line = raw_input()
while line != "end":
   g2.append(line)
   line = raw_input()

i = 0
while i < len(g2):
   print g2[i]
   i = i + 1

i = 0
while i < len(g1):
   print g1[i]
   i = i + 1
```
(Finally fixed indentation after many attempts)

=== REAL TRACE 3: Sum Positive/Negative (from DCU Python dataset) ===
Task: Read numbers until 0, print sum of negatives and sum of positives

Attempt 1 [FAILED]:
```python
total   =0
totalneg=0
numberneg=0

i = 0
number =input()

while number !=0:
   total  = total + number
   number =input()
   if number < 0:
      numberneg = number
      totalneg = totalneg +numberneg
   else:
      total  = total + number    # BUG: adding number twice
   i = i + 1

print totalneg ,total
```
(Logic error: number added to total before the if/else, then added again in else)

Attempt 3 [PASSED]:
```python
conti = True
totalPo = 0
totalNe = 0

while (conti):
	 n = input()
	 if (n < 0): totalNe = totalNe +n
	 else: totalPo = totalPo +n
	 if (n == 0): conti = False

print totalNe, totalPo
```
(Rewrote with cleaner logic)

=== REAL TRACE 4: Simple Addition (from DCU Python dataset) ===
Task: Read two numbers and print their sum

Attempt 1 [FAILED]:
```python
#!/usr/bin/env python

"$1" + "$2"
```
(Student confused Python with Bash shell syntax!)

Attempt 2 [FAILED]:
```python
#!/usr/bin/env python

x=int(raw_input())
y=int(raw_input())

print "x+y"  # BUG: printing string literal instead of expression
```
(Fixed input but now printing literal string "x+y" instead of the sum)

Attempt 3 [PASSED]:
```python
#!/usr/bin/env python

x=int(raw_input())
y=int(raw_input())

print x+y
```
(Removed quotes - finally works)
"""

# =============================================================================
# Prompts - Following LLM-SS Framework
# =============================================================================

SYSTEM_PROMPT = """You are simulating a novice university student learning Python programming.
You are NOT a professional programmer. You are NOT an AI assistant. You are a real student who makes mistakes.

PERFORMANCE LEVEL: {performance_level}

{persona_description}

## CRITICAL: Learn from Real Student Behavior

Below are REAL debugging traces from actual university students learning Python.
Study these carefully - they show how REAL students:
- Make typos and don't notice them (temp vs tmp)
- Confuse syntax from other languages (bash $1 in Python)
- Have messy indentation (mixing tabs and spaces)
- Make logic errors (adding numbers twice)
- Sometimes rewrite from scratch instead of debugging
- Submit the same broken code multiple times before noticing the bug

{real_traces}

## Your Task

You must simulate a student who behaves LIKE the students in these traces:
- Make the SAME types of mistakes (typos, wrong operators, logic errors)
- Show the SAME debugging patterns (not finding bugs immediately, trying random changes)
- Use similar coding style (inconsistent spacing, short variable names, comments like "# idk")
- Sometimes get stuck and try a completely different approach

{extra_instructions}

Remember: A "good" simulation means you struggle realistically like real students, not that you solve efficiently.
"""

USER_PROMPT = """PROBLEM: {problem_description}

CURRENT CODE:
```python
{current_code}
```

LAST OUTPUT:
{last_output}

STEP {step_number} of {max_steps}:

Based on the real student traces above, simulate what a real novice student would do next.
Think and act like the students in the examples - make realistic mistakes, show confusion, and struggle.
"""

# =============================================================================
# Output Models
# =============================================================================


class LLMSSStepOutput(BaseModel):
    """Output structure for an LLM-SS step."""
    thinking: str = Field(
        ...,
        max_length=800,
        description="Your internal monologue as a confused student (2-3 sentences, ~50 words MAX, like 'wait why isnt this working', 'maybe if I try...', 'ugh')."
    )
    action: str = Field(
        ...,
        max_length=50,
        description="One of: CONSTRUCTING, DEBUGGING, or ASSESSING"
    )
    code: str = Field(
        ...,
        max_length=8000,
        description="The complete updated code with realistic novice mistakes, or 'NO_CHANGE'"
    )


@dataclass
class LLMSSStepResult:
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
class LLMSSSimulationResult:
    """Result of a complete LLM-SS simulation."""
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
    baseline_type: str = "llmss"

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# LLM-SS Student Class
# =============================================================================


class LLMSSStudent:
    """
    LLM-SS (LLM for Student Synthesis) inspired baseline.

    Uses REAL student traces from DCU dataset for in-context conditioning.
    Tests whether exemplar-based prompting can match BEAGLE's architectural approach.

    Key difference from vanilla/CoT:
    - Uses actual student debugging trajectories as exemplars
    - Follows peer-reviewed LLM-SS methodology
    - Conditions on real error patterns, not synthetic examples
    """

    def __init__(
        self,
        performance_level: Literal["low", "high"] = "low",
        llm_model: str = "google-gla:gemini-2.0-flash",
    ):
        self.performance_level = performance_level
        self.llm_client = LLMClient(model=llm_model)

        # Build persona based on performance level
        if performance_level == "low":
            self.persona = """As a LOW performer (like the struggling students in the traces), you:
- Get confused easily and make many conceptual errors
- Often try random changes without understanding why
- Make typos and don't notice them for several attempts
- Mix up syntax from different contexts (like using bash in Python)
- Have messy, inconsistent indentation
- Sometimes give up and rewrite from scratch
- Your mindset: "I don't really get this..." or "Maybe if I change this..."
"""
            self.extra_instructions = "Follow the patterns from the FAILED attempts closely. Make REAL mistakes like the students did."
        else:
            self.persona = """As a HIGH performer (more systematic but still a novice), you:
- More organized but still make beginner errors
- Try to understand error messages but sometimes misinterpret them
- Still make typos occasionally
- Better indentation but still inconsistent sometimes
- Debug more systematically but still get stuck
- Your mindset: "Let me think about this..." or "I think I see the problem..."
"""
            self.extra_instructions = "Be more systematic but still make the types of mistakes shown in the traces."

    async def _run_step(
        self,
        problem_description: str,
        current_code: str,
        last_output: str,
        step_number: int,
        max_steps: int,
    ) -> LLMSSStepOutput:
        """Run a single simulation step with real student exemplars."""

        system_prompt = SYSTEM_PROMPT.format(
            performance_level=self.performance_level.upper(),
            persona_description=self.persona,
            real_traces=REAL_STUDENT_TRACES,
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
            output_type=LLMSSStepOutput,
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
        """Run the LLM-SS simulation on a problem."""
        oracle = IDEOracle(save_history=True)

        current_code = ""
        last_output = "(No output yet)"
        history = []
        solved = False

        for step in range(1, max_steps + 1):
            try:
                output = asyncio.get_event_loop().run_until_complete(
                    self._run_step(
                        problem_description=problem_description,
                        current_code=current_code,
                        last_output=last_output,
                        step_number=step,
                        max_steps=max_steps,
                    )
                )

                action = output.action.upper()
                if action not in ["CONSTRUCTING", "DEBUGGING", "ASSESSING"]:
                    if "DEBUG" in action:
                        action = "DEBUGGING"
                    elif "ASSESS" in action or "REFLECT" in action:
                        action = "ASSESSING"
                    else:
                        action = "CONSTRUCTING"

                new_code = output.code
                if new_code and new_code.upper() not in ["NO_CHANGE", "NO CHANGE"]:
                    # Clean up code (remove markdown fences if present)
                    new_code = re.sub(r'^```python\s*', '', new_code.strip())
                    new_code = re.sub(r'\s*```$', '', new_code.strip())
                    current_code = new_code

                # Execute code for DEBUGGING and ASSESSING
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

                step_result = LLMSSStepResult(
                    step=step,
                    action=action,
                    thinking=output.thinking,
                    code=current_code,
                    output=last_output,
                    tests_passed=tests_passed,
                    tests_total=tests_total,
                    success=success,
                )

                step_dict = asdict(step_result)
                # Add fields for evaluator compatibility
                step_dict['cognitive_state'] = action
                step_dict['monologue'] = output.thinking
                history.append(step_dict)

                logger.info(
                    f"  Step {step}: {action} - {'SOLVED!' if success else f'{tests_passed}/{tests_total} tests'}"
                )

                if success:
                    solved = True
                    break

            except Exception as e:
                logger.error(f"Step {step} failed: {e}")
                history.append({
                    "step": step,
                    "error": str(e),
                    "action": "ERROR",
                    "code": current_code,
                    "output": last_output,
                    "success": False,
                })
                continue

        return history


# =============================================================================
# Simulation Runner
# =============================================================================


def run_llmss_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str,
) -> LLMSSSimulationResult:
    """
    Run a single LLM-SS simulation.

    Args:
        run_id: Unique identifier for this run
        performance_level: 'low' or 'high'
        problem_id: Problem identifier
        max_steps: Maximum simulation steps
        llm_model: LLM model to use

    Returns:
        LLMSSSimulationResult with full history
    """
    logger.info(f"Starting LLM-SS run {run_id} ({performance_level} performer)")

    start_time = time.time()

    # Load problem
    problem_def = IDEOracle.load_problem(problem_id)
    problem_description = problem_def.description

    student = LLMSSStudent(
        performance_level=performance_level,
        llm_model=llm_model,
    )

    history = student.solve_problem(
        problem_description=problem_description,
        problem_id=problem_id,
        max_steps=max_steps,
    )

    duration = time.time() - start_time

    # Compute stats
    solved = any(h.get("success", False) for h in history)
    action_counts = {"CONSTRUCTING": 0, "DEBUGGING": 0, "ASSESSING": 0}
    for h in history:
        action = h.get("action", "UNKNOWN")
        if action in action_counts:
            action_counts[action] += 1

    final_step = history[-1] if history else {}

    result = LLMSSSimulationResult(
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
        baseline_type="llmss",
    )

    logger.info(
        f"  Run {run_id} complete: {'SOLVED' if solved else 'NOT SOLVED'} in {result.total_steps} steps ({duration:.1f}s)"
    )

    return result


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Running LLM-SS baseline test...")
    result = run_llmss_simulation(
        run_id=1,
        performance_level="low",
        problem_id="particle_simulator",
        max_steps=10,
        llm_model="google-gla:gemini-2.5-flash",
    )

    print(f"\nResult: {'SOLVED' if result.solved else 'NOT SOLVED'} in {result.total_steps} steps")
    print(f"Actions: {result.action_counts}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    # Show first few steps
    print("\nFirst 3 steps:")
    for step in result.history[:3]:
        print(f"  Step {step.get('step')}: {step.get('action')}")
        print(f"    Thinking: {step.get('thinking', '')[:100]}...")
